# Migração do dashboard para EC2 compartilhada — plano de dev (repositório)

Este é o plano do que muda **no código deste repositório** para o dashboard
rodar atrás do nginx compartilhado na EC2. O passo a passo de infraestrutura
(AWS, SSH, nginx, systemd na máquina) está em
`docs/migracao_ec2_passo_a_passo_aws.md` — este documento aqui só cobre o que
sai como commit.

Ponto de partida: o `.md` original do colega presumia que o dashboard ainda
rodava localmente em cada NUC e precisava ganhar heartbeat pull, multi-estação
e autenticação de estação do zero. Não é o caso — auditei o repo antes de
escrever este plano. O heartbeat pull (NUC sempre inicia a conversa, comandos
voltam na resposta do próprio heartbeat) **já existe e já bate com o que o
colega propôs**; é inclusive uma mudança em relação ao design original do
projeto (`README.md` de abril, "Fase 6"), que previa o dashboard chamando a
NUC de volta (`POST /config`, `POST /session/stop`) — modelo que só funciona
na mesma LAN. Isso já foi corrigido antes desta migração entrar em pauta, ver
`src/dashboard/app.py:173` (`POST /api/heartbeats`) e
`DashboardStore.drain_commands` em `src/dashboard/store.py`.

Também foi decidido manter **um repositório só** (não separar o dashboard),
seguindo a razão já registrada em `docs/roles.md`: `dlib`/`opencv` continuam
necessários nos dois papéis (o enrollment por S3 roda reconhecimento facial no
próprio dashboard) e separar custaria sincronizar o contrato de heartbeat
entre dois repos sem ganho real hoje.

## O que já existe (não é trabalho deste plano)

- Heartbeat pull NUC → dashboard: `POST /api/heartbeats`
  (`src/dashboard/app.py:173`, cliente em `src/core/dashboard_sync.py`).
- Multi-estação: `StationRecord` por `station_id`, comandos enfileirados
  (`APPLY_CONFIG`, `SET_AUTOSTART`, `STOP_SESSION`, `UNBLOCK_SESSION`) em
  `src/dashboard/store.py`.
- URLs pré-assinadas S3 para o player de vídeo
  (`DashboardStore._hydrate_asset`, `src/dashboard/store.py:465`).
- Auth do professor (Basic Auth, usuário/senha único, hash PBKDF2 no banco)
  — **isso não muda neste plano**, ver item 4.
- systemd unit gerado dinamicamente por `scripts/install_dashboard_service.sh`
  e bootstrap dedicado (`scripts/bootstrap_dashboard.sh`).
- Runbook de instalação em máquina genérica: `docs/setup_dashboard.md`.

Nada disso precisa ser reescrito do zero. O que muda de verdade: trocar o
storage para Postgres e trocar a credencial de estação compartilhada por
token individual — ambos pedidos explicitamente — mais as duas lacunas de
infra (porta interna, nginx) já identificadas antes.

## Mudanças

### 1. Parar de expor a porta da app diretamente — nginx assume 80/443

Hoje `scripts/install_dashboard_service.sh` usa `--host 0.0.0.0` e porta `80`
por padrão (com `CAP_NET_BIND_SERVICE`). Numa EC2 compartilhada, isso colide
com o nginx que já ocupa 80/443 para o outro serviço. Mudar para:

- Bind em `127.0.0.1` (a app não precisa mais ser alcançável direto de fora).
- Porta interna não-privilegiada — reaproveitar o default que já existe em
  `DashboardConfig.base_url` (`src/core/config.py:245`): `8010`. Já é o valor
  default do `.env.example`, só nunca foi usado como porta real de bind no
  script de instalação.
- Remover `AmbientCapabilities=CAP_NET_BIND_SERVICE`/`CapabilityBoundingSet`
  do unit gerado (não é mais porta <1024).

Arquivos: `scripts/install_dashboard_service.sh` (trocar `PORT` default de
`80` para `8010`, trocar `--host 0.0.0.0` para `--host 127.0.0.1`).

Também corrigir a deriva entre esse script e o unit estático versionado em
`systemd/proctor-dashboard.service` (hoje ele já usa porta `8010`, mas
`WorkingDirectory=/opt/proctor` não bate com o caminho que
`docs/setup_dashboard.md` manda clonar — `~/proctor-station`). Alinhar os
dois ou deixar claro no arquivo que é só exemplo de referência, não gerado
pelo script.

### 2. Bloco nginx do subdomínio (com suporte a WebSocket)

Criar `scripts/nginx/proctor-dashboard.conf` como template versionado
(o operador copia pra `/etc/nginx/sites-available/` na EC2 — passo manual,
ver o doc de AWS). Conteúdo mínimo:

- `server_name` no subdomínio (ex: `proctoring.dominio.com.br`).
- `proxy_pass http://127.0.0.1:8010;`.
- **Importante e fácil de esquecer:** `/ws/stations` é WebSocket
  (`src/dashboard/app.py:276`, atualiza o dashboard em tempo real). Sem
  `proxy_http_version 1.1;` + `proxy_set_header Upgrade $http_upgrade;` +
  `proxy_set_header Connection "upgrade";` no bloco, o upgrade falha
  silenciosamente e o painel para de atualizar em tempo real, caindo pra
  nada (não há fallback de polling implementado hoje) — vale um teste manual
  específico pra isso depois do deploy.
- `proxy_set_header Host $host;` e `X-Forwarded-For` para logs/IP corretos.

TLS (certbot) é configurado por fora do bloco (o certbot edita o server
block); não precisa ir no template do repo. **Não é opcional** — o próprio
`docs/setup_dashboard.md` ("Risco aceito por enquanto") já registra que
Basic Auth em HTTP puro expõe a senha do professor em texto claro, e isso só
era aceitável em rede local isolada. Expondo o subdomínio na internet
pública, TLS passa de "resolve depois" para pré-requisito do cutover — sem
ele, tanto a senha do professor quanto o token de estação (item 4) trafegam
em claro. Sequenciamento certo no doc de AWS: nginx sobe e certbot roda na
mesma janela, sem ninguém logar ou apontar NUC pro subdomínio entre um passo
e outro.

### 3. Trocar SQLite por Postgres

Minha recomendação anterior era manter SQLite (menor pegada de RAM). Você
pediu pra trocar por Postgres — troquei. Vale registrar o trade-off que isso
reabre: Postgres nativo consome mais RAM de base que o SQLite embutido, então
o passo 1 do doc de AWS (checar `free -h`) fica mais decisivo — numa
t3.micro compartilhada, é bem provável que isso empurre pra t3.small. Deixei
essa nota reforçada no doc de AWS.

**Onde roda:** Postgres nativo na própria EC2 (não RDS) — mantém a decisão
de "sem infraestrutura AWS gerenciada nova" desta rodada de migração, que é
manual/portal, sem Terraform. RDS free-tier fica como opção futura se quiser
backups gerenciados; não é bloqueador agora.

**Mudanças no código:**

- `pyproject.toml`: adicionar `psycopg[binary]>=3.1` ao extra `dashboard`
  (só o dashboard fala com o banco; a estação não precisa desse driver).
- `src/core/config.py`: novo campo em `DashboardConfig` (ou em `AppConfig`,
  a decidir na hora), `database_url` — DSN completo via env var
  `PROCTOR_DASHBOARD_DATABASE_URL`
  (ex: `postgresql://proctor_dashboard:<senha>@127.0.0.1:5432/proctor_dashboard`).
- `src/dashboard/store.py`: `DashboardStore` troca `sqlite3.connect` por
  `psycopg.connect(dsn)`. A troca é só na camada SQL, **não** numa reescrita
  de arquitetura — o padrão atual (lock em memória + linha por registro como
  JSON) continua igual, só muda o dialeto:
  - `?` → `%s` nos placeholders.
  - Colunas `payload TEXT` → `payload JSONB` (Postgres nativo; abre a porta
    pra consultar campos específicos no futuro sem quebrar nada hoje).
  - `INTEGER PRIMARY KEY AUTOINCREMENT` (tabela `configs`) → `BIGSERIAL
    PRIMARY KEY`.
  - `INSERT OR REPLACE` → `INSERT ... ON CONFLICT (id) DO UPDATE SET ...`.
  - `INSERT OR IGNORE` (credenciais) → `INSERT ... ON CONFLICT DO NOTHING`.
  - A tabela nova de tokens de estação (item 4) já nasce nesse schema
    Postgres — não vale a pena criar em SQLite primeiro.
- Sem dado real de produção hoje (nada foi ao ar na EC2 ainda), então não
  precisa de script de migração de dados — o schema nasce direto no
  Postgres.

**Testes — o custo real dessa troca:** hoje `tests/test_dashboard.py`
instancia `DashboardStore(tmp_path / "dashboard")` e ganha um arquivo SQLite
isolado de graça a cada teste, sem infraestrutura nenhuma. Isso acaba com
Postgres — os testes passam a precisar de um Postgres de verdade rodando.

**Implementado:** fixture `dashboard_database_url` em `tests/conftest.py`
conecta num Postgres local/container e faz `CREATE DATABASE test_<uuid>` por
teste (banco inteiro, não schema — mais simples que isolar por
`search_path` e evita depender de codificação de DSN via query string),
dropando com `WITH (FORCE)` no fim (funciona mesmo se a conexão do teste
ainda estiver aberta). `docker-compose.yml` novo sobe um `postgres:16` local.
Validado rodando a suíte inteira contra um Postgres real via Docker: 151
passando, 1 falha pré-existente sem relação (`test_recorder.py`, confirmada
também em `main` sem essas mudanças). Ainda falta, se houver CI configurado
(não há workflow no repo hoje): declarar `services: postgres:` no job de
teste — sem isso o CI quebra assim que este PR for integrado lá.

### 4. Autenticação por token individual de estação

Você confirmou a ideia — aqui vai o desenho completo e como ela convive com
o login do professor.

**Login do professor não muda.** Continua Basic Auth
(`PROCTOR_DASHBOARD_ADMIN_USER`/`PASSWORD`), hash PBKDF2 gravado no banco no
primeiro boot, exatamente como hoje (`src/dashboard/auth.py`,
`src/dashboard/app.py:56`). Essa credencial passa a valer **só** para as
rotas de navegador/administração: `/`, `/config`, `/enrollment`,
`/sessions/{id}`, `/partials/*`, `/api/reports/*`, `/api/enrollment/*`,
`/ws/stations`, `/api/configs`, `/api/stations/{id}/*` (os comandos que o
professor dispara pelo painel).

**O que muda é a autenticação das NUCs.** Hoje elas usam a mesma senha do
professor. Passa a ser: cada NUC tem seu próprio token, gerado na hora do
provisionamento, guardado só como hash no banco (igual à senha do professor
hoje) — se o `.env` de uma NUC vazar, dá acesso só àquela estação, não ao
painel inteiro nem às outras NUCs.

**Desenho:**

- Nova tabela `station_tokens` (station_id PK, token_hash, created_at,
  label opcional) — vive no `DashboardStore`, mesmo padrão de hash (PBKDF2)
  já usado pra credencial do professor em `src/dashboard/auth.py`.
- Cada chamada da NUC ao dashboard carrega dois headers dedicados, setados
  uma vez no `httpx.Client` (`_default_client_factory` em
  `src/core/dashboard_sync.py`), não no Basic Auth: `X-Station-Id` e
  `X-Station-Token`.
- No dashboard, uma dependency nova (`require_station_token`, não o
  middleware global de Basic Auth) protege só as quatro rotas que hoje só a
  NUC chama: `POST /api/heartbeats`, `POST /api/sessions`,
  `POST /api/sessions/{id}/finalize`, `POST /api/sessions/{id}/events`. Ela
  lê os dois headers, busca o hash daquele `station_id` em `station_tokens`,
  verifica — 401 se não bater.
- **Checagem extra que vale a pena:** o corpo dessas rotas também carrega
  `station_id` (`StationHeartbeat.station_id`, `SessionRecord.station_id`).
  A dependency deve conferir que o `station_id` do header bate com o do
  corpo, e rejeitar se não bater — sem isso, uma estação autenticada
  corretamente ainda poderia declarar `station_id` de outra estação no
  corpo e "falar" por ela no store.
- `require_basic_auth` (middleware global atual) passa a ignorar
  explicitamente essas quatro rotas — hoje ele intercepta tudo.

**Provisionamento (como o token nasce):** diferente da senha do professor,
que se autopreenche no primeiro boot a partir do `.env`
(`app.py:53`), o token de estação **não** tem bootstrap automático — é
provisionado explicitamente antes da NUC ir pra produção, porque é segredo
por máquina, não algo pra repetir num `.env.example` compartilhado.

- Novo script `scripts/issue_station_token.py <station_id> [--label NOME]`:
  gera um token aleatório (`secrets.token_urlsafe`), grava o hash em
  `station_tokens` via `DashboardStore`, imprime o token em texto puro **uma
  vez** (o operador copia pro `.env` da NUC correspondente). Roda na própria
  EC2, com acesso ao Postgres.
- Re-executar pro mesmo `station_id` sobrescreve o hash — é como se revoga e
  roda um token (o antigo para de funcionar imediatamente).
- Fica como melhoria futura (não neste plano): um endpoint
  `POST /api/stations/{id}/token` protegido por Basic Auth do professor, pra
  gerar/rotacionar token pelo próprio painel sem precisar SSH na EC2.

**Arquivos:** `src/dashboard/store.py` (tabela + hash/verify),
`src/dashboard/app.py` (dependency nova, escopo do middleware), 
`src/core/config.py` (`station_token` no lado NUC de `DashboardConfig`),
`src/core/dashboard_sync.py` (headers no client), `scripts/issue_station_token.py`
(novo), `.env.example`, `docs/setup_dashboard.md`, `docs/setup_nuc.md`,
`docs/roles.md` (tabela de `.env` por papel).

**Testes:** estender `tests/test_dashboard_auth.py` e `tests/test_dashboard.py`
— heartbeat com token da própria estação passa; com token de outra estação
ou nenhum token, 401; Basic Auth do professor **não** autentica mais o
heartbeat (era o comportamento atual, muda); token não autentica rotas de
navegador; `station_id` do header divergente do corpo é rejeitado.

### 5. Atualizar `MILESTONES.md` com checklist da migração e registro de decisões

Adicionar seção nova (ex: `## M8 — Migração do dashboard para EC2
compartilhada 🟡`) espelhando os passos deste plano e do doc de AWS como
checklist `- [ ]`, no mesmo formato usado em M7. Fecha também o item já
pendente em M7 ("Atualizar Terraform para (...) EC2 dashboard") com nota de
que esta rodada é manual (via portal AWS) por decisão do time, IaC fica pra
depois.

Adicionar na tabela "Registro de decisões técnicas" (`MILESTONES.md:544`):

| Milestone | Decisão | Motivo |
|---|---|---|
| Migração EC2 | Postgres em vez de SQLite no dashboard | Prioridade sobre a menor pegada de RAM do SQLite; ganha backup/gerenciamento mais formal e caminho aberto pra RDS depois. Custo aceito: instância provavelmente precisa subir pra t3.small, e testes passam a precisar de um Postgres disponível (local ou CI) |
| Migração EC2 | Token individual por estação em vez de credencial única compartilhada | Uma NUC comprometida ou com `.env` vazado só compromete aquela estação — não o painel do professor nem as demais NUCs |

## Fora de escopo deste plano (explicitamente)

- Terraform/Ansible para a EC2 do dashboard — item já listado como pendente
  em `MILESTONES.md`, não faz parte desta migração pontual.
- RDS gerenciado — Postgres nativo na EC2 por agora (ver item 3); RDS fica
  como evolução futura, não bloqueador.
- Endpoint no painel pra rotacionar token de estação pela UI — só o script
  CLI por enquanto (ver item 4).
- Separar o dashboard em outro repositório — decisão tomada, ver
  `docs/roles.md` e o topo deste documento.

## Ordem sugerida de execução (PRs)

1. **Postgres** (item 3) — é a base: a tabela de tokens do item 4 já nasce
   nesse schema, evita criar em SQLite e reescrever depois.
2. **Token por estação** (item 4) — depende do PR 1.
3. **Porta interna + nginx template** (itens 1 e 2) — independente dos
   dois primeiros, mas só faz sentido habilitar o deploy depois deles.
4. **Docs + `MILESTONES.md`** (item 5) — fecha o registro, referencia os
   PRs anteriores.

Só depois de 1–3 mergeados o passo a passo em
`docs/migracao_ec2_passo_a_passo_aws.md` pode ser seguido de ponta a ponta.
