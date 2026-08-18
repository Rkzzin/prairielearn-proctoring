# Configurar o dashboard do zero (ex: EC2)

Passo a passo para levar uma máquina Linux sem nada instalado até rodar
`proctor-dashboard.service`. Ver `docs/roles.md` para o que diferencia este
papel do papel de estação (`docs/setup_nuc.md`) — em resumo, o dashboard não
precisa de GNOME, X11, câmera nem Chromium; é só um servidor web.

## 1. Provisionar a máquina (manual)

- Ubuntu 24.04 ou Amazon Linux 2023 (Server serve — não precisa de sessão
  gráfica). O bootstrap do passo 3 detecta sozinho qual é (`apt` vs `dnf`);
  o passo 4 (Postgres) tem os dois comandos, use o que bater com a sua.
- A app do dashboard escuta só em `127.0.0.1:8010` (ver passo 6) — quem
  precisa de porta pública aberta no security group/firewall é o nginx na
  frente dela (`80/443/tcp`), não a app diretamente. Se este for um teste
  isolado sem nginx (`PROCTOR_DASHBOARD_HOST=0.0.0.0` no passo 6), aí sim
  abra a porta escolhida.
- Como a autenticação é por credencial (Basic Auth do professor, token por
  estação — ver passo 8), não é obrigatório restringir por IP de origem, mas
  reduz superfície se você souber de antemão a faixa de rede das NUCs.

## 2. Clonar o repositório (manual)

```bash
git clone <url-do-repo> ~/proctor-station
cd ~/proctor-station
```

## 3. Rodar o bootstrap do dashboard (script)

```bash
chmod +x scripts/bootstrap_dashboard.sh
./scripts/bootstrap_dashboard.sh
```

Instala só o necessário para o papel de dashboard (build tools, Python 3.12,
bibliotecas numéricas — `dlib` ainda compila aqui, ver `docs/roles.md` sobre o
porquê), cria o venv, instala `pip install -e ".[dashboard,dev]"`, cria o
`.env` a partir do `.env.example` (se não existir) e roda a suíte de testes.

## 4. Provisionar o Postgres (manual)

O dashboard persiste tudo (estações, sessões, enrollments, configs,
credenciais) em Postgres — não sobe sem um banco acessível. Numa máquina só
para o dashboard, o mais simples é instalar nativo.

**Ubuntu/Debian (`apt`)** — instala, inicializa e já sobe o serviço sozinho:

```bash
sudo apt update && sudo apt install -y postgresql
```

**Amazon Linux/RHEL (`dnf`)** — precisa inicializar o cluster e habilitar o
serviço manualmente (o `apt` faz isso por trás; o `dnf` não). O nome exato do
pacote varia por versão disponível no repositório da máquina — confira antes
com `dnf list available 'postgresql*-server'` se o comando abaixo falhar:

```bash
sudo dnf install -y postgresql15 postgresql15-server
sudo postgresql-setup --initdb
sudo systemctl enable --now postgresql
```

Se `systemctl enable --now postgresql` reclamar que a unit não existe, rode
`systemctl list-units --type=service | grep postgres` pra achar o nome real
do serviço (pode vir versionado, ex: `postgresql-15`) e use esse.

**Nos dois casos**, criar o banco/role exclusivos do dashboard:

```bash
sudo -u postgres psql -c "CREATE ROLE proctor_dashboard WITH LOGIN PASSWORD '<senha-forte>';"
sudo -u postgres psql -c "CREATE DATABASE proctor_dashboard OWNER proctor_dashboard;"
```

O DSN resultante (`postgresql://proctor_dashboard:<senha>@127.0.0.1:5432/proctor_dashboard`)
vai no `.env` no próximo passo. As tabelas são criadas automaticamente no
primeiro boot do serviço — não precisa rodar migração manual.

Se a EC2 já roda outro serviço (dividindo a máquina), ver
`docs/migracao_ec2_passo_a_passo_aws.md` — lá o Postgres também é nativo na
mesma instância, mas com um banco/role isolado do outro serviço.

Para desenvolvimento local, `docker-compose.yml` na raiz do repo sobe um
Postgres descartável (`docker compose up -d`), já compatível com o exemplo
de DSN no `.env.example`.

## 5. Preencher o `.env` à mão (manual)

Os campos que importam para o dashboard (o resto do `.env.example` é herança
compartilhada com a estação e não faz efeito aqui — ver `docs/roles.md`):

```dotenv
# Credenciais AWS — geram as URLs pré-assinadas das gravações e alimentam o
# enrollment via S3 (/enrollment no painel)
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=

PROCTOR_S3_BUCKET=proctor-station
PROCTOR_S3_REGION=sa-east-1

# OBRIGATÓRIO — sem isto o dashboard nem sobe (falha no boot com erro
# explícito). DSN do banco criado no passo 4.
PROCTOR_DASHBOARD_DATABASE_URL=postgresql://proctor_dashboard:<senha>@127.0.0.1:5432/proctor_dashboard

# OBRIGATÓRIO — sem isto o painel fica acessível sem senha para qualquer um
# que descubra o IP/porta. É só o login do professor — as NUCs usam um
# token próprio (passo 8), não esta senha.
PROCTOR_DASHBOARD_ADMIN_USER=professor
PROCTOR_DASHBOARD_ADMIN_PASSWORD=<escolha uma senha forte>
```

A senha é hasheada (PBKDF2) e gravada no Postgres do dashboard **no primeiro
boot do serviço**. Depois disso, `PROCTOR_DASHBOARD_ADMIN_PASSWORD` pode ser
removida do `.env` — só o hash no banco importa a partir daí. Trocar a senha
depois exige apagar a linha correspondente na tabela `credentials` (ex:
`DELETE FROM credentials WHERE username = 'professor';` via `psql`) e
reiniciar o serviço (não existe rota de "trocar senha" na UI ainda).

## 6. Instalar o serviço systemd (script)

```bash
cd ~/proctor-station   # sempre cd antes — o script resolve import a partir do cwd
sudo bash scripts/install_dashboard_service.sh
```

Isso cria e habilita `proctor-dashboard.service` escutando só em
`127.0.0.1:8010` por padrão — pensado para rodar atrás de um nginx (mesma
máquina ou compartilhado com outro serviço, ver
`docs/migracao_ec2_passo_a_passo_aws.md`) que termina TLS e faz proxy pra
essa porta interna; `scripts/nginx/proctor-dashboard.conf` tem o template do
server block. `PROCTOR_DASHBOARD_PORT`/`PROCTOR_DASHBOARD_HOST` no shell
mudam isso se precisar (ex: `sudo env PROCTOR_DASHBOARD_HOST=0.0.0.0 bash
scripts/install_dashboard_service.sh` pra expor direto, sem nginx, num teste
rápido — não use isso em produção sem TLS na frente).

## 7. Verificação (script)

```bash
systemctl status proctor-dashboard --no-pager
curl -sS -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8010/       # espera 401 sem credencial
curl -sS -o /dev/null -w '%{http_code}\n' -u professor:<senha> http://127.0.0.1:8010/   # espera 200
```

Isso confirma que a app subiu — mas ela só escuta local. De fora da máquina,
quem confirma que as NUCs vão alcançar o painel é o nginx na frente (ver
`docs/migracao_ec2_passo_a_passo_aws.md`, passos 6-7), não a app diretamente:

```bash
curl -sS -o /dev/null -w '%{http_code}\n' https://<subdominio-do-dashboard>/
```

## 8. Emitir um token por estação e apontar as NUCs

A senha do professor (`PROCTOR_DASHBOARD_ADMIN_USER/PASSWORD`) não autentica
mais o heartbeat — cada NUC tem seu próprio token, emitido aqui no dashboard:

```bash
cd ~/proctor-station
venv/bin/python scripts/issue_station_token.py nuc-01 --label "Sala 3, estação 1"
```

O token é impresso em texto puro **uma única vez** — copiar imediatamente,
não fica recuperável depois (só o hash fica no Postgres). Repetir o comando
pro mesmo `station_id` revoga o token anterior e emite um novo. Repetir para
cada NUC.

Em cada NUC (ver `docs/setup_nuc.md`, passo 4), `.env`:

```dotenv
PROCTOR_DASHBOARD_ENABLED=true
PROCTOR_DASHBOARD_BASE_URL=https://<subdominio-do-dashboard>
PROCTOR_DASHBOARD_STATION_ID=nuc-01
PROCTOR_DASHBOARD_STATION_TOKEN=<o token impresso acima>
```

`PROCTOR_DASHBOARD_STATION_ID` precisa bater exatamente com o `station_id`
usado ao emitir o token — o dashboard rejeita se um divergir do outro. Sem
token (ou com o errado), o heartbeat de saída recebe 401 do dashboard e a
estação nunca aparece no painel — não é um erro silencioso, mas aparece só
no log (`journalctl -u proctor -f`), não na UI.

## Risco aceito por enquanto

Se você não estiver rodando atrás do nginx do passo 6 (ex: teste isolado com
`PROCTOR_DASHBOARD_HOST=0.0.0.0`), o tráfego entre NUC e dashboard
(heartbeat, comandos, token de estação) roda em HTTP puro, sem TLS — o token
viaja em texto claro dentro do header `X-Station-Token`. Aceitável só para
um teste pontual numa rede confiável; para uso continuado, o nginx +
certificado (Let's Encrypt) do passo 6 é obrigatório, não opcional.
