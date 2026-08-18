# Migração do dashboard para EC2 compartilhada — passo a passo do operador

Passo a passo manual (console AWS + SSH na instância) para colocar o
dashboard no ar na EC2 já existente, compartilhada com outro serviço, atrás
de nginx e num subdomínio próprio, com Postgres e token individual por
estação. Pré-requisito: os PRs descritos em `docs/migracao_ec2_plano_dev.md`
já mergeados (Postgres, token de estação, porta interna) — os passos 5 em
diante abaixo não fazem sentido sem eles.

Este documento assume que `docs/setup_dashboard.md` já existe como runbook
genérico de instalação — aqui só entram os passos **específicos de dividir a
EC2 com outro serviço**: nginx compartilhado, subdomínio, DNS, capacidade,
Postgres e emissão de token por estação.

---

## 1. Validar capacidade da EC2 atual (console AWS + SSH)

```bash
free -h
df -h
nproc
```

A instância é pequena (t3.micro/t3.small) e já roda outro serviço. Com a
troca de storage pra Postgres (decisão registrada no plano de dev, item 3),
essa checagem fica mais decisiva do que quando o plano ainda previa SQLite:
Postgres sozinho, mesmo com `shared_buffers` reduzido, costuma consumir bem
mais que os poucos MB do SQLite embutido, e ainda soma com FastAPI +
`dlib`/`opencv` do enrollment sob demanda. **Expectativa realista numa
t3.micro (1GB) compartilhada: provavelmente vai apertar.**

- **Console AWS → EC2 → Instances → selecionar a instância → Instance state
  → Stop.** Depois **Actions → Instance settings → Change instance type** →
  `t3.small`. **Start** de novo. (Isso tira a instância do ar por alguns
  minutos — coordenar com quem depende do outro serviço que já roda ali.)
- Se quiser tentar manter t3.micro mesmo assim: instalar Postgres com
  `shared_buffers` baixo (ex: 32-64MB) e acompanhar `free -h` de perto nas
  primeiras provas reais antes de confiar nisso em produção.

## 2. Security group (console AWS)

**EC2 → Instances → instância → aba Security → Security groups → editar
inbound rules:**

- Confirmar que `80/tcp` e `443/tcp` já estão liberados (o outro serviço
  provavelmente já abriu isso para o nginx compartilhado — não duplicar
  regra).
- **Não** abrir a porta `8010` (nem nenhuma outra porta da app) para fora —
  ela passa a ouvir só em `127.0.0.1`, o nginx é o único ponto de entrada
  público. Se `8010` já estava aberta de testes antigos, remover a regra.
- **Não** abrir a porta do Postgres (`5432`) para fora — ele também escuta
  só em `127.0.0.1` (mesmo host, sem acesso externo; nem a app roda em
  máquina separada, então não há motivo pra expor).
- Deixar a porta usada pelo outro serviço como já está.

## 3. DNS do subdomínio (mesma zona do domínio que já existe)

A EC2 já responde por um domínio próprio com HTTPS (o outro serviço já
configurou isso) — não é preciso montar zona/DNS/certificado do zero, só
**estender** o que já existe com um subdomínio novo pro dashboard. Onde
quer que esse domínio esteja registrado (Route 53, Registro.br, Cloudflare
etc.), criar um registro novo na mesma zona:

- Tipo `A` (ou `CNAME` se preferir apontar para um DNS público da EC2 em vez
  do IP).
- Nome: o subdomínio escolhido (ex: `proctoring`, resultando em
  `proctoring.dominio.com.br` — precisa ser diferente do domínio/subdomínio
  já usado pelo outro serviço).
- Valor: o mesmo IP público (ou Elastic IP) que o domínio existente já
  aponta — é a mesma EC2. Confirme em **EC2 → Instances → coluna "Public
  IPv4 address"**; se não houver Elastic IP associado e a instância for
  reiniciada, o IP muda e todo DNS que aponta pra ela quebra, não só o novo
  — vale checar isso antes de seguir (**EC2 → Elastic IPs**).

Se o domínio estiver no Route 53: **Route 53 → Hosted zones → zona do
domínio → Create record.** Em outro provedor, o equivalente é criar o mesmo
registro A/CNAME no painel de lá.

Confirmar propagação antes de seguir:

```bash
dig +short proctoring.dominio.com.br
```

## 4. Instalar e configurar Postgres (SSH na EC2)

Comando de instalação depende da distro da EC2 — confirme com
`cat /etc/os-release` se não tiver certeza.

**Ubuntu/Debian (`apt`)** — instala, inicializa e já sobe o serviço:

```bash
sudo apt update
sudo apt install -y postgresql
```

**Amazon Linux/RHEL (`dnf`)** — precisa inicializar o cluster e habilitar o
serviço à mão (o nome exato do pacote pode variar; confira com
`dnf list available 'postgresql*-server'` se o comando abaixo falhar):

```bash
sudo dnf install -y postgresql15 postgresql15-server
sudo postgresql-setup --initdb
sudo systemctl enable --now postgresql
```

Se `systemctl enable --now postgresql` reclamar que a unit não existe, rode
`systemctl list-units --type=service | grep postgres` pra achar o nome real
(pode vir versionado, ex: `postgresql-15`) — use esse nome nos comandos
`systemctl` daqui pra frente.

**Nos dois casos**, criar banco e role exclusivos do dashboard — **sem
overlap** de schema com o outro serviço que já roda na mesma instância:

```bash
sudo -u postgres psql -c "CREATE ROLE proctor_dashboard WITH LOGIN PASSWORD '<senha-forte>';"
sudo -u postgres psql -c "CREATE DATABASE proctor_dashboard OWNER proctor_dashboard;"
```

Se a instância ficou em t3.micro (passo 1), reduzir o consumo de base do
Postgres. O caminho do `postgresql.conf` muda por distro/empacotamento —
em vez de adivinhar, pergunte ao próprio Postgres:

```bash
sudo -u postgres psql -c "SHOW config_file;"
```

Editar `shared_buffers = 32MB` nesse arquivo, depois:

```bash
sudo systemctl restart postgresql   # ou o nome da unit achado acima
```

O DSN resultante (`postgresql://proctor_dashboard:<senha>@127.0.0.1:5432/proctor_dashboard`)
vai para o `.env` do dashboard no próximo passo — **não** para o `.env` de
nenhuma NUC.

## 5. Deploy da aplicação (SSH na EC2)

Seguir `docs/setup_dashboard.md` passos 2–4 (clonar repo,
`bootstrap_dashboard.sh`, preencher `.env`) normalmente, com estas diferenças
em relação ao doc genérico:

- No `.env`, preencher `PROCTOR_DASHBOARD_DATABASE_URL` com o DSN do passo 4.
- `PROCTOR_DASHBOARD_ADMIN_USER/PASSWORD` continuam preenchidos normalmente
  — é só o login do professor, não muda com a migração.
- **Não** preencher nenhuma credencial de estação aqui — token é por NUC,
  emitido no passo 8, não fica no `.env` do dashboard.
- Instalar o serviço já na porta interna, **não** na 80:
  ```bash
  cd ~/proctor-station
  sudo env PROCTOR_DASHBOARD_PORT=8010 bash scripts/install_dashboard_service.sh
  ```
  (depois do PR do item 1 do plano de dev, isso já deve ser o default —
  passar explícito aqui é só cinto e suspensório.)

Verificar que a app só escuta local, não pública, e que subiu conectada ao
Postgres:

```bash
sudo ss -tlnp | grep 8010          # deve mostrar 127.0.0.1:8010, não 0.0.0.0:8010
curl -sS -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8010/   # espera 401
sudo journalctl -u proctor-dashboard -n 50 --no-pager   # sem erro de conexão ao Postgres
```

## 6. Nginx: novo server block (SSH na EC2)

O nginx já está instalado e rodando nessa EC2 pelo outro serviço — nada a
instalar aqui, só **adicionar** um server block novo ao lado do que já
existe (mesmo processo nginx, dois domínios). Ainda assim, o subdomínio
**novo** (`proctoring.dominio.com.br`) começa sem certificado próprio até o
passo 7 rodar — o domínio existente já ter HTTPS não estende
automaticamente pro subdomínio novo, cada nome precisa do seu certificado.

**Por isso, fazer este passo e o passo 7 (TLS) na mesma janela, sem
interrupção.** Entre um e outro, o subdomínio novo fica servindo em HTTP
puro — Basic Auth do professor é só base64 (não é criptografia) e o token
de estação (passo 8) também trafega em claro num header; qualquer um
capturando o tráfego nessa janela decodifica os dois numa linha. Não fazer
login como professor, não emitir/usar token de estação e não apontar
nenhuma NUC pro subdomínio novo antes do passo 7 estar confirmado.

Antes de copiar o arquivo, confirme **onde** o nginx existente espera
server blocks — `sites-available`/`sites-enabled` (convenção Debian/Ubuntu)
e `conf.d/` (convenção Amazon Linux/RHEL) não são a mesma coisa, e colocar o
arquivo no lugar errado não dá erro nenhum — o `nginx -t` passa, só que o
bloco novo nunca é carregado:

```bash
sudo grep -i include /etc/nginx/nginx.conf
```

- Se aparecer algo como `include /etc/nginx/sites-enabled/*;` → use
  `sites-available`/`sites-enabled` (comandos abaixo já são esse caso).
- Se aparecer `include /etc/nginx/conf.d/*.conf;` (mais comum em Amazon
  Linux) → copie direto para `/etc/nginx/conf.d/proctor-dashboard.conf`,
  sem symlink nenhum.

```bash
# Debian/Ubuntu (sites-available/sites-enabled):
sudo cp ~/proctor-station/scripts/nginx/proctor-dashboard.conf \
  /etc/nginx/sites-available/proctor-dashboard
sudo ln -s /etc/nginx/sites-available/proctor-dashboard \
  /etc/nginx/sites-enabled/proctor-dashboard

# Amazon Linux/RHEL (conf.d) — use este OU o de cima, não os dois:
sudo cp ~/proctor-station/scripts/nginx/proctor-dashboard.conf \
  /etc/nginx/conf.d/proctor-dashboard.conf
```

Editar `server_name` no arquivo copiado para o subdomínio real, se o
template não vier preenchido.

**Sempre testar antes de recarregar** — um erro aqui derruba os dois
domínios juntos, já que o processo nginx é compartilhado com o outro
serviço:

```bash
sudo nginx -t
sudo systemctl reload nginx
```

## 7. TLS (certbot, SSH na EC2)

O certbot também já está instalado e configurado nessa EC2 (é o que gera o
HTTPS que o outro domínio já usa) — não é preciso instalar nem configurar
nada novo, só pedir mais um certificado, para o subdomínio novo, com a
mesma ferramenta já provada em produção:

```bash
sudo certbot --nginx -d proctoring.dominio.com.br
```

**Escolher a opção de redirecionar HTTP → HTTPS quando o certbot
perguntar — não pular essa opção.** É ela que fecha de vez a janela em
claro do passo 6: depois disso, qualquer requisição em `http://` vira
redirect para `https://` automaticamente, então mesmo um cliente mal
configurado (ex: uma NUC com `PROCTOR_DASHBOARD_BASE_URL=http://...` por
engano) não consegue mais mandar credencial em claro sem perceber.

Confirmar que isso **não** mexeu no certificado do outro domínio (certbot
lista os certificados existentes; `sudo certbot certificates` mostra os
dois separados). Renovação automática já vem configurada por padrão em
instalações recentes do certbot (`systemctl status certbot.timer`).

Só depois de confirmar `https://proctoring.dominio.com.br/` respondendo com
certificado válido é que os passos 8 em diante (emitir token, apontar NUCs,
logar como professor) devem acontecer.

## 8. Emitir um token por estação e apontar as NUCs (SSH na EC2 + cada NUC)

Na EC2, um token por NUC (não reaproveitar entre estações — é o ponto todo
do desenho, ver plano de dev item 4):

```bash
cd ~/proctor-station
sudo -u proctor-dashboard venv/bin/python scripts/issue_station_token.py nuc-01 --label "Sala 3, estação 1"
```

O script imprime o token em texto puro **uma única vez** — copiar
imediatamente, não fica recuperável depois (só o hash fica no Postgres).
Repetir para cada `station_id` (`nuc-02`, `nuc-03`, ...).

Em cada NUC, `.env` (ver `docs/setup_nuc.md`, passo 4):

```dotenv
PROCTOR_DASHBOARD_ENABLED=true
PROCTOR_DASHBOARD_BASE_URL=https://proctoring.dominio.com.br
PROCTOR_DASHBOARD_STATION_ID=nuc-01
PROCTOR_DASHBOARD_STATION_NAME=<nome amigável>
PROCTOR_DASHBOARD_STATION_TOKEN=<o token impresso pra este station_id>
```

`PROCTOR_DASHBOARD_STATION_ID` **precisa bater exatamente** com o
`station_id` usado ao emitir o token — o dashboard rejeita se o header e o
corpo da requisição divergirem (ver plano de dev, item 4). Reiniciar
`proctor.service` em cada NUC e confirmar no painel que a estação aparece
(`journalctl -u proctor -f` mostra 401 se o token ou o `station_id`
estiverem errados — não aparece na UI, só no log).

Perdeu ou suspeita que um token vazou? Rodar `issue_station_token.py` de novo
para o mesmo `station_id` — o token antigo para de funcionar imediatamente
(sobrescreve o hash).

## 9. Verificação de ponta a ponta

```bash
# De fora da EC2:
curl -sS -o /dev/null -w '%{http_code}\n' https://proctoring.dominio.com.br/            # espera 401
curl -sS -o /dev/null -w '%{http_code}\n' -u professor:<senha> https://proctoring.dominio.com.br/  # espera 200
```

Testar o WebSocket manualmente pelo navegador: abrir o painel logado,
confirmar que o status de uma estação atualiza sozinho na tela sem dar
refresh (é o `/ws/stations`, o item mais fácil de passar batido se o bloco
nginx não tiver os headers de upgrade — ver plano de dev, item 2).

## 10. Cutover controlado

Rodar 1–2 sessões de teste completas (identificação → gravação → upload →
finalização) apontando para o subdomínio novo antes de considerar isso
produção. Confirmar no painel: estação aparece, sessão aparece, gravação
reproduz (URL pré-assinada S3), CSV exporta.

Só depois disso, considerar o subdomínio novo como o endereço definitivo do
painel e comunicar aos professores.

## 11. Monitoramento inicial

Acompanhar `free -h` e `journalctl -u proctor-dashboard -f` nas primeiras
provas reais — é o primeiro sinal de que a instância está apertada demais
para dividir com o outro serviço (Postgres é o componente novo que mais
pesa, ver passo 1). Alarmes CloudWatch (ex: `CPUUtilization`,
`StatusCheckFailed`) são opcionais nesta rodada — IaC completa (Terraform)
fica para depois, por decisão já registrada em `MILESTONES.md`.

Backup do Postgres: `pg_dump` periódico (ex: cron diário) pro mesmo bucket
S3 já usado pelas gravações, com prefixo separado — não faz parte deste
plano por padrão, mas é barato de adicionar e vale considerar já que a
troca pra Postgres foi justamente para ganhar esse tipo de durabilidade.

## Plano de rollback

Se a EC2 cair (nginx, app, Postgres ou a instância inteira), a NUC — pelo
desenho já existente do agente — termina a prova em andamento sozinha e
buferiza localmente eventos e gravação até a conexão com o dashboard voltar.
Não é preciso ação manual imediata na NUC; ao restabelecer o serviço na EC2,
o próximo heartbeat resincroniza o estado.
