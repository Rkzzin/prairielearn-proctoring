# Configurar o dashboard do zero (ex: EC2)

Passo a passo para levar uma máquina Linux sem nada instalado até rodar
`proctor-dashboard.service`. Ver `docs/roles.md` para o que diferencia este
papel do papel de estação (`docs/setup_nuc.md`) — em resumo, o dashboard não
precisa de GNOME, X11, câmera nem Chromium; é só um servidor web.

## 1. Provisionar a máquina (manual)

- Ubuntu 24.04 (Server serve — não precisa de sessão gráfica).
- Abra a porta `80/tcp` (ou a que você escolher) no security group / firewall
  para quem vai acessar o painel (professor e as NUCs). Como a autenticação é
  usuário/senha (ver passo 5), não é obrigatório restringir por IP, mas reduz
  superfície se você souber de antemão a faixa de rede das NUCs.
- Não é preciso restringir por IP de origem se as NUCs/professor acessarem de
  redes variáveis — é exatamente o cenário para o qual a Basic Auth do passo 5
  foi feita.

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
para o dashboard, o mais simples é instalar nativo:

```bash
sudo apt update && sudo apt install -y postgresql
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
# que descubra o IP/porta. Mesmo usuário/senha que vai para as NUCs.
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

Isso cria e habilita `proctor-dashboard.service` na porta `80` por padrão
(`PROCTOR_DASHBOARD_PORT` no shell muda isso, ex:
`sudo env PROCTOR_DASHBOARD_PORT=8010 bash scripts/install_dashboard_service.sh`).
O unit já ganha `AmbientCapabilities=CAP_NET_BIND_SERVICE`, necessário para
abrir uma porta <1024 sem rodar como root.

## 7. Verificação (script)

```bash
systemctl status proctor-dashboard --no-pager
curl -sS -o /dev/null -w '%{http_code}\n' http://127.0.0.1/       # espera 401 sem credencial
curl -sS -o /dev/null -w '%{http_code}\n' -u professor:<senha> http://127.0.0.1/   # espera 200
```

De fora da máquina, confirme que a porta está acessível de onde as NUCs vão
estar amanhã — bloqueio de saída em rede de terceiros é comum:

```bash
curl -sS -o /dev/null -w '%{http_code}\n' http://<ip-ou-dns-do-ec2>/
```

## 8. Apontar as NUCs para cá

Em cada NUC (ver `docs/setup_nuc.md`, passo 4), `.env`:

```dotenv
PROCTOR_DASHBOARD_ENABLED=true
PROCTOR_DASHBOARD_BASE_URL=http://<ip-ou-dns-do-ec2>
PROCTOR_DASHBOARD_ADMIN_USER=professor
PROCTOR_DASHBOARD_ADMIN_PASSWORD=<a mesma senha>
```

Sem usuário/senha configurados na NUC, o heartbeat de saída recebe 401 do
dashboard e a estação nunca aparece no painel — não é um erro silencioso, mas
aparece só no log (`journalctl -u proctor -f`), não na UI.

## Risco aceito por enquanto

O tráfego entre NUC e dashboard (heartbeat, comandos, Basic Auth) roda em HTTP
puro, sem TLS. A senha viaja em texto claro dentro do header `Authorization`.
Aceitável para um teste pontual numa rede confiável; para uso continuado,
colocar um nginx com certificado (Let's Encrypt) na frente resolve.
