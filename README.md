# Proctoring Station

Sistema de estações de prova presencial baseadas em Intel NUC, com reconhecimento facial, gravação com proctoring automatizado, browser lockdown e integração com PrairieLearn. O objetivo é substituir quizzes que consomem 1h de aula por sessões controladas em máquinas dedicadas, fora do horário de aula.

Status atual, histórico e decisões técnicas: [`MILESTONES.md`](MILESTONES.md) — é a fonte de verdade do progresso do projeto, não este README.

---

## Como o sistema funciona

O repositório serve **dois deploys independentes, em máquinas diferentes**:

- **Estação** — roda em cada NUC (`proctor.service`), identifica o aluno, grava e monitora a prova.
- **Dashboard** — roda numa máquina central (hoje: EC2, `proctor-dashboard.service`), é o painel do professor.

| Caminho | Papel | Observação |
|---|---|---|
| `src/api/` | Estação | API FastAPI local (`proctor.service`, porta 8000, só rede local). |
| `src/kiosk/` | Estação | Chromium controlado, allowlist, lockdown, overlays. |
| `src/recorder/` | Estação | Captura FFmpeg e upload incremental S3. |
| `src/proctor/` | Estação | Engine de proctoring (gaze, ausência, múltiplos rostos). |
| `src/core/session.py`, `camera.py`, `teardown.py`, `cpu_affinity.py`, `autostart.py`, `dashboard_sync.py` | Estação | FSM de sessão, câmera, teardown, afinidade de CPU e o cliente de heartbeat que fala com o dashboard — nada disso é importado por `src/dashboard/`. |
| `src/dashboard/` | Dashboard | App FastAPI do painel (`proctor-dashboard.service`, porta 8010 atrás de nginx). |
| `src/face/` | **Compartilhado** | `FaceRecognizer` (dlib): estação usa para identificar aluno durante a prova; dashboard usa para o enrollment via S3 (`src/dashboard/enrollment_service.py`). |
| `src/core/config.py`, `states.py`, `s3_client.py`, `dashboard_payload.py` | **Compartilhado** | Sem dependências pesadas específicas de um papel só; usados nos dois lados. |

`dlib`/`opencv`/`Pillow` continuam em `dependencies` (não em `station`) porque o dashboard também roda `FaceRecognizer` na tela `/enrollment` — não dá pra excluir essas dependências pesadas do dashboard sem perder essa feature.

**Um repositório só, um `pyproject.toml` só** (extras nomeados `station`/`dashboard`/`dev`) — decisão deliberada, não default: separar em dois repos não paga o custo de sincronizar mudanças de schema/protocolo entre eles.

### Entrypoints por papel

| | Estação | Dashboard |
|---|---|---|
| Setup do zero | `scripts/bootstrap.sh` | `scripts/bootstrap_dashboard.sh` |
| Extra de dependências | `pip install -e ".[station,dev]"` | `pip install -e ".[dashboard,dev]"` |
| Instala o serviço | `sudo bash scripts/install_systemd_service.sh` | `sudo bash scripts/install_dashboard_service.sh` |
| Unit systemd | `proctor.service` | `proctor-dashboard.service` |
| App ASGI | `src.api.server:app` | `src.dashboard.app:create_app` (`--factory`) |
| Porta padrão | `8000` (só rede local) | `8010` (atrás de nginx — professor e NUC têm autenticações separadas, Basic Auth e token de estação) |
| Hardware necessário | webcam UVC, Matchbox/Xorg, Chromium; GNOME mantido para manutenção | nenhum — é só um servidor web |

### `.env` — o que pertence a cada papel

O mesmo `.env.example` serve os dois papéis (`AppConfig`/`DashboardConfig` compartilhados), mas `PROCTOR_DASHBOARD_ADMIN_USER`/`PASSWORD` (login do professor) e `PROCTOR_DASHBOARD_STATION_TOKEN` (autentica só o heartbeat de uma estação, emitido via `scripts/issue_station_token.py`) **não são a mesma credencial**. Campos de estação (`PROCTOR_FACE_*`, `PROCTOR_GAZE_*`, `PROCTOR_REC_*`, `STATION_ID`/`TOKEN`...) não têm efeito no dashboard e vice-versa — ver `.env.example` para a lista completa comentada.

## Configurar uma máquina do zero

- Estação (NUC): [`docs/setup_nuc.md`](docs/setup_nuc.md)
- Dashboard (professor): [`docs/setup_dashboard.md`](docs/setup_dashboard.md)

## Instalar uma nova NUC

Este é o procedimento de produção completo. Ele parte de uma instalação limpa
do **Ubuntu 24.04 Desktop**, com acesso à internet e uma conta administrativa
chamada **`proctor`**. Esse nome é obrigatório atualmente: a sessão automática,
o helper de policies do Chromium e o modo de manutenção usam essa conta.

### 1. Preparar a máquina

Durante a instalação do Ubuntu:

- crie o usuário `proctor` com senha forte;
- mantenha GNOME e GDM instalados;
- conecte a webcam USB e, preferencialmente, Ethernet durante o setup;
- aplique as atualizações do sistema e reinicie antes de continuar.

```bash
sudo apt update
sudo apt full-upgrade -y
sudo reboot
```

### 2. Clonar o projeto

Entre como `proctor` e clone o repositório no caminho padrão usado nas NUCs:

```bash
git clone git@github.com:Rkzzin/prairielearn-proctoring.git ~/prairielearn-proctoring
cd ~/prairielearn-proctoring
```

Não copie `.env`, tokens ou diretórios de dados de outra NUC. Cada estação
precisa de `station_id` e token próprios.

### 3. Cadastrar a estação e gerar o token

No dashboard, clique em **Nova estação**, informe um nome exclusivo e copie o
bloco de configuração exibido. O ID é derivado do nome e o token aparece uma
única vez; apenas seu hash fica armazenado no servidor.

Como alternativa administrativa, no servidor do dashboard é possível emitir o
token pela linha de comando:

```bash
cd ~/prairielearn-proctoring
source venv/bin/activate
python scripts/issue_station_token.py nuc-descomp-03
```

Guarde o token exibido. O dashboard armazena somente o hash e não consegue
mostrar o valor novamente. Se ele for perdido, emita outro token.

### 4. Executar o bootstrap da NUC

Na NUC:

```bash
cd ~/prairielearn-proctoring
./scripts/bootstrap.sh
```

O bootstrap instala Python 3.12, FFmpeg, Chromium, Matchbox, bibliotecas do
dlib/OpenCV, cria `venv/`, baixa os modelos faciais, instala o hardening base do
Chromium e executa os testes compatíveis com a estação.

### 5. Configurar `.env`

O bootstrap cria `.env` a partir de `.env.example` quando necessário. Edite-o:

```bash
nano ~/prairielearn-proctoring/.env
```

Confira pelo menos:

```dotenv
AWS_ACCESS_KEY_ID=<credencial com acesso mínimo ao bucket>
AWS_SECRET_ACCESS_KEY=<segredo correspondente>
AWS_DEFAULT_REGION=sa-east-1
PROCTOR_S3_BUCKET=proctor-station

PROCTOR_DASHBOARD_ENABLED=true
PROCTOR_DASHBOARD_BASE_URL=https://proctoring.descompvalidator.click
PROCTOR_DASHBOARD_STATION_ID=nuc-descomp-03
PROCTOR_DASHBOARD_STATION_NAME=NUC Sala 03
PROCTOR_DASHBOARD_STATION_TOKEN=<token emitido no passo 3>

PROCTOR_APP_PROXY_SERVER=<URL do proxy, ou vazio para acesso direto>
PROCTOR_REC_DISPLAY=:0
```

Regras importantes:

- `PROCTOR_DASHBOARD_STATION_ID` deve ser exclusivo;
- o token da estação não é a senha do professor;
- não use o mesmo token em duas NUCs;
- não versione `.env` nem credenciais AWS;
- deixe `PROCTOR_APP_PROXY_SERVER=` vazio se a rede não exigir proxy.

### 6. Detectar câmera e áudio

Com a webcam definitiva conectada:

```bash
cd ~/prairielearn-proctoring
bash scripts/detect_camera_audio.sh
source venv/bin/activate
python scripts/test_camera.py --headless
```

Revise no `.env` o índice de câmera e o dispositivo de áudio detectados antes
de instalar o serviço.

### 7. Instalar Matchbox e o serviço

```bash
cd ~/prairielearn-proctoring
sudo bash scripts/install_matchbox_session.sh
sudo bash scripts/install_chromium_hardening.sh
sudo bash scripts/install_systemd_service.sh
```

Esses scripts:

- configuram autologin de `proctor` na sessão Matchbox/Xorg;
- bloqueiam troca de VT e atalhos de fuga durante a prova;
- instalam a policy de allowlist do Chromium e seu helper privilegiado;
- criam `/opt/proctor/data` com o proprietário correto;
- habilitam `proctor.service` no boot.

### 8. Instalar o modo de manutenção GRUB

Gere uma senha PBKDF2. Digite a senha duas vezes e copie a linha inteira que
começa com `grub.pbkdf2.sha512`:

```bash
grub-mkpasswd-pbkdf2
```

Instale o menu usando o hash, nunca a senha em texto puro:

```bash
sudo env GRUB_PASSWORD_HASH='<cole-o-hash-completo>' \
  bash scripts/install_grub_maintenance.sh
```

No próximo boot, o GRUB mostrará por 8 segundos:

- **Infraestrutura de Provas (padrao)**: opção padrão, inicia Matchbox e o
  serviço automaticamente;
- **Manutencao GNOME (administrador)**: exige usuário GRUB `proctor-admin` e a
  senha definida acima, desativa o autologin, não inicia `proctor.service` e
  abre o login do GNOME; o GNOME também exige a senha da conta `proctor`.

### 9. Reiniciar e validar

```bash
sudo reboot
```

Sem escolher nada no GRUB, a NUC deve entrar na infraestrutura de provas. De
outro computador, valide:

```bash
ssh proctor@<ip-da-nuc>
systemctl is-active proctor.service
curl --fail http://127.0.0.1:8000/status
journalctl -u proctor.service -n 100 --no-pager
```

Resultado esperado:

- `proctor.service` está `active`;
- `/status` informa `IDLE`, `WAITING_STUDENT` ou `IDENTIFYING`;
- a estação aparece online no dashboard em até alguns heartbeats;
- antes da autenticação, o overlay mostra somente **Iniciar prova**, com a
  câmera e o Chromium fechados;
- ao clicar em **Iniciar prova**, a câmera abre, identifica o aluno e apresenta
  a confirmação da avaliação;
- no início da prova, Chromium abre e sites fora da allowlist são bloqueados.

Também faça um boot manual em **Manutencao GNOME (administrador)** e confirme
que o GNOME pede senha, o terminal e as configurações de rede estão acessíveis
e `proctor.service` não inicia nesse modo.

### 10. Carregar as fotos da turma

Depois que a estação aparecer no dashboard e estiver sem prova ativa:

1. clique em **Atualizar reconhecimento facial** no cartão da NUC;
2. selecione as turmas;
3. clique em **Atualizar imagens e modelo**;
4. acompanhe o estado no cartão da estação.

Isso executa na NUC `scripts/enroll.py --turma <turma> --force`, baixa as fotos
do S3 e gera `data/encodings/<turma>.pkl`. Para diagnóstico local, o mesmo
comando pode ser executado manualmente dentro de `venv/`.

### Checklist antes de liberar a NUC

- estação tem ID e token exclusivos;
- câmera, microfone e gravação foram testados;
- turma foi processada e o aluno de teste foi reconhecido;
- PrairieLearn/PrairieTest abre pelo proxy configurado;
- domínio externo é bloqueado pela allowlist;
- bloqueios por ausência, múltiplas faces e usuário diferente foram testados;
- encerramento envia os segmentos e retorna para `WAITING_STUDENT`;
- boot padrão abre Matchbox e boot administrativo abre GNOME com senha.

## Preparar uma NUC clonada de outra estação

Use este procedimento quando o disco ou a imagem completa de uma NUC existente
for duplicado. Não conecte a máquina clonada à rede de produção antes de trocar
suas identidades: enquanto ela mantiver o mesmo `station_id`, token, `machine-id`
e estado do Tailscale, poderá sobrescrever a estação original no dashboard ou
aparecer como o mesmo equipamento na rede.

### 1. Iniciar em manutenção

No GRUB, escolha **Manutencao GNOME (administrador)**. Nesse modo é esperado que
`systemctl is-active proctor.service` retorne `inactive`: a condição
`proctor.maintenance=1` impede o serviço de provas de iniciar durante a
manutenção.

Confirme o modo atual:

```bash
cat /proc/cmdline
```

A linha deve conter `proctor.maintenance=1`.

### 2. Trocar a identidade do Ubuntu

Defina um hostname exclusivo e regenere o `machine-id` e as chaves SSH. Execute
estes comandos somente na cópia, nunca na estação original:

```bash
sudo hostnamectl set-hostname nuc-descomp-04

sudo truncate -s 0 /etc/machine-id
sudo rm -f /var/lib/dbus/machine-id
sudo systemd-machine-id-setup
sudo ln -sf /etc/machine-id /var/lib/dbus/machine-id

sudo rm -f /etc/ssh/ssh_host_*
sudo ssh-keygen -A
sudo systemctl restart ssh
```

Depois da troca das chaves SSH, clientes que acessaram a imagem original podem
precisar remover a entrada antiga com `ssh-keygen -R <ip-ou-hostname>`.

### 3. Gerar uma estação nova no dashboard

No dashboard, clique em **Nova estação**, informe o nome e copie o bloco gerado.
O token aparece uma única vez. Não reutilize o ID ou token da NUC original.

Na NUC clonada, edite:

```bash
nano ~/prairielearn-proctoring/.env
```

Substitua as três variáveis pelo bloco emitido pelo dashboard e preserve as
demais configurações da estação:

```dotenv
PROCTOR_DASHBOARD_STATION_ID=nuc-descomp-04
PROCTOR_DASHBOARD_STATION_NAME=NUC Sala 04
PROCTOR_DASHBOARD_STATION_TOKEN=<token-novo>
```

Confira também:

```dotenv
PROCTOR_DASHBOARD_ENABLED=true
PROCTOR_DASHBOARD_BASE_URL=https://proctoring.descompvalidator.click
```

### 4. Remover o estado persistido da estação original

Não crie `station-config.json` manualmente. O caminho correto é
`/opt/proctor/data/station-config.json`, e o sistema o recria quando uma
configuração de prova é enviada pelo dashboard. A cópia existente deve ser
arquivada porque contém o `station_id` antigo e pode sobrescrever o `.env`:

```bash
sudo systemctl stop proctor.service

if [ -f /opt/proctor/data/station-config.json ]; then
  sudo mv /opt/proctor/data/station-config.json \
    /opt/proctor/data/station-config.json.clonado
fi

sudo install -d -o proctor -g proctor /opt/proctor/data
```

Revise também `/opt/proctor/data/sessions/` antes de liberar a máquina. Sessões
da estação original não devem permanecer na cópia; preserve-as em backup seguro
ou remova-as conforme a política de retenção adotada.

### 5. Gerar uma identidade nova no Tailscale

Se a imagem inclui Tailscale, o estado clonado também identifica as duas NUCs
como o mesmo dispositivo. Somente na máquina clonada:

```bash
sudo systemctl stop tailscaled
sudo rm -f /var/lib/tailscale/tailscaled.state
sudo systemctl start tailscaled
sudo tailscale up
```

Conclua a autenticação exibida e confirme que a nova NUC aparece como um nó
separado. Pule este passo se Tailscale não estiver instalado.

### 6. Detectar o hardware e reinstalar o serviço

A webcam e o dispositivo de áudio podem receber índices diferentes na nova
máquina. Com a webcam definitiva conectada:

```bash
cd ~/prairielearn-proctoring
bash scripts/detect_camera_audio.sh
source venv/bin/activate
python scripts/test_camera.py --headless
deactivate

sudo bash scripts/install_systemd_service.sh
```

O instalador recria a unit com o diretório e o usuário corretos, ajusta as
permissões de `/opt/proctor/data` e habilita o serviço no boot.

### 7. Reiniciar no modo de provas e validar

```bash
sudo reboot
```

No GRUB, escolha **Infraestrutura de Provas (padrao)** ou aguarde o timeout. Não
escolha manutenção: nesse modo o serviço continuará inativo por projeto.

Depois do boot:

```bash
cat /proc/cmdline
systemctl is-active proctor.service
curl --fail http://127.0.0.1:8000/status
journalctl -u proctor.service -n 100 --no-pager
```

Resultado esperado:

- `/proc/cmdline` não contém `proctor.maintenance=1`;
- `proctor.service` retorna `active`;
- a nova estação aparece como um cartão separado no dashboard;
- a estação original continua online com seu próprio ID e token;
- câmera, microfone, reconhecimento facial e gravação passam nos testes.

## Desenvolvimento local

```bash
git clone <repo> proctor-station && cd proctor-station
cp .env.example .env   # preencher credenciais AWS e o resto necessário pro papel que for testar
./scripts/bootstrap.sh              # ou bootstrap_dashboard.sh, conforme o papel
source venv/bin/activate
```

Testes automatizados isolam S3 com doubles/patches — rodam sem credenciais AWS reais (exceto `tests/test_dashboard.py`, que precisa de Postgres — ver `docker-compose.yml`).

```bash
pytest tests/ -v                                            # suíte completa
python scripts/enroll.py --turma ES2025-T1                  # enrollment de turma
python scripts/calibrate_gaze.py                             # calibrar thresholds de gaze
python scripts/test_camera.py --headless                     # validar câmera + dlib
python scripts/test_integration.py --turma ES2025-T1         # teste de ponta a ponta (ao vivo)
python scripts/test_integration.py --turma ES2025-T1 --no-record  # idem, sem gravação
```

---

## Bucket S3

```
proctor-station/
├── fotos/
│   └── {turma_id}/
│       └── {nome_aluno}.png    ← fotos de enrollment
└── gravacoes/
    └── {session_id}/
        ├── webcam_000.mp4
        ├── screen_000.mp4
        └── ...
```

**Lifecycle rules:** `gravacoes/` expira em 90 dias; `fotos/` tem limpeza manual semestral (janeiro e julho) por turma.

## Custos estimados (mensal, 10 NUCs)

Estimativa de referência, não cobrada automaticamente — confira o console AWS pra números reais.

| Item | Custo estimado |
|---|---|
| S3 storage gravações (~100GB/mês, expiram em 90d) | ~$2,30 |
| S3 storage fotos (pequeno, permanente) | ~$0,05 |
| EC2 t3.small (dashboard) | ~$15 |
| Data transfer (upload) | ~$0,90 |
| **Total mensal AWS** | **~$18** |
| Hardware (NUC + webcam + monitor) × 10 | ~$7.000 (one-time) |

## Riscos e mitigações

| Risco | Impacto | Mitigação |
|---|---|---|
| Falso positivo de gaze (aluno pensando olha pro lado) | Bloqueio injusto | Threshold generoso + `gaze_duration_sec` longo + review pós-prova |
| dlib lento na NUC sem GPU | Lag no reconhecimento | HOG detector com `detection_scale=0.5`; benchmark com `test_camera.py` |
| Aluno parecido confunde o sistema | Acesso errado | Threshold restritivo (0.45) + confirmação visual |
| Internet instável durante prova | Upload falha | Fila local com retry; gravação 100% local mesmo sem internet |
| Chromium atualiza e quebra browser controlado | Prova comprometida | `snap refresh --hold chromium` + teste da extensão/policies antes de prova |
| Câmera ocupada por outro processo | FFmpeg falha | NUC dedicada; nenhum app extra instalado |
| Perfil do browser retém login do aluno | Vazamento de sessão | Perfil descartável por prova, policies de limpeza e cleanup explícito ao fim |
| Lockdown não restaura após erro | NUC difícil de operar | `finally`/cleanup, comando de recuperação (`scripts/recover_exam_mode.sh`) e acesso administrativo ao GNOME |
| Credenciais AWS expostas | Acesso indevido ao bucket | `~/.aws/credentials` ou IAM role, nunca commitar credenciais; política IAM mínima |
| Privacidade / LGPD | Legal | Consentimento no enrollment; retenção máxima 90 dias para vídeos; acesso restrito |
