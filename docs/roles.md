# Papéis: estação (NUC) × dashboard (professor)

O repositório serve dois deploys independentes, em máquinas diferentes:

- **Estação** — roda em cada NUC, identifica o aluno, grava e monitora a prova.
- **Dashboard** — roda numa máquina central (hoje: EC2), é o painel do professor.

Este documento mapeia o que pertence a cada um, para deixar explícito o que já
era verdade no grafo de imports mas não estava documentado em lugar nenhum.

## Diretórios e módulos

| Caminho | Papel | Observação |
|---|---|---|
| `src/api/` | Estação | API FastAPI local (`proctor.service`, porta 8000). |
| `src/kiosk/` | Estação | Chromium controlado, allowlist, lockdown, overlays. |
| `src/recorder/` | Estação | Captura FFmpeg e upload incremental S3. |
| `src/proctor/` | Estação | Engine de proctoring (gaze, ausência, múltiplos rostos). |
| `src/dashboard/` | Dashboard | App FastAPI do painel (`proctor-dashboard.service`, porta 80). |
| `src/face/` | **Compartilhado** | `FaceRecognizer` (dlib): estação usa para identificar aluno durante a prova; dashboard usa para o enrollment via S3 (`src/dashboard/enrollment_service.py`). |
| `src/core/config.py`, `states.py`, `s3_client.py`, `dashboard_payload.py` | **Compartilhado** | Sem dependências pesadas específicas de um papel só; usados nos dois lados. |
| `src/core/session.py`, `camera.py`, `teardown.py`, `cpu_affinity.py`, `autostart.py`, `dashboard_sync.py` | Estação | FSM de sessão, câmera, teardown, afinidade de CPU e o cliente de heartbeat que fala com o dashboard — nada disso é importado por `src/dashboard/`. |

Confirmação: `src/dashboard/*.py` importa só `src.core.config`, `src.core.s3_client`,
`src.face.recognizer` e módulos do próprio `src.dashboard` — nunca `src.core.session`,
`src.api` ou `src.kiosk`. Essa fronteira já existia no código; este documento só
a explicita.

## Por que `dlib`/`opencv`/`Pillow` continuam em `dependencies` (não em `station`)

O dashboard **não faz reconhecimento durante a prova**, mas a tela `/enrollment`
gera os `.pkl` de encodings a partir das fotos do S3 rodando `FaceRecognizer` no
próprio servidor do dashboard (`enrollment_service.py`). Por isso essas
dependências pesadas são exigidas nos dois papéis hoje — não é possível excluir
`dlib` do dashboard sem remover essa feature. Ver `pyproject.toml` para o
comentário completo.

## Entrypoints por papel

| | Estação | Dashboard |
|---|---|---|
| Setup do zero | `scripts/bootstrap.sh` | `scripts/bootstrap_dashboard.sh` |
| Extra de dependências | `pip install -e ".[station,dev]"` | `pip install -e ".[dashboard,dev]"` |
| Instala o serviço | `sudo bash scripts/install_systemd_service.sh` | `sudo bash scripts/install_dashboard_service.sh` |
| Unit systemd | `proctor.service` | `proctor-dashboard.service` |
| App ASGI | `src.api.server:app` | `src.dashboard.app:app` |
| Porta padrão | `8000` (`--host 0.0.0.0`, só a rede local deveria alcançar) | `80` (`--host 0.0.0.0`, pensado para ser público — por isso tem Basic Auth) |
| Guia passo a passo | `docs/setup_nuc.md` | `docs/setup_dashboard.md` |
| Hardware necessário | webcam UVC, sessão GNOME/X11, Chromium | nenhum — é só um servidor web |

## `.env` — o que pertence a cada papel

O mesmo `.env.example` serve os dois, porque `AppConfig`/`DashboardConfig` são
compartilhados (ex: `PROCTOR_DASHBOARD_ADMIN_USER`/`PASSWORD` são lidos tanto
pelo dashboard, que semeia a credencial, quanto pela estação, que autentica
saída com ela). Por papel:

- **Estação**: tudo em `PROCTOR_FACE_*`, `PROCTOR_GAZE_*`/`PROCTOR_ABSENCE_*`/
  `PROCTOR_MULTI_FACE_BLOCK`, `PROCTOR_REC_*`, `PROCTOR_APP_PROXY_SERVER`,
  `PROCTOR_DASHBOARD_STATION_ID`/`STATION_NAME`/`BASE_URL`/`ENABLED`. AWS
  credenciais para ler fotos e gravar em S3.
- **Dashboard**: `AWS_*` (para gerar URLs pré-assinadas das gravações e rodar
  o enrollment via S3), `PROCTOR_S3_*`, `PROCTOR_DASHBOARD_ADMIN_USER`/
  `PASSWORD` (obrigatórios aqui — sem eles o painel fica sem autenticação).
  `PROCTOR_FACE_*`/`PROCTOR_GAZE_*`/`PROCTOR_REC_*`/`PROCTOR_APP_PROXY_SERVER`/
  `STATION_ID`/`STATION_NAME` não têm efeito nenhum no dashboard — ficam com o
  default e podem ser ignorados.

## O que NÃO foi separado (de propósito)

- **Um repositório só**, não dois. `src/core` compartilhado economiza duplicar
  `config.py`/`s3_client.py`, e o ganho de separar em dois repos não paga o
  custo de sincronizar mudanças de schema/protocolo entre eles.
- **Um `pyproject.toml` só**, com extras nomeados (`station`, `dashboard`,
  `dev`) em vez de dois pacotes publicados — mais simples de manter, e o
  ganho real de reduzir o instalado no dashboard é pequeno hoje (só
  `jinja2`/`python-multipart` deixam de entrar; `dlib`/`opencv` continuam
  necessários nos dois, ver acima).
