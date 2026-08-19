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
| Hardware necessário | webcam UVC, sessão GNOME/X11, Chromium | nenhum — é só um servidor web |

### `.env` — o que pertence a cada papel

O mesmo `.env.example` serve os dois papéis (`AppConfig`/`DashboardConfig` compartilhados), mas `PROCTOR_DASHBOARD_ADMIN_USER`/`PASSWORD` (login do professor) e `PROCTOR_DASHBOARD_STATION_TOKEN` (autentica só o heartbeat de uma estação, emitido via `scripts/issue_station_token.py`) **não são a mesma credencial**. Campos de estação (`PROCTOR_FACE_*`, `PROCTOR_GAZE_*`, `PROCTOR_REC_*`, `STATION_ID`/`TOKEN`...) não têm efeito no dashboard e vice-versa — ver `.env.example` para a lista completa comentada.

## Configurar uma máquina do zero

- Estação (NUC): [`docs/setup_nuc.md`](docs/setup_nuc.md)
- Dashboard (professor): [`docs/setup_dashboard.md`](docs/setup_dashboard.md)

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
