# Proctoring Station

Sistema de estações de prova presencial baseadas em Intel NUC, com reconhecimento facial, gravação com proctoring automatizado, browser lockdown e integração com PrairieLearn. O objetivo é substituir quizzes que consomem 1h de aula por sessões controladas em máquinas dedicadas, fora do horário de aula.

---

## Status de implementação

| Fase | Descrição | Status |
|---|---|---|
| 0 — Infraestrutura | Hardware, stack, estrutura do repo | ✅ Completo |
| 1 — Face recognition | Enrollment via S3 + identificação dlib | ✅ Completo |
| 2 — Proctoring engine | Gaze estimation + FSM + event log | ✅ Completo |
| 3 — Gravação e upload | FFmpeg + upload incremental S3 | ✅ Completo |
| 4 — Browser lockdown | Base de Chromium controlado + bloqueio/reidentificação | ✅ Completo |
| 5 — Session manager | Orquestrador E2E + FastAPI local | ✅ Completo |
| 6 — Dashboard | Interface do professor | ✅ Completo |
| 7 — Modo prova final | GNOME atual endurecido, allowlist e hardening | 🟡 Em validação na NUC |
| 8 — Terraform + Ansible | IaC completo | 🔲 Pendente |
| 9 — Deploy | Go-live | 🔲 Pendente |

---

## Fase 0 — Infraestrutura e decisões de projeto

### Hardware por estação

| Componente | Especificação mínima | Justificativa |
|---|---|---|
| NUC | Intel NUC 12/13 (i5), 16GB RAM, SSD 256GB | Encode H.264 via Quick Sync |
| Webcam | Logitech C920/C922 (1080p, wide-angle) | Boa qualidade, UVC nativo no Linux |
| Monitor | 21"+ Full HD | Tamanho padrão de prova |
| Teclado + Mouse | USB com fio | Sem risco de Bluetooth spoofing |
| Microfone | Embutido na webcam (C920 já tem) | Captura áudio ambiente |

### Stack tecnológica

```
OS:           Ubuntu 24.04 LTS Desktop
Linguagem:    Python 3.12
CV/ML:        OpenCV 4.x + dlib (HOG detector + ResNet 128-d encoding)
Gaze:         dlib shape_predictor_68 + solvePnP (pose estimation)
Gravação:     FFmpeg 6.x (H.264 via libx264, sem áudio)
Browser:      Chromium controlado no GNOME atual + extensão allowlist
Orquestrador: FastAPI (API local na NUC)
Upload:       boto3 → AWS S3 real (sa-east-1)
IaC:          Terraform + Ansible (pendente)
Dashboard:    FastAPI + HTMX
```

### Estrutura do repositório

```
proctor-station/
├── models/                      # Modelos dlib (download_models.sh)
│   ├── shape_predictor_68_face_landmarks.dat
│   ├── shape_predictor_5_face_landmarks.dat
│   ├── dlib_face_recognition_resnet_model_v1.dat
│   └── mmod_human_face_detector.dat
├── data/
│   ├── encodings/               # .pkl por turma (gerados pelo enroll.py)
│   └── sessions/                # logs JSONL e gravações por sessão
│       └── {session_id}/
│           ├── events.jsonl
│           └── recordings/
│               ├── webcam_000.mp4
│               └── screen_000.mp4
├── src/
│   ├── core/
│   │   ├── config.py            # Configuração central (pydantic-settings)
│   │   ├── models.py            # Dataclasses do sistema
│   │   └── s3_client.py         # Cliente S3 real + factory get_s3_client()
│   ├── face/
│   │   ├── recognizer.py        # Enrollment + identificação
│   │   └── detector.py          # Detecção rápida para proctoring contínuo
│   ├── proctor/
│   │   ├── engine.py            # FSM principal de monitoramento
│   │   ├── gaze.py              # Pose estimation via dlib + solvePnP
│   │   └── events.py            # Event bus + serialização JSONL
│   ├── recorder/
│   │   ├── capture.py           # Gerencia processos FFmpeg (webcam + tela)
│   │   └── uploader.py          # Upload incremental S3 com retry
│   ├── kiosk/                   # Chromium controlado, lockdown, overlays (Fase 4/M7)
│   ├── api/                     # FastAPI local da NUC (Fase 5)
│   └── dashboard/               # Dashboard do professor (Fase 6)
├── scripts/
│   ├── bootstrap.sh             # Setup completo de uma NUC do zero
│   ├── download_models.sh       # Baixa modelos dlib (~100MB)
│   ├── enroll.py                # CLI de enrollment via S3
│   ├── calibrate_gaze.py        # Calibração visual de thresholds de gaze
│   ├── test_camera.py           # Validação de hardware (câmera + dlib)
│   └── test_integration.py      # Teste manual Fase 1+2+3 ao vivo
├── tests/
│   ├── test_face_recognition.py # Testes Fase 1 (39 casos)
│   ├── test_proctor_engine.py   # Testes Fase 2
│   ├── test_recorder.py         # Testes Fase 3 (14 casos)
│   ├── test_kiosk.py            # Testes Fase 4
│   ├── test_session_manager.py  # Testes Fase 5
│   ├── test_dashboard.py        # Testes Fase 6
│   └── test_dashboard_sync.py   # Sync NUC → dashboard
├── .env.example                 # Referência de todas as variáveis
├── pyproject.toml
└── README.md
```

---

## Fase 1 — Face recognition ✅

### Fluxo de enrollment

Alunos sobem suas fotos nas primeiras semanas de aula para o S3. Após o prazo, o operador roda o enrollment para gerar o `.pkl` local na NUC.

```
Layout no S3:
  s3://{bucket}/fotos/{turma_id}/{nome_do_aluno}.png

Fluxo:
  1. Alunos fazem upload de suas fotos para o S3
  2. Operador roda: python scripts/enroll.py --turma ES2025-T1
  3. Script baixa fotos do S3, gera encodings 128-d via dlib ResNet
  4. Salva em data/encodings/ES2025-T1.pkl
```

```bash
# Enrollment completo de uma turma
python scripts/enroll.py --turma ES2025-T1

# Enrollment de aluno individual
python scripts/enroll.py --turma ES2025-T1 --aluno joao_silva

# Listar turmas cadastradas
python scripts/enroll.py --list

# Ver alunos de uma turma
python scripts/enroll.py --turma ES2025-T1 --info

# Remover aluno
python scripts/enroll.py --turma ES2025-T1 --remove joao_silva

# Reprocessar turma do zero
python scripts/enroll.py --turma ES2025-T1 --force
```

### Fluxo de identificação

Detecção HOG → encoding ResNet 128-d → distância euclidiana contra `.pkl` da turma.

**Decisões técnicas:**
- Threshold de distância: `0.45` (mais restritivo que o default `0.6`)
- 3 jitters por foto no enrollment para maior robustez
- `identify_best_of_n()` para identificação com múltiplos frames

### Entregáveis

- [x] Script de enrollment via S3 com CLI completa
- [x] `recognizer.py` com `identify()` e `identify_best_of_n()`
- [x] `detector.py` para detecção leve no loop de proctoring
- [x] 16 testes unitários cobrindo todos os fluxos

---

## Fase 2 — Proctoring engine ✅

### Gaze estimation

Usa `shape_predictor_68_face_landmarks` + `cv2.solvePnP` para estimar pose da cabeça (yaw/pitch/roll). Sem dependência de MediaPipe.

```
Pontos 3D de referência: nariz (30), queixo (8), cantos dos olhos (36, 45), cantos da boca (48, 54)

Normalização:
  yaw_ratio   = abs(yaw) / 90.0
  pitch_ratio = abs(abs(pitch) - 180.0) / 90.0   ← pitch neutro ≈ 180° no solvePnP
```

### Parâmetros tunáveis

| Variável de ambiente | Default | Descrição |
|---|---|---|
| `PROCTOR_GAZE_H_THRESHOLD` | 0.35 | Limiar horizontal do olhar (ratio 0–1) |
| `PROCTOR_GAZE_V_THRESHOLD` | 0.30 | Limiar vertical do olhar (ratio 0–1) |
| `PROCTOR_GAZE_DURATION_SEC` | 5.0 | Segundos de desvio antes de BLOCKED |
| `PROCTOR_ABSENCE_TIMEOUT_SEC` | 5.0 | Segundos sem rosto antes de BLOCKED |
| `PROCTOR_MULTI_FACE_BLOCK` | true | Bloquear imediatamente se >1 rosto |

Use `python scripts/calibrate_gaze.py` para encontrar os valores ideais para seu ambiente.

### Máquina de estados

```
                ┌──────────┐
                │  NORMAL  │ ← estado padrão durante a prova
                └────┬─────┘
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
    ┌──────────┐ ┌────────┐ ┌──────────┐
    │GAZE_WARN │ │ABSENCE │ │MULTI_FACE│  (imediato)
    └────┬─────┘ └───┬────┘ └────┬─────┘
         │           │           │
    gaze_duration_  absence_      imediato
    sec             timeout_sec
         │           │           │
         └───────────┴───────────┘
                     ▼
              ┌─────────────┐
              │   BLOCKED   │ ← só sai via engine.unblock()
              └──────┬──────┘
                     │ face re-match OK
                     ▼
                ┌──────────┐
                │  NORMAL  │
                └──────────┘
```

### Log de eventos

Cada sessão gera um arquivo JSONL em `data/sessions/{session_id}/events.jsonl`:

```json
{"timestamp": 1719000000.0, "frame": 120, "type": "GAZE_WARNING", "severity": "WARNING", "details": {"yaw": 35.2, "pitch": 172.1}}
{"timestamp": 1719000015.0, "frame": 570, "type": "GAZE_BLOCKED", "severity": "CRITICAL", "details": {"reason": "GAZE"}}
{"timestamp": 1719000022.0, "frame": 780, "type": "SESSION_RESUMED", "severity": "INFO", "details": {"unblocked_after": "GAZE"}}
```

Tipos de evento: `SESSION_STARTED`, `SESSION_ENDED`, `GAZE_WARNING`, `GAZE_BLOCKED`, `ABSENCE_WARNING`, `ABSENCE_BLOCKED`, `MULTI_FACE_BLOCKED`, `SESSION_RESUMED`.

### Entregáveis

- [x] `gaze.py` com pose estimation via solvePnP + correção de pitch
- [x] `engine.py` com FSM completa e suavização de yaw (janela de 10 frames)
- [x] `events.py` com EventLogger, enums tipados e roundtrip JSONL
- [x] `calibrate_gaze.py` com overlay visual de thresholds em tempo real
- [x] 23 testes unitários cobrindo todos os estados e transições da FSM

---

## Fase 3 — Gravação e upload ✅

### Streams FFmpeg

Durante uma sessão com gravação ativa, a webcam física fica com **um único dono**:
o FFmpeg. O OpenCV usa `/dev/video0` apenas na identificação inicial; depois do
match, o `SessionManager` libera a câmera, inicia o `Capture` e passa a consumir
o preview UDP publicado pelo FFmpeg. Se a sessão for iniciada com `no_record`,
o proctoring continua lendo a câmera física diretamente.

Fluxo real:

1. `SessionManager` abre `/dev/video0` por OpenCV só para identificar o aluno.
2. Após o match, a câmera física é liberada.
3. `Capture.start()` sobe dois processos FFmpeg independentes: webcam e tela.
4. O FFmpeg da webcam abre `/dev/video0` via `v4l2` e aplica `split=2`.
5. O ramo `record` grava segmentos `webcam_%03d.mp4`.
6. O ramo `preview` publica MPEG-TS em `udp://127.0.0.1:18181?pkt_size=1316`.
7. O OpenCV reconecta em `udp://127.0.0.1:18181?overrun_nonfatal=1&fifo_size=5000000` para gaze, ausência e reidentificação.
8. O FFmpeg da tela usa `x11grab`; o tamanho real do display é detectado via `xrandr --current` e redimensionado para `PROCTOR_REC_SCREEN_SIZE`.

Comandos equivalentes:

```bash
# Stream webcam
ffmpeg -f v4l2 -thread_queue_size 512 -input_format mjpeg \
       -use_wallclock_as_timestamps 1 \
       -framerate 30 -video_size 1280x720 -i /dev/video0 \
       -filter_complex "[0:v]split=2[record][preview];[preview]fps=10,scale=640:360[preview_out]" \
       -map "[record]" \
       -c:v libx264 -preset veryfast -crf 23 -profile:v high -pix_fmt yuv420p \
       -threads 1 -fps_mode passthrough \
       -f segment -segment_time 300 -segment_format_options movflags=+faststart \
       -reset_timestamps 1 \
       data/sessions/{id}/recordings/webcam_%03d.mp4 \
       -map "[preview_out]" \
       -c:v libx264 -preset ultrafast -tune zerolatency -crf 35 -pix_fmt yuv420p \
       -g 10 -keyint_min 10 -sc_threshold 0 -x264-params repeat-headers=1:aud=1 \
       -threads 1 -f mpegts "udp://127.0.0.1:18181?pkt_size=1316"

# Stream tela
ffmpeg -f x11grab -thread_queue_size 512 \
       -use_wallclock_as_timestamps 1 \
       -video_size <resolucao_real_do_display> -framerate 15 -i :1 \
       -vf "scale=1280:720:flags=fast_bilinear,setsar=1" \
       -c:v libx264 -preset veryfast -crf 28 -profile:v high -pix_fmt yuv420p \
       -threads 1 -fps_mode passthrough \
       -f segment -segment_time 300 -segment_format_options movflags=+faststart \
       -reset_timestamps 1 \
       data/sessions/{id}/recordings/screen_%03d.mp4
```

**Decisões:**
- `/dev/video0` não fica mais aberto em paralelo durante a sessão; o FFmpeg é o dono da webcam
- O proctoring contínuo lê um preview local de baixa latência, não a câmera física
- Webcam e tela usam timestamps de relógio real (`use_wallclock_as_timestamps`) e `fps_mode passthrough`
- Os MP4s finais são gravados em H.264 `High` + `yuv420p` + `faststart` para compatibilidade com navegador/dashboard
- A resolução de captura da tela vem do display X11 real; `PROCTOR_REC_SCREEN_SIZE` é a resolução final após downscale
- Afinidade de CPU pode reservar os últimos núcleos para o FFmpeg e dividir esses núcleos entre webcam/tela
- Segmentos de 5 minutos — upload incremental, não espera o fim da prova
- `stop()` envia `SIGINT`, aguarda o trailer do MP4 e faz flush do último segmento antes do upload
- CRF 23 para webcam (qualidade facial), CRF 28 para tela (texto legível, arquivo menor)
- 15fps para tela é suficiente para capturar interação com mouse/teclado

### Upload incremental

Cada segmento fechado pelo FFmpeg é detectado automaticamente e enfileirado para upload:

```
Layout no S3:
  s3://{bucket}/gravacoes/{session_id}/
      webcam_000.mp4
      webcam_001.mp4
      screen_000.mp4
      screen_001.mp4
```

- Retry 3x com backoff exponencial (2s → 4s → 8s)
- Arquivo local deletado após upload bem-sucedido (configurável)
- Segmentos que falharam após todos os retries ficam em `uploader.failed_segments`
- O upload usa exclusivamente AWS S3 real via `boto3`

### Retenção S3

- Gravações: lifecycle rule expira automaticamente após **90 dias**
- Fotos de enrollment: limpeza manual semestral (janeiro e julho) por turma

### Entregáveis

- [x] `capture.py` gerenciando dois subprocessos FFmpeg com detecção de segmentos
- [x] `uploader.py` com fila thread-safe, retry e registro dos segmentos enviados/falhos
- [x] `RecorderConfig` integrado ao `AppConfig`
- [x] 14 testes unitários cobrindo lógica de segmentos e upload sem FFmpeg real

---

## Fase 4 — Browser lockdown ✅

### Base M4 de Chromium

```bash
chromium-browser \
    --kiosk \
    --no-first-run \
    --disable-translate \
    --disable-dev-tools \
    --disable-default-apps \
    --disable-background-networking \
    --disable-sync \
    --incognito \
    --start-fullscreen \
    "https://prairielearn.exemplo.edu.br"
```

Esta foi a base implementada para validar fullscreen, pausa por `SIGSTOP`,
retomada por `SIGCONT`, overlay de bloqueio e lockdown dinâmico de teclado.
A Fase 7 evolui esse modelo para Chromium maximizado com abas normais, extensão
allowlist local e limpeza explícita de perfil.

### Lockdown do SO

- Atalhos GNOME/WM desativados temporariamente via `gsettings` durante a sessão
- Super sozinho desativado via `org.gnome.mutter overlay-key=''` enquanto a prova roda
- Segunda camada via `xbindkeys` temporário para engolir `Alt+F4`, `Alt+Tab`, Super+teclas, `Ctrl+Alt+T`, `Ctrl+L`, `Ctrl+W`, screenshots e combinações similares
- Teclas especiais do servidor X tratadas via `setxkbmap -option srvrkeys:none` para reduzir escape por `Ctrl+Alt+Fn`
- Todos os valores alterados são restaurados por `Lockdown.disable()` ao encerrar a sessão
- Hardening permanente de produção ainda deve cobrir USB mass storage, Bluetooth, firewall/auditd e política de usuário
- No modo final com abas normais, atalhos úteis do navegador como `Ctrl+T` e `Ctrl+L` podem ser liberados; a barreira principal passa a ser allowlist + policies do Chromium, não bloqueio cego de todo atalho do browser

### Tela de bloqueio

Quando o proctoring engine emite `BLOCKED`:
1. SIGSTOP no processo Chromium (congela a prova)
2. Overlay fullscreen local informa claramente que a sessão foi bloqueada
3. Aguarda `engine.unblock()` após face re-match
4. SIGCONT no Chromium e retoma a sessão

### Entregáveis

- [x] `src/kiosk/chromium.py` — launcher + lifecycle
- [x] `src/kiosk/reidentify.py` — loop de re-identificação durante bloqueio
- [x] `src/kiosk/lockdown.py` — bloqueio temporário de atalhos de fuga com restauração
- [x] Fullscreen via `wmctrl` buscando janela pelo PID
- [x] BLOCKED → `SIGSTOP` → re-match → `SIGCONT`
- [x] Início da sessão → lockdown de teclas; fim da sessão → restauração
- [x] Limpeza garantida com restauração das extensões do Gnome no encerramento
- [x] Cobertura unitária do fluxo de kiosk e re-identificação
- [x] Fase 7 evolui o browser para modo controlado com extensão allowlist e limpeza de perfil

---

## Fase 5 — Session manager ✅

### Fluxo completo de uma sessão

```
NUC em modo manutenção (GNOME / usuário proctor)
  │
  ├─ 1. Operador aplica config no dashboard
  ├─ 2. Operador ativa auto-start no dashboard
  ├─ 3. NUC entra em modo prova no GNOME atual do usuário proctor
  ├─ 4. Overlay: "Sente-se e olhe para a câmera para iniciar a prova"
  ├─ 5. Face detection detecta rosto
  ├─ 6. Face recognition identifica aluno (student_id)
  ├─ 7. Verifica se aluno já fez a prova → rejeita se sim
  ├─ 8. Inicia gravação (capture.py + uploader.py)
  ├─ 9. Inicia proctoring engine
  ├─ 10. Abre Chromium controlado → PrairieLearn
  ├─ 11. Timer começa
  │
  ├─ DURANTE A PROVA
  │   ├─ Engine monitora continuamente
  │   ├─ Eventos logados em JSONL
  │   ├─ Segmentos upados incrementalmente
  │   ├─ Extensão/policies bloqueiam navegação fora da allowlist
  │   └─ Se BLOCKED: overlay + pausa + re-match → unblock()
  │
  └─ FIM (timer ou submit)
      ├─ Fecha Chromium
      ├─ Limpa cookies, storage, cache, histórico, senhas e perfil descartável do Chromium
      ├─ Para gravação
      ├─ Faz flush do último segmento MP4
      ├─ Aguarda uploader drenar a fila de segmentos
      ├─ Coleta eventos JSONL e gravações para o payload do dashboard
      └─ Remove lockdown/overlays e retorna o GNOME ao modo manutenção
```

### API local (FastAPI, porta 8000)

```
GET  /health              # Healthcheck: {status, state, mode, camera, s3}
GET  /status              # Estado de sessão e modo da estação
GET  /session             # Dados da sessão ativa
POST /session/start       # Início manual (admin)
POST /session/stop        # Fim forçado (professor)
POST /session/unblock     # Desbloqueio manual
POST /config              # Atualiza config da próxima sessão
POST /exam-mode/recover   # Recuperação manual: restaura manutenção
```

### Entregáveis

- [x] `src/core/session.py` com FSM completa
- [x] `src/api/server.py` + `src/api/routes.py`
- [x] Integração com face, proctor, recorder e módulo de browser/lockdown
- [x] `systemd/proctor.service` para autostart na NUC
- [x] `systemd/proctor-dashboard.service` para dashboard local reproduzível
- [x] Testes automatizados da FSM e da API local
- [x] Teste E2E do fluxo completo em hardware real

---

## Fase 6 — Dashboard do professor ✅

### Funcionalidades

- **Visão em tempo real** — status de cada NUC, qual aluno está sentado, tempo restante, contagem de flags
- **Configuração de prova** — criar sessão (turma, assessment URL, timer, thresholds), distribuir config para as NUCs
- **Revisão pós-prova** — lista de sessões com player de vídeo e timeline de eventos (clicar no evento pula para o timestamp no vídeo)
- **Exportar relatório** — CSV com eventos por aluno
- **Layout responsivo** — telas principais utilizáveis em celular

### Stack

```
Backend:  FastAPI (Python)
Frontend: HTMX + Jinja2 (sem build step)
Hosting:  EC2 t3.small na mesma região do S3
```

### Entregáveis

- [x] `src/dashboard/app.py` com FastAPI central
- [x] Templates Jinja2 + HTMX
- [x] Player de vídeo com timeline de eventos
- [x] API para comunicação NUC → dashboard

---

## Fase 7 — Modo prova final, allowlist e hardening 🟡

### Objetivo

Transformar a implementação atual de kiosk em um fluxo operacional de produção:
a NUC continua no GNOME atual do usuário `proctor`, mas entra em modo prova
somente quando o operador autoriza. O aluno usa Chromium maximizado com abas
normais, extensão allowlist local, policies do Chromium, perfil descartável e
lockdown forte de GNOME/X11 para impedir saída do ambiente autorizado.

### Arquitetura alvo

```
Modo manutenção
  └── usuário proctor + GNOME completo + dashboard/API local

Operador aplica config e ativa auto-start
  └── serviço entra em WAITING_STUDENT no próprio GNOME

Modo prova
  ├── overlay WAITING_STUDENT
  ├── reconhecimento facial inicial
  ├── FFmpeg grava webcam e tela do display GNOME atual
  ├── Chromium maximizado com abas normais
  ├── extensão allowlist local pré-instalada
  ├── policies bloqueiam incognito, devtools, sync, extensões e URLs internas
  ├── perfil descartável do Chromium por prova
  └── lockdown bloqueia fuga pelo GNOME/OS

Fim da prova
  ├── Chromium encerra
  ├── perfil do browser é limpo
  ├── gravação e upload finalizam
  ├── lockdown e overlays são removidos
  └── NUC permanece no GNOME de manutenção
```

### Componentes implementados no repositório

- **Modo prova no GNOME atual** — entrada/saída controladas por API/dashboard no usuário `proctor`.
- **Espera de aluno** — overlay fullscreen orienta o aluno a sentar e olhar para a câmera antes da identificação.
- **Browser controlado** — Chromium maximizado com abas normais, extensão allowlist local, bloqueio DNR e fallback por navegação para bloquear sites fora da allowlist.
- **Privacidade entre alunos** — perfil dedicado do browser e limpeza obrigatória de cookies, storage, cache, histórico, senhas e autofill ao final.
- **Operação pelo dashboard** — config pode ser distribuída sem derrubar manutenção; ativar auto-start entra no modo prova e desativar auto-start retorna à manutenção.
- **Hardening e validação** — lockdown de GNOME/X11, gravação no display atual, testes E2E em NUC real, contingência e provisionamento reproduzível.
- **Recuperação manual** — `POST /exam-mode/recover` e `scripts/recover_exam_mode.sh` restauram GNOME/atalhos se browser, overlay ou lockdown travarem.

### Estado atual

A parte versionada da M7 está implementada: FSM de modo da estação, endpoints
locais, comandos do dashboard, lockdown já no `WAITING_STUDENT`, overlay de
espera fixo, câmera mantida aberta durante auto-start, dock/Show Apps
desabilitado durante prova, display X11 explícito via `PROCTOR_REC_DISPLAY`,
extensão allowlist estática com bloqueio DNR, fallback por navegação e
hosts obrigatórios do PrairieLearn, Chromium maximizado com perfil descartável,
relaunch controlado, cleanup de perfil,
policy de hardening do Chromium, recuperação manual e testes unitários. A validação manual no GNOME real da NUC e
o hardening físico continuam centralizados em `MILESTONES.md`, seção `M7 — Modo
Prova Final, Allowlist e Hardening`.

---

## Fase 8 — Terraform + Ansible 🔲

### Recursos AWS (Terraform)

- S3 bucket `proctor-station` com lifecycle rules
- IAM user `proctor-station-nuc` com política mínima
- EC2 t3.small para dashboard
- CloudWatch alarms

### Provisionamento de NUCs (Ansible)

```yaml
roles:
  - base        # packages, usuário proctor, firewall, NTP, auditd
  - browser     # Chromium, extensão allowlist, policies e cleanup
  - proctor     # venv, código, credenciais AWS, systemd units
  - monitoring  # node_exporter, promtail
```

### Adicionar uma nova NUC

```bash
# 1. Instalar Ubuntu Desktop 24.04 na NUC
# 2. Adicionar ao inventory
echo "nuc-lab-05 ansible_host=192.168.1.105" >> ansible/inventory/hosts.yml
# 3. Rodar playbook
ansible-playbook playbooks/setup-nuc.yml --limit nuc-lab-05
# 4. Validar
curl http://192.168.1.105:8000/health
```

---

## Setup e desenvolvimento

### Pré-requisitos

- Ubuntu 24.04 Desktop (NUC) ou qualquer Linux (desenvolvimento)
- Python 3.12
- Webcam USB (Logitech C920 recomendada)
- Credenciais AWS com acesso ao bucket `proctor-station`

### Instalação

```bash
git clone <repo> /opt/proctor
cd /opt/proctor

# Configurar ambiente
cp .env.example .env
nano .env  # preencher credenciais

# Bootstrap completo (instala tudo, baixa modelos, roda testes)
chmod +x scripts/bootstrap.sh
./scripts/bootstrap.sh

# Em NUC já provisionada, reinstalar apenas policies do Chromium
sudo bash scripts/install_chromium_hardening.sh

source venv/bin/activate
```

### Configuração — variáveis de ambiente

Todas as configurações via `.env` na raiz do projeto (veja `.env.example` para referência completa).

Variáveis mais importantes:

```dotenv
# AWS
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=sa-east-1

# S3
PROCTOR_S3_BUCKET=proctor-station
PROCTOR_S3_REGION=sa-east-1
PROCTOR_S3_PHOTOS_PREFIX=fotos
PROCTOR_S3_RECORDINGS_PREFIX=gravacoes
PROCTOR_S3_SEGMENT_DURATION_SEC=300

# Gaze (calibrar com scripts/calibrate_gaze.py)
PROCTOR_GAZE_H_THRESHOLD=0.35
PROCTOR_GAZE_V_THRESHOLD=0.30
PROCTOR_GAZE_DURATION_SEC=5.0
PROCTOR_ABSENCE_TIMEOUT_SEC=5.0
PROCTOR_MULTI_FACE_BLOCK=true

# Gravação
PROCTOR_REC_DISPLAY=:1                # display X11 capturado no GNOME atual da NUC
PROCTOR_REC_SCREEN_SIZE=1280x720      # resolução final do vídeo de tela
PROCTOR_REC_WEBCAM_INPUT_FORMAT=mjpeg
PROCTOR_REC_FFMPEG_THREADS=1
PROCTOR_REC_PREVIEW_HOST=127.0.0.1
PROCTOR_REC_PREVIEW_PORT=18181
PROCTOR_REC_PREVIEW_WIDTH=640
PROCTOR_REC_PREVIEW_HEIGHT=360
PROCTOR_REC_PREVIEW_FPS=10
PROCTOR_REC_FFMPEG_CPU_CORES=6-7      # opcional
PROCTOR_REC_PROCTOR_CPU_CORES=0-5     # opcional
```

### Testes sem credenciais AWS

Enrollment e upload reais exigem credenciais AWS com acesso ao bucket configurado.
Os testes automatizados isolam S3 com doubles e patches, então podem ser
executados sem credenciais.

### Comandos úteis

```bash
# Testes automatizados (87 casos, sem câmera)
pytest tests/ -v

# Enrollment de turma
python scripts/enroll.py --turma ES2025-T1

# Calibrar thresholds de gaze
python scripts/calibrate_gaze.py

# Validar câmera + dlib
python scripts/test_camera.py --headless

# Teste de integração ao vivo (identificação + proctoring + gravação)
python scripts/test_integration.py --turma ES2025-T1

# Sem gravação (desenvolvimento local)
python scripts/test_integration.py --turma ES2025-T1 --no-record
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

**Lifecycle rules:**
- `gravacoes/` — objetos expiram após **90 dias**
- `fotos/` — limpeza manual semestral (janeiro e julho) por turma

---

## Custos estimados (mensal, 10 NUCs)

| Item | Custo estimado |
|---|---|
| S3 storage gravações (~100GB/mês, expiram em 90d) | ~$2.30 |
| S3 storage fotos (pequeno, permanente) | ~$0.05 |
| EC2 t3.small (dashboard) | ~$15 |
| Data transfer (upload) | ~$0.90 |
| **Total mensal AWS** | **~$18** |
| Hardware (NUC + webcam + monitor) × 10 | ~$7.000 (one-time) |

---

## Riscos e mitigações

| Risco | Impacto | Mitigação |
|---|---|---|
| Falso positivo de gaze (aluno pensando olha pro lado) | Bloqueio injusto | Threshold generoso + `gaze_duration_sec` longo + review pós-prova |
| dlib lento na NUC sem GPU | Lag no reconhecimento | HOG detector com `detection_scale=0.5`; benchmark com `test_camera.py` |
| Aluno parecido confunde o sistema | Acesso errado | Threshold restritivo (0.45) + confirmação visual |
| Internet instável durante prova | Upload falha | Fila local com retry; gravação 100% local mesmo sem internet |
| Chromium atualiza e quebra browser controlado | Prova comprometida | `apt-mark hold chromium-browser` + teste da extensão/policies antes de prova |
| Câmera ocupada por outro processo | FFmpeg falha | NUC dedicada; nenhum app extra instalado |
| Perfil do browser retém login do aluno | Vazamento de sessão | Perfil descartável por prova, policies de limpeza e cleanup explícito ao fim |
| Lockdown não restaura após erro | NUC difícil de operar | `finally`/cleanup, comando de saída pelo dashboard e acesso administrativo ao GNOME |
| Credenciais AWS expostas | Acesso indevido ao bucket | Usar `~/.aws/credentials`, nunca commitar credenciais; política IAM mínima |
| Privacidade / LGPD | Legal | Consentimento no enrollment; retenção máxima 90 dias para vídeos; acesso restrito |
