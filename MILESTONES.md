# Proctor Station — Milestones

Documento de referência do projeto. Cada milestone tem escopo fechado,
critério de conclusão objetivo e lista de arquivos afetados.
Nada avança para a próxima milestone sem os critérios da atual estarem
100% satisfeitos.

---

## Como usar este documento

- **Status**: `✅ Concluído` | `🔧 Em andamento` | `🔲 Pendente`
- Cada milestone começa com uma revisão do que foi herdado da anterior
- Bugs encontrados durante uma milestone são corrigidos nela, não adiante
- O critério de conclusão é o único árbitro — não basta "parecer funcionar"

---

## M0 — Setup e Ideação ✅

**Objetivo:** Definir o problema, escolher a stack e estruturar o repositório.

### Decisões tomadas

- **Hardware:** Intel NUC 12/13 (i5), 16GB RAM, SSD 256GB, webcam Logitech C920
- **OS:** Ubuntu 24.04 LTS Desktop — **sessão X11** (Wayland desabilitado via `/etc/gdm3/custom.conf`)
- **Linguagem:** Python 3.12
- **CV/ML:** OpenCV 4.x + dlib (HOG detector + ResNet 128-d)
- **Gaze:** dlib shape_predictor_68 + cv2.solvePnP (sem MediaPipe)
- **Gravação:** FFmpeg 6.x (H.264/libx264) — sem áudio
- **Upload:** boto3 → AWS S3 (sa-east-1)
- **Browser:** Chromium kiosk + extensão custom
- **Orquestrador:** FastAPI (API local na NUC)
- **Dashboard:** FastAPI + HTMX (EC2 t3.small)
- **IaC:** Terraform + Ansible

### Estrutura do repositório definida

```
proctor-station/
├── models/                      # Modelos dlib (download_models.sh)
├── data/
│   ├── encodings/               # .pkl por turma
│   └── sessions/                # logs JSONL e gravações por sessão
├── src/
│   ├── core/                    # config, models, s3_client
│   ├── face/                    # recognizer, detector
│   ├── proctor/                 # engine, gaze, events
│   ├── recorder/                # capture, uploader
│   ├── kiosk/                   # (M4)
│   ├── api/                     # (M5)
│   └── dashboard/               # (M6)
├── scripts/                     # CLIs e utilitários
├── tests/
├── .env
├── pyproject.toml
└── MILESTONES.md
```

### Critério de conclusão

- [x] Stack definida e documentada
- [x] Estrutura de diretórios criada
- [x] `pyproject.toml` com dependências declaradas
- [x] `bootstrap.sh` funcional (instala tudo do zero em Ubuntu 24.04)
- [x] `download_models.sh` baixa os 4 modelos dlib
- [x] `.env` com todas as variáveis documentadas

---

## M1 — Face Recognition ✅

**Objetivo:** Enrollment via S3 e identificação facial confiável via dlib.

### O que foi implementado

- `src/core/s3_client.py` — lista e baixa fotos do S3 por turma
- `src/face/recognizer.py` — enrollment + `identify()` + `identify_best_of_n()`
- `src/face/detector.py` — detecção leve HOG para o loop de proctoring
- `scripts/enroll.py` — CLI completa (enroll, info, list, remove, force)

### Decisões técnicas

- `student_id` = identificador institucional único gerado pela faculdade (ex: `henriquels5`)
- Threshold de distância: `0.45` (mais restritivo que o default `0.6`)
- 3 jitters por foto no enrollment para maior robustez
- HOG detector com `detection_scale=0.5` para performance na NUC

### Bugs corrigidos nesta milestone

- `local_s3_client.py` removido — mock S3 eliminado, sistema usa AWS real
- `config.py`: variáveis `PROCTOR_S3_MOCK` e `PROCTOR_S3_MOCK_DIR` removidas
- `.env`: limpeza de variáveis obsoletas e organização por seções

### Critério de conclusão

- [x] `pytest tests/test_face_recognition.py` — 39 casos passando
- [x] `enroll.py --turma T2026-T1` — baixa fotos do S3 e gera `.pkl`
- [x] Identificação correta de aluno real na webcam (confiança > 0.5)

---

## M2 — Proctoring Engine ✅

**Objetivo:** FSM de monitoramento com gaze estimation e log de eventos.

### O que foi implementado

- `src/proctor/gaze.py` — pose estimation via solvePnP
- `src/proctor/engine.py` — FSM: NORMAL → GAZE_WARN → BLOCKED
- `src/proctor/events.py` — EventLogger com roundtrip JSONL
- `scripts/calibrate_gaze.py` — calibração visual de thresholds

### Bugs corrigidos nesta milestone

- `engine.py`: timer warn→block usava `gaze_block_sec` em vez de `gaze_duration_sec`
- `engine.py`: `_handle_no_face` ignorava estado `GAZE_WARN` — rosto sumindo em GAZE_WARN ficava preso; corrigido para transitar para `ABSENCE` em qualquer estado não-BLOCKED
- `test_proctor_engine.py`: helper `_gaze()` usava `pitch=0.0` como neutro, mas solvePnP retorna pitch ≈ 180° com cabeça ereta — corrigido para `pitch=180.0`
- `config.py`: campo morto `gaze_block_sec` removido do `ProctorConfig`
- `README.md`: `PROCTOR_GAZE_BLOCK_SEC` removido da tabela de parâmetros

### Critério de conclusão

- [x] `pytest tests/test_proctor_engine.py` — 40 casos passando
- [x] FSM transita corretamente entre todos os estados incluindo GAZE_WARN → ABSENCE
- [x] Log JSONL gerado e legível após sessão

---

## M3 — Gravação, Upload e Loop Unificado ✅

**Objetivo:** Gravação de webcam + tela via FFmpeg com upload incremental ao S3,
mantendo o proctoring separado da gravação para preservar FPS e reduzir contenção de CPU.

### Arquitetura definida

**A câmera física não fica mais aberta em paralelo durante a sessão.**
O OpenCV usa `/dev/video0` só na identificação inicial. Depois disso, a câmera
é liberada e o FFmpeg vira o único dono da webcam.

```
/dev/video0 ──► OpenCV (identificação inicial)
                  │
                  └── aluno identificado → libera /dev/video0

/dev/video0 ──► FFmpeg v4l2 ──► split=2
                                ├──► [record]  ──► webcam_%03d.mp4 ──► S3
                                └──► [preview] ──► udp://127.0.0.1:18181 ──► OpenCV (gaze + reidentify)

FFmpeg separado: x11grab → screen_%03d.mp4 ──► S3
```

### Decisões técnicas

- O FFmpeg é o único dono de `/dev/video0` durante a sessão ativa
- O proctoring contínuo lê um preview local de baixa latência, não a câmera física
- Gravação de webcam via `v4l2` direto no FFmpeg — não depende do FPS do proctoring
- Sem áudio — simplifica o pipeline e reduz CPU
- Captura de tela via x11grab — requer sessão X11 (Wayland desabilitado)
- Webcam e tela usam `use_wallclock_as_timestamps` + `fps_mode passthrough`
- MP4s finais saem em H.264 `High` + `yuv420p` + `faststart` para compatibilidade com browser/dashboard
- Segmentação de 5min com upload incremental ao S3
- Afinidade de CPU configurável — últimos núcleos podem ser reservados para o FFmpeg e divididos entre webcam/tela
- `stop()` encerra webcam/tela por SIGINT e flush do segmento final — garante arquivo válido

### Bugs corrigidos nesta milestone

- `capture.py`: identificação inicial separada do proctoring contínuo — evita disputa permanente por `/dev/video0`
- `capture.py`: gravação da webcam desacoplada do loop do proctoring, evitando vídeo acelerado/degradado
- `capture.py`: preview local de webcam adicionado para gaze/re-identificação sem reabrir a câmera física
- `capture.py`: timestamps de relógio real e `fps_mode passthrough` adicionados para reduzir drift entre webcam e tela
- `capture.py`: saída MP4 padronizada em `yuv420p` + `faststart` para reprodução confiável no dashboard
- `capture.py`: afinidade opcional de CPU adicionada aos processos FFmpeg, com divisão entre webcam e tela
- `capture.py`: áudio removido de todos os comandos FFmpeg
- `test_integration.py`: câmera aberta uma única vez, nunca fechada entre fases
- `test_integration.py`: janela OpenCV removida — output no terminal, encerramento via Ctrl+C
- `.env`: `PROCTOR_REC_DISPLAY=:1`, `PROCTOR_REC_WEBCAM_INPUT_FORMAT`, `PROCTOR_REC_FFMPEG_THREADS` e afinidade de CPU adicionados

### Critério de conclusão

- [x] Roda sem `Device or resource busy`
- [x] Identificação → proctoring sem fechar/reabrir câmera
- [x] Ctrl+C encerra limpo — sem traceback, `finally` executa
- [x] `webcam_000.mp4` e `screen_000.mp4` gerados corretamente
- [x] Upload confirmado em `s3://proctor-station/gravacoes/`
- [x] Arquivos de webcam e tela reproduzíveis no browser/dashboard (`H.264 High`, `yuv420p`)
- [x] Vídeo webcam gravado com duração próxima da tela sem depender do FPS do proctoring
- [x] `pytest tests/` — todos os testes passando

---

## M4 — Browser Lockdown ✅

**Objetivo:** Chromium em modo kiosk com bloqueio e re-identificação facial.

### O que foi implementado

- `src/kiosk/chromium.py` — launcher fullscreen via wmctrl (PID), SIGSTOP/SIGCONT
- `src/kiosk/reidentify.py` — loop de re-identificação facial durante bloqueio
- `src/kiosk/lockdown.py` — placeholder para M7

### Decisões tomadas

- Fullscreen via `wmctrl -i -r <win_id> -b add,fullscreen` pelo PID — evita pegar janela errada por nome
- Extensões do Gnome (`ubuntu-dock`, `tiling-assistant`) desabilitadas durante sessão e restauradas no `finally`
- Lockdown de teclas (Alt+F4, Super, Ctrl+Alt+T) movido para M7 — hardening de produção
- Allowlist de domínios deixada para a próxima etapa
- Encerramento em produção (timer, submit, professor) definido na M5 — por ora Ctrl+C

### Bugs corrigidos nesta milestone

- `chromium.py`: `wmctrl` buscava janela por nome e pegava VSCode/Firefox — corrigido para buscar pelo PID
- `test_integration.py`: extensões do Gnome não eram restauradas se script morria abruptamente — `finally` garante restauração
- `lockdown.py`: tentativas de xbindkeys e gsettings falhavam silenciosamente — simplificado para placeholder

### Critério de conclusão

- [x] Chromium abre em fullscreen
- [x] BLOCKED → Chromium congela (SIGSTOP)
- [x] Aluno olha para câmera → re-identificado → Chromium retoma (SIGCONT)
- [x] Ctrl+C encerra limpo — extensões do Gnome restauradas
- [x] `pytest tests/` — todos passando
- [ ] Lockdown de teclas — M7
- [ ] Allowlist de domínios — próxima etapa

---

## M5 — Session Manager ✅

**Objetivo:** Orquestrador E2E que gerencia o ciclo de vida completo de uma sessão
de prova, exposto via FastAPI local na NUC.

### Escopo

FSM de sessão de alto nível:

```
IDLE → IDENTIFYING → SESSION → BLOCKED → SESSION → UPLOADING → IDLE
```

- `src/core/session.py` — FSM principal, integra face + proctor + recorder + kiosk
- `src/api/server.py` + `src/api/routes.py` — FastAPI com os endpoints abaixo
- `systemd` unit para autostart na NUC

### API

```
GET  /health          → {status, state, camera_ok, s3_ok}
GET  /status          → estado atual da FSM
GET  /session         → dados da sessão ativa
POST /session/start   → início manual
POST /session/stop    → fim forçado
POST /session/unblock → desbloqueio manual
POST /config          → atualiza config da próxima sessão
```

### Critério de conclusão

- [x] `src/core/session.py` implementado com FSM `IDLE → IDENTIFYING → SESSION → BLOCKED → UPLOADING → IDLE`
- [x] `src/api/server.py` + `src/api/routes.py` expõem `/health`, `/status`, `/session`, `/session/start`, `/session/stop`, `/session/unblock`, `/config`
- [x] Integração de código com face, proctor, recorder e kiosk
- [x] `systemd/proctor.service` adicionado ao repositório
- [x] `/health` retorna 200 com `status`, `state`, `camera_ok`, `s3_ok`
- [x] `pytest` cobre FSM e endpoints principais
- [x] `systemctl start proctor` validado em NUC real
- [x] Aluno senta → identificado → prova inicia automaticamente em hardware real
- [x] BLOCKED → tela bloqueia → re-identificação → retoma validado em hardware real
- [x] Fim da prova → gravação encerrada → upload completo → IDLE validado em hardware real
- [x] Reinício da NUC → serviço sobe automaticamente

---

## M6 — Dashboard do Professor ✅

**Objetivo:** Interface web para o professor monitorar sessões em tempo real
e revisar gravações pós-prova.

### Escopo

- Status em tempo real de cada NUC (aluno, estado, tempo restante, flags)
- Configuração de prova (turma, URL, timer, thresholds)
- Revisão pós-prova: player de vídeo + timeline de eventos
- Exportar relatório CSV por turma
- Layout responsivo para uso em celular

### Stack

```
Backend:  FastAPI (Python) — EC2 t3.small
Frontend: HTMX + Jinja2
```

### Critério de conclusão

- [x] Dashboard acessível via navegador na rede local
- [x] Status de todas as NUCs atualiza em tempo real (polling ou SSE)
- [x] Player de vídeo sincronizado com timeline de eventos JSONL
- [x] Exportar CSV com eventos por aluno funciona
- [x] Páginas principais utilizáveis em tela de celular

---

## M7 — Testes, Hardening e IaC 🔲

**Objetivo:** Sistema pronto para produção — seguro, reproduzível e monitorado.

### Escopo

**Testes**
- Suite E2E com câmera real (pelo menos 1 NUC)
- Teste de carga: 10 NUCs simultâneas fazendo upload

**Hardening**
- Firewall (ufw): só portas 22 e 8000 abertas
- `auditd` para log de acesso a arquivos sensíveis
- User `proctor` sem sudo
- `apt-mark hold` nos pacotes críticos (chromium, ffmpeg, dlib)

**IaC**
- Terraform: S3 bucket + lifecycle rules + IAM user com política mínima + EC2
- Ansible: roles `base`, `kiosk`, `proctor`, `monitoring`
- `node_exporter` + `promtail` para métricas e logs

### Critério de conclusão

- [ ] `ansible-playbook setup-nuc.yml` configura NUC limpa do zero sem intervenção
- [ ] `terraform apply` sobe infraestrutura AWS completa
- [ ] 10 NUCs simultâneas: CPU < 80%, sem drops de upload
- [ ] Penetration test básico: usuário `proctor` não consegue escalar privilégios
- [ ] Plano de contingência documentado (NUC falha, internet cai, câmera trava)

---

## Registro de decisões técnicas

| Milestone | Decisão | Motivo |
|---|---|---|
| M0 | dlib em vez de MediaPipe | Sem dependência de GPU, roda bem no NUC i5 |
| M0 | solvePnP em vez de eye tracking puro | Mais robusto com óculos e iluminação variável |
| M0 | FFmpeg segmentado em vez de gravação contínua | Upload incremental sem esperar o fim da prova |
| M0 | X11 em vez de Wayland | x11grab não funciona com Wayland; kiosk em X11 é mais estável |
| M1 | Threshold 0.45 em vez do default 0.6 do dlib | Reduz falsos positivos em ambiente controlado |
| M1 | `student_id` = identificador institucional | Gerado pela faculdade, único, não precisa de mapeamento |
| M2 | `gaze_duration_sec` como único timer warn→block | `gaze_block_sec` era redundante e causava bug |
| M3 | FFmpeg grava a webcam direto via v4l2 | A gravação não herda o FPS reduzido do proctoring |
| M3 | Afinidade de CPU separa FFmpeg e proctoring | Reduz contenção no mesmo núcleo durante a prova |
| M3 | Sem áudio na gravação | Reduz CPU e simplifica o pipeline |
| M3 | `PROCTOR_FACE_PIPE_FPS` configurável | FPS real com dlib (~8fps) difere do nominal (30fps) — declarar errado acelera o vídeo |
| M4 | wmctrl por PID em vez de nome | Evita fullscreen na janela errada (VSCode, Firefox) |
| M4 | Lockdown de teclas movido para M7 | Em desenvolvimento sempre precisa de saída de emergência |
| M4 | Allowlist de domínios deixada para a próxima etapa | Ainda depende de definir a estratégia final de navegação |
