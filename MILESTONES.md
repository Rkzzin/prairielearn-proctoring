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
- **OS:** Ubuntu 24.04 LTS Desktop — GNOME para manutenção e prova com lockdown temporário
- **Linguagem:** Python 3.12
- **CV/ML:** OpenCV 4.x + dlib (HOG detector + ResNet 128-d)
- **Gaze:** dlib shape_predictor_68 + cv2.solvePnP (sem MediaPipe)
- **Gravação:** FFmpeg 6.x (H.264/libx264) — sem áudio
- **Upload:** boto3 → AWS S3 (sa-east-1)
- **Browser:** Chromium controlado no GNOME atual + extensão allowlist custom
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
│   ├── kiosk/                   # browser controlado, lockdown e overlays (M4/M7)
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

- `s3_client.py`: fluxo de enrollment consolidado em AWS S3 real via `boto3`
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
- `README.md`: tabela de parâmetros alinhada ao timer único `gaze_duration_sec`

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

FFmpeg separado: x11grab + xrandr → scale(PROCTOR_REC_SCREEN_SIZE) → screen_%03d.mp4 ──► S3
```

### Decisões técnicas

- O FFmpeg é o único dono de `/dev/video0` durante a sessão ativa
- O proctoring contínuo lê um preview local de baixa latência, não a câmera física
- Gravação de webcam via `v4l2` direto no FFmpeg — não depende do FPS do proctoring
- Sem áudio — simplifica o pipeline e reduz CPU
- Captura de tela via `x11grab` — requer sessão X11 (Wayland desabilitado)
- Resolução real da tela detectada por `xrandr --current`; `PROCTOR_REC_SCREEN_SIZE` é a resolução final após downscale
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
- `capture.py`: captura de tela agora detecta a resolução real via `xrandr` e faz downscale para a saída configurada
- `capture.py`: afinidade opcional de CPU adicionada aos processos FFmpeg, com divisão entre webcam e tela
- `capture.py`: áudio removido de todos os comandos FFmpeg
- `session.py`: câmera física liberada após identificação e OpenCV reconectado ao preview local gerado pelo FFmpeg
- `test_integration.py`: janela OpenCV removida — output no terminal, encerramento via Ctrl+C
- `.env`: `PROCTOR_REC_DISPLAY`, `PROCTOR_REC_WEBCAM_INPUT_FORMAT`, `PROCTOR_REC_FFMPEG_THREADS`, preview UDP e afinidade de CPU adicionados

### Critério de conclusão

- [x] Roda sem `Device or resource busy`
- [x] Identificação pela câmera física → gravação via FFmpeg → proctoring pelo preview local
- [x] Ctrl+C encerra limpo — sem traceback, `finally` executa
- [x] `webcam_000.mp4` e `screen_000.mp4` gerados corretamente
- [x] Upload confirmado em `s3://proctor-station/gravacoes/`
- [x] Arquivos de webcam e tela reproduzíveis no browser/dashboard (`H.264 High`, `yuv420p`)
- [x] Vídeo webcam gravado com duração próxima da tela sem depender do FPS do proctoring
- [x] `pytest tests/` — todos os testes passando

---

## M4 — Browser Lockdown ✅

**Objetivo:** Validar a base de browser controlado, bloqueio temporário e
re-identificação facial. Esta milestone implementou o primeiro envelope em
fullscreen/kiosk; a arquitetura final de produção será concluída no M7 com
GNOME atual endurecido, Chromium maximizado e allowlist real.

### O que foi implementado

- `src/kiosk/chromium.py` — launcher fullscreen via wmctrl (PID), SIGSTOP/SIGCONT
- `src/kiosk/reidentify.py` — loop de re-identificação facial durante bloqueio
- `src/kiosk/lockdown.py` — bloqueio temporário de atalhos de fuga com restauração

### Decisões tomadas

- Fullscreen via `wmctrl -i -r <win_id> -b add,fullscreen` pelo PID — evita pegar janela errada por nome
- Extensões do Gnome (`ubuntu-dock`, `tiling-assistant`) desabilitadas durante sessão e restauradas no `finally`; isso passa a ser parte do lockdown do GNOME atual
- Lockdown de teclas aplicado dinamicamente na sessão: `gsettings` para GNOME/WM e Super sozinho (`org.gnome.mutter overlay-key`), `xbindkeys` como segunda camada e `setxkbmap -option srvrkeys:none` para teclas especiais do X
- Allowlist de domínios e UX de abas deixadas para M7
- O modo final não deve depender de `--incognito`, porque o aluno precisa usar abas normais e a extensão allowlist; limpeza do perfil será responsabilidade explícita do M7
- Encerramento em produção (timer, submit, professor) definido na M5 — por ora Ctrl+C

### Bugs corrigidos nesta milestone

- `chromium.py`: `wmctrl` buscava janela por nome e pegava VSCode/Firefox — corrigido para buscar pelo PID
- `test_integration.py`: extensões do Gnome não eram restauradas se script morria abruptamente — `finally` garante restauração
- `lockdown.py`: bloqueio leve, sem polling, com restauração explícita no encerramento

### Critério de conclusão

- [x] Chromium abre em fullscreen
- [x] BLOCKED → Chromium congela (SIGSTOP)
- [x] Aluno olha para câmera → re-identificado → Chromium retoma (SIGCONT)
- [x] Ctrl+C encerra limpo — extensões do Gnome restauradas
- [x] Lockdown de teclas aplicado durante a prova e restaurado no fim
- [x] `pytest tests/` — todos passando
- [x] Substituição do kiosk/incognito por browser maximizado + allowlist no GNOME atual — M7

---

## M5 — Session Manager ✅

**Objetivo:** Orquestrador E2E que gerencia o ciclo de vida completo de uma sessão
de prova, exposto via FastAPI local na NUC.

### Escopo

FSM de sessão de alto nível:

```
IDLE → IDENTIFYING → SESSION → BLOCKED → SESSION → UPLOADING → IDLE
```

- `src/core/session.py` — FSM principal, integra face + proctor + recorder + browser/lockdown
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
- [x] Integração de código com face, proctor, recorder e módulo de browser/lockdown
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

## M7 — Modo Prova Final, Allowlist e Hardening 🟡

**Objetivo:** Transformar a base atual em um modo de prova de produção no GNOME
atual do usuário `proctor`. O modo prova usa Chromium maximizado com abas
normais, allowlist real de sites, perfil descartável, limpeza garantida e
lockdown forte de GNOME/X11.

### Arquitetura alvo

```
GNOME / proctor / manutenção
  ├── dashboard local ou central distribui config
  └── operador ativa auto-start, que é o próprio modo prova

WAITING_STUDENT
  ├── GNOME atual entra em lockdown temporário
  ├── overlay: "Sente-se e olhe para a câmera para iniciar a prova"
  └── loop de identificação facial tenta aluno elegível

SESSION
  ├── FFmpeg grava webcam e tela do display GNOME atual
  ├── OpenCV lê preview UDP gerado pelo FFmpeg
  ├── Chromium maximizado com abas normais
  ├── extensão allowlist local controla UX e navegação
  ├── policies Chromium bloqueiam rotas internas/perigosas
  ├── perfil descartável isola cookies/storage/login por prova
  └── lockdown bloqueia saída do GNOME/OS

FINALIZAÇÃO
  ├── encerra Chromium
  ├── limpa perfil/cookies/storage/cache/histórico/senhas
  ├── encerra gravação/upload
  ├── registra payload final no dashboard
  └── restaura atalhos/overlays e retorna ao GNOME de manutenção
```

### Escopo detalhado

**Controle de modo no GNOME atual**
- [x] Fundir auto-start e modo prova: `SET_AUTOSTART=true` entra em `WAITING_STUDENT`; `false` retorna à manutenção.
- [x] Garantir que ativar auto-start aplique lockdown antes de permitir interação do aluno.
- [x] Garantir que desativar auto-start restaure atalhos/estado visual mesmo em caso de erro.
- [x] Criar mecanismo de recuperação manual caso o lockdown ou Chromium trave.

**FSM e API**
- [x] Adicionar estados de modo de estação separados dos estados de sessão, ou expandir a FSM com `MAINTENANCE`, `EXAM_READY` e `WAITING_STUDENT`.
- [x] Preservar a FSM de sessão `IDENTIFYING → SESSION → BLOCKED → UPLOADING`.
- [x] Ajustar `SessionAutoStartWorker` para operar somente quando a estação está em `WAITING_STUDENT`.
- [x] Com auto-start ativo, manter loop de identificação até aluno elegível ser encontrado.
- [x] Se aluno já concluiu a mesma configuração, manter espera por outro aluno e mostrar mensagem apropriada.
- [x] Expor status do modo atual no `/status` e no heartbeat do dashboard.
- [x] Garantir que receber `APPLY_CONFIG` não altera o modo da estação automaticamente.
- [x] Garantir que `SET_AUTOSTART` é o comando único de entrada/saída do modo prova.
- [x] Garantir que `STOP_SESSION` encerra prova, limpa browser e devolve a estação ao estado esperado.

**Overlay e experiência do aluno**
- [x] Adicionar modo `waiting` em `src/kiosk/overlay_app.py`.
- [x] Mostrar overlay fullscreen/topmost com "Sente-se e olhe para a câmera para iniciar a prova".
- [x] Manter o overlay `waiting` padrão durante falhas de identificação; detalhes ficam em logs/dashboard.
- [x] Manter o overlay `waiting` padrão quando o aluno já fez a prova nesta configuração.
- [x] Evitar flicker do overlay durante tentativas repetidas de auto-start.
- [x] Manter overlay `blocked` para `ABSENCE`, `GAZE` e múltiplos rostos.
- [x] Garantir que overlay não capture a câmera; identificação continua pelo `SessionManager`.
- [x] Garantir que overlay fecha limpo ao iniciar prova, bloquear sessão ou sair do modo prova.
- [x] Testar overlay no display real do GNOME da NUC.

**Chromium controlado**
- [x] Renomear/refatorar `ChromiumKiosk` para refletir browser controlado, ou manter compatibilidade com nome antigo e documentar.
- [x] Remover `--kiosk` do modo final.
- [x] Remover `--incognito` do modo final.
- [x] Remover `--disable-extensions` quando a extensão allowlist for usada.
- [x] Abrir Chromium maximizado ou fullscreen gerenciado pelo WM, preservando abas, botão `+` e barra de endereço.
- [x] Usar perfil descartável dedicado por prova, sem usar o perfil pessoal do Chromium do `proctor`.
- [x] Definir diretório operacional estável e limpo para perfil descartável, por exemplo `/tmp/proctor-chromium-profile` ou `data/runtime/chromium-profile`.
- [x] Bloquear first-run, translate, sync, default apps, crash dialogs e background networking desnecessário.
- [x] Impedir DevTools por policy e por atalhos.
- [x] Impedir incognito por policy.
- [ ] Impedir instalação ou ativação de extensões não autorizadas por policy.
- [x] Impedir gerenciador de senhas, autofill e salvamento de credenciais.

**Allowlist e extensão**
- [x] Criar extensão local estática versionada no repositório, por exemplo `src/kiosk/allowlist_extension/`.
- [x] A extensão deve ser pré-criada; em runtime só pode ser escrito arquivo pequeno de config por prova.
- [x] Definir schema do config: URL inicial, domínios permitidos, nomes amigáveis, modo estrito e textos.
- [x] Implementar normalização de allowlist: domínio raiz, subdomínios, scheme opcional e portas se necessário.
- [x] Incluir automaticamente hosts obrigatórios da URL inicial do PrairieLearn.
- [x] Implementar página de nova aba da extensão com links para sites permitidos.
- [x] Implementar campo de navegação/pesquisa na nova aba que oriente o aluno para sites permitidos.
- [x] Implementar bloqueio de navegação principal fora da allowlist via `declarativeNetRequest`.
- [x] Adicionar fallback via `webNavigation`/`tabs.onUpdated` para bloquear navegação principal caso DNR não aplique.
- [x] Reescrever `config.json` da extensão ao aplicar config/ativar modo prova para evitar allowlist stale.
- [x] Começar bloqueando `main_frame` e documentar que subresources/CDNs ficam liberados para não quebrar sites permitidos.
- [x] Adicionar opção futura de modo estrito para `sub_frame`/subresources se necessário.
- [x] Criar página de bloqueio explicando que o site não é permitido.
- [x] Bloquear redirects para fora da allowlist.
- [ ] Testar digitação direta na barra de endereço, links, nova aba, reload, histórico e redirects.
- [x] Decidir se Google será permitido só quando explicitamente incluído; padrão recomendado: não liberar Google amplo.

**Policies do Chromium**
- [ ] Definir estratégia de policies para a NUC inteira ou flags/runtime sem quebrar manutenção normal do `proctor`.
- [ ] Configurar `URLBlocklist` e `URLAllowlist` como barreira adicional à extensão.
- [x] Bloquear `chrome://settings`, `chrome://extensions`, `chrome://policy`, `chrome://flags`, `devtools://`, `file://` e esquemas não necessários.
- [x] Configurar `DeveloperToolsAvailability` para bloquear DevTools.
- [x] Configurar `IncognitoModeAvailability` para desabilitar incognito.
- [x] Configurar `SyncDisabled` e/ou browser sign-in para impedir sync do aluno.
- [x] Criar `scripts/install_chromium_hardening.sh` para instalar policy gerenciada que bloqueia páginas internas e esquemas perigosos.
- [ ] Configurar policies de extensões para permitir apenas a extensão local de allowlist.
- [x] Configurar `ClearBrowsingDataOnExitList` com cookies, site data, cache, histórico, downloads, senhas, autofill e hosted app data.
- [ ] Testar `chrome://policy` durante desenvolvimento e bloquear acesso na prova final.

**Limpeza de perfil e privacidade**
- [x] Implementar cleanup explícito do perfil do Chromium no fim da prova.
- [x] Limpar cookies e site storage antes da próxima prova.
- [x] Limpar IndexedDB, Local Storage, Session Storage, Cache Storage e Service Workers.
- [x] Limpar histórico, downloads, autofill e senhas salvas.
- [x] Garantir que logout de PrairieLearn e demais sites não dependa do aluno clicar em "sair".
- [x] Fazer cleanup depois que o browser fecha, não antes de abrir, para não atrasar o início.
- [ ] Se cleanup falhar, bloquear início de nova prova até resolver ou recriar perfil.
- [x] Registrar evento/log de cleanup bem-sucedido ou falho.
- [ ] Testar duas provas consecutivas com alunos diferentes na mesma NUC.

**Lockdown de OS no modo prova**
- [x] Aplicar lockdown temporário no GNOME atual.
- [x] Usar `gsettings`, XKB e `xbindkeys` temporário com restauração automática.
- [x] Manter bloqueios de fuga: `Alt+Tab`, `Alt+F4`, Super, `Ctrl+Alt+Fn`, `Ctrl+Alt+T`, logout, menu do WM, screenshots e terminal.
- [x] Bloquear hot corner/Activities com `enable-hot-corners=false` e guard overlay no topo do display.
- [x] Desabilitar temporariamente Ubuntu Dock/Show Apps e esconder o botão de aplicativos durante lockdown.
- [x] Reavaliar bloqueio de `Ctrl+T` e `Ctrl+L`: devem ser liberados se allowlist/policies estiverem válidas.
- [x] Bloquear `Ctrl+N`, `Ctrl+Shift+N`, `Ctrl+W`, `Ctrl+Q`, `Alt+Left/Right` se necessário.
- [ ] Validar que mouse não alcança painel, dock, menu de aplicações ou área de trabalho útil.
- [ ] Validar comportamento quando Chromium fecha, crasha ou é minimizado.
- [x] Implementar relaunch controlado ou violação registrada se Chromium sair durante a prova.

**Gravação e display**
- [x] Confirmar `PROCTOR_REC_DISPLAY=:1` para a sessão GNOME atual da NUC.
- [x] Validar `xrandr --current` no display real da NUC.
- [x] Validar `x11grab` gravando a tela real vista pelo aluno.
- [ ] Garantir que o overlay `waiting` aparece na gravação apenas antes da prova, se desejado.
- [ ] Garantir que overlay `blocked` aparece na gravação durante bloqueio.
- [x] Confirmar que FFmpeg continua único dono da webcam durante sessão ativa.
- [x] Confirmar que OpenCV consome preview UDP sem disputar `/dev/video0`.
- [x] Manter a câmera física aberta enquanto a estação está em `WAITING_STUDENT` com auto-start ativo.
- [x] Liberar a câmera física ao sair do modo prova para manutenção.

**Dashboard e operação**
- [x] Usar botão único de auto-start como entrada/saída do modo prova por estação.
- [x] Mostrar modo da estação separado do status da sessão.
- [x] Mostrar quando a NUC está `WAITING_STUDENT`.
- [ ] Mostrar erro operacional de câmera, browser, cleanup ou policies.
- [x] Evitar que config distribuída automaticamente derrube manutenção ativa.
- [x] Registrar no dashboard quando a NUC entra e sai do modo prova.
- [ ] Documentar rotina diária: ligar NUC, validar saúde, aplicar config, entrar modo prova, acompanhar, encerrar, voltar manutenção.

**Instalação, hardening e IaC**
- [x] Atualizar `scripts/bootstrap.sh` com `xbindkeys`, `wmctrl`, Chromium e dependências do lockdown atual.
- [x] Criar script idempotente para instalar policies do browser controlado compatíveis com o GNOME atual.
- [ ] Criar script de reversão para remover policies/artefatos de modo prova sem apagar dados de sessões.
- [x] Manter `systemd/proctor.service` como serviço único da estação.
- [ ] Definir ownership de arquivos: código, configs, extensão, policies e dados.
- [ ] Aplicar firewall (`ufw`) com portas mínimas.
- [ ] Configurar `auditd` para arquivos sensíveis e ações de modo prova.
- [ ] Bloquear USB mass storage se não necessário.
- [ ] Desabilitar Bluetooth se não usado.
- [ ] Aplicar `apt-mark hold` em Chromium, FFmpeg e pacotes críticos depois de validar versão.
- [ ] Atualizar Ansible com roles `base`, `browser`, `proctor` e `monitoring`.
- [ ] Atualizar Terraform para S3, lifecycle, IAM mínimo, EC2 dashboard e alarmes (M8 usa portal AWS manual por decisão, não Terraform — ver M8).
- [ ] Configurar métricas/logs com `node_exporter` e `promtail` ou equivalente.

**Testes**
- [x] Teste unitário para normalização de allowlist.
- [x] Teste unitário para geração de config da extensão.
- [x] Teste unitário para flags do Chromium controlado.
- [x] Teste unitário para cleanup de perfil.
- [x] Teste unitário para estados `WAITING_STUDENT`/modo de estação.
- [x] Teste unitário para overlay `waiting`.
- [x] Teste unitário para auto-start como comando de entrar/sair modo prova.
- [x] Teste unitário para lockdown aplicado já no `WAITING_STUDENT`.
- [x] Teste unitário para recuperação manual de modo prova.
- [x] Teste unitário para relaunch controlado do Chromium.
- [x] Teste unitário para overlay fixo e câmera persistente no auto-start.
- [x] Teste manual em NUC: GNOME manutenção → modo prova no mesmo usuário → waiting overlay.
- [x] Teste manual em NUC: identificação → gravação → Chromium maximizado.
- [ ] Teste manual em NUC: abrir nova aba pelo botão `+`.
- [ ] Teste manual em NUC: site permitido funciona.
- [ ] Teste manual em NUC: site não permitido é bloqueado.
- [ ] Teste manual em NUC: `Alt+Tab`, `Alt+F4`, Super, `Ctrl+Alt+Fn` não escapam.
- [x] Teste manual em NUC: bloqueio por ausência/gaze congela prova, mostra overlay e retoma após reidentificação.
- [ ] Teste manual em NUC: finalização limpa perfil e remove login.
- [ ] Teste manual em NUC: segunda prova não herda sessão do aluno anterior.
- [ ] Teste de carga: 10 NUCs simultâneas fazendo upload e heartbeat.
- [ ] Teste de contingência: internet cai, dashboard cai, câmera trava, Chromium crasha e NUC reinicia.

### Registro de validação em NUC real — 2026-07-29

Base de código: `7be1601`. Webcam Logitech C920 em `/dev/video0`, display `:1`
a 1920x1080, turma `T2026-T1`. Duas sessões completas de ponta a ponta pelo
serviço local, com `systemctl` ativo.

| Sessão | Observado |
|---|---|
| `T2026-T1_henriquels5_20260729_222146` | identificação em ~1 s; 593 frames de proctoring em 80 s (~7,4 fps); **6 `GAZE_BLOCKED` + 1 `ABSENCE_BLOCKED` com 7 `SESSION_RESUMED`** (SIGSTOP → reidentificação → SIGCONT); 2 segmentos no S3 (7,2 MB) |
| `T2026-T1_henriquels5_20260729_225635` | identificação em ~3 s; `screen_000.mp4` h264 1280x720 yuv420p **68,2 s** e `webcam_000.mp4` h264 + **aac** 67,9 s, ambos baixados do S3 e validados com `ffprobe`; dashboard `COMPLETED` com 2 gravações |

Confirmado em ambas: `xrandr --current` detectou 1920x1080 e o `x11grab` gravou
a tela real; `fuser /dev/video0` mostrou **somente o ffmpeg** durante a sessão,
com o OpenCV lendo o preview UDP (9,6 fps medidos); Chromium maximizado com abas
e barra de endereço, extensão carregada, perfil descartável por sessão; ao final,
perfil apagado, `gsettings` (`switch-applications`, `close`, `terminal`,
`overlay-key`, `enable-hot-corners`) e opções XKB restaurados, arquivo de estado
de lockdown removido, zero processos residuais de ffmpeg/overlay/xbindkeys/Chromium.

**O que este registro não cobre** — e por isso os itens correspondentes seguem
desmarcados: nenhuma navegação foi feita no browser (site permitido, site
proibido, nova aba, URL digitada, redirect), nenhuma tentativa real de fuga pelo
teclado ou mouse, e as duas sessões foram do **mesmo aluno**, então a limpeza de
login entre alunos diferentes continua não validada.

### Critério de conclusão

- [x] GNOME continua utilizável para manutenção sem alterações permanentes indesejadas.
- [x] Modo prova só inicia por ação explícita do operador.
- [x] Overlay de espera orienta o aluno antes da identificação.
- [ ] Chromium abre com abas normais e barra de endereço, sem acesso ao OS.
- [ ] Allowlist bloqueia navegação fora dos sites autorizados.
- [ ] Páginas internas perigosas do Chromium, DevTools, incognito, sync e extensões não autorizadas estão bloqueadas.
- [ ] Perfil do Chromium é limpo ao fim e não preserva login/cookies/storage do aluno.
- [x] Gravação de webcam e tela continua correta no display GNOME atual.
- [x] Bloqueio/reidentificação continua funcionando.
- [x] A estação restaura lockdown/overlays e volta ao modo manutenção após prova ou cancelamento.
- [ ] Teste E2E completo em NUC real aprovado.
- [ ] Plano de contingência documentado e validado.

---

## M8 — Migração do dashboard para EC2 compartilhada 🟡

**Objetivo:** tirar o dashboard de teste local e colocá-lo no ar numa EC2 já
existente, compartilhada com outro serviço via nginx, com storage e
autenticação adequados para ficar exposto num subdomínio público. Planos
completos em `docs/archive/migracao_ec2_plano_dev.md` (mudanças de repositório) e
`docs/archive/migracao_ec2_passo_a_passo_aws.md` (portal AWS + shell na EC2).

Esta rodada é deliberadamente manual (console AWS + SSH), não Terraform —
fecha o item de IaC pendente em M7 com uma decisão explícita, não um
esquecimento: IaC completa fica para depois, se a operação em mais de uma
EC2/ambiente justificar o investimento.

### Repositório (feito neste milestone)

- [x] Trocar storage do `DashboardStore` de SQLite para Postgres (`psycopg`),
      mantendo o mesmo padrão de lock + linha como JSON.
- [x] Autenticação por token individual de estação (`X-Station-Id`/
      `X-Station-Token`), separada da senha do professor — reduz o raio de
      dano de uma NUC comprometida ao expor o painel na internet pública.
- [x] `scripts/issue_station_token.py` para emitir/rotacionar token por NUC.
- [x] App para de escutar em porta pública direta (`127.0.0.1:8010` por
      padrão) — nginx assume 80/443 como ponto de entrada único.
- [x] Template de server block nginx (`scripts/nginx/proctor-dashboard.conf`),
      com os headers de upgrade que `/ws/stations` (WebSocket) exige.
- [x] Fixture de teste (`tests/conftest.py`) isola cada teste num banco
      Postgres novo; `docker-compose.yml` sobe um Postgres local pra dev.
- [x] Docs (`setup_dashboard.md`, `setup_nuc.md`, `roles.md`, `.env.example`)
      atualizadas para o storage e a autenticação novos.

### Operador — portal AWS + EC2 (pendente, ver `docs/archive/migracao_ec2_passo_a_passo_aws.md`)

- [ ] Validar capacidade da EC2 atual (`free -h`/`df -h`) e decidir
      t3.micro vs t3.small com o Postgres novo rodando.
- [ ] Confirmar security group: só 80/443 públicos, nada da app/Postgres
      exposto diretamente.
- [ ] Registro DNS do subdomínio do dashboard, na mesma zona do domínio que
      a EC2 já usa para o outro serviço.
- [ ] Provisionar Postgres nativo na EC2 (banco/role isolados do outro
      serviço).
- [ ] Deploy da aplicação (clone, bootstrap, `.env`, systemd) na porta
      interna.
- [ ] Server block nginx novo + certbot para o subdomínio novo, na mesma
      janela (sem tráfego em claro entre um passo e outro).
- [ ] Emitir token de cada NUC e apontá-las pro subdomínio novo (`https://`).
- [ ] Cutover controlado: 1-2 sessões de teste de ponta a ponta antes de
      considerar produção.
- [ ] Monitoramento inicial (`free -h`, `journalctl`) nas primeiras provas
      reais na EC2 compartilhada.

### Critério de conclusão

- [x] Dashboard roda com Postgres e token por estação, testado localmente
      (157 testes automatizados, incluindo o fluxo de auth novo).
- [ ] Dashboard acessível via HTTPS no subdomínio novo, atrás do nginx
      compartilhado da EC2.
- [ ] Pelo menos uma NUC completando uma prova de ponta a ponta apontando
      pro subdomínio novo (não mais para um endereço de teste local).
- [ ] Rollback validado: queda da EC2 não impede a NUC de terminar uma
      prova em andamento e buferizar localmente até a conexão voltar.

---

## Registro de decisões técnicas

| Milestone | Decisão | Motivo |
|---|---|---|
| M0 | dlib em vez de MediaPipe | Sem dependência de GPU, roda bem no NUC i5 |
| M0 | solvePnP em vez de eye tracking puro | Mais robusto com óculos e iluminação variável |
| M0 | FFmpeg segmentado em vez de gravação contínua | Upload incremental sem esperar o fim da prova |
| M0 | X11 em vez de Wayland | x11grab não funciona com Wayland; sessão de prova controlada em X11 é mais previsível |
| M1 | Threshold 0.45 em vez do default 0.6 do dlib | Reduz falsos positivos em ambiente controlado |
| M1 | `student_id` = identificador institucional | Gerado pela faculdade, único, não precisa de mapeamento |
| M2 | `gaze_duration_sec` como único timer warn→block | `gaze_block_sec` era redundante e causava bug |
| M3 | FFmpeg grava a webcam direto via v4l2 | A gravação não herda o FPS reduzido do proctoring |
| M3 | Afinidade de CPU separa FFmpeg e proctoring | Reduz contenção no mesmo núcleo durante a prova |
| M3 | Sem áudio na gravação | Reduz CPU e simplifica o pipeline |
| M3 | `PROCTOR_FACE_PIPE_FPS` configurável | FPS real com dlib (~8fps) difere do nominal (30fps) — declarar errado acelera o vídeo |
| M4 | wmctrl por PID em vez de nome | Evita fullscreen na janela errada (VSCode, Firefox) |
| M4 | Lockdown de teclas dinâmico em vez de Xorg permanente | Reduz risco de travar a NUC fora da prova e permite restauração automática |
| M4 | Kiosk/incognito tratado como base temporária | Validou lockdown e reidentificação, mas não atende UX de abas/pesquisa |
| M7 | Modo prova no GNOME atual do usuário `proctor` | Mantém manutenção simples e permite lockdown temporário com restauração automática |
| M7 | Chromium maximizado em vez de `--kiosk` | Permite abas normais e barra de endereço para pesquisa em sites autorizados |
| M7 | Extensão allowlist pré-criada | Evita custo de gerar extensão/perfil inteiro no início da prova |
| M7 | Limpeza explícita de perfil | `--incognito` será removido; cookies e logins precisam sumir ao final da prova |
| M8 | Postgres em vez de SQLite no dashboard | Prioridade sobre a menor pegada de RAM do SQLite; ganha backup/gerenciamento mais formal e caminho aberto pra RDS depois. Custo aceito: instância provavelmente precisa subir pra t3.small, e testes passam a precisar de um Postgres disponível (local ou CI) |
| M8 | Token individual por estação em vez de credencial única compartilhada | Uma NUC comprometida ou com `.env` vazado só compromete aquela estação — não o painel do professor nem as demais NUCs |
| M8 | Token/auth de estação como `Depends` do FastAPI por rota, não extensão do middleware de Basic Auth | Evita ambiguidade de path-matching entre rotas que a NUC chama e rotas homônimas do professor (ex: `POST /api/sessions` da NUC vs `GET /api/sessions`/`POST /api/sessions/clear` do professor) |
| M8 | Postgres nativo na EC2 em vez de RDS | Sem infraestrutura AWS gerenciada nova nesta rodada, que é manual/portal, não Terraform; RDS fica como evolução futura |
