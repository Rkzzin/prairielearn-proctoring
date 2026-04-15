# Proctoring Station — Plano de implementação completo

## Visão geral do projeto

Sistema de estações de prova presencial baseadas em Intel NUC, com reconhecimento facial, gravação com proctoring automatizado, browser lockdown e integração com PrairieLearn. O objetivo é substituir quizzes que consomem 1h de aula por sessões controladas em máquinas dedicadas, fora do horário de aula.

---

## Fase 0 — Infraestrutura e decisões de projeto

### 0.1 Hardware por estação

| Componente | Especificação mínima | Justificativa |
|---|---|---|
| NUC | Intel NUC 12/13 (i5), 16GB RAM, SSD 256GB | Encode H.264 via Quick Sync, headless OK |
| Webcam | Logitech C920/C922 (1080p, wide-angle) | Boa qualidade, UVC nativo no Linux |
| Monitor | 21"+ Full HD | Tamanho padrão de prova |
| Teclado + Mouse | USB com fio | Sem risco de Bluetooth spoofing |
| Microfone | Embutido na webcam (C920 já tem) | Captura áudio ambiente |

### 0.2 Stack tecnológica

```
OS:           Ubuntu 24.04 LTS Server + Xorg minimal
Linguagem:    Python 3.12 (orquestração, CV, API)
CV/ML:        OpenCV 4.x + MediaPipe + face_recognition (dlib)
Gravação:     FFmpeg 6.x (H.264 via libx264 ou vaapi)
Browser:      Chromium em modo kiosk + extensão custom
Orquestrador: FastAPI (API local na NUC)
Upload:       boto3 → AWS S3
IaC:          Terraform + Ansible
Dashboard:    FastAPI + HTMX (servidor central)
```

### 0.3 Repositório — estrutura proposta

```
proctor-station/
├── ansible/
│   ├── playbooks/
│   │   ├── setup-nuc.yml
│   │   ├── update-nuc.yml
│   │   └── enroll-faces.yml
│   ├── roles/
│   │   ├── base/           # packages, users, firewall
│   │   ├── kiosk/          # Xorg, Chromium, lockdown
│   │   ├── proctor/        # Python app, systemd units
│   │   └── monitoring/     # Prometheus node_exporter
│   └── inventory/
│       └── hosts.yml
├── terraform/
│   ├── main.tf             # S3, IAM, dashboard EC2/ECS
│   ├── variables.tf
│   └── outputs.tf
├── src/
│   ├── core/
│   │   ├── session.py      # Session manager (orquestrador)
│   │   ├── config.py       # Configuração central
│   │   └── models.py       # Pydantic schemas
│   ├── face/
│   │   ├── recognizer.py   # Enrollment + identificação
│   │   ├── detector.py     # Contagem de faces
│   │   └── encodings/      # .pkl por turma
│   ├── proctor/
│   │   ├── engine.py       # Loop principal de monitoramento
│   │   ├── gaze.py         # Gaze estimation via MediaPipe
│   │   └── events.py       # Event bus + log de timestamps
│   ├── recorder/
│   │   ├── capture.py      # Gerencia processos FFmpeg
│   │   └── uploader.py     # Upload incremental S3
│   ├── kiosk/
│   │   ├── chromium.py     # Launcher + lifecycle
│   │   ├── allowlist.json  # Domínios permitidos
│   │   └── extension/      # Chrome extension de bloqueio
│   ├── api/
│   │   ├── server.py       # FastAPI local
│   │   └── routes.py       # Endpoints da estação
│   └── dashboard/
│       ├── app.py           # FastAPI central
│       ├── templates/       # Jinja2 + HTMX
│       └── static/
├── scripts/
│   ├── enroll.py           # CLI para cadastrar rostos
│   ├── provision.sh        # Bootstrap inicial de uma NUC
│   └── test_camera.py      # Validação rápida de hardware
├── tests/
├── docker-compose.yml      # Dev environment
├── pyproject.toml
└── README.md
```

---

## Fase 1 — Face recognition (gate de acesso)

**Duração estimada: 1 semana**

### 1.1 Enrollment de alunos

```
Fluxo:
1. Professor executa `python scripts/enroll.py --turma "ES2025"`
2. Script captura N frames do rosto do aluno via webcam
3. Gera encoding 128-d via face_recognition.face_encodings()
4. Salva em src/face/encodings/ES2025.pkl (dict: {RA: encoding})
5. Opcional: importar fotos do sistema acadêmico via CSV
```

**Decisões técnicas:**
- Armazenar 3-5 encodings por aluno (ângulos diferentes) e usar média ou match-any.
- Threshold de distância: `0.45` (mais restritivo que o default `0.6`) para evitar falsos positivos entre colegas.
- Fallback: se não reconhecer em 3 tentativas, exibir tela de "Chamar o professor" + tirar foto para log.

### 1.2 Fluxo de identificação

```python
# Pseudocódigo simplificado
def identify_student(frame, turma_encodings, threshold=0.45):
    face_locations = face_recognition.face_locations(frame, model="hog")
    if len(face_locations) == 0:
        return None, "NO_FACE"
    if len(face_locations) > 1:
        return None, "MULTIPLE_FACES"
    
    encoding = face_recognition.face_encodings(frame, face_locations)[0]
    distances = face_recognition.face_distance(
        list(turma_encodings.values()), encoding
    )
    best_idx = distances.argmin()
    if distances[best_idx] < threshold:
        ra = list(turma_encodings.keys())[best_idx]
        return ra, "MATCH"
    return None, "NO_MATCH"
```

### 1.3 Entregáveis

- [ ] Script de enrollment funcional com CLI
- [ ] Módulo `recognizer.py` com `identify_student()`
- [ ] Testes unitários com fotos de teste
- [ ] Documentação de calibração de threshold por turma

---

## Fase 2 — Proctoring engine

**Duração estimada: 1.5 semanas**

### 2.1 Gaze estimation

Usa MediaPipe Face Mesh (468 landmarks) para estimar direção do olhar.

```
Landmarks relevantes:
- Iris: pontos 468-477 (iris landmarks do MediaPipe)
- Contorno dos olhos: pontos 33, 133, 362, 263 (cantos)
- Nariz: ponto 1 (referência central)

Algoritmo:
1. Calcular posição relativa da íris dentro do contorno do olho
2. Normalizar para ratio -1.0 (esquerda) a +1.0 (direita)
3. Se |ratio_horizontal| > 0.35 por mais de 3 segundos → FLAG
4. Se |ratio_vertical| > 0.30 por mais de 3 segundos → FLAG
```

**Parâmetros tunáveis (config.py):**

| Parâmetro | Default | Descrição |
|---|---|---|
| `GAZE_H_THRESHOLD` | 0.35 | Limiar horizontal do olhar |
| `GAZE_V_THRESHOLD` | 0.30 | Limiar vertical do olhar |
| `GAZE_DURATION_SEC` | 3.0 | Tempo antes de flagar |
| `ABSENCE_TIMEOUT_SEC` | 5.0 | Tempo sem rosto antes de bloquear |
| `MULTI_FACE_BLOCK` | True | Bloquear se >1 rosto |

### 2.2 Detecção de múltiplas faces

```python
# Roda a cada frame no loop principal
def check_faces(frame):
    faces = face_recognition.face_locations(frame, model="hog")
    if len(faces) == 0:
        return ProctorEvent.ABSENCE
    if len(faces) > 1:
        return ProctorEvent.MULTIPLE_FACES
    return ProctorEvent.OK
```

### 2.3 Event bus e log de timestamps

```python
@dataclass
class ProctorEvent:
    timestamp: float          # time.time()
    frame_number: int
    event_type: str           # GAZE_LEFT | GAZE_RIGHT | ABSENCE | MULTI_FACE | RETURN
    severity: str             # WARNING | BLOCK
    details: dict             # Dados extras (gaze ratio, face count, etc.)

# Log salvo como JSONL:
# {"ts": 1719000000.0, "frame": 4520, "type": "GAZE_LEFT", "severity": "WARNING", ...}
# {"ts": 1719000003.2, "frame": 4616, "type": "GAZE_LEFT", "severity": "BLOCK", ...}
```

### 2.4 Máquina de estados do proctoring

```
                ┌──────────┐
                │  NORMAL  │ ← estado padrão durante a prova
                └────┬─────┘
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
    ┌──────────┐ ┌────────┐ ┌──────────┐
    │ GAZE_WARN│ │ABSENCE │ │MULTI_FACE│
    └────┬─────┘ └───┬────┘ └────┬─────┘
         │           │           │
    duração >        duração >   imediato
    threshold        timeout
         │           │           │
         ▼           ▼           ▼
    ┌──────────────────────────────┐
    │          BLOCKED             │
    │  (tela travada, grava flag)  │
    └──────────┬───────────────────┘
               │
        aluno retorna +
        face re-match OK
               │
               ▼
          ┌──────────┐
          │  NORMAL  │
          └──────────┘
```

### 2.5 Entregáveis

- [ ] `gaze.py` com estimação via MediaPipe
- [ ] `engine.py` com loop principal e FSM
- [ ] `events.py` com event bus e serialização JSONL
- [ ] Testes com vídeos gravados (replay de cenários)
- [ ] Script de calibração de thresholds

---

## Fase 3 — Gravação e upload

**Duração estimada: 1 semana**

### 3.1 Streams FFmpeg

```bash
# Stream 1: Webcam (rosto + áudio)
ffmpeg -f v4l2 -video_size 1280x720 -framerate 30 -i /dev/video0 \
       -f pulse -i default \
       -c:v libx264 -preset fast -crf 23 \
       -c:a aac -b:a 128k \
       -f segment -segment_time 300 -reset_timestamps 1 \
       webcam_%03d.mp4

# Stream 2: Captura de tela
ffmpeg -f x11grab -video_size 1920x1080 -framerate 15 -i :0.0 \
       -c:v libx264 -preset fast -crf 28 \
       -f segment -segment_time 300 -reset_timestamps 1 \
       screen_%03d.mp4
```

**Decisões:**
- Segmentos de 5 minutos para upload incremental (não esperar o fim da prova).
- CRF 23 para webcam (boa qualidade facial), CRF 28 para tela (texto legível, tamanho menor).
- Framerate da tela: 15fps é suficiente para capturar interação.

### 3.2 Upload incremental para S3

```python
# Estrutura no S3:
# s3://proctor-recordings/
#   └── {turma}/{data}/{ra}/
#       ├── webcam_000.mp4
#       ├── webcam_001.mp4
#       ├── screen_000.mp4
#       ├── screen_001.mp4
#       ├── events.jsonl
#       └── session.json   # metadados da sessão

class Uploader:
    def __init__(self, bucket, prefix):
        self.s3 = boto3.client('s3')
        self.bucket = bucket
        self.prefix = prefix  # turma/data/ra
    
    def watch_and_upload(self, directory):
        """Observa diretório e faz upload de segmentos completos."""
        # Usa inotify para detectar segmentos finalizados
        # Upload com multipart para arquivos > 100MB
        # Retry com exponential backoff
        # Deleta local após upload confirmado
```

### 3.3 Entregáveis

- [ ] `capture.py` gerenciando subprocessos FFmpeg
- [ ] `uploader.py` com watch + upload incremental
- [ ] Teste de gravação simultânea webcam + tela por 30min
- [ ] Validação de integridade pós-upload (ETag check)

---

## Fase 4 — Browser lockdown

**Duração estimada: 1 semana**

### 4.1 Chromium kiosk mode

```bash
# Lançamento do Chromium
chromium-browser \
    --kiosk \
    --no-first-run \
    --disable-translate \
    --disable-extensions-except=/opt/proctor/extension \
    --load-extension=/opt/proctor/extension \
    --disable-dev-tools \
    --disable-default-apps \
    --disable-background-networking \
    --disable-sync \
    --disable-plugins \
    --incognito \
    --start-fullscreen \
    "https://prairielearn.exemplo.edu.br"
```

### 4.2 Chrome extension — allowlist enforcer

```javascript
// extension/manifest.json
{
  "manifest_version": 3,
  "name": "Proctor Allowlist",
  "permissions": ["webRequest", "declarativeNetRequest"],
  "background": { "service_worker": "background.js" },
  "declarativeNetRequest": {
    "ruleResources": [{
      "id": "allowlist",
      "enabled": true,
      "path": "rules.json"
    }]
  }
}

// extension/rules.json — bloqueia tudo exceto allowlist
// Rule 1: Bloqueia TUDO
// Rule 2-N: Permite domínios específicos (prioridade maior)
```

**Domínios no allowlist (exemplo):**
- `prairielearn.exemplo.edu.br`
- `*.prairielearn.org` (se usar cloud)
- CDNs do PrairieLearn (MathJax, etc.)

### 4.3 Lockdown do SO

```yaml
# Ansible role: kiosk
# Desabilitar:
- Alt+F4, Alt+Tab, Ctrl+Alt+Del    # via Xorg config
- TTY switching (Ctrl+Alt+F1-F6)   # kernel: sysrq=0 + /etc/X11/xorg.conf DontVTSwitch
- USB mass storage                  # udev rule: blacklist usb-storage
- Bluetooth                         # systemctl disable bluetooth
- Terminal emulators                # não instalar
- File manager                      # não instalar
- Print screen                      # Xorg grab
- Right-click no Chromium           # --disable-context-menu (via extension)
```

```ini
# /etc/X11/xorg.conf.d/99-lockdown.conf
Section "ServerFlags"
    Option "DontVTSwitch" "true"
    Option "DontZap"      "true"
EndSection
```

### 4.4 Tela de bloqueio (proctoring trigger)

Quando o proctoring engine emite `BLOCK`, o session manager:

1. Envia sinal SIGSTOP ao processo do Chromium (congela).
2. Sobrepõe uma janela fullscreen X11 (Python + tkinter ou GTK) com a mensagem: *"Sessão pausada — olhe para a câmera para continuar."*
3. Aguarda face re-match + gaze OK.
4. Envia SIGCONT ao Chromium e remove o overlay.

### 4.5 Entregáveis

- [ ] Chrome extension com allowlist configurável
- [ ] Xorg lockdown config
- [ ] Overlay de bloqueio funcional
- [ ] Teste E2E: trigger de bloqueio → re-match → desbloqueio
- [ ] Ansible role `kiosk` completo

---

## Fase 5 — Session manager (orquestrador)

**Duração estimada: 1.5 semanas**

### 5.1 Fluxo completo de uma sessão

```
┌─ NUC em estado IDLE ──────────────────────────────────────────┐
│                                                                │
│  1. Tela mostra "Sente-se e olhe para a câmera"               │
│  2. Câmera detecta rosto                                       │
│  3. Face recognition identifica o aluno (RA)                   │
│  4. Tela mostra "Olá, {nome}! Iniciando sessão..."            │
│  5. Verifica se aluno já fez a prova → rejeita se sim         │
│  6. Inicia gravação (webcam + tela + áudio)                    │
│  7. Inicia proctoring engine (loop de monitoramento)           │
│  8. Abre Chromium kiosk → PrairieLearn login                   │
│  9. Timer começa (configurado pelo professor)                  │
│                                                                │
│  ── DURANTE A PROVA ──                                         │
│  • Proctoring engine monitora continuamente                    │
│  • Eventos são logados em JSONL                                │
│  • Segmentos de vídeo são uploaded incrementalmente            │
│  • Se BLOCK: overlay + pausa + re-match                        │
│                                                                │
│  ── FIM DA SESSÃO (timer ou submit) ──                         │
│  10. Fecha Chromium                                            │
│  11. Para gravação                                             │
│  12. Faz upload final (último segmento + events.jsonl)         │
│  13. Gera session.json com metadados                           │
│  14. Reseta para estado IDLE                                   │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 5.2 API local (FastAPI)

```python
# Endpoints da estação (rodam na NUC, porta 8000)

GET  /status              # Estado atual: IDLE | SESSION | BLOCKED | UPLOADING
GET  /session             # Dados da sessão ativa (aluno, tempo restante, eventos)
POST /session/start       # Força início manual (admin)
POST /session/stop        # Força fim (admin/professor)
POST /session/unblock     # Desbloqueia manualmente (professor com auth)
GET  /health              # Healthcheck para monitoramento
POST /config              # Atualiza config (timer, allowlist, thresholds)
```

### 5.3 Configuração por prova

```json
{
  "turma": "ES2025-T1",
  "prova": "Quiz-03",
  "timer_minutes": 45,
  "prairielearn_url": "https://pl.exemplo.edu.br/course/123/assessment/456",
  "allowlist": [
    "pl.exemplo.edu.br",
    "cdn.prairielearn.org",
    "cdnjs.cloudflare.com"
  ],
  "proctoring": {
    "gaze_h_threshold": 0.35,
    "gaze_duration_sec": 3.0,
    "absence_timeout_sec": 5.0,
    "multi_face_block": true
  },
  "s3_prefix": "ES2025-T1/2025-07-15/Quiz-03"
}
```

### 5.4 Entregáveis

- [ ] `session.py` com FSM completa
- [ ] FastAPI server com todos os endpoints
- [ ] Integração com todos os módulos (face, proctor, recorder, kiosk)
- [ ] Teste E2E do fluxo completo
- [ ] systemd unit para autostart

---

## Fase 6 — Dashboard do professor

**Duração estimada: 1 semana**

### 6.1 Funcionalidades

- **Visão em tempo real**: status de cada NUC (IDLE/SESSION/BLOCKED), qual aluno está sentado, tempo restante.
- **Configuração de prova**: criar sessão (turma, assessment, timer, thresholds), distribuir config para as NUCs.
- **Revisão pós-prova**: lista de sessões com contagem de flags, player de vídeo com timeline de eventos (clicar no evento pula para o timestamp no vídeo), exportar relatório.
- **Enrollment**: interface para cadastrar rostos (webcam ao vivo ou upload de fotos).

### 6.2 Stack

```
Backend:    FastAPI (Python)
Frontend:   HTMX + Jinja2 templates (leve, sem build step)
Vídeo:      HLS via S3 presigned URLs (sem proxy de vídeo)
WebSocket:  Para status real-time das NUCs
DB:         SQLite (simples) ou PostgreSQL (se escalar)
Auth:       OAuth via sistema acadêmico ou basic auth
```

### 6.3 Comunicação NUC ↔ Dashboard

```
NUC → Dashboard:
  • WebSocket heartbeat a cada 5s (status, aluno, tempo, eventos recentes)
  • POST /api/sessions ao iniciar e finalizar sessão

Dashboard → NUC:
  • POST /config para enviar config de prova
  • POST /session/stop para forçar fim
  • POST /session/unblock para desbloquear remotamente
```

### 6.4 Entregáveis

- [ ] Dashboard funcional com HTMX
- [ ] Tela de monitoramento real-time
- [ ] Player de vídeo com timeline de eventos
- [ ] Tela de configuração de prova
- [ ] Comunicação bidirecional NUC ↔ Dashboard

---

## Fase 7 — Testes e hardening

**Duração estimada: 1 semana**

### 7.1 Matriz de testes

| Cenário | Tipo | Esperado |
|---|---|---|
| Aluno reconhecido corretamente | Happy path | Sessão inicia |
| Aluno não cadastrado | Edge case | Tela de "chamar professor" |
| Dois alunos na câmera | Proctoring | BLOCK imediato |
| Aluno olha para o lado 4s | Proctoring | WARNING → BLOCK |
| Aluno sai da frente da câmera | Proctoring | BLOCK após timeout |
| Aluno tenta abrir nova aba | Lockdown | Bloqueado pela extension |
| Aluno tenta Ctrl+Alt+F2 | Lockdown | Nada acontece |
| Aluno conecta USB flash | Lockdown | Dispositivo ignorado |
| Internet cai durante prova | Resiliência | Gravação local continua, upload retry |
| NUC reinicia durante prova | Resiliência | Session recovery via state file |
| Upload S3 falha | Resiliência | Queue local, retry exponential |
| Prova de 45min, 720p + tela | Performance | CPU < 60%, sem frame drop |

### 7.2 Hardening de segurança

- Firewall (ufw): permitir apenas saída para S3, PrairieLearn, dashboard.
- Usuário `proctor` sem sudo, sem shell interativo.
- AppArmor profile para o Chromium.
- Disable USB storage via udev rules.
- LUKS no SSD (dados em trânsito protegidos at rest).
- Logs de auditoria (auditd) para ações privilegiadas.

### 7.3 Entregáveis

- [ ] Suite de testes automatizados
- [ ] Testes de carga (simulação de 30min contínuos)
- [ ] Relatório de hardening aplicado
- [ ] Playbook de disaster recovery

---

## Fase 8 — Infraestrutura como código (Terraform + Ansible)

**Duração estimada: 1.5 semanas**

### 8.1 Terraform — recursos cloud (AWS)

```hcl
# terraform/main.tf

# S3 bucket para gravações
resource "aws_s3_bucket" "recordings" {
  bucket = "proctor-recordings-${var.environment}"
}

resource "aws_s3_bucket_lifecycle_configuration" "recordings_lifecycle" {
  bucket = aws_s3_bucket.recordings.id
  rule {
    id     = "archive-old-recordings"
    status = "Enabled"
    transition {
      days          = 90
      storage_class = "GLACIER"
    }
    expiration {
      days = 365  # Retenção de 1 ano
    }
  }
}

# IAM user para as NUCs (upload only)
resource "aws_iam_user" "nuc_uploader" {
  name = "proctor-nuc-uploader-${var.environment}"
}

resource "aws_iam_user_policy" "nuc_upload_policy" {
  name = "s3-upload-only"
  user = aws_iam_user.nuc_uploader.name
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = ["s3:PutObject", "s3:PutObjectAcl"]
        Resource = "${aws_s3_bucket.recordings.arn}/*"
      }
    ]
  })
}

# Dashboard (ECS Fargate ou EC2 pequeno)
resource "aws_instance" "dashboard" {
  ami           = var.ubuntu_ami
  instance_type = "t3.small"
  key_name      = var.ssh_key_name
  
  vpc_security_group_ids = [aws_security_group.dashboard.id]
  
  user_data = templatefile("${path.module}/userdata.sh", {
    environment = var.environment
  })

  tags = {
    Name = "proctor-dashboard-${var.environment}"
  }
}

# Security group do dashboard
resource "aws_security_group" "dashboard" {
  name = "proctor-dashboard-${var.environment}"
  
  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = [var.campus_cidr]  # Apenas rede do campus
  }
  
  ingress {
    from_port   = 8443
    to_port     = 8443
    protocol    = "tcp"
    cidr_blocks = [var.campus_cidr]  # WebSocket das NUCs
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# CloudWatch alarms
resource "aws_cloudwatch_metric_alarm" "s3_errors" {
  alarm_name          = "proctor-s3-upload-errors"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "4xxErrors"
  namespace           = "AWS/S3"
  period              = 300
  statistic           = "Sum"
  threshold           = 10
  alarm_actions       = [aws_sns_topic.alerts.arn]
}

resource "aws_sns_topic" "alerts" {
  name = "proctor-alerts-${var.environment}"
}
```

```hcl
# terraform/variables.tf

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "campus_cidr" {
  description = "CIDR block da rede do campus"
  type        = string
}

variable "ubuntu_ami" {
  description = "AMI do Ubuntu 24.04 LTS"
  type        = string
}

variable "ssh_key_name" {
  description = "SSH key pair name"
  type        = string
}
```

### 8.2 Ansible — provisionamento das NUCs

```yaml
# ansible/playbooks/setup-nuc.yml
---
- name: Provisionar NUC como proctoring station
  hosts: nucs
  become: true
  
  vars:
    proctor_user: proctor
    proctor_home: /opt/proctor
    python_version: "3.12"
    chromium_version: latest
    
  roles:
    - role: base
      # Instala packages, cria user, configura firewall, NTP
    - role: kiosk
      # Xorg minimal, Chromium, lockdown configs
    - role: proctor
      # Python venv, código da aplicação, systemd units
    - role: monitoring
      # node_exporter, promtail para logs centralizados
```

```yaml
# ansible/roles/base/tasks/main.yml
---
- name: Instalar packages base
  apt:
    name:
      - python3.12
      - python3.12-venv
      - ffmpeg
      - v4l-utils
      - chromium-browser
      - xorg
      - xinit
      - libgl1-mesa-dri
      - cmake              # Para compilar dlib
      - libdlib-dev
      - ufw
      - auditd
    state: present
    update_cache: true

- name: Criar usuário proctor
  user:
    name: "{{ proctor_user }}"
    shell: /usr/sbin/nologin
    home: "{{ proctor_home }}"
    system: true

- name: Configurar firewall
  ufw:
    rule: allow
    direction: out
    to_ip: "{{ item }}"
  loop:
    - "{{ s3_endpoint_ip }}"
    - "{{ dashboard_ip }}"
    - "{{ prairielearn_ip }}"

- name: Bloquear USB mass storage
  copy:
    dest: /etc/udev/rules.d/99-no-usb-storage.rules
    content: |
      ACTION=="add", SUBSYSTEMS=="usb", DRIVERS=="usb-storage", \
      ATTR{authorized}="0"

- name: Desabilitar Bluetooth
  systemd:
    name: bluetooth
    state: stopped
    enabled: false
```

```yaml
# ansible/roles/proctor/tasks/main.yml
---
- name: Copiar código da aplicação
  synchronize:
    src: "{{ playbook_dir }}/../../src/"
    dest: "{{ proctor_home }}/src/"
    rsync_opts:
      - "--exclude=__pycache__"

- name: Criar Python venv e instalar deps
  pip:
    requirements: "{{ proctor_home }}/src/requirements.txt"
    virtualenv: "{{ proctor_home }}/venv"
    virtualenv_command: python3.12 -m venv

- name: Copiar credenciais AWS
  template:
    src: aws_credentials.j2
    dest: "{{ proctor_home }}/.aws/credentials"
    owner: "{{ proctor_user }}"
    mode: "0600"

- name: Instalar systemd units
  template:
    src: "{{ item }}.service.j2"
    dest: "/etc/systemd/system/{{ item }}.service"
  loop:
    - proctor-session    # Orquestrador principal
    - proctor-api        # FastAPI local
  notify: Reload systemd

- name: Habilitar e iniciar serviços
  systemd:
    name: "{{ item }}"
    state: started
    enabled: true
  loop:
    - proctor-session
    - proctor-api

- name: Configurar autologin e autostart do Xorg
  template:
    src: xinitrc.j2
    dest: "{{ proctor_home }}/.xinitrc"
    owner: "{{ proctor_user }}"
```

### 8.3 Workflow para adicionar uma nova NUC

```bash
# 1. Boot na NUC com Ubuntu 24.04 minimal (USB ou PXE)

# 2. Bootstrap mínimo (rede + SSH)
ssh root@nova-nuc 'bash -s' < scripts/provision.sh

# 3. Adicionar ao inventory do Ansible
echo "nuc-lab-05 ansible_host=192.168.1.105" >> ansible/inventory/hosts.yml

# 4. Rodar playbook
cd ansible && ansible-playbook playbooks/setup-nuc.yml --limit nuc-lab-05

# 5. Enrollment de rostos (se turma já existe, basta copiar .pkl)
ansible-playbook playbooks/enroll-faces.yml --limit nuc-lab-05 \
    -e turma=ES2025-T1

# 6. Validar
curl http://192.168.1.105:8000/health
# {"status": "ok", "state": "IDLE", "camera": true, "s3": true}
```

### 8.4 Escalabilidade — cenários

| Cenário | Solução |
|---|---|
| 1→10 NUCs | Ansible inventory + playbook (30min) |
| 10→50 NUCs | PXE boot + cloud-init + Ansible pull mode |
| Novo campus | Terraform workspace + novo inventory |
| Atualização de código | `ansible-playbook update-nuc.yml` (rolling update) |
| Nova turma | `enroll.py` + sync do .pkl via Ansible |
| Múltiplos professores | Dashboard multi-tenant com roles |

---

## Fase 9 — Deploy e operação

### 9.1 Checklist de go-live

- [ ] Terraform apply no ambiente prod
- [ ] Ansible setup em todas as NUCs
- [ ] Enrollment de todos os alunos da turma
- [ ] Teste E2E com 1 aluno voluntário em cada NUC
- [ ] Configurar monitoramento (Prometheus + Grafana ou CloudWatch)
- [ ] Documentação para o professor (como configurar prova, como revisar)
- [ ] Documentação para o aluno (o que esperar, regras)
- [ ] Plano de contingência (NUC falha durante prova, internet cai)

### 9.2 Operação recorrente

```
ANTES DE CADA PROVA:
1. Professor configura prova no dashboard (timer, turma, assessment URL)
2. Dashboard distribui config para NUCs
3. NUCs reiniciam em estado IDLE esperando alunos

DIA DA PROVA:
4. Alunos sentam nas estações, são identificados, fazem a prova
5. Professor monitora no dashboard (real-time)
6. Ao finalizar, gravações são uploadadas automaticamente

APÓS A PROVA:
7. Professor revisa flags no dashboard
8. Gravações ficam disponíveis por 1 ano (S3 → Glacier após 90 dias)
```

### 9.3 Custos estimados (mensal, 10 NUCs)

| Item | Custo estimado |
|---|---|
| S3 storage (Hot, ~100GB/mês) | ~$2.30 |
| S3 storage (Glacier, acumulado) | ~$0.40/100GB |
| EC2 t3.small (dashboard) | ~$15 |
| Data transfer (upload) | ~$0.90 |
| **Total mensal AWS** | **~$19** |
| Hardware (NUC + webcam + monitor) × 10 | ~$7.000 (one-time) |

---

## Cronograma resumido

| Fase | Semana | Entrega |
|---|---|---|
| 0 — Setup e decisões | S1 | Repo, hardware, stack definidos |
| 1 — Face recognition | S2 | Enrollment + identificação funcional |
| 2 — Proctoring engine | S3-S4 | Gaze + multi-face + FSM |
| 3 — Gravação e upload | S4-S5 | FFmpeg + S3 incremental |
| 4 — Browser lockdown | S5-S6 | Kiosk + extension + lockdown |
| 5 — Session manager | S6-S7 | Orquestrador E2E |
| 6 — Dashboard | S8 | Interface do professor |
| 7 — Testes e hardening | S9 | Suite de testes + segurança |
| 8 — Terraform + Ansible | S10-S11 | IaC completo |
| 9 — Deploy | S11-S12 | Go-live |

**Total estimado: 12 semanas (3 meses) para MVP funcional.**

---

## Riscos e mitigações

| Risco | Impacto | Mitigação |
|---|---|---|
| Falso positivo de gaze (aluno pensando olha pro lado) | Bloqueio injusto | Threshold generoso + review pós-prova |
| dlib lento na NUC sem GPU | Lag no reconhecimento | Usar modelo HOG (CPU), não CNN; ou fallback para MediaPipe face detection |
| Aluno parecido confunde o sistema | Acesso errado | Threshold restritivo (0.45) + confirmação visual na tela |
| Internet instável | Upload falha | Queue local com retry; gravação 100% local |
| Chromium atualiza e quebra kiosk | Prova comprometida | Pinnar versão do Chromium via apt hold |
| Privacidade / LGPD | Legal | Consentimento no enrollment; política de retenção clara; acesso restrito |