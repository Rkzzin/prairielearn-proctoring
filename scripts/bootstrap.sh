#!/usr/bin/env bash
# ============================================================================
#  proctor-station bootstrap — papel de ESTAÇÃO (NUC)
#  Leva um Ubuntu 24.04 Desktop limpo até proctor.service rodando.
#  Para o dashboard do professor (outra máquina, ex. EC2), use
#  scripts/bootstrap_dashboard.sh em vez deste.
#
#  Uso:
#    chmod +x scripts/bootstrap.sh
#    ./scripts/bootstrap.sh
#
#  O que faz:
#    1. Instala pacotes do sistema (apt)
#    2. Verifica sessão gráfica (X11)
#    3. Cria Python venv
#    4. Instala dependências Python
#    5. Baixa modelos dlib (~100MB)
#    6. Copia .env.example -> .env (se não existir) e tenta detectar
#       automaticamente câmera (índice V4L2) e microfone (card ALSA)
#    7. Roda testes
#    8. Testa câmera (se disponível)
# ============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()   { echo -e "${GREEN}[✓]${NC} $*"; }
warn()  { echo -e "${YELLOW}[!]${NC} $*"; }
fail()  { echo -e "${RED}[✗]${NC} $*"; exit 1; }

# ── Detectar diretório do projeto ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"
log "Diretório do projeto: $PROJECT_DIR"

# ── 1. Pacotes do sistema ──
echo ""
echo "=========================================="
echo "  1/6  Instalando pacotes do sistema"
echo "=========================================="

sudo apt update -qq

sudo apt install -y -qq \
    build-essential \
    cmake \
    pkg-config \
    > /dev/null 2>&1
log "Build essentials instalados"

sudo apt install -y -qq \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-tk \
    > /dev/null 2>&1
log "Python 3.12 + tkinter instalados"

sudo apt install -y -qq \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    > /dev/null 2>&1
log "Bibliotecas numéricas instaladas"

sudo apt install -y -qq \
    ffmpeg \
    v4l-utils \
    alsa-utils \
    > /dev/null 2>&1
log "FFmpeg, v4l-utils e alsa-utils instalados"

sudo apt install -y -qq \
    git \
    curl \
    lsof \
    xauth \
    bzip2 \
    > /dev/null 2>&1
log "Git, curl, lsof, xauth e bzip2 instalados"

sudo apt install -y -qq \
    x11-xserver-utils \
    xbindkeys \
    wmctrl \
    chromium-browser \
    > /dev/null 2>&1
log "Pacotes do browser controlado e lockdown instalados"

chmod +x scripts/install_chromium_hardening.sh
sudo bash scripts/install_chromium_hardening.sh
log "Policies de hardening do Chromium instaladas"

# ── 2. Sessão gráfica ──
echo ""
echo "=========================================="
echo "  2/8  Verificando sessão gráfica"
echo "=========================================="

if [ "${XDG_SESSION_TYPE:-desconhecido}" != "x11" ]; then
    warn "Sessão atual é '${XDG_SESSION_TYPE:-desconhecido}', não 'x11'."
    warn "A prova depende de X11 (captura de tela via x11grab, wmctrl, xbindkeys)."
    warn "Troque para 'Ubuntu on Xorg' na tela de login do GNOME antes de rodar a prova."
else
    log "Sessão X11 confirmada"
fi

# ── 3. Python venv ──
echo ""
echo "=========================================="
echo "  3/8  Criando Python virtual environment"
echo "=========================================="

if [ -d ".venv" ] || [ -d "venv" ]; then
    warn "Ambiente virtual existente encontrado — recriando..."
    rm -rf .venv
    rm -rf venv
fi

python3.12 -m venv venv
source venv/bin/activate
log "venv criado e ativado: $(which python3)"

python3 - <<'PY'
import sys
assert sys.version_info[:2] == (3, 12), sys.version
PY
log "Python do venv confirmado em 3.12"

python3 -m pip install --upgrade pip --quiet
log "pip atualizado: $(pip --version | cut -d' ' -f2)"

# ── 4. Instalar dependências Python ──
echo ""
echo "=========================================="
echo "  4/8  Instalando dependências Python"
echo "=========================================="
echo "       (dlib compila do source — pode levar 3-5 min)"

python3 -m pip install -e ".[station,dev]"
log "Dependências instaladas (papel: estação)"

# ── 5. Baixar modelos dlib ──
echo ""
echo "=========================================="
echo "  5/8  Baixando modelos dlib"
echo "=========================================="

chmod +x scripts/download_models.sh
./scripts/download_models.sh models
log "Modelos prontos"

# ── 6. Preparar .env ──
echo ""
echo "=========================================="
echo "  6/8  Preparando .env"
echo "=========================================="

if [ -f ".env" ]; then
    warn ".env já existe — não foi sobrescrito. Confira manualmente se está completo."
else
    cp .env.example .env
    log ".env criado a partir de .env.example"
    warn "AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY e as credenciais do dashboard"
    warn "(PROCTOR_DASHBOARD_ADMIN_USER/PASSWORD) continuam em branco — preencha o .env à mão."
fi

warn "Câmera/microfone NÃO são detectados aqui: o bootstrap costuma rodar antes"
warn "da webcam definitiva estar conectada na NUC. Depois de plugar a webcam de"
warn "verdade, rode:"
warn "    bash scripts/detect_camera_audio.sh"

# ── 7. Testes ──
echo ""
echo "=========================================="
echo "  7/8  Rodando testes"
echo "=========================================="

python3 -m pytest tests/ -v --tb=short
PYTEST_EXIT=$?

if [ "$PYTEST_EXIT" -eq 0 ]; then
    log "Todos os testes passaram"
else
    warn "Alguns testes falharam — verifique a saída acima"
fi

# ── 8. Teste de câmera ──
echo ""
echo "=========================================="
echo "  8/8  Testando câmera"
echo "=========================================="

if ls /dev/video* > /dev/null 2>&1; then
    python3 scripts/test_camera.py --headless
else
    warn "Nenhuma câmera detectada (/dev/video* não encontrado)"
    warn "Pule este passo em VMs — teste na NUC física com webcam USB"
fi

# ── Resumo ──
echo ""
echo "=========================================="
echo "  Setup completo!"
echo "=========================================="
echo ""
echo "  Para ativar o venv:"
echo "    source venv/bin/activate"
echo ""
echo "  Para cadastrar alunos:"
echo "    python3 scripts/enroll.py --turma MINHA-TURMA"
echo ""
echo "  Para rodar testes:"
echo "    pytest tests/ -v"
echo ""
