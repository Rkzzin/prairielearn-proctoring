#!/usr/bin/env bash
# ============================================================================
#  proctor-station bootstrap
#  Leva um Ubuntu 24.04 limpo até o projeto rodando.
#
#  Uso:
#    chmod +x scripts/bootstrap.sh
#    ./scripts/bootstrap.sh
#
#  O que faz:
#    1. Instala pacotes do sistema (apt)
#    2. Cria Python venv
#    3. Instala dependências Python
#    4. Baixa modelos dlib (~100MB)
#    5. Roda testes
#    6. Testa câmera (se disponível)
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
    > /dev/null 2>&1
log "Python 3.12 instalado"

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
    > /dev/null 2>&1
log "FFmpeg e v4l-utils instalados"

sudo apt install -y -qq \
    git \
    curl \
    bzip2 \
    > /dev/null 2>&1
log "Git, curl e bzip2 instalados"

# ── 2. Python venv ──
echo ""
echo "=========================================="
echo "  2/6  Criando Python virtual environment"
echo "=========================================="

if [ -d ".venv" ]; then
    warn "venv já existe — recriando..."
    rm -rf .venv
fi

python3.12 -m venv .venv
source .venv/bin/activate
log "venv criado e ativado: $(which python3)"

python3 -m pip install --upgrade pip --quiet
log "pip atualizado: $(pip --version | cut -d' ' -f2)"

# ── 3. Instalar dependências Python ──
echo ""
echo "=========================================="
echo "  3/6  Instalando dependências Python"
echo "=========================================="
echo "       (dlib compila do source — pode levar 3-5 min)"

pip install -e ".[dev]" 2>&1 | tail -1
log "Dependências instaladas"

# ── 4. Baixar modelos dlib ──
echo ""
echo "=========================================="
echo "  4/6  Baixando modelos dlib"
echo "=========================================="

chmod +x scripts/download_models.sh
./scripts/download_models.sh models
log "Modelos prontos"

# ── 5. Testes ──
echo ""
echo "=========================================="
echo "  5/6  Rodando testes"
echo "=========================================="

python3 -m pytest tests/ -v --tb=short 2>&1 | tail -20
PYTEST_EXIT=${PIPESTATUS[0]:-$?}

if [ "$PYTEST_EXIT" -eq 0 ]; then
    log "Todos os testes passaram"
else
    warn "Alguns testes falharam — verifique a saída acima"
fi

# ── 6. Teste de câmera ──
echo ""
echo "=========================================="
echo "  6/6  Testando câmera"
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
echo "    source .venv/bin/activate"
echo ""
echo "  Para cadastrar alunos:"
echo "    python3 scripts/enroll.py --turma MINHA-TURMA --ra 12345 --nome 'Nome'"
echo ""
echo "  Para rodar testes:"
echo "    pytest tests/ -v"
echo ""