#!/usr/bin/env bash
# ============================================================================
#  proctor-station bootstrap — papel de DASHBOARD (professor)
#  Leva uma máquina Linux limpa (ex. EC2 Ubuntu 24.04) até
#  proctor-dashboard.service rodando.
#
#  Diferença deliberada para scripts/bootstrap.sh (papel de ESTAÇÃO/NUC):
#  nada de GNOME, X11, câmera, Chromium ou ferramentas de lockdown — o
#  dashboard é só um servidor web. Ver docs/roles.md.
#
#  Uso:
#    chmod +x scripts/bootstrap_dashboard.sh
#    ./scripts/bootstrap_dashboard.sh
# ============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()   { echo -e "${GREEN}[✓]${NC} $*"; }
warn()  { echo -e "${YELLOW}[!]${NC} $*"; }
fail()  { echo -e "${RED}[✗]${NC} $*"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"
log "Diretório do projeto: $PROJECT_DIR"

# ── 1. Pacotes do sistema ──
echo ""
echo "=========================================="
echo "  1/5  Instalando pacotes do sistema"
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
    > /dev/null 2>&1
log "Bibliotecas numéricas instaladas (dlib compila mesmo aqui — ver docs/roles.md)"

sudo apt install -y -qq \
    git \
    curl \
    lsof \
    > /dev/null 2>&1
log "Git, curl e lsof instalados"

# ── 2. Python venv ──
echo ""
echo "=========================================="
echo "  2/5  Criando Python virtual environment"
echo "=========================================="

if [ -d ".venv" ] || [ -d "venv" ]; then
    warn "Ambiente virtual existente encontrado — recriando..."
    rm -rf .venv
    rm -rf venv
fi

python3.12 -m venv venv
source venv/bin/activate
log "venv criado e ativado: $(which python3)"

python3 -m pip install --upgrade pip --quiet
log "pip atualizado: $(pip --version | cut -d' ' -f2)"

# ── 3. Instalar dependências Python ──
echo ""
echo "=========================================="
echo "  3/5  Instalando dependências Python"
echo "=========================================="
echo "       (dlib compila do source — pode levar 3-5 min; é usado pelo"
echo "        enrollment via S3 do dashboard, não só pela estação)"

python3 -m pip install -e ".[dashboard,dev]"
log "Dependências instaladas (papel: dashboard)"

# ── 4. Preparar .env ──
echo ""
echo "=========================================="
echo "  4/5  Preparando .env"
echo "=========================================="

if [ -f ".env" ]; then
    warn ".env já existe — não foi sobrescrito. Confira manualmente se está completo."
else
    cp .env.example .env
    log ".env criado a partir de .env.example"
fi

warn "Preencha à mão no .env: AWS_ACCESS_KEY_ID/SECRET, PROCTOR_DASHBOARD_ADMIN_USER"
warn "e PROCTOR_DASHBOARD_ADMIN_PASSWORD (obrigatórios para o dashboard exigir login)."
warn "Campos de estação (câmera, gaze, proxy, station_id) não se aplicam aqui —"
warn "ver docs/setup_dashboard.md."

# ── 5. Testes ──
echo ""
echo "=========================================="
echo "  5/5  Rodando testes"
echo "=========================================="

python3 -m pytest tests/ -v --tb=short
PYTEST_EXIT=$?

if [ "$PYTEST_EXIT" -eq 0 ]; then
    log "Todos os testes passaram"
else
    warn "Alguns testes falharam — verifique a saída acima"
fi

# ── Resumo ──
echo ""
echo "=========================================="
echo "  Setup completo!"
echo "=========================================="
echo ""
echo "  Para instalar o serviço systemd:"
echo "    cd $PROJECT_DIR && sudo bash scripts/install_dashboard_service.sh"
echo ""
