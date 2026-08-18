#!/usr/bin/env bash
# ============================================================================
#  proctor-station bootstrap — papel de DASHBOARD (professor)
#  Leva uma máquina Linux limpa (Ubuntu/apt ou Amazon Linux/dnf — detectado
#  automaticamente) até proctor-dashboard.service rodando.
#
#  Diferença deliberada para scripts/bootstrap.sh (papel de ESTAÇÃO/NUC, só
#  Ubuntu — NUCs físicas): nada de GNOME, X11, câmera, Chromium ou
#  ferramentas de lockdown — o dashboard é só um servidor web. Ver docs/roles.md.
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

# Só suprime stdout, nunca stderr — silenciar os dois já escondeu um erro
# real (conflito curl/curl-minimal no dnf) sem deixar rastro nenhum pro
# operador depurar. `set -e` já mata o script no primeiro erro; o mínimo é
# ele aparecer na tela.

if command -v apt-get >/dev/null 2>&1; then
    log "Gerenciador de pacotes: apt (Ubuntu/Debian)"
    sudo apt update -qq

    sudo apt install -y -qq build-essential cmake pkg-config > /dev/null
    log "Build essentials instalados"

    sudo apt install -y -qq python3.12 python3.12-venv python3.12-dev > /dev/null
    log "Python 3.12 instalado"

    if sudo apt install -y -qq libopenblas-dev liblapack-dev > /dev/null 2>/dev/null; then
        log "Bibliotecas numéricas instaladas (dlib compila mesmo aqui — ver docs/roles.md)"
    else
        warn "libopenblas-dev/liblapack-dev indisponíveis — dlib compila sem elas (BLAS/LAPACK internos, mais lento, não é bloqueante)"
    fi

    sudo apt install -y -qq git curl lsof > /dev/null
    log "Git, curl e lsof instalados"

elif command -v dnf >/dev/null 2>&1; then
    log "Gerenciador de pacotes: dnf (Amazon Linux/RHEL/Fedora)"
    # --allowerasing: a imagem base do Amazon Linux já vem com curl-minimal,
    # que conflita com o pacote curl completo — sem essa flag o dnf recusa a
    # transação inteira (e sem isso o script já morreu aqui uma vez, calado
    # porque o erro estava indo pro /dev/null junto com o resto).

    sudo dnf install -y -q --allowerasing gcc gcc-c++ make cmake pkgconf-pkg-config > /dev/null
    log "Build essentials instalados"

    if ! sudo dnf install -y -q --allowerasing python3.12 python3.12-devel > /dev/null; then
        fail "python3.12 não encontrado nos repositórios dnf desta máquina. Rode \`dnf list available 'python3.12*'\` para ver o que existe — se a versão exata não estiver disponível, instale via outra fonte (ex: compilar do source) e rode este script de novo."
    fi
    log "Python 3.12 instalado"

    if sudo dnf install -y -q --allowerasing openblas-devel lapack-devel > /dev/null 2>/dev/null; then
        log "Bibliotecas numéricas instaladas (dlib compila mesmo aqui — ver docs/roles.md)"
    else
        warn "openblas-devel/lapack-devel indisponíveis — dlib compila sem elas (BLAS/LAPACK internos, mais lento, não é bloqueante)"
    fi

    sudo dnf install -y -q --allowerasing git curl lsof > /dev/null
    log "Git, curl e lsof instalados"

else
    fail "Nenhum gerenciador de pacotes suportado encontrado nesta máquina (esperado apt ou dnf)."
fi

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

warn "Preencha à mão no .env: AWS_ACCESS_KEY_ID/SECRET, PROCTOR_DASHBOARD_DATABASE_URL"
warn "(Postgres — ver docs/setup_dashboard.md passo 4) e PROCTOR_DASHBOARD_ADMIN_USER"
warn "/PASSWORD (login do professor; obrigatórios para o dashboard exigir login)."
warn "Campos de estação (câmera, gaze, proxy, station_id, station_token) não se"
warn "aplicam aqui — ver docs/setup_dashboard.md."

# ── 5. Testes ──
echo ""
echo "=========================================="
echo "  5/5  Rodando testes"
echo "=========================================="
warn "tests/test_dashboard.py fica de fora — precisa de Postgres já configurado"
warn "(PROCTOR_DASHBOARD_DATABASE_URL, passo 4 do runbook), o que normalmente ainda"
warn "não existe nesta hora do bootstrap. Depois de configurar o Postgres, rode"
warn "\`pytest tests/test_dashboard.py\` à parte pra validar essa parte."

python3 -m pytest tests/ -v --tb=short --ignore=tests/test_dashboard.py
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
