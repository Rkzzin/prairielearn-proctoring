#!/usr/bin/env bash
# Recuperacao manual do modo prova no GNOME atual.
# Pode ser executado pelo usuario proctor quando overlay/browser/lockdown travarem.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
API_PORT="${PROCTOR_APP_API_PORT:-8000}"
DISPLAY_VALUE="${DISPLAY:-:1}"

log() {
    printf '[recover-exam-mode] %s\n' "$*"
}

cd "$PROJECT_DIR"

if command -v curl >/dev/null 2>&1; then
    if curl -fsS -X POST "http://127.0.0.1:${API_PORT}/exam-mode/recover" >/dev/null 2>&1; then
        log "API local restaurou o modo prova"
    else
        log "API local indisponivel; aplicando limpeza local best-effort"
    fi
else
    log "curl ausente; aplicando limpeza local best-effort"
fi

pkill -f "src.kiosk.overlay_app" >/dev/null 2>&1 || true
pkill -f "xbindkeys -n -f /tmp/proctor-lockdown-" >/dev/null 2>&1 || true
pkill -f "chromium.*proctor-chromium-profile" >/dev/null 2>&1 || true
pkill -f "chrome.*proctor-chromium-profile" >/dev/null 2>&1 || true

if [ -x "venv/bin/python" ]; then
    venv/bin/python -m src.kiosk.recover_lockdown --display "$DISPLAY_VALUE"
else
    python3 -m src.kiosk.recover_lockdown --display "$DISPLAY_VALUE"
fi

log "recuperacao concluida"
