#!/usr/bin/env bash
# ============================================================================
#  detect_camera_audio.sh
#  Detecta a webcam USB e o microfone interno do notebook, ajustando
#  PROCTOR_FACE_CAMERA_INDEX e PROCTOR_REC_WEBCAM_AUDIO_DEVICE no .env.
#
#  Rode isto DEPOIS de conectar a webcam definitiva da estação, não durante
#  o bootstrap (que costuma rodar antes do hardware final estar plugado).
#  Rode de novo sempre que a webcam for trocada ou reconectada numa porta
#  USB diferente — o índice /dev/videoN e o card ALSA podem mudar.
#
#  Uso:
#    bash scripts/detect_camera_audio.sh
# ============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()   { echo -e "${GREEN}[✓]${NC} $*"; }
warn()  { echo -e "${YELLOW}[!]${NC} $*"; }
fail()  { echo -e "${RED}[✗]${NC} $*"; exit 1; }

set_env() {
    local key="$1"
    local value="$2"
    if grep -q "^${key}=" .env; then
        sed -i "s#^${key}=.*#${key}=${value}#" .env
    else
        printf '\n%s=%s\n' "$key" "$value" >> .env
    fi
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

if [ ! -f ".env" ]; then
    fail ".env não encontrado. Rode primeiro: cp .env.example .env (ou scripts/bootstrap.sh)"
fi

# ── Câmera ──
# Procura o primeiro /dev/videoN que fala MJPG (o nó de captura de imagem;
# webcams UVC costumam expor mais de um /dev/videoN — metadata/ISOC — e só
# um deles serve frames de vídeo de verdade).
CAMERA_DEV=""
CAMERA_INDEX=""
if command -v v4l2-ctl > /dev/null 2>&1; then
    for dev in /dev/video*; do
        [ -e "$dev" ] || continue
        if v4l2-ctl -d "$dev" --list-formats 2>/dev/null | grep -qi "MJPG\|Motion-JPEG"; then
            CAMERA_DEV="$dev"
            CAMERA_INDEX="${dev#/dev/video}"
            break
        fi
    done
else
    warn "v4l2-ctl não encontrado (pacote v4l-utils). Instale com: sudo apt install v4l-utils"
fi

if [ -n "$CAMERA_INDEX" ]; then
    set_env PROCTOR_FACE_CAMERA_INDEX "$CAMERA_INDEX"
    log "Câmera detectada em $CAMERA_DEV — PROCTOR_FACE_CAMERA_INDEX=$CAMERA_INDEX"
else
    warn "Nenhuma câmera com suporte a MJPG encontrada."
    warn "Confira se está conectada: v4l2-ctl --list-devices"
    warn "PROCTOR_FACE_CAMERA_INDEX no .env não foi alterado."
fi

# ── Microfone interno ──
PULSE_SOURCE=""
SOURCE_ID=""
ALSA_CARD=""
if command -v wpctl > /dev/null 2>&1; then
    SOURCE_LINE="$(XDG_RUNTIME_DIR="/run/user/$(id -u)" wpctl status -n 2>/dev/null \
        | grep -m1 -E 'alsa_input\.pci[^ ]*analog-stereo' || true)"
    PULSE_SOURCE="$(printf '%s' "$SOURCE_LINE" \
        | sed -E 's/.*[0-9]+\. (alsa_input\.pci[^ ]*analog-stereo).*/\1/')"
    SOURCE_ID="$(printf '%s' "$SOURCE_LINE" | sed -E 's/.* ([0-9]+)\. .*/\1/')"
    if [ -n "$SOURCE_ID" ]; then
        ALSA_CARD="$(XDG_RUNTIME_DIR="/run/user/$(id -u)" wpctl inspect "$SOURCE_ID" 2>/dev/null \
            | sed -nE 's/^[[:space:]]*alsa\.card = "([0-9]+)"/\1/p')"
    fi
fi

if [ -n "$PULSE_SOURCE" ] && [ -n "$ALSA_CARD" ]; then
    set_env PROCTOR_REC_WEBCAM_AUDIO_DEVICE "$PULSE_SOURCE"
    set_env PROCTOR_REC_WEBCAM_AUDIO_ALSA_CARD "$ALSA_CARD"
    set_env PROCTOR_REC_WEBCAM_AUDIO_CAPTURE_PERCENT "30"
    log "Microfone interno detectado: ${PULSE_SOURCE} — ALSA card ${ALSA_CARD}, ganho 30%"
else
    warn "Nenhuma entrada PipeWire interna foi identificada automaticamente."
    warn "Confira manualmente: wpctl status -n"
    warn "e ajuste PROCTOR_REC_WEBCAM_AUDIO_DEVICE no .env à mão."
fi

echo ""
if [ -z "$CAMERA_INDEX" ] || [ -z "$PULSE_SOURCE" ] || [ -z "$ALSA_CARD" ]; then
    warn "Detecção incompleta — confira o .env manualmente antes de subir o serviço."
else
    log "Detecção completa. Reinicie o serviço para aplicar: sudo systemctl restart proctor"
fi
