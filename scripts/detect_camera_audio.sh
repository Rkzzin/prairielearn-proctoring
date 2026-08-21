#!/usr/bin/env bash
# ============================================================================
#  detect_camera_audio.sh
#  Detecta a webcam USB conectada agora e ajusta PROCTOR_FACE_CAMERA_INDEX e
#  PROCTOR_REC_WEBCAM_AUDIO_DEVICE no .env.
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
    sed -i "s#^PROCTOR_FACE_CAMERA_INDEX=.*#PROCTOR_FACE_CAMERA_INDEX=${CAMERA_INDEX}#" .env
    log "Câmera detectada em $CAMERA_DEV — PROCTOR_FACE_CAMERA_INDEX=$CAMERA_INDEX"
else
    warn "Nenhuma câmera com suporte a MJPG encontrada."
    warn "Confira se está conectada: v4l2-ctl --list-devices"
    warn "PROCTOR_FACE_CAMERA_INDEX no .env não foi alterado."
fi

# ── Microfone ──
# Procura o primeiro card ALSA de captura que pareça ser webcam USB (a
# maioria se anuncia como 'USB Audio' no nome do dispositivo).
CARD_LINE=""
if command -v arecord > /dev/null 2>&1; then
    CARD_LINE="$(arecord -l 2>/dev/null | grep -im1 -E 'usb|webcam|c920|c922|c930')"
else
    warn "arecord não encontrado (pacote alsa-utils). Instale com: sudo apt install alsa-utils"
fi

if [ -n "$CARD_LINE" ]; then
    CARD_NAME="$(echo "$CARD_LINE" | sed -E 's/^card [0-9]+: ([^ ]+).*/\1/')"
    if [ -n "$CARD_NAME" ]; then
        sed -i "s#^PROCTOR_REC_WEBCAM_AUDIO_DEVICE=.*#PROCTOR_REC_WEBCAM_AUDIO_DEVICE=default#" .env
        log "Microfone da webcam detectado: ${CARD_NAME}; usando Pulse default para compartilhar via PipeWire"
    fi
else
    warn "Nenhum card ALSA de webcam identificado automaticamente."
    warn "Confira manualmente: arecord -l"
    warn "e ajuste PROCTOR_REC_WEBCAM_AUDIO_DEVICE no .env à mão."
fi

echo ""
if [ -z "$CAMERA_INDEX" ] || [ -z "${CARD_NAME:-}" ]; then
    warn "Detecção incompleta — confira o .env manualmente antes de subir o serviço."
else
    log "Detecção completa. Reinicie o serviço para aplicar: sudo systemctl restart proctor"
fi
