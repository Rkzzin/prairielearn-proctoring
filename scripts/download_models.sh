#!/usr/bin/env bash
# Download dlib and passive-liveness pre-trained model files.
# These are the same models that face_recognition_models shipped,
# but fetched directly from dlib's official source.
#
# Usage:
#   ./scripts/download_models.sh [target_dir]
#   Default target: ./models/

set -euo pipefail

TARGET="${1:-models}"
mkdir -p "$TARGET"

BASE_URL="http://dlib.net/files"

MODELS=(
    "shape_predictor_68_face_landmarks.dat.bz2"
    "shape_predictor_5_face_landmarks.dat.bz2"
    "dlib_face_recognition_resnet_model_v1.dat.bz2"
    "mmod_human_face_detector.dat.bz2"
)

echo "Downloading dlib models to $TARGET/"
echo ""

for archive in "${MODELS[@]}"; do
    dat="${archive%.bz2}"
    if [ -f "$TARGET/$dat" ]; then
        echo "  ✓ $dat (already exists)"
        continue
    fi

    echo "  ↓ $dat ..."
    curl -fSL "$BASE_URL/$archive" -o "$TARGET/$archive"
    bunzip2 -f "$TARGET/$archive"
    echo "  ✓ $dat"
done

LIVENESS_MODEL="minifasnet_v2.onnx"
LIVENESS_URL="https://huggingface.co/garciafido/minifasnet-v2-anti-spoofing-onnx/resolve/d29c87568ca9b5662da803b10f217c4db20b142b/$LIVENESS_MODEL"
LIVENESS_SHA256="d7b3cd9ba8a7ceb13baa8c4720902e27ca3112eff52f926c08804af6b6eecc7b"

if [ -f "$TARGET/$LIVENESS_MODEL" ] && printf '%s  %s\n' "$LIVENESS_SHA256" "$TARGET/$LIVENESS_MODEL" | sha256sum --check --status; then
    echo "  ✓ $LIVENESS_MODEL (already exists)"
else
    echo "  ↓ $LIVENESS_MODEL ..."
    curl -fSL "$LIVENESS_URL" -o "$TARGET/$LIVENESS_MODEL.tmp"
    printf '%s  %s\n' "$LIVENESS_SHA256" "$TARGET/$LIVENESS_MODEL.tmp" | sha256sum --check --status
    mv "$TARGET/$LIVENESS_MODEL.tmp" "$TARGET/$LIVENESS_MODEL"
    echo "  ✓ $LIVENESS_MODEL"
fi

DEVICE_MODEL="object_detection_yolox.onnx"
DEVICE_MODEL_URL="https://github.com/opencv/opencv_zoo/raw/47534e27c9851bb1128ccc0102f1145e27f23f98/models/object_detection_yolox/object_detection_yolox_2022nov.onnx"
DEVICE_MODEL_SHA256="c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063"

if [ -f "$TARGET/$DEVICE_MODEL" ] && printf '%s  %s\n' "$DEVICE_MODEL_SHA256" "$TARGET/$DEVICE_MODEL" | sha256sum --check --status; then
    echo "  ✓ $DEVICE_MODEL (already exists)"
else
    echo "  ↓ $DEVICE_MODEL ..."
    curl -fSL "$DEVICE_MODEL_URL" -o "$TARGET/$DEVICE_MODEL.tmp"
    printf '%s  %s\n' "$DEVICE_MODEL_SHA256" "$TARGET/$DEVICE_MODEL.tmp" | sha256sum --check --status
    mv "$TARGET/$DEVICE_MODEL.tmp" "$TARGET/$DEVICE_MODEL"
    echo "  ✓ $DEVICE_MODEL"
fi

echo ""
echo "All models ready in $TARGET/"
