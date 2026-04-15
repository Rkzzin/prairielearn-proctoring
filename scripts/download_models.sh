#!/usr/bin/env bash
# Download dlib pre-trained model files.
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

echo ""
echo "All models ready in $TARGET/"