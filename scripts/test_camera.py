#!/usr/bin/env python3
"""Validação rápida de hardware: câmera + detecção facial.

Uso:
    python scripts/test_camera.py              # câmera 0
    python scripts/test_camera.py --camera 1   # câmera 1
    python scripts/test_camera.py --headless   # sem janela (CI/SSH)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import dlib
import numpy as np

# Adicionar raiz do projeto ao path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def test_camera(camera_index: int = 0, headless: bool = False) -> bool:
    print(f"\n{'='*50}")
    print(f"  Teste de câmera — /dev/video{camera_index}")
    print(f"{'='*50}\n")

    # 0. Verificar modelos
    sp_path = MODELS_DIR / "shape_predictor_68_face_landmarks.dat"
    rec_path = MODELS_DIR / "dlib_face_recognition_resnet_model_v1.dat"

    if not sp_path.exists() or not rec_path.exists():
        print("  ✗ Modelos dlib não encontrados")
        print(f"    Rode: ./scripts/download_models.sh {MODELS_DIR}")
        return False
    print("  ✓ Modelos dlib encontrados")

    # 1. Abrir câmera
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("  ✗ Não foi possível abrir a câmera")
        return False

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"  ✓ Câmera aberta: {w}x{h} @ {fps:.0f}fps")

    # 2. Capturar frame de teste
    for _ in range(10):
        cap.grab()

    ret, frame = cap.read()
    if not ret:
        print("  ✗ Falha ao capturar frame")
        cap.release()
        return False
    print(f"  ✓ Frame capturado: {frame.shape}")

    # 3. Testar detecção facial (HOG)
    detector = dlib.get_frontal_face_detector()
    print("  … Testando detecção facial (HOG)...")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    t0 = time.time()
    faces = detector(rgb, 1)
    dt = time.time() - t0
    print(f"  ✓ Detecção: {len(faces)} rosto(s) em {dt*1000:.0f}ms")

    # 4. Testar encoding (se rosto detectado)
    if len(faces) > 0:
        sp = dlib.shape_predictor(str(sp_path))
        encoder = dlib.face_recognition_model_v1(str(rec_path))

        shape = sp(rgb, faces[0])
        t0 = time.time()
        encoding = encoder.compute_face_descriptor(rgb, shape, 1)
        dt = time.time() - t0
        print(f"  ✓ Encoding: {len(encoding)}d em {dt*1000:.0f}ms")

    # 5. Benchmark: FPS de detecção
    print("  … Benchmark de detecção (20 frames)...")
    t0 = time.time()
    for _ in range(20):
        ret, frame = cap.read()
        if ret:
            small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            detector(rgb_small, 1)
    dt = time.time() - t0
    detection_fps = 20 / dt
    print(f"  ✓ Detection FPS (scale=0.5): {detection_fps:.1f} fps")

    if detection_fps < 5:
        print("  ⚠ FPS baixo — considere reduzir detection_scale ou usar GPU")
    elif detection_fps < 15:
        print("  ○ FPS aceitável para proctoring (alvo: >10fps)")
    else:
        print("  ✓ FPS excelente para proctoring")

    # 6. Preview (se não headless)
    if not headless:
        print("\n  Mostrando preview — pressione Q para sair\n")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            rects = detector(rgb_small, 1)

            for r in rects:
                s = 2
                cv2.rectangle(
                    frame,
                    (r.left() * s, r.top() * s),
                    (r.right() * s, r.bottom() * s),
                    (0, 255, 0),
                    2,
                )

            cv2.putText(
                frame,
                f"Faces: {len(rects)} | FPS: {detection_fps:.0f} | Q=sair",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Camera Test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n{'='*50}")
    print("  Teste concluído com sucesso")
    print(f"{'='*50}\n")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Teste de câmera e detecção facial")
    parser.add_argument("--camera", type=int, default=0, help="Índice da câmera")
    parser.add_argument("--headless", action="store_true", help="Sem janela (para SSH/CI)")
    args = parser.parse_args()

    success = test_camera(args.camera, args.headless)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()