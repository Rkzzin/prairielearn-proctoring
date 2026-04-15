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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import face_recognition


def test_camera(camera_index: int = 0, headless: bool = False) -> bool:
    print(f"\n{'='*50}")
    print(f"  Teste de câmera — /dev/video{camera_index}")
    print(f"{'='*50}\n")

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
    # Descartar primeiros frames (warm-up)
    for _ in range(10):
        cap.grab()

    ret, frame = cap.read()
    if not ret:
        print("  ✗ Falha ao capturar frame")
        cap.release()
        return False
    print(f"  ✓ Frame capturado: {frame.shape}")

    # 3. Testar detecção facial
    print("  … Testando detecção facial (HOG)...")
    t0 = time.time()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb, model="hog")
    dt = time.time() - t0
    print(f"  ✓ Detecção: {len(locations)} rosto(s) em {dt*1000:.0f}ms")

    # 4. Testar encoding (se rosto detectado)
    if locations:
        t0 = time.time()
        encodings = face_recognition.face_encodings(rgb, locations)
        dt = time.time() - t0
        print(f"  ✓ Encoding: {len(encodings[0])}d em {dt*1000:.0f}ms")

    # 5. Benchmark: FPS de detecção
    print("  … Benchmark de detecção (20 frames)...")
    t0 = time.time()
    for _ in range(20):
        ret, frame = cap.read()
        if ret:
            small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            face_recognition.face_locations(rgb_small, model="hog")
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
            locs = face_recognition.face_locations(rgb_small, model="hog")

            for top, right, bottom, left in locs:
                s = 2  # scale back
                cv2.rectangle(frame, (left*s, top*s), (right*s, bottom*s), (0, 255, 0), 2)

            cv2.putText(
                frame,
                f"Faces: {len(locs)} | FPS: {detection_fps:.0f} | Q=sair",
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
