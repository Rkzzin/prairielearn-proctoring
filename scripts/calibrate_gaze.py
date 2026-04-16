#!/usr/bin/env python3
"""Calibrador visual de gaze — cabeça + olhos.

Mostra o feed da webcam com overlay de yaw, pitch e eye ratio
em tempo real. Use para encontrar os thresholds ideais para
seu ambiente antes de ajustar o .env.

Uso:
    python scripts/calibrate_gaze.py
    python scripts/calibrate_gaze.py --camera 1
    python scripts/calibrate_gaze.py --no-eye   # desativa ratio ocular

Referência de valores:
    Yaw ratio  > 0.35 (≈ 31°) → GAZE_WARNING horizontal
    Pitch ratio > 0.30 (≈ 27°) → GAZE_WARNING vertical
    Eye ratio  < 0.5 ou > 1.5  → desvio ocular

    Ajuste PROCTOR_GAZE_H_THRESHOLD e PROCTOR_GAZE_V_THRESHOLD no .env.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config import FaceConfig, ProctorConfig
from src.proctor.gaze import GazeEstimator


def run_calibration(camera_index: int = 0, enable_eye: bool = True) -> None:
    face_cfg = FaceConfig()
    proctor_cfg = ProctorConfig()

    estimator = GazeEstimator(face_config=face_cfg, enable_eye_gaze=enable_eye)
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print(f"  ✗ Não foi possível abrir a câmera {camera_index}")
        sys.exit(1)

    print("=" * 55)
    print("  CALIBRADOR DE GAZE — pressione Q para sair")
    print("=" * 55)
    print(f"  Thresholds atuais (.env):")
    print(f"    Horizontal : {proctor_cfg.gaze_h_threshold}")
    print(f"    Vertical   : {proctor_cfg.gaze_v_threshold}")
    print(f"    Duração    : {proctor_cfg.gaze_duration_sec}s")
    print("=" * 55)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        data = estimator.process_frame(frame)

        if data is None:
            cv2.putText(frame, "SEM ROSTO", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        else:
            yaw_ratio   = abs(data.yaw) / 90.0
            pitch_centered = abs(data.pitch) - 180.0
            pitch_ratio = abs(pitch_centered) / 90.0

            h_ok    = yaw_ratio   <= proctor_cfg.gaze_h_threshold
            v_ok    = pitch_ratio <= proctor_cfg.gaze_v_threshold
            gaze_ok = h_ok and v_ok

            status_color = (0, 255, 0) if gaze_ok else (0, 0, 255)
            status_text  = "OK" if gaze_ok else "DESVIO"

            # Indicador de status
            cv2.circle(frame, (frame.shape[1] - 50, 50), 20, status_color, -1)
            cv2.putText(frame, status_text, (frame.shape[1] - 110, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

            # Valores numéricos
            lines = [
                (f"Yaw   : {data.yaw:+.1f}  (ratio {yaw_ratio:.2f} / lim {proctor_cfg.gaze_h_threshold})",
                 (0, 255, 0) if h_ok else (0, 0, 255)),
                (f"Pitch : {data.pitch:+.1f} → centrado {pitch_centered:+.1f}  (ratio {pitch_ratio:.2f} / lim {proctor_cfg.gaze_v_threshold})",
                 (0, 255, 0) if v_ok else (0, 0, 255)),
                (f"Roll  : {data.roll:+.1f}", (200, 200, 200)),
            ]
            if data.eye_ratio is not None:
                eye_ok = 0.5 <= data.eye_ratio <= 1.5
                lines.append((
                    f"Eye   : {data.eye_ratio:.2f}  (ok: 0.5–1.5)",
                    (0, 255, 0) if eye_ok else (0, 0, 255),
                ))

            for i, (text, color) in enumerate(lines):
                cv2.putText(frame, text, (20, 50 + i * 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Calibrador de Gaze", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrador visual de gaze")
    parser.add_argument("--camera", type=int, default=0, help="Índice da câmera")
    parser.add_argument("--no-eye", action="store_true",
                        help="Desativar ratio ocular")
    args = parser.parse_args()
    run_calibration(camera_index=args.camera, enable_eye=not args.no_eye)


if __name__ == "__main__":
    main()