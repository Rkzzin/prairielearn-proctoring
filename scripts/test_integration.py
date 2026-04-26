#!/usr/bin/env python3
"""Teste de integração manual — Fases 1 + 2 + 3 + 4.

Fluxo:
  1. Abre câmera UMA vez
  2. Fase 1: identifica o aluno — output no terminal
  3. Chromium kiosk abre no PrairieLearn
  4. Fase 2: proctoring + gravação
     - NORMAL: Chromium rodando, aluno faz prova
     - BLOCKED: Chromium congela → re-identificação facial
     - Re-identificado: Chromium retoma
  5. Ctrl+C encerra (produção: session manager via M5)

Uso:
    python scripts/test_integration.py --turma T2026-T1
    python scripts/test_integration.py --turma T2026-T1 --no-record
    python scripts/test_integration.py --turma T2026-T1 --no-kiosk
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config import AppConfig, FaceConfig, ProctorConfig
from src.face.recognizer import FaceRecognizer
from src.kiosk.chromium import ChromiumKiosk
from src.kiosk.lockdown import Lockdown
from src.kiosk.reidentify import run_reidentify
from src.proctor.engine import ProctorEngine, ProctorState
from src.recorder.capture import Capture
from src.recorder.uploader import Uploader

PRAIRIELEARN_URL = "https://prairielearn.org/pl"


def phase_identify(
    recognizer: FaceRecognizer,
    cap: cv2.VideoCapture,
    max_attempts: int,
) -> str | None:
    """Fase 1 — identifica o aluno na webcam. Output no terminal."""
    print(f"\n  Fase 1 — Identificação (até {max_attempts} tentativas)")
    print("  Olhe para a câmera | Ctrl+C = cancelar\n")

    attempt = 0
    while attempt < max_attempts:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        attempt += 1
        result = recognizer.identify(frame)

        if result.is_match:
            print(f"  ✓ Identificado: {result.student_name} "
                  f"(confiança: {result.confidence:.2f})")
            return result.student_id

        status_msgs = {
            "NO_FACE":        "Nenhum rosto detectado",
            "MULTIPLE_FACES": f"{result.face_count} rostos — fique sozinho",
            "NO_MATCH":       "Rosto não reconhecido",
        }
        msg = status_msgs.get(result.status.value, "...")
        print(f"  [{attempt:>2}/{max_attempts}] {msg}", end="\r")

    print(f"\n  ✗ Não identificado após {max_attempts} tentativas.")
    return None


def phase_proctor(
    engine: ProctorEngine,
    cap: cv2.VideoCapture,
    student_id: str,
    student_name: str,
    recognizer: FaceRecognizer,
    kiosk: ChromiumKiosk | None = None,
    capture: Capture | None = None,
    uploader: Uploader | None = None,
) -> None:
    """Fase 2 — loop de proctoring ao vivo com gravação e kiosk.

    BLOCKED → Chromium congela → re-identificação → Chromium retoma.
    Encerrar com Ctrl+C.
    """
    print(f"\n  Fase 2 — Proctoring ativo para: {student_name}")
    print("  Ctrl+C = encerrar\n")

    engine.start()
    if uploader:
        uploader.start()
    if capture:
        capture.start()
        print("  Gravação iniciada (webcam via v4l2 + tela via x11grab)\n")

    last_state  = None
    frame_count = 0
    t_status    = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            frame_count += 1
            state = engine.update(frame)

            # Mudança de estado
            if state != last_state:
                suffix = (f" — {engine.block_reason.value}"
                          if engine.block_reason else "")
                print(f"  [{time.strftime('%H:%M:%S')}] "
                      f"Estado: {state.value}{suffix}")
                last_state = state

                # BLOCKED → congelar Chromium e re-identificar
                if state == ProctorState.BLOCKED:
                    if kiosk:
                        kiosk.block()

                    ok = run_reidentify(
                        recognizer=recognizer,
                        cap=cap,
                        expected_student_id=student_id,
                        timeout_sec=60.0,
                        required_matches=3,
                    )

                    if ok:
                        engine.unblock()
                        if kiosk:
                            kiosk.unblock()
                        last_state = None  # força reimpressão do novo estado

            # Status periódico a cada 10s
            now = time.time()
            if now - t_status >= 10.0:
                t_status = now
                print(f"  [{time.strftime('%H:%M:%S')}] "
                      f"frame={frame_count}  estado={state.value}")

    except KeyboardInterrupt:
        print("\n  Encerrando...")

    finally:
        if kiosk:
            kiosk.stop()
        if capture:
            capture.stop()
        engine.stop()
        if uploader:
            uploader.stop()
        print(f"\n  Sessão encerrada. Log: {engine._logger.log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Teste de integração — identificação + proctoring + kiosk + gravação"
    )
    parser.add_argument("--turma",     required=True, help="ID da turma (ex: T2026-T1)")
    parser.add_argument("--camera",    type=int, default=0, help="Índice da câmera")
    parser.add_argument("--attempts",  type=int, default=30,
                        help="Máx tentativas de identificação (default: 30)")
    parser.add_argument("--session-id", dest="session_id", default=None,
                        help="ID da sessão (default: turma_timestamp)")
    parser.add_argument("--no-record", dest="no_record", action="store_true",
                        help="Desativar gravação")
    parser.add_argument("--no-kiosk",  dest="no_kiosk",  action="store_true",
                        help="Desativar Chromium kiosk (só proctoring)")
    parser.add_argument("--url",       default=PRAIRIELEARN_URL,
                        help=f"URL do kiosk (default: {PRAIRIELEARN_URL})")
    args = parser.parse_args()

    face_cfg    = FaceConfig()
    proctor_cfg = ProctorConfig()
    app_cfg     = AppConfig()
    rec_cfg     = app_cfg.recorder
    s3_cfg      = app_cfg.s3

    session_id = args.session_id or f"{args.turma}_{time.strftime('%Y%m%d_%H%M%S')}"

    # Carregar turma
    recognizer = FaceRecognizer(face_cfg)
    try:
        recognizer.load_turma(args.turma)
    except FileNotFoundError:
        print(f"\n  ✗ Turma '{args.turma}' não encontrada.")
        print(f"  Rode: python scripts/enroll.py --turma {args.turma}")
        sys.exit(1)

    print(f"\n  Turma '{args.turma}' carregada "
          f"({recognizer.turma.student_count} alunos)")

    # ── Câmera aberta UMA vez ──
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  face_cfg.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, face_cfg.camera_height)
    cap.set(cv2.CAP_PROP_FPS,          face_cfg.camera_fps)

    if not cap.isOpened():
        print(f"\n  ✗ Não foi possível abrir a câmera {args.camera}")
        sys.exit(1)

    # ── Lockdown de teclado ──
    lockdown = Lockdown(display=rec_cfg.display)

    try:
        # ── Fase 1: identificação ──
        student_id = phase_identify(recognizer, cap, args.attempts)
        if student_id is None:
            sys.exit(1)

        student_name = recognizer.turma.students[student_id].student_name

        # ── Kiosk ──
        kiosk = None
        if not args.no_kiosk:
            kiosk = ChromiumKiosk(display=rec_cfg.display)
            kiosk.start(args.url)
            lockdown.enable()

        # ── Engine ──
        engine = ProctorEngine(
            session_id=session_id,
            proctor_config=proctor_cfg,
            face_config=face_cfg,
            app_config=app_cfg,
            enable_eye_gaze=False,
        )

        # ── Recorder ──
        capture  = None
        uploader = None
        if not args.no_record:
            uploader = Uploader(
                session_id=session_id,
                s3_config=s3_cfg,
                app_config=app_cfg,
                delete_after_upload=rec_cfg.delete_after_upload,
            )
            capture = Capture(
                session_id=session_id,
                s3_config=s3_cfg,
                face_config=face_cfg,
                app_config=app_cfg,
                on_segment_ready=uploader.enqueue,
                display=rec_cfg.display,
                screen_size=rec_cfg.screen_size,
            )

        # ── Fase 2: proctoring + kiosk + gravação ──
        phase_proctor(
            engine=engine,
            cap=cap,
            student_id=student_id,
            student_name=student_name,
            recognizer=recognizer,
            kiosk=kiosk,
            capture=capture,
            uploader=uploader,
        )

    finally:
        lockdown.disable()
        cap.release()
        # Garantir que extensões do Gnome são sempre restauradas
        # mesmo se kiosk.stop() não foi chamado (ex: exceção inesperada)
        import subprocess, os
        env = os.environ.copy()
        env["DISPLAY"] = rec_cfg.display
        for ext in ["ubuntu-dock@ubuntu.com", "tiling-assistant@ubuntu.com"]:
            subprocess.run(
                ["gnome-extensions", "enable", ext],
                env=env, capture_output=True,
            )


if __name__ == "__main__":
    main()
