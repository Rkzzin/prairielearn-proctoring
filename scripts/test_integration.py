#!/usr/bin/env python3
"""Teste de integração manual — Fase 1 + Fase 2.

Fluxo:
  1. Carrega o .pkl da turma
  2. Abre a webcam
  3. Tenta identificar o aluno (até max_attempts)
  4. Se identificado, inicia o proctoring engine ao vivo
  5. Exibe estado na tela em tempo real
  6. Pressione Q para encerrar

Uso:
    python scripts/test_integration.py --turma SUA_TURMA
    python scripts/test_integration.py --turma SUA_TURMA --camera 1
    python scripts/test_integration.py --turma SUA_TURMA --attempts 5
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
from src.proctor.engine import ProctorEngine, ProctorState
from src.recorder.capture import Capture
from src.recorder.uploader import Uploader

# ── Cores BGR ──
GREEN  = (0, 200, 0)
YELLOW = (0, 200, 255)
RED    = (0, 0, 220)
WHITE  = (255, 255, 255)
GRAY   = (160, 160, 160)

STATE_COLOR = {
    ProctorState.NORMAL:    GREEN,
    ProctorState.GAZE_WARN: YELLOW,
    ProctorState.ABSENCE:   YELLOW,
    ProctorState.BLOCKED:   RED,
}


def _overlay(frame, lines: list[tuple[str, tuple]]) -> None:
    """Escreve linhas de texto no canto superior esquerdo."""
    for i, (text, color) in enumerate(lines):
        cv2.putText(frame, text, (20, 40 + i * 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def phase_identify(
    recognizer: FaceRecognizer,
    cap: cv2.VideoCapture,
    max_attempts: int,
) -> str | None:
    """Fase 1 — identifica o aluno na webcam.

    Exibe o feed ao vivo e tenta identificar por até max_attempts frames.
    Retorna o student_id se identificado, None caso contrário.
    """
    print(f"\n  Fase 1 — Identificação (até {max_attempts} tentativas)")
    print("  Olhe para a câmera | Q = cancelar\n")

    attempt = 0
    while attempt < max_attempts:
        ret, frame = cap.read()
        if not ret:
            continue

        display = frame.copy()
        attempt += 1

        result = recognizer.identify(frame)

        if result.is_match:
            # Desenhar retângulo verde no rosto
            if result.face_location:
                top, right, bottom, left = result.face_location
                cv2.rectangle(display, (left, top), (right, bottom), GREEN, 2)

            _overlay(display, [
                (f"IDENTIFICADO: {result.student_name}", GREEN),
                (f"Confianca: {result.confidence:.2f}", GREEN),
                (f"Tentativa {attempt}/{max_attempts}", GRAY),
            ])
            cv2.imshow("Proctor — Identificacao", display)
            cv2.waitKey(1000)
            cv2.destroyWindow("Proctor — Identificacao")
            print(f"  ✓ Identificado: {result.student_name} "
                  f"(confiança: {result.confidence:.2f})")
            return result.student_id

        # Feedback visual por status
        status_msgs = {
            "NO_FACE":        ("Nenhum rosto detectado", YELLOW),
            "MULTIPLE_FACES": (f"{result.face_count} rostos — fique sozinho", RED),
            "NO_MATCH":       ("Rosto nao reconhecido", RED),
        }
        msg, color = status_msgs.get(result.status.value, ("...", GRAY))

        _overlay(display, [
            (msg, color),
            (f"Tentativa {attempt}/{max_attempts}", GRAY),
        ])
        cv2.imshow("Proctor — Identificacao", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            print("\n  Cancelado pelo operador.")
            return None

    cv2.destroyWindow("Proctor — Identificacao")
    print(f"  ✗ Não identificado após {max_attempts} tentativas.")
    return None


def phase_proctor(
    engine: ProctorEngine,
    cap: cv2.VideoCapture,
    student_name: str,
    capture: Capture | None = None,
    uploader: Uploader | None = None,
) -> None:
    """Fase 2 — loop de proctoring ao vivo.

    Exibe estado da FSM na tela em tempo real.
    Pressione Q para encerrar, U para desbloquear manualmente.
    """
    print(f"\n  Fase 2 — Proctoring ativo para: {student_name}")
    print("  Q = encerrar | U = desbloquear (simula re-identificação)\n")

    engine.start()
    if uploader:
        uploader.start()
    if capture:
        capture.start()
        print("  Gravação iniciada (webcam + tela)")
    frame_times: list[float] = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            t0 = time.time()
            state = engine.update(frame)
            frame_times.append(time.time() - t0)
            if len(frame_times) > 30:
                frame_times.pop(0)

            fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0
            color = STATE_COLOR.get(state, GRAY)

            lines = [
                (f"Aluno : {student_name}", WHITE),
                (f"Estado: {state.value}", color),
                (f"FPS   : {fps:.1f}", GRAY),
            ]

            if state == ProctorState.GAZE_WARN:
                elapsed = time.time() - engine._warn_start
                remaining = max(0, engine._cfg.gaze_duration_sec - elapsed)
                lines.append((f"Bloqueio em: {remaining:.1f}s", YELLOW))

            if state == ProctorState.ABSENCE:
                elapsed = time.time() - engine._absence_start
                remaining = max(0, engine._cfg.absence_timeout_sec - elapsed)
                lines.append((f"Ausencia: {remaining:.1f}s", YELLOW))

            if state == ProctorState.BLOCKED:
                lines.append((f"BLOQUEADO — motivo: {engine.block_reason.value}", RED))
                lines.append(("Pressione U para desbloquear", GRAY))

            display = frame.copy()
            _overlay(display, lines)

            # Borda colorida indica estado
            cv2.rectangle(display, (0, 0),
                          (display.shape[1]-1, display.shape[0]-1), color, 6)

            cv2.imshow("Proctor — Monitoramento", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("u") and state == ProctorState.BLOCKED:
                engine.unblock()
                print("  → Sessão desbloqueada manualmente")

    finally:
        if capture:
            capture.stop()
        engine.stop()
        if uploader:
            uploader.stop()
        cv2.destroyAllWindows()
        print(f"\n  Sessão encerrada. Log salvo em:")
        print(f"  {engine._logger.log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Teste de integração — identificação + proctoring ao vivo"
    )
    parser.add_argument("--turma", required=True, help="ID da turma (ex: SUA_TURMA)")
    parser.add_argument("--camera", type=int, default=0, help="Índice da câmera")
    parser.add_argument("--attempts", type=int, default=30,
                        help="Máx de tentativas de identificação (default: 30)")
    parser.add_argument("--session-id", dest="session_id", default=None,
                        help="ID da sessão para o log (default: turma_timestamp)")
    parser.add_argument("--no-record", dest="no_record", action="store_true",
                        help="Desativar gravação (só identificação + proctoring)")
    args = parser.parse_args()

    # Configs
    face_cfg    = FaceConfig()
    proctor_cfg = ProctorConfig()
    app_cfg     = AppConfig()
    rec_cfg     = app_cfg.recorder
    s3_cfg      = app_cfg.s3

    session_id = args.session_id or (
        f"{args.turma}_{time.strftime('%Y%m%d_%H%M%S')}"
    )

    # Carregar turma
    recognizer = FaceRecognizer(face_cfg)
    try:
        recognizer.load_turma(args.turma)
    except FileNotFoundError:
        print(f"\n  ✗ Turma '{args.turma}' não encontrada.")
        print(f"  Rode primeiro: python scripts/enroll.py --turma {args.turma}")
        sys.exit(1)

    print(f"\n  Turma '{args.turma}' carregada "
          f"({recognizer.turma.student_count} alunos)")

    # Abrir câmera
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, face_cfg.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, face_cfg.camera_height)
    cap.set(cv2.CAP_PROP_FPS, face_cfg.camera_fps)

    if not cap.isOpened():
        print(f"\n  ✗ Não foi possível abrir a câmera {args.camera}")
        sys.exit(1)

    try:
        # Fase 1 — identificação
        student_id = phase_identify(recognizer, cap, args.attempts)
        if student_id is None:
            sys.exit(1)

        student_name = recognizer.turma.students[student_id].student_name

        # Fase 2 — proctoring
        engine = ProctorEngine(
            session_id=session_id,
            proctor_config=proctor_cfg,
            face_config=face_cfg,
            app_config=app_cfg,
            enable_eye_gaze=False,
        )
        # Recorder (opcional)
        capture = None
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

        phase_proctor(engine, cap, student_name, capture=capture, uploader=uploader)

    finally:
        cap.release()


if __name__ == "__main__":
    main()