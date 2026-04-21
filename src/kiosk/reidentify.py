"""Tela de re-identificação facial durante bloqueio de proctoring.

Exibida quando o proctoring engine emite BLOCKED. O aluno olha para
a câmera e o sistema tenta re-identificá-lo. Se bem-sucedido, a sessão
é retomada. Se o timeout expirar, o bloqueio permanece.

Sem janela OpenCV — output no terminal (igual ao test_integration.py).
Em produção (M5), isso será substituído por uma tela GTK fullscreen.

Uso:
    ok = run_reidentify(
        recognizer=recognizer,
        cap=cap,
        expected_student_id="henriquels5",
        timeout_sec=60.0,
    )
    if ok:
        engine.unblock()
        kiosk.unblock()
"""

from __future__ import annotations

import logging
import time

import cv2

from src.face.recognizer import FaceRecognizer

logger = logging.getLogger(__name__)


def run_reidentify(
    recognizer: FaceRecognizer,
    cap: cv2.VideoCapture,
    expected_student_id: str,
    timeout_sec: float = 60.0,
    required_matches: int = 3,
) -> bool:
    """Tenta re-identificar o aluno na câmera.

    Lê frames continuamente até identificar o aluno correto
    `required_matches` vezes consecutivas, ou até o timeout.

    Args:
        recognizer: FaceRecognizer com a turma já carregada.
        cap: VideoCapture já aberto (mesmo cap do proctoring).
        expected_student_id: ID do aluno que deve ser identificado.
        timeout_sec: Tempo máximo de espera em segundos.
        required_matches: Quantas identificações consecutivas para confirmar.

    Returns:
        True se re-identificado com sucesso, False se timeout.
    """
    print(f"\n  ⚠  SESSÃO BLOQUEADA")
    print(f"  Olhe para a câmera para retomar a prova.")
    print(f"  Tempo limite: {int(timeout_sec)}s\n")

    t0 = time.time()
    consecutive = 0

    while time.time() - t0 < timeout_sec:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        result = recognizer.identify(frame)
        elapsed = time.time() - t0
        remaining = max(0.0, timeout_sec - elapsed)

        if result.is_match and result.student_id == expected_student_id:
            consecutive += 1
            print(f"  [{elapsed:>5.1f}s] Identificado: {result.student_name} "
                  f"(confiança: {result.confidence:.2f}) "
                  f"[{consecutive}/{required_matches}]",
                  end="\r")

            if consecutive >= required_matches:
                print(f"\n  ✓ Re-identificação bem-sucedida — retomando sessão\n")
                logger.info(
                    "Re-identificação OK: %s após %.1fs",
                    expected_student_id, elapsed,
                )
                return True

        elif result.is_match and result.student_id != expected_student_id:
            # Outro aluno tentando desbloquear
            consecutive = 0
            print(f"  [{elapsed:>5.1f}s] Aluno incorreto — acesso negado          ",
                  end="\r")
            logger.warning(
                "Re-identificação: aluno errado '%s' (esperado '%s')",
                result.student_id, expected_student_id,
            )

        else:
            consecutive = 0
            status = {
                "NO_FACE":        "Nenhum rosto detectado",
                "MULTIPLE_FACES": f"{result.face_count} rostos — fique sozinho",
                "NO_MATCH":       "Rosto não reconhecido",
            }.get(result.status.value, "...")
            print(f"  [{elapsed:>5.1f}s] {status} — {remaining:.0f}s restantes     ",
                  end="\r")

    print(f"\n  ✗ Timeout — sessão permanece bloqueada\n")
    logger.warning(
        "Re-identificação timeout após %.1fs — sessão permanece bloqueada",
        timeout_sec,
    )
    return False