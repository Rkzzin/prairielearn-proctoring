"""Executa scripts/enroll.py em background para o comando RUN_ENROLL do dashboard.

O enroll via S3 que o próprio dashboard oferece (`/enrollment`) só registra o
enrollment no banco do dashboard — não gera o `.pkl` local que a estação usa
pra identificar aluno (`FaceRecognizer.load_turma`, `src/face/recognizer.py`).
Por isso o professor precisa de um jeito de disparar `scripts/enroll.py` na
NUC de verdade; este módulo é o lado que roda lá, acionado por um comando
RUN_ENROLL vindo no heartbeat (ver `src/core/dashboard_sync.py`).

Roda cada turma como subprocesso (`enroll.py` já usa `sys.exit()` em caminhos
de erro — importar e chamar a função direto mataria o `proctor.service`) numa
thread própria, pra não travar o heartbeat (que roda a cada
`heartbeat_interval_sec`, hoje 5s, e ficaria "atrasado" na visão do dashboard
se `_apply_command` bloqueasse por minutos aqui).
"""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from src.core.session import SessionManager, SessionState

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

#: Teto de tempo por turma — fotos+encoding de uma turma não deveriam passar
#: disso; existe só pra não deixar um enroll travado preso pra sempre.
_ENROLL_TIMEOUT_SEC = 1800


class EnrollRunner:
    def __init__(
        self,
        *,
        session_manager: SessionManager,
        project_root: Path | None = None,
        python_bin: str | None = None,
    ):
        self._session_manager = session_manager
        self._project_root = project_root or _PROJECT_ROOT
        self._python_bin = python_bin or sys.executable
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._status = "idle"
        self._message = ""

    def status_dict(self) -> dict[str, Any]:
        with self._lock:
            return {"enroll_status": self._status, "enroll_message": self._message}

    def start(self, turma_ids: list[str]) -> None:
        """Dispara o enroll das turmas listadas. Não bloqueia."""
        with self._lock:
            if self._status == "running":
                logger.warning("RUN_ENROLL recebido com um enroll já em andamento — ignorado")
                return
            if not turma_ids:
                self._status = "error"
                self._message = "nenhuma turma selecionada"
                return
            if self._session_manager.state != SessionState.IDLE:
                self._status = "recusado"
                self._message = f"prova ativa nesta estação ({self._session_manager.state.value})"
                logger.warning(
                    "RUN_ENROLL recusado: estação em %s, não em IDLE",
                    self._session_manager.state.value,
                )
                return
            self._status = "running"
            self._message = f"0/{len(turma_ids)}"
            self._thread = threading.Thread(
                target=self._run,
                args=(list(turma_ids),),
                name="enroll-runner",
                daemon=True,
            )
            self._thread.start()

    def _run(self, turma_ids: list[str]) -> None:
        failed: list[str] = []
        for index, turma_id in enumerate(turma_ids):
            with self._lock:
                self._message = f"{index}/{len(turma_ids)} — processando '{turma_id}'"
            try:
                result = subprocess.run(
                    [self._python_bin, "scripts/enroll.py", "--turma", turma_id, "--force"],
                    cwd=self._project_root,
                    capture_output=True,
                    text=True,
                    timeout=_ENROLL_TIMEOUT_SEC,
                )
                if result.returncode != 0:
                    failed.append(turma_id)
                    logger.warning(
                        "Enroll falhou para turma '%s' (código %d): %s",
                        turma_id,
                        result.returncode,
                        result.stderr.strip()[-2000:],
                    )
                else:
                    logger.info("Enroll concluído para turma '%s'", turma_id)
            except subprocess.TimeoutExpired:
                failed.append(turma_id)
                logger.warning("Enroll da turma '%s' excedeu %ds — abortado", turma_id, _ENROLL_TIMEOUT_SEC)
            except Exception as exc:  # pragma: no cover - subprocess/OS path
                failed.append(turma_id)
                logger.warning("Enroll da turma '%s' falhou: %s", turma_id, exc)

        with self._lock:
            if failed:
                self._status = "error"
                self._message = f"falhou: {', '.join(failed)}"
            else:
                self._status = "done"
                self._message = f"{len(turma_ids)} turma(s) processada(s)"
