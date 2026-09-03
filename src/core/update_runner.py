"""Atualiza o checkout da estação e reinicia sob comando do dashboard."""

from __future__ import annotations

import logging
import subprocess
import threading
from pathlib import Path
from typing import Any

from src.core.session import SessionManager, SessionState

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class UpdateRunner:
    def __init__(self, *, session_manager: SessionManager, project_root: Path | None = None):
        self._session_manager = session_manager
        self._project_root = project_root or _PROJECT_ROOT
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._status = "idle"
        self._message = ""

    def status_dict(self) -> dict[str, Any]:
        with self._lock:
            return {"update_status": self._status, "update_message": self._message}

    def start(self) -> None:
        """Atualiza apenas fora de prova e reinicia após um pull bem-sucedido."""
        with self._lock:
            if self._status == "running":
                logger.warning("UPDATE_AND_REBOOT ignorado: atualização já em andamento")
                return
            if self._session_manager.state != SessionState.IDLE:
                self._status = "recusado"
                self._message = f"prova ativa nesta estação ({self._session_manager.state.value})"
                logger.warning("UPDATE_AND_REBOOT recusado: estação em %s", self._session_manager.state.value)
                return
            self._status = "running"
            self._message = "atualizando repositório"
            self._thread = threading.Thread(target=self._run, name="update-runner", daemon=True)
            self._thread.start()

    def _run(self) -> None:
        try:
            result = subprocess.run(
                ["git", "pull", "--ff-only", "origin", "main"],
                cwd=self._project_root,
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:  # pragma: no cover - subprocess/OS path
            logger.warning("Falha ao atualizar repositório: %s", exc)
            self._set_error(str(exc))
            return

        if result.returncode != 0:
            message = (result.stderr or result.stdout).strip()[-500:]
            logger.warning("git pull falhou: %s", message)
            self._set_error(message or f"git pull terminou com código {result.returncode}")
            return

        with self._lock:
            self._status = "reiniciando"
            self._message = "repositório atualizado; reiniciando estação"
        subprocess.Popen(["sudo", "/usr/bin/systemctl", "reboot"])

    def _set_error(self, message: str) -> None:
        with self._lock:
            self._status = "error"
            self._message = message
