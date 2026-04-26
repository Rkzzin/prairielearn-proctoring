from __future__ import annotations

import logging
import threading

from src.core.session import SessionError, SessionManager, SessionState

logger = logging.getLogger(__name__)


class SessionAutoStartWorker:
    def __init__(
        self,
        *,
        session_manager: SessionManager,
        interval_sec: float = 2.0,
        enabled: bool = True,
        log_failures: bool = False,
    ):
        self._session_manager = session_manager
        self._interval_sec = interval_sec
        self._enabled = enabled
        self._log_failures = log_failures
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not self._enabled:
            logger.info("Auto-start local desabilitado")
            return
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="session-autostart",
            daemon=True,
        )
        self._thread.start()
        logger.info("Auto-start local iniciado")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)

    def run_once(self) -> None:
        cfg = self._session_manager.next_config
        if not cfg.auto_start or not cfg.turma_id:
            return
        if self._session_manager.state != SessionState.IDLE:
            return

        try:
            self._session_manager.start_session()
            logger.info("Sessão iniciada automaticamente")
        except SessionError as exc:
            if self._log_failures:
                logger.info("Auto-start aguardando aluno elegível: %s", exc)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.run_once()
            except Exception as exc:  # pragma: no cover
                logger.warning("Falha no worker de auto-start: %s", exc)
            if self._stop_event.wait(self._interval_sec):
                break
