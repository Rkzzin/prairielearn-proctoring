"""Heartbeat da NUC para o dashboard do professor."""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable

import httpx

from src.core.config import DashboardConfig
from src.core.dashboard_payload import event_to_payload, session_events_path
from src.core.enroll_runner import EnrollRunner
from src.core.session import SessionError, SessionManager, SessionState, StationMode
from src.core.update_runner import UpdateRunner
from src.proctor.events import ProctorEvent

logger = logging.getLogger(__name__)

#: Quantos eventos recentes acompanham cada heartbeat.
_RECENT_EVENTS_KEPT = 10


class DashboardHeartbeatWorker:
    def __init__(
        self,
        *,
        config: DashboardConfig,
        session_manager: SessionManager,
        client_factory: Callable[[], httpx.Client] | None = None,
        enroll_runner: EnrollRunner | None = None,
        update_runner: UpdateRunner | None = None,
    ):
        self._config = config
        self._session_manager = session_manager
        self._client_factory = client_factory or self._default_client_factory
        self._enroll_runner = enroll_runner or EnrollRunner(session_manager=session_manager)
        self._update_runner = update_runner or UpdateRunner(session_manager=session_manager)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._registered_sessions: set[str] = set()
        self._finalized_sessions: set[str] = set()
        self._event_offsets: dict[str, int] = {}
        self._recent_events: dict[str, list[dict[str, Any]]] = {}

    def start(self) -> None:
        if not self._config.enabled:
            logger.info("Dashboard heartbeat desabilitado")
            return
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="dashboard-heartbeat",
            daemon=True,
        )
        self._thread.start()
        logger.info("Dashboard heartbeat iniciado: %s", self._config.base_url)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)

    def run_once(self) -> None:
        with self._client_factory() as client:
            self._sync_session(client)
            payload = self._build_heartbeat_payload()
            response = client.post("/api/heartbeats", json=payload)
            response.raise_for_status()
            body = response.json()
        for command in body.get("commands", []):
            self._apply_command(command)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.run_once()
            except Exception as exc:  # pragma: no cover - network path
                logger.warning("Falha ao sincronizar heartbeat com dashboard: %s", exc)
            if self._stop_event.wait(self._config.heartbeat_interval_sec):
                break

    def _apply_command(self, command: dict[str, Any]) -> None:
        command_type = command.get("command_type")
        payload = command.get("payload") or {}

        if command_type == "APPLY_CONFIG":
            config = self._session_manager.apply_dashboard_config(payload)
            if config.auto_start:
                self._enter_exam_mode_if_idle()
            logger.info("Configuração aplicada a partir do dashboard")
            return

        if command_type == "SET_AUTOSTART":
            enabled = bool(payload.get("auto_start"))
            self._session_manager.update_config(auto_start=enabled)
            if enabled:
                self._enter_exam_mode_if_idle()
            else:
                self._session_manager.exit_exam_mode()
            logger.info("SET_AUTOSTART processado como modo prova: %s", enabled)
            return

        if command_type == "STOP_SESSION":
            self._session_manager.stop_session(reason="dashboard_command")
            logger.info("STOP_SESSION processado")
            return

        if command_type == "UNBLOCK_SESSION":
            if self._session_manager.state == SessionState.BLOCKED:
                self._session_manager.unblock_session()
                logger.info("UNBLOCK_SESSION processado")
            return

        if command_type == "RUN_ENROLL":
            turma_ids = [str(t) for t in payload.get("turma_ids") or []]
            self._enroll_runner.start(turma_ids)
            logger.info("RUN_ENROLL recebido para %d turma(s)", len(turma_ids))
            return

        if command_type == "UPDATE_AND_REBOOT":
            self._update_runner.start()
            logger.info("UPDATE_AND_REBOOT processado")
            return

        logger.warning("Comando desconhecido do dashboard: %s", command_type)

    def _enter_exam_mode_if_idle(self) -> None:
        if self._session_manager.state != SessionState.IDLE:
            return
        if self._session_manager.mode == StationMode.WAITING_STUDENT:
            return
        try:
            self._session_manager.enter_exam_mode()
        except SessionError as exc:
            logger.warning("Falha ao entrar no modo prova via auto-start: %s", exc)

    def _sync_session(self, client: httpx.Client) -> None:
        current = self._session_manager.dashboard_session_payload()
        if current is None:
            return

        session_id = current["session_id"]
        if current["ended_at"] is None:
            if session_id not in self._registered_sessions:
                client.post("/api/sessions", json=current).raise_for_status()
                self._registered_sessions.add(session_id)
            return

        if session_id in self._finalized_sessions:
            return

        client.post("/api/sessions", json=current).raise_for_status()
        if current["events"]:
            client.post(f"/api/sessions/{session_id}/events", json=current["events"]).raise_for_status()
        client.post(f"/api/sessions/{session_id}/finalize").raise_for_status()
        self._finalized_sessions.add(session_id)

    def _build_heartbeat_payload(self) -> dict[str, Any]:
        payload = self._session_manager.dashboard_snapshot()
        payload.update(self._enroll_runner.status_dict())
        payload.update(self._update_runner.status_dict())
        session = self._session_manager.dashboard_session_payload(include_completed=False)
        if session is None:
            return payload

        session_id = session["session_id"]
        recent_events = self._read_recent_events(session_id)
        payload["recent_events"] = recent_events
        payload["last_event"] = recent_events[-1] if recent_events else None
        return payload

    def _read_recent_events(self, session_id: str) -> list[dict[str, Any]]:
        """Lê incrementalmente o JSONL da sessão, guardando o offset de arquivo.

        Formato do evento vem de ``event_to_payload`` — mesmo formato usado na
        coleta final da sessão, para os dois não divergirem.
        """
        log_path = session_events_path(self._session_manager.data_dir, session_id)
        if not log_path.exists():
            return self._recent_events.get(session_id, [])

        offset = self._event_offsets.get(session_id, 0)
        cached = self._recent_events.get(session_id, [])
        with open(log_path, encoding="utf-8") as handle:
            handle.seek(offset)
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                cached.append(event_to_payload(ProctorEvent.from_json(line)))
            self._event_offsets[session_id] = handle.tell()
        self._recent_events[session_id] = cached[-_RECENT_EVENTS_KEPT:]
        return self._recent_events[session_id]

    def _default_client_factory(self) -> httpx.Client:
        headers = {}
        if self._config.station_token:
            headers["X-Station-Id"] = self._config.station_id
            headers["X-Station-Token"] = self._config.station_token
        return httpx.Client(
            base_url=self._config.base_url,
            timeout=self._config.timeout_sec,
            headers=headers,
        )
