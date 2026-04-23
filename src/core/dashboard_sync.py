"""Heartbeat da NUC para o dashboard do professor."""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any, Callable

import httpx

from src.core.config import DashboardConfig
from src.core.session import SessionManager, SessionState
from src.proctor.events import ProctorEvent

logger = logging.getLogger(__name__)


class DashboardHeartbeatWorker:
    def __init__(
        self,
        *,
        config: DashboardConfig,
        session_manager: SessionManager,
        client_factory: Callable[[], httpx.Client] | None = None,
    ):
        self._config = config
        self._session_manager = session_manager
        self._client_factory = client_factory or self._default_client_factory
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
            self._session_manager.apply_dashboard_config(payload)
            logger.info("Configuração aplicada a partir do dashboard")
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

        logger.warning("Comando desconhecido do dashboard: %s", command_type)

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
        session = self._session_manager.dashboard_session_payload(include_completed=False)
        if session is None:
            return payload

        session_id = session["session_id"]
        recent_events = self._read_recent_events(session_id)
        payload["recent_events"] = recent_events
        payload["last_event"] = recent_events[-1] if recent_events else None
        return payload

    def _read_recent_events(self, session_id: str) -> list[dict[str, Any]]:
        log_path = self._session_manager._app_cfg.data_dir / "sessions" / session_id / "events.jsonl"
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
                event = ProctorEvent.from_json(line)
                cached.append(
                    {
                        "timestamp": datetime.fromtimestamp(event.timestamp, tz=timezone.utc).isoformat(),
                        "frame_number": event.frame,
                        "event_type": event.type,
                        "severity": event.severity,
                        "details": event.details,
                    }
                )
            self._event_offsets[session_id] = handle.tell()
        self._recent_events[session_id] = cached[-10:]
        return self._recent_events[session_id]

    def _default_client_factory(self) -> httpx.Client:
        return httpx.Client(
            base_url=self._config.base_url,
            timeout=self._config.timeout_sec,
        )
