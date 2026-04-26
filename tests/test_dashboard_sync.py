from __future__ import annotations

import json
import httpx
from pathlib import Path

from src.core.config import AppConfig

from src.core.config import DashboardConfig
from src.core.dashboard_sync import DashboardHeartbeatWorker
from src.core.session import SessionState


class FakeSessionManager:
    def __init__(self):
        self.state = SessionState.IDLE
        self.applied_payloads: list[dict] = []
        self.updated_payloads: list[dict] = []
        self.stop_reasons: list[str] = []
        self.unblock_calls = 0
        self._app_cfg = AppConfig(data_dir=Path("/tmp/proctor-dashboard-sync"))
        self.session_payload = None

    def dashboard_snapshot(self):
        return {
            "station_id": "nuc-01",
            "station_name": "NUC Sala 1",
            "status": "IDLE",
            "student": None,
            "active_session_id": None,
            "assessment": "Quiz-03",
            "turma": "T2026-T1",
            "auto_start_enabled": True,
            "seconds_remaining": None,
            "recent_events": [],
        }

    def dashboard_session_payload(self, *, include_completed: bool = True):
        return self.session_payload

    def apply_dashboard_config(self, payload):
        self.applied_payloads.append(payload)

    def update_config(self, **kwargs):
        self.updated_payloads.append(kwargs)

    def stop_session(self, *, reason: str):
        self.stop_reasons.append(reason)

    def unblock_session(self):
        self.unblock_calls += 1


def test_dashboard_worker_applies_config_and_stop_command():
    seen_payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_payloads.append(request.read().decode())
        return httpx.Response(
            200,
            json={
                "station": {"station_id": "nuc-01"},
                "commands": [
                    {
                        "command_type": "APPLY_CONFIG",
                        "payload": {
                            "turma": "T2026-T1",
                            "assessment": "Quiz-03",
                            "timer_minutes": 45,
                            "prairielearn_url": "https://prairielearn.org/pl",
                            "allowlist": ["prairielearn.org"],
                            "auto_start": True,
                            "gaze_h_threshold": 0.4,
                            "gaze_duration_sec": 4.0,
                            "absence_timeout_sec": 6.0,
                            "multi_face_block": True,
                            "s3_prefix": "T2026-T1/quiz-03",
                        },
                    },
                    {
                        "command_type": "STOP_SESSION",
                        "payload": {},
                    },
                ],
            },
        )

    manager = FakeSessionManager()
    worker = DashboardHeartbeatWorker(
        config=DashboardConfig(enabled=True, base_url="http://dashboard.test"),
        session_manager=manager,
        client_factory=lambda: httpx.Client(
            transport=httpx.MockTransport(handler),
            base_url="http://dashboard.test",
        ),
    )

    worker.run_once()

    assert seen_payloads
    assert manager.applied_payloads[0]["assessment"] == "Quiz-03"
    assert manager.stop_reasons == ["dashboard_command"]


def test_dashboard_worker_unblocks_only_when_station_is_blocked():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "station": {"station_id": "nuc-01"},
                "commands": [{"command_type": "UNBLOCK_SESSION", "payload": {}}],
            },
        )

    manager = FakeSessionManager()
    worker = DashboardHeartbeatWorker(
        config=DashboardConfig(enabled=True, base_url="http://dashboard.test"),
        session_manager=manager,
        client_factory=lambda: httpx.Client(
            transport=httpx.MockTransport(handler),
            base_url="http://dashboard.test",
        ),
    )

    worker.run_once()
    assert manager.unblock_calls == 0

    manager.state = SessionState.BLOCKED
    worker.run_once()
    assert manager.unblock_calls == 1


def test_dashboard_worker_updates_autostart_flag():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "station": {"station_id": "nuc-01"},
                "commands": [{"command_type": "SET_AUTOSTART", "payload": {"auto_start": False}}],
            },
        )

    manager = FakeSessionManager()
    worker = DashboardHeartbeatWorker(
        config=DashboardConfig(enabled=True, base_url="http://dashboard.test"),
        session_manager=manager,
        client_factory=lambda: httpx.Client(
            transport=httpx.MockTransport(handler),
            base_url="http://dashboard.test",
        ),
    )

    worker.run_once()

    assert manager.updated_payloads == [{"auto_start": False}]


def test_dashboard_worker_registers_and_finalizes_completed_session():
    requests: list[tuple[str, dict | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = None
        if request.content:
            payload = json.loads(request.content.decode())
        requests.append((f"{request.method} {request.url.path}", payload))
        if request.url.path == "/api/heartbeats":
            return httpx.Response(200, json={"station": {"station_id": "nuc-01"}, "commands": []})
        return httpx.Response(200, json={})

    manager = FakeSessionManager()
    manager.session_payload = {
        "session_id": "sess-1",
        "station_id": "nuc-01",
        "turma": "T2026-T1",
        "assessment": "Quiz-03",
        "started_at": "2026-04-22T20:00:00+00:00",
        "ended_at": "2026-04-22T20:30:00+00:00",
        "timer_minutes": 45,
        "student": {"student_id": "123", "student_name": "Alice"},
        "status": "UPLOADING",
        "flags_count": 1,
        "events": [
            {
                "timestamp": "2026-04-22T20:10:00+00:00",
                "frame_number": 2400,
                "event_type": "GAZE_LEFT",
                "severity": "WARNING",
                "details": {"ratio": 0.52},
            }
        ],
        "recordings": [
            {
                "label": "Webcam 000",
                "s3_bucket": "proctor-station",
                "s3_key": "gravacoes/sess-1/webcam_000.mp4",
                "kind": "video",
            }
        ],
    }

    worker = DashboardHeartbeatWorker(
        config=DashboardConfig(enabled=True, base_url="http://dashboard.test"),
        session_manager=manager,
        client_factory=lambda: httpx.Client(
            transport=httpx.MockTransport(handler),
            base_url="http://dashboard.test",
        ),
    )

    worker.run_once()

    paths = [item[0] for item in requests]
    assert "POST /api/sessions" in paths
    assert "POST /api/sessions/sess-1/events" in paths
    assert "POST /api/sessions/sess-1/finalize" in paths
