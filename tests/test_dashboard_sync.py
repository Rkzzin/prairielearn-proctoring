from __future__ import annotations

import json
import httpx
from pathlib import Path

from src.core.config import DashboardConfig
from src.core.dashboard_sync import DashboardHeartbeatWorker
from src.core.session import SessionConfig, SessionState, StationMode


class FakeSessionManager:
    def __init__(self):
        self.state = SessionState.IDLE
        self.mode = StationMode.MAINTENANCE
        self.applied_payloads: list[dict] = []
        self.updated_payloads: list[dict] = []
        self.stop_reasons: list[str] = []
        self.unblock_calls = 0
        self.enter_mode_calls = 0
        self.exit_mode_calls = 0
        # Interface pública que o worker consome (antes ele alcançava _app_cfg).
        self.data_dir = Path("/tmp/proctor-dashboard-sync")
        self.session_payload = None

    def dashboard_snapshot(self):
        return {
            "station_id": "nuc-01",
            "station_name": "NUC Sala 1",
            "status": "IDLE",
            "mode": "MAINTENANCE",
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
        return SessionConfig(auto_start=bool(payload.get("auto_start")))

    def update_config(self, **kwargs):
        self.updated_payloads.append(kwargs)

    def stop_session(self, *, reason: str):
        self.stop_reasons.append(reason)

    def unblock_session(self):
        self.unblock_calls += 1

    def enter_exam_mode(self):
        self.enter_mode_calls += 1
        self.mode = StationMode.WAITING_STUDENT

    def exit_exam_mode(self):
        self.exit_mode_calls += 1
        self.mode = StationMode.MAINTENANCE


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
    assert manager.enter_mode_calls == 1
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


def test_dashboard_worker_updates_autostart_flag_and_exits_exam_mode():
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
    assert manager.exit_mode_calls == 1


def test_dashboard_worker_enabling_autostart_enters_exam_mode():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "station": {"station_id": "nuc-01"},
                "commands": [{"command_type": "SET_AUTOSTART", "payload": {"auto_start": True}}],
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

    assert manager.updated_payloads == [{"auto_start": True}]
    assert manager.enter_mode_calls == 1


class FakeEnrollRunner:
    def __init__(self):
        self.started_with: list[list[str]] = []
        self._status = "idle"
        self._message = ""

    def start(self, turma_ids):
        self.started_with.append(list(turma_ids))
        self._status = "running"

    def status_dict(self):
        return {"enroll_status": self._status, "enroll_message": self._message}


class FakeUpdateRunner:
    def __init__(self):
        self.started = 0

    def start(self):
        self.started += 1

    def status_dict(self):
        return {"update_status": "idle", "update_message": ""}


def test_dashboard_worker_update_and_reboot_command_starts_runner():
    manager = FakeSessionManager()
    update_runner = FakeUpdateRunner()
    worker = DashboardHeartbeatWorker(
        config=DashboardConfig(enabled=True, base_url="http://dashboard.test"),
        session_manager=manager,
        update_runner=update_runner,
    )

    worker._apply_command({"command_type": "UPDATE_AND_REBOOT", "payload": {}})

    assert update_runner.started == 1


def test_dashboard_worker_run_enroll_command_starts_the_runner():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "station": {"station_id": "nuc-01"},
                "commands": [
                    {
                        "command_type": "RUN_ENROLL",
                        "payload": {"turma_ids": ["ES2026-T1", "ES2026-T2"]},
                    }
                ],
            },
        )

    manager = FakeSessionManager()
    enroll_runner = FakeEnrollRunner()
    worker = DashboardHeartbeatWorker(
        config=DashboardConfig(enabled=True, base_url="http://dashboard.test"),
        session_manager=manager,
        enroll_runner=enroll_runner,
        client_factory=lambda: httpx.Client(
            transport=httpx.MockTransport(handler),
            base_url="http://dashboard.test",
        ),
    )

    worker.run_once()

    assert enroll_runner.started_with == [["ES2026-T1", "ES2026-T2"]]


def test_heartbeat_payload_includes_enroll_status():
    seen_payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_payloads.append(json.loads(request.read().decode()))
        return httpx.Response(200, json={"station": {"station_id": "nuc-01"}, "commands": []})

    manager = FakeSessionManager()
    enroll_runner = FakeEnrollRunner()
    enroll_runner._status = "running"
    enroll_runner._message = "1/2 — processando 'ES2026-T2'"
    worker = DashboardHeartbeatWorker(
        config=DashboardConfig(enabled=True, base_url="http://dashboard.test"),
        session_manager=manager,
        enroll_runner=enroll_runner,
        client_factory=lambda: httpx.Client(
            transport=httpx.MockTransport(handler),
            base_url="http://dashboard.test",
        ),
    )

    worker.run_once()

    assert seen_payloads[0]["enroll_status"] == "running"
    assert seen_payloads[0]["enroll_message"] == "1/2 — processando 'ES2026-T2'"


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


def test_default_client_factory_sends_station_headers_when_token_configured():
    config = DashboardConfig(
        enabled=True,
        base_url="http://dashboard.test",
        station_id="nuc-07",
        station_token="s3cr3t",
    )
    worker = DashboardHeartbeatWorker(config=config, session_manager=FakeSessionManager())

    client = worker._default_client_factory()

    assert client.headers["x-station-id"] == "nuc-07"
    assert client.headers["x-station-token"] == "s3cr3t"


def test_default_client_factory_omits_station_headers_without_token():
    config = DashboardConfig(enabled=True, base_url="http://dashboard.test", station_id="nuc-07")
    worker = DashboardHeartbeatWorker(config=config, session_manager=FakeSessionManager())

    client = worker._default_client_factory()

    assert "x-station-id" not in client.headers
    assert "x-station-token" not in client.headers
