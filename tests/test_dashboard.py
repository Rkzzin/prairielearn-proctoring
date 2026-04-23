from __future__ import annotations

from datetime import datetime, timezone

import pytest
from httpx import ASGITransport, AsyncClient

from src.core.config import AppConfig
from src.dashboard.app import create_app
from src.dashboard.models import (
    ExamConfigPayload,
    EventSeverity,
    RecordingAsset,
    SessionEventPayload,
    SessionRecord,
    StationStatus,
    StudentInfo,
)
from src.dashboard.store import DashboardStore


def _make_app(tmp_path):
    config = AppConfig(data_dir=tmp_path)
    return create_app(config=config)


@pytest.mark.asyncio
async def test_dashboard_home_renders(tmp_path):
    async with AsyncClient(transport=ASGITransport(app=_make_app(tmp_path)), base_url="http://testserver") as client:
        response = await client.get("/")

    assert response.status_code == 200
    assert "Estações em tempo real" in response.text


@pytest.mark.asyncio
async def test_heartbeat_returns_pending_config_command(tmp_path):
    async with AsyncClient(transport=ASGITransport(app=_make_app(tmp_path)), base_url="http://testserver") as client:
        config_payload = {
            "turma": "ES2025-T1",
            "assessment": "Quiz-03",
            "timer_minutes": 45,
            "prairielearn_url": "https://pl.exemplo.edu.br/course/123/assessment/456",
            "allowlist": ["pl.exemplo.edu.br"],
            "target_station_ids": ["nuc-01"],
            "gaze_h_threshold": 0.35,
            "gaze_duration_sec": 3.0,
            "absence_timeout_sec": 5.0,
            "multi_face_block": True,
            "s3_prefix": "ES2025-T1/2026-04-16/Quiz-03",
        }
        config_response = await client.post("/api/configs", json=config_payload)
        assert config_response.status_code == 201

        heartbeat = {
            "station_id": "nuc-01",
            "station_name": "NUC Sala 1",
            "status": "IDLE",
            "student": None,
            "active_session_id": None,
            "assessment": None,
            "turma": None,
            "seconds_remaining": None,
            "recent_events": [],
        }
        response = await client.post("/api/heartbeats", json=heartbeat)

    assert response.status_code == 200
    payload = response.json()
    assert payload["station"]["station_id"] == "nuc-01"
    assert payload["commands"][0]["command_type"] == "APPLY_CONFIG"
    assert payload["commands"][0]["payload"]["assessment"] == "Quiz-03"


@pytest.mark.asyncio
async def test_register_session_and_append_events(tmp_path):
    async with AsyncClient(transport=ASGITransport(app=_make_app(tmp_path)), base_url="http://testserver") as client:
        session_payload = SessionRecord(
            session_id="sess-1",
            station_id="nuc-01",
            turma="ES2025-T1",
            assessment="Quiz-03",
            started_at=datetime(2026, 4, 16, 18, 0, tzinfo=timezone.utc),
            student=StudentInfo(student_id="123", student_name="Alice"),
            status=StationStatus.SESSION,
            recordings=[
                RecordingAsset(
                    label="Webcam",
                    url="https://example.com/webcam.m3u8",
                )
            ],
        ).model_dump(mode="json")

        create_response = await client.post("/api/sessions", json=session_payload)
        assert create_response.status_code == 201

        event_payload = [
            SessionEventPayload(
                timestamp=datetime(2026, 4, 16, 18, 10, tzinfo=timezone.utc),
                frame_number=2400,
                event_type="GAZE_LEFT",
                severity=EventSeverity.WARNING,
                details={"ratio": 0.52},
            ).model_dump(mode="json")
        ]
        event_response = await client.post("/api/sessions/sess-1/events", json=event_payload)
        assert event_response.status_code == 200
        assert event_response.json()["flags_count"] == 1

        review_response = await client.get("/sessions/sess-1")
        assert review_response.status_code == 200
        assert "Timeline de eventos" in review_response.text
        assert "Webcam" in review_response.text
        assert "Ir para 600s" in review_response.text

        csv_response = await client.get("/api/reports/events.csv?turma=ES2025-T1")
        assert csv_response.status_code == 200
        assert "text/csv" in csv_response.headers["content-type"]
        csv_text = csv_response.text
        assert "session_id,station_id,turma,assessment,student_id,student_name" in csv_text
        assert "sess-1,nuc-01,ES2025-T1,Quiz-03,123,Alice" in csv_text
        assert "GAZE_LEFT" in csv_text


@pytest.mark.asyncio
async def test_enrollment_form_updates_partial(tmp_path):
    async with AsyncClient(transport=ASGITransport(app=_make_app(tmp_path)), base_url="http://testserver") as client:
        response = await client.post(
            "/api/enrollment",
            data={
                "turma": "ES2025-T1",
                "student_id": "12345",
                "student_name": "Alice Silva",
                "source": "upload",
            },
            files={"files": ("alice.jpg", b"fake-image", "image/jpeg")},
        )

    assert response.status_code == 200
    assert "Alice Silva" in response.text
    assert "alice.jpg" in response.text


@pytest.mark.asyncio
async def test_station_command_endpoints_enqueue_commands(tmp_path):
    async with AsyncClient(transport=ASGITransport(app=_make_app(tmp_path)), base_url="http://testserver") as client:
        stop_response = await client.post("/api/stations/nuc-01/session/stop")
        unblock_response = await client.post("/api/stations/nuc-01/session/unblock")
        heartbeat_response = await client.post(
            "/api/heartbeats",
            json={
                "station_id": "nuc-01",
                "station_name": "NUC Sala 1",
                "status": "BLOCKED",
                "student": None,
                "active_session_id": None,
                "assessment": None,
                "turma": None,
                "seconds_remaining": None,
                "recent_events": [],
            },
        )

    assert stop_response.status_code == 202
    assert unblock_response.status_code == 202
    commands = heartbeat_response.json()["commands"]
    assert [item["command_type"] for item in commands] == ["STOP_SESSION", "UNBLOCK_SESSION"]


def test_dashboard_store_persists_across_restarts(tmp_path):
    store = DashboardStore(tmp_path / "dashboard")
    store.create_config(
        ExamConfigPayload(
            turma="ES2025-T1",
            assessment="Quiz-03",
            timer_minutes=45,
            prairielearn_url="https://pl.exemplo.edu.br/quiz-03",
            allowlist=["pl.exemplo.edu.br"],
            target_station_ids=["nuc-01"],
            s3_prefix="ES2025-T1/quiz-03",
        )
    )
    store.add_enrollment(
        turma="ES2025-T1",
        student_id="123",
        student_name="Alice Silva",
        source="upload",
        file_names=["alice.jpg"],
    )
    store.register_session(
        SessionRecord(
            session_id="sess-1",
            station_id="nuc-01",
            turma="ES2025-T1",
            assessment="Quiz-03",
            started_at=datetime(2026, 4, 16, 18, 0, tzinfo=timezone.utc),
            student=StudentInfo(student_id="123", student_name="Alice Silva"),
            status=StationStatus.SESSION,
        )
    )

    reloaded = DashboardStore(tmp_path / "dashboard")
    snapshot = reloaded.snapshot()

    assert snapshot["configs"][0].assessment == "Quiz-03"
    assert snapshot["enrollments"][0].student_name == "Alice Silva"
    assert snapshot["sessions"][0].session_id == "sess-1"


def test_dashboard_store_generates_presigned_url_for_s3_assets(tmp_path):
    class FakeS3:
        def generate_presigned_url(self, _operation, Params, ExpiresIn):
            return f"https://signed.example/{Params['Bucket']}/{Params['Key']}?exp={ExpiresIn}"

    store = DashboardStore(tmp_path / "dashboard", app_config=AppConfig(data_dir=tmp_path), s3_client=FakeS3())
    store.register_session(
        SessionRecord(
            session_id="sess-s3",
            station_id="nuc-01",
            turma="ES2025-T1",
            assessment="Quiz-03",
            started_at=datetime(2026, 4, 16, 18, 0, tzinfo=timezone.utc),
            student=StudentInfo(student_id="123", student_name="Alice Silva"),
            status=StationStatus.SESSION,
            recordings=[
                RecordingAsset(
                    label="Webcam",
                    s3_bucket="proctor-station",
                    s3_key="gravacoes/sess-s3/webcam_000.mp4",
                )
            ],
        )
    )

    session = store.get_session("sess-s3")
    assert session is not None
    assert session.recordings[0].url == "https://signed.example/proctor-station/gravacoes/sess-s3/webcam_000.mp4?exp=3600"
