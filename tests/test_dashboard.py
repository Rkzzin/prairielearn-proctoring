from __future__ import annotations

from datetime import datetime, timezone

import pytest
from httpx import ASGITransport, AsyncClient

from src.core.config import AppConfig
from src.dashboard.app import create_app
from src.dashboard.models import (
    EventSeverity,
    RecordingAsset,
    SessionEventPayload,
    SessionRecord,
    StationStatus,
    StudentInfo,
)


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
