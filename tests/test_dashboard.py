from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient
from jinja2 import Environment, FileSystemLoader

from src.core.config import AppConfig, DashboardConfig
from src.core.states import known_station_statuses
from src.dashboard.app import (
    _build_timeline,
    _event_counts,
    _format_duration,
    _format_session_datetime,
    _format_relative_time,
    _parse_roster_csv,
    _station_id_from_name,
    create_app,
)
from src.dashboard.auth import hash_password
from src.dashboard.enrollment_service import S3EnrollmentStudent, S3EnrollmentSummary
from src.dashboard.integrity_score import compute_integrity_score
from src.dashboard.models import (
    CommandType,
    EventSeverity,
    ExamConfigPayload,
    NotificationSettings,
    RecordingAsset,
    SessionEventPayload,
    SessionRecord,
    SessionReviewStatus,
    StationHeartbeat,
    StationStatus,
    StudentInfo,
)
from src.dashboard.store import DashboardStore


def _make_app(
    tmp_path,
    database_url,
    *,
    admin_auth: bool = False,
    event_snapshot_processor=None,
    session_report_mailer=None,
):
    dashboard = DashboardConfig(
        database_url=database_url,
        admin_user="prof" if admin_auth else None,
        admin_password="secret" if admin_auth else None,
    )
    config = AppConfig(data_dir=tmp_path, dashboard=dashboard)
    return create_app(
        config=config,
        event_snapshot_processor=event_snapshot_processor,
        session_report_mailer=session_report_mailer,
    )


def test_dashboard_exam_config_defaults_are_tolerant_but_keep_liveness_blocking():
    config = ExamConfigPayload(
        turma="T2026-T2",
        assessment="Quiz-01",
        prairielearn_url="https://us.prairietest.com",
    )

    assert config.gaze_h_threshold == 0.60
    assert config.gaze_v_threshold == 0.60
    assert config.gaze_duration_sec == 10.0
    assert config.absence_timeout_sec == 30.0
    assert config.multi_face_block is True
    assert config.flexible_mode is True
    assert config.liveness_enabled is True
    assert config.liveness_average_threshold == 0.80
    assert config.liveness_shadow_mode is False


def test_session_datetime_uses_sao_paulo_time():
    value = datetime(2026, 9, 18, 15, 30, tzinfo=timezone.utc)

    assert _format_session_datetime(value) == "18/09/2026 12:30"


@pytest.mark.asyncio
async def test_notification_settings_persist_email_list_and_unlimited_images(
    tmp_path, dashboard_database_url
):
    app = _make_app(tmp_path, dashboard_database_url, admin_auth=True)
    payload = {
        "enabled": True,
        "sender_email": "Proctor@Example.edu ",
        "recipient_emails": ["Teacher@Example.edu", "teacher@example.edu"],
        "image_link_limit": 0,
        "ses_region": "sa-east-1",
        "public_dashboard_url": "https://dashboard.example.edu/",
    }

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        response = await client.post(
            "/api/notification-settings",
            json=payload,
            auth=("prof", "secret"),
        )

    assert response.status_code == 200
    assert response.json()["sender_email"] == "proctor@example.edu"
    assert response.json()["recipient_emails"] == ["teacher@example.edu"]
    assert response.json()["image_link_limit"] == 0
    assert response.json()["public_dashboard_url"] == "https://dashboard.example.edu"
    assert app.state.store.get_notification_settings().enabled is True


@pytest.mark.asyncio
async def test_notification_settings_require_admin_auth_and_valid_https_url(
    tmp_path, dashboard_database_url
):
    payload = {
        "enabled": True,
        "sender_email": "proctor@example.edu",
        "recipient_emails": ["teacher@example.edu"],
        "image_link_limit": 12,
        "ses_region": "sa-east-1",
        "public_dashboard_url": "https://dashboard.example.edu",
    }
    app_without_auth = _make_app(tmp_path, dashboard_database_url)
    async with AsyncClient(
        transport=ASGITransport(app=app_without_auth), base_url="http://testserver"
    ) as client:
        no_auth_config = await client.post("/api/notification-settings", json=payload)

    app_with_auth = _make_app(tmp_path, dashboard_database_url, admin_auth=True)
    async with AsyncClient(
        transport=ASGITransport(app=app_with_auth), base_url="http://testserver"
    ) as client:
        invalid_url = await client.post(
            "/api/notification-settings",
            json={
                **payload,
                "public_dashboard_url": "https://dashboard.example.edu?token=unsafe",
            },
            auth=("prof", "secret"),
        )
        too_many_recipients = await client.post(
            "/api/notification-settings",
            json={
                **payload,
                "public_dashboard_url": "https://dashboard.example.edu",
                "recipient_emails": [f"teacher-{index}@example.edu" for index in range(51)],
            },
            auth=("prof", "secret"),
        )

    assert no_auth_config.status_code == 422
    assert invalid_url.status_code == 422
    assert too_many_recipients.status_code == 422


def test_waiting_email_report_persists_and_can_be_activated(dashboard_database_url):
    store = DashboardStore(dashboard_database_url)
    now = datetime.now(timezone.utc)
    store.register_session(
        SessionRecord(
            session_id="session-durable-email",
            station_id="nuc-01",
            turma="T1",
            assessment="Quiz",
            started_at=now,
            ended_at=now,
            status=StationStatus.COMPLETED,
        )
    )
    settings = NotificationSettings(
        enabled=True,
        sender_email="proctor@example.edu",
        recipient_emails=["teacher@example.edu"],
        ses_region="sa-east-1",
        public_dashboard_url="https://dashboard.example.edu",
    )

    assert store.queue_email_report(
        "session-durable-email", settings, status="waiting_snapshots"
    )
    reloaded = DashboardStore(dashboard_database_url)
    assert reloaded.waiting_email_report_ids() == ["session-durable-email"]
    assert reloaded.activate_waiting_email_report("session-durable-email")
    assert reloaded.claim_email_report("session-durable-email").status == "sending"


@pytest.mark.asyncio
async def test_finalize_persists_email_intent_before_snapshot_processing(
    tmp_path, dashboard_database_url
):
    calls = []

    class FakeMailer:
        def resume_pending(self):
            pass

        def prepare(self, session_id):
            calls.append(("prepare", session_id))
            return True

    class FakeProcessor:
        def resume_pending(self):
            pass

        def enqueue(self, session_id, *, retry_failed=False):
            calls.append(("snapshots", session_id))
            return 0

    app = _make_app(
        tmp_path,
        dashboard_database_url,
        admin_auth=True,
        event_snapshot_processor=FakeProcessor(),
        session_report_mailer=FakeMailer(),
    )
    app.state.store.register_session(
        SessionRecord(
            session_id="session-finalize-email",
            station_id="nuc-01",
            turma="T1",
            assessment="Quiz",
            started_at=datetime.now(timezone.utc),
            status=StationStatus.SESSION,
        )
    )
    headers = _station_headers(app, "nuc-01")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        response = await client.post(
            "/api/sessions/session-finalize-email/finalize", headers=headers
        )

    assert response.status_code == 200
    assert calls == [
        ("prepare", "session-finalize-email"),
        ("snapshots", "session-finalize-email"),
    ]


@pytest.mark.asyncio
async def test_manual_email_report_uses_configured_mailer(
    tmp_path, dashboard_database_url
):
    class FakeMailer:
        def __init__(self):
            self.calls = []

        def resume_pending(self):
            pass

        def enqueue(self, session_id, *, force=False):
            self.calls.append((session_id, force))
            return True

    mailer = FakeMailer()
    app = _make_app(
        tmp_path,
        dashboard_database_url,
        admin_auth=True,
        session_report_mailer=mailer,
    )
    now = datetime.now(timezone.utc)
    app.state.store.register_session(
        SessionRecord(
            session_id="session-email",
            station_id="nuc-01",
            turma="T1",
            assessment="Quiz",
            started_at=now,
            ended_at=now,
            status=StationStatus.COMPLETED,
        )
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        response = await client.post(
            "/api/sessions/session-email/email-report/send",
            auth=("prof", "secret"),
        )

    assert response.status_code == 202
    assert mailer.calls == [("session-email", True)]


@pytest.mark.asyncio
async def test_manual_email_report_waits_for_event_snapshots(
    tmp_path, dashboard_database_url
):
    class FakeMailer:
        def resume_pending(self):
            pass

        def enqueue(self, session_id, *, force=False):
            raise AssertionError("report must not be queued before snapshots")

    class FakeProcessor:
        def __init__(self):
            self.calls = []

        def resume_pending(self):
            pass

        def enqueue(self, session_id, *, retry_failed=False):
            self.calls.append((session_id, retry_failed))
            return 1

    processor = FakeProcessor()
    app = _make_app(
        tmp_path,
        dashboard_database_url,
        admin_auth=True,
        event_snapshot_processor=processor,
        session_report_mailer=FakeMailer(),
    )
    now = datetime.now(timezone.utc)
    app.state.store.register_session(
        SessionRecord(
            session_id="email-waiting-images",
            station_id="nuc-01",
            turma="T1",
            assessment="Quiz",
            started_at=now,
            ended_at=now,
            status=StationStatus.COMPLETED,
            events=[
                SessionEventPayload(
                    timestamp=now,
                    event_type="GAZE_WARNING",
                    severity=EventSeverity.WARNING,
                )
            ],
        )
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        response = await client.post(
            "/api/sessions/email-waiting-images/email-report/send",
            auth=("prof", "secret"),
        )

    assert response.status_code == 409
    assert "imagens ainda estão sendo processadas" in response.json()["detail"]
    assert processor.calls == [("email-waiting-images", True)]


@pytest.mark.asyncio
async def test_finished_session_queues_and_renders_event_snapshot_grid(
    tmp_path, dashboard_database_url
):
    class FakeProcessor:
        def __init__(self):
            self.calls = []

        def resume_pending(self):
            pass

        def enqueue(self, session_id, *, retry_failed=False):
            self.calls.append((session_id, retry_failed))
            return 2

    processor = FakeProcessor()
    app = _make_app(
        tmp_path,
        dashboard_database_url,
        event_snapshot_processor=processor,
    )
    now = datetime.now(timezone.utc)
    session = SessionRecord(
        session_id="session-images",
        station_id="nuc-01",
        turma="T1",
        assessment="Quiz",
        started_at=now,
        ended_at=now,
        status=StationStatus.COMPLETED,
        events=[
            SessionEventPayload(
                timestamp=now,
                event_type="SESSION_STARTED",
                severity=EventSeverity.INFO,
            ),
            SessionEventPayload(
                timestamp=now,
                event_type="GAZE_WARNING",
                severity=EventSeverity.WARNING,
            ),
            SessionEventPayload(
                timestamp=now,
                event_type="MULTI_FACE_ALERT",
                severity=EventSeverity.CRITICAL,
            ),
        ],
    )
    app.state.store.register_session(session)
    assert app.state.store.queue_event_snapshots(session.session_id) == 2

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        detail = await client.get(f"/sessions/{session.session_id}")
        response = await client.post(
            f"/api/sessions/{session.session_id}/event-snapshots/process"
        )

    assert detail.status_code == 200
    assert 'class="event-snapshot-grid"' in detail.text
    assert detail.text.count('class="event-snapshot-card') == 2
    assert 'aria-label="Filtrar fotos por severidade"' in detail.text
    assert "Informações técnicas" not in detail.text
    assert "ID da sessão" in detail.text
    assert "Relatório por e-mail" not in detail.text
    assert "Processar imagens dos alertas" in detail.text
    assert response.status_code == 202
    assert response.json() == {"status": "queued", "queued": 2}
    assert processor.calls == [(session.session_id, True)]


def _station_headers(app, station_id: str, token: str = "test-token") -> dict[str, str]:
    """Emite (sobrescrevendo se já existir) um token pra `station_id` e retorna os headers de auth."""
    app.state.store.set_station_token_hash(station_id, hash_password(token))
    return {"X-Station-Id": station_id, "X-Station-Token": token}


class FakeS3EnrollmentService:
    def __init__(self):
        self.calls = []
        self.photo_calls = []

    def list_turmas(self):
        return ["ES2025-T1", "ES2025-T2"]

    def student_photo_url_candidates(self, turma: str, student_name: str):
        self.photo_calls.append((turma, student_name))
        if student_name == "felipehl":
            return [f"https://s3.example.com/fotos/{turma}/felipehl.jpg?sig=1"]
        return []

    def enroll_turma(self, turma: str, *, force: bool = False):
        self.calls.append((turma, force))
        return S3EnrollmentSummary(
            turma=turma,
            total=2,
            ok=1,
            failed=1,
            pkl_path=Path("data/encodings") / f"{turma}.pkl",
            students=[
                S3EnrollmentStudent(
                    student_id="alice",
                    student_name="alice",
                    s3_key=f"fotos/{turma}/alice.jpg",
                    success=True,
                    samples_captured=3,
                    message="3 samples capturados com sucesso.",
                ),
                S3EnrollmentStudent(
                    student_id="bob",
                    student_name="bob",
                    s3_key=f"fotos/{turma}/bob.jpg",
                    success=False,
                    samples_captured=0,
                    message="verifique se a foto contém exatamente 1 rosto.",
                ),
            ],
        )


@pytest.mark.asyncio
async def test_dashboard_has_no_auth_when_admin_user_unset(tmp_path, dashboard_database_url):
    async with AsyncClient(transport=ASGITransport(app=_make_app(tmp_path, dashboard_database_url)), base_url="http://testserver") as client:
        response = await client.get("/")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_dashboard_requires_basic_auth_when_admin_user_set(tmp_path, dashboard_database_url):
    config = AppConfig(
        data_dir=tmp_path,
        dashboard={
            "admin_user": "prof",
            "admin_password": "senha-forte",
            "database_url": dashboard_database_url,
        },
    )
    app = create_app(config=config)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        unauthenticated = await client.get("/")
        assert unauthenticated.status_code == 401
        assert unauthenticated.headers["www-authenticate"].startswith("Basic")

        wrong_password = await client.get("/", auth=("prof", "senha-errada"))
        assert wrong_password.status_code == 401

        wrong_user = await client.get("/", auth=("outro", "senha-forte"))
        assert wrong_user.status_code == 401

        authenticated = await client.get("/", auth=("prof", "senha-forte"))
        assert authenticated.status_code == 200


@pytest.mark.asyncio
async def test_dashboard_home_renders(tmp_path, dashboard_database_url):
    async with AsyncClient(transport=ASGITransport(app=_make_app(tmp_path, dashboard_database_url)), base_url="http://testserver") as client:
        response = await client.get("/")

    assert response.status_code == 200
    assert "Estações em tempo real" in response.text
    assert "Limpar sessões" in response.text
    assert "Atualizar reconhecimento facial" in response.text
    assert "scripts/enroll.py --force" in response.text
    assert "Nova estação" in response.text
    assert "Gerar estação e token" in response.text
    assert "Configurar estações" in response.text
    assert "Todas" in response.text
    assert "Ativar controle de tempo" in response.text
    assert "https://us.prairietest.com" in response.text
    assert "Câmera principal" in response.text
    assert "Câmera ambiente" in response.text
    assert "Fotografar câmeras" in response.text
    assert "Fotos das câmeras" in response.text


@pytest.mark.asyncio
async def test_camera_check_queues_idle_station_and_skips_active_session(tmp_path, dashboard_database_url):
    app = _make_app(tmp_path, dashboard_database_url, admin_auth=True)
    app.state.store.upsert_station_heartbeat(
        StationHeartbeat(
            station_id="nuc-idle",
            station_name="NUC Livre",
            status=StationStatus.WAITING_STUDENT,
            mode="WAITING_STUDENT",
            available_cameras=[
                {"index": 0, "name": "Integrated Camera", "device": "/dev/video0"}
            ],
        )
    )
    app.state.store.upsert_station_heartbeat(
        StationHeartbeat(
            station_id="nuc-busy",
            station_name="NUC Ocupada",
            status=StationStatus.SESSION,
            active_session_id="session-1",
            available_cameras=[
                {"index": 0, "name": "Integrated Camera", "device": "/dev/video0"}
            ],
        )
    )

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
        auth=("prof", "secret"),
    ) as client:
        response = await client.post("/api/camera-checks")

    assert response.status_code == 202
    payload = response.json()
    assert payload["queued_station_ids"] == ["nuc-idle"]
    assert payload["skipped"] == [{"station_id": "nuc-busy", "reason": "com avaliação ativa"}]
    command = app.state.store.get_station("nuc-idle").pending_commands[0]
    assert command.command_type == CommandType.CAPTURE_CAMERA_SNAPSHOTS
    assert command.payload["calibrate_electronics"] is False


@pytest.mark.asyncio
async def test_electronic_calibration_reuses_camera_snapshot_command(
    tmp_path,
    dashboard_database_url,
):
    app = _make_app(tmp_path, dashboard_database_url, admin_auth=True)
    app.state.store.upsert_station_heartbeat(
        StationHeartbeat(
            station_id="nuc-idle",
            status=StationStatus.IDLE,
            electronic_device_calibration_supported=True,
            available_cameras=[
                {"index": 0, "name": "Integrated Camera", "device": "/dev/video0"}
            ],
        )
    )

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
        auth=("prof", "secret"),
    ) as client:
        response = await client.post("/api/electronic-device-calibration")

    assert response.status_code == 202
    command = app.state.store.get_station("nuc-idle").pending_commands[0]
    assert command.command_type == CommandType.CAPTURE_CAMERA_SNAPSHOTS
    assert command.payload["calibrate_electronics"] is True


@pytest.mark.asyncio
async def test_station_uploads_camera_snapshot_and_gallery_displays_it(tmp_path, dashboard_database_url):
    app = _make_app(tmp_path, dashboard_database_url, admin_auth=True)
    headers = _station_headers(app, "nuc-01")
    app.state.store.upsert_station_heartbeat(
        StationHeartbeat(
            station_id="nuc-01",
            station_name="NUC Sala 1",
            status=StationStatus.IDLE,
            available_cameras=[
                {"index": 2, "name": "C922 Pro Stream Webcam", "device": "/dev/video2"}
            ],
        )
    )

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
        auth=("prof", "secret"),
    ) as client:
        queued = (await client.post("/api/camera-checks")).json()
        batch_id = queued["batch_id"]
        response = await client.post(
            "/api/camera-snapshots",
            headers=headers,
            json={
                "batch_id": batch_id,
                "camera_index": 2,
                "camera_name": "C922 Pro Stream Webcam",
                "device": "/dev/video2",
                "image_base64": base64.b64encode(b"\xff\xd8jpeg").decode("ascii"),
            },
        )
        gallery = await client.get("/partials/camera-gallery")
        image_url = app.state.store.get_station("nuc-01").camera_snapshots[0].image_url
        image = await client.get(image_url)

    assert response.status_code == 201
    assert "C922 Pro Stream Webcam" in gallery.text
    assert "/camera-snapshots/" in gallery.text
    assert image.status_code == 200
    assert image.content == b"\xff\xd8jpeg"


@pytest.mark.asyncio
async def test_camera_capture_fails_closed_without_admin_auth(tmp_path, dashboard_database_url):
    app = _make_app(tmp_path, dashboard_database_url)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        response = await client.post("/api/camera-checks")

    assert response.status_code == 503


@pytest.mark.asyncio
async def test_station_partial_offers_exit_when_station_reports_waiting_student(tmp_path, dashboard_database_url):
    app = _make_app(tmp_path, dashboard_database_url)
    app.state.store.upsert_station_heartbeat(
        StationHeartbeat(
            station_id="nuc-01",
            station_name="NUC Sala 1",
            status=StationStatus.WAITING_STUDENT,
            mode="WAITING_STUDENT",
            auto_start_enabled=False,
        )
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        response = await client.get("/partials/stations")

    assert response.status_code == 200
    assert "Sair do modo prova" in response.text
    assert "/autostart/disable" in response.text
    assert "Atualizar reconhecimento facial" in response.text


@pytest.mark.asyncio
async def test_station_partial_highlights_critical_liveness_event(
    tmp_path, dashboard_database_url
):
    app = _make_app(tmp_path, dashboard_database_url)
    app.state.store.upsert_station_heartbeat(
        StationHeartbeat(
            station_id="nuc-01",
            station_name="NUC Sala 1",
            status=StationStatus.SESSION,
            last_event=SessionEventPayload(
                timestamp=datetime.now(timezone.utc),
                event_type="LIVENESS_FAILED",
                severity=EventSeverity.CRITICAL,
            ),
        )
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        response = await client.get("/partials/stations")

    assert response.status_code == 200
    assert "event-line-critical" in response.text
    assert "Tentativa reprovada na prova de vida" in response.text
    assert "CRÍTICO" in response.text


@pytest.mark.asyncio
async def test_dashboard_queues_update_and_reboot_only_when_station_is_idle(tmp_path, dashboard_database_url):
    app = _make_app(tmp_path, dashboard_database_url)
    app.state.store.upsert_station_heartbeat(
        StationHeartbeat(station_id="nuc-01", station_name="NUC Sala 1", status=StationStatus.IDLE)
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        response = await client.post("/api/stations/nuc-01/update-and-reboot")

    assert response.status_code == 202
    assert response.json()["command_type"] == "UPDATE_AND_REBOOT"


@pytest.mark.asyncio
async def test_dashboard_defers_update_and_reboot_until_station_is_idle(
    tmp_path, dashboard_database_url
):
    app = _make_app(tmp_path, dashboard_database_url)
    app.state.store.upsert_station_heartbeat(
        StationHeartbeat(
            station_id="nuc-01",
            station_name="NUC Sala 1",
            status=StationStatus.SESSION,
            active_session_id="sess-1",
        )
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        response = await client.post("/api/stations/nuc-01/update-and-reboot")

    assert response.status_code == 202
    assert app.state.store.drain_commands("nuc-01") == []

    app.state.store.upsert_station_heartbeat(
        StationHeartbeat(
            station_id="nuc-01",
            station_name="NUC Sala 1",
            status=StationStatus.IDLE,
            active_session_id=None,
        )
    )
    commands = app.state.store.drain_commands("nuc-01")
    assert [command.command_type for command in commands] == [CommandType.UPDATE_AND_REBOOT]


@pytest.mark.asyncio
async def test_heartbeat_returns_pending_config_command(tmp_path, dashboard_database_url):
    app = _make_app(tmp_path, dashboard_database_url)
    station_headers = _station_headers(app, "nuc-01")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        config_payload = {
            "turma": "ES2025-T1",
            "assessment": "Quiz-03",
            "timer_minutes": 45,
            "prairielearn_url": "https://prairielearn.org/pl",
            "allowlist": ["prairielearn.org"],
            "auto_start": True,
            "allow_repeat_attempts": False,
            "target_station_ids": ["nuc-01"],
            "gaze_h_threshold": 0.35,
            "gaze_v_threshold": 0.60,
            "gaze_duration_sec": 3.0,
            "absence_timeout_sec": 5.0,
            "multi_face_block": True,
            "liveness_enabled": True,
            "liveness_average_threshold": 0.82,
            "liveness_shadow_mode": True,
            "electronic_device_primary_enabled": True,
            "electronic_device_secondary_enabled": True,
            "s3_prefix": "ES2025-T1/2026-04-16/Quiz-03",
            "primary_camera_index": 0,
            "secondary_camera_index": 2,
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
            "auto_start_enabled": True,
            "seconds_remaining": None,
            "recent_events": [],
            "available_cameras": [
                {"index": 0, "name": "Integrated Camera", "device": "/dev/video0"},
                {"index": 2, "name": "Logitech BRIO", "device": "/dev/video2"},
            ],
        }
        response = await client.post("/api/heartbeats", json=heartbeat, headers=station_headers)

    assert response.status_code == 200
    payload = response.json()
    assert payload["station"]["station_id"] == "nuc-01"
    assert payload["commands"][0]["command_type"] == "APPLY_CONFIG"
    assert payload["commands"][0]["payload"]["assessment"] == "Quiz-03"
    assert payload["commands"][0]["payload"]["allow_repeat_attempts"] is False
    assert payload["commands"][0]["payload"]["primary_camera_index"] == 0
    assert payload["commands"][0]["payload"]["secondary_camera_index"] == 2
    assert payload["commands"][0]["payload"]["liveness_average_threshold"] == 0.82
    assert payload["commands"][0]["payload"]["liveness_shadow_mode"] is True
    assert payload["commands"][0]["payload"]["electronic_device_primary_enabled"] is True
    assert payload["commands"][0]["payload"]["electronic_device_secondary_enabled"] is True
    assert payload["station"]["available_cameras"][1]["name"] == "Logitech BRIO"


def test_legacy_heartbeat_does_not_erase_reported_cameras(tmp_path, dashboard_database_url):
    app = _make_app(tmp_path, dashboard_database_url)
    app.state.store.upsert_station_heartbeat(
        StationHeartbeat(
            station_id="nuc-01",
            status=StationStatus.IDLE,
            available_cameras=[
                {"index": 0, "name": "Integrated Camera", "device": "/dev/video0"}
            ],
        )
    )

    station = app.state.store.upsert_station_heartbeat(
        StationHeartbeat(station_id="nuc-01", status=StationStatus.IDLE)
    )

    assert station.available_cameras[0].name == "Integrated Camera"


@pytest.mark.asyncio
async def test_register_session_and_append_events(tmp_path, dashboard_database_url):
    app = _make_app(tmp_path, dashboard_database_url)
    station_headers = _station_headers(app, "nuc-01")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
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

        create_response = await client.post("/api/sessions", json=session_payload, headers=station_headers)
        assert create_response.status_code == 201

        event_payload = [
            SessionEventPayload(
                timestamp=datetime(2026, 4, 16, 18, 10, tzinfo=timezone.utc),
                frame_number=2400,
                event_type="GAZE_LEFT",
                severity=EventSeverity.WARNING,
                details={"raw_metric_secret": 0.52},
            ).model_dump(mode="json")
        ]
        event_response = await client.post(
            "/api/sessions/sess-1/events", json=event_payload, headers=station_headers
        )
        assert event_response.status_code == 200
        assert event_response.json()["flags_count"] == 1

        review_response = await client.get("/sessions/sess-1")
        assert review_response.status_code == 200
        assert "Timeline de eventos" in review_response.text
        assert "Webcam" in review_response.text
        assert "Ver vídeos do evento" in review_response.text

        csv_response = await client.get("/api/reports/events.csv?turma=ES2025-T1")
        assert csv_response.status_code == 200
        assert "text/csv" in csv_response.headers["content-type"]
        csv_text = csv_response.text
        assert "session_id,station_id,turma,assessment,student_id,student_name" in csv_text
        assert "sess-1,nuc-01,ES2025-T1,Quiz-03,123,Alice" in csv_text
        assert "GAZE_LEFT" in csv_text


@pytest.mark.asyncio
async def test_heartbeat_requires_valid_station_token(tmp_path, dashboard_database_url):
    app = _make_app(tmp_path, dashboard_database_url)
    _station_headers(app, "nuc-01", token="correct-token")
    heartbeat = {
        "station_id": "nuc-01",
        "station_name": "NUC Sala 1",
        "status": "IDLE",
        "student": None,
        "active_session_id": None,
        "assessment": None,
        "turma": None,
        "auto_start_enabled": True,
        "seconds_remaining": None,
        "recent_events": [],
    }

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        no_token = await client.post("/api/heartbeats", json=heartbeat)
        assert no_token.status_code == 401

        wrong_token = await client.post(
            "/api/heartbeats",
            json=heartbeat,
            headers={"X-Station-Id": "nuc-01", "X-Station-Token": "wrong-token"},
        )
        assert wrong_token.status_code == 401

        ok = await client.post(
            "/api/heartbeats",
            json=heartbeat,
            headers={"X-Station-Id": "nuc-01", "X-Station-Token": "correct-token"},
        )
        assert ok.status_code == 200


def test_flexible_station_heartbeat_queues_emergency_unblock(dashboard_database_url):
    store = DashboardStore(dashboard_database_url)
    store.create_config(
        ExamConfigPayload(
            turma="T1",
            assessment="Quiz",
            prairielearn_url="https://us.prairietest.com",
            target_station_ids=["nuc-01"],
            flexible_mode=True,
        )
    )
    store.drain_commands("nuc-01")

    store.upsert_station_heartbeat(
        StationHeartbeat(
            station_id="nuc-01",
            status=StationStatus.BLOCKED,
            mode="SESSION",
            active_session_id="session-1",
        )
    )

    commands = store.drain_commands("nuc-01")
    assert [command.command_type for command in commands] == [CommandType.UNBLOCK_SESSION]


def test_dashboard_ignores_unconfirmed_different_user_events(dashboard_database_url):
    store = DashboardStore(dashboard_database_url)
    now = datetime.now(timezone.utc)
    store.register_session(
        SessionRecord(
            session_id="session-1",
            station_id="nuc-01",
            turma="T1",
            assessment="Quiz",
            started_at=now,
            status=StationStatus.SESSION,
        )
    )
    ignored = SessionEventPayload(
        timestamp=now,
        event_type="DIFFERENT_USER_ALERT",
        severity=EventSeverity.CRITICAL,
        details={"detected_status": "NO_MATCH", "detected_confidence": 0.51},
    )
    confirmed = SessionEventPayload(
        timestamp=now + timedelta(seconds=20),
        event_type="DIFFERENT_USER_ALERT",
        severity=EventSeverity.CRITICAL,
        details={"detected_status": "MATCH", "detected_student_id": "456"},
    )

    session = store.append_events("session-1", [ignored, confirmed])
    store.upsert_station_heartbeat(
        StationHeartbeat(
            station_id="nuc-01",
            status=StationStatus.SESSION,
            last_event=ignored,
            recent_events=[confirmed, ignored],
        )
    )

    assert [event.details["detected_status"] for event in session.events] == ["MATCH"]
    station = store.get_station("nuc-01")
    assert station.last_event.details["detected_status"] == "MATCH"
    assert [event.details["detected_status"] for event in station.recent_events] == ["MATCH"]


@pytest.mark.asyncio
async def test_create_and_delete_station_manages_its_token(tmp_path, dashboard_database_url):
    app = _make_app(tmp_path, dashboard_database_url)
    heartbeat = {
        "station_id": "nuc-sala-04",
        "station_name": "NUC Sala 04",
        "status": "IDLE",
        "student": None,
        "active_session_id": None,
        "assessment": None,
        "turma": None,
        "auto_start_enabled": True,
        "seconds_remaining": None,
        "recent_events": [],
    }

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        create_response = await client.post(
            "/api/stations",
            json={"station_name": "NUC Sala 04"},
        )

        assert create_response.status_code == 201
        created = create_response.json()
        assert created["station"]["station_id"] == "nuc-sala-04"
        assert created["station"]["station_name"] == "NUC Sala 04"
        assert created["station"]["status"] == "OFFLINE"
        assert len(created["station_token"]) >= 32

        station_headers = {
            "X-Station-Id": "nuc-sala-04",
            "X-Station-Token": created["station_token"],
        }
        heartbeat_response = await client.post(
            "/api/heartbeats",
            json=heartbeat,
            headers=station_headers,
        )
        assert heartbeat_response.status_code == 200

        duplicate_response = await client.post(
            "/api/stations",
            json={"station_name": "NUC Sala 04"},
        )
        assert duplicate_response.status_code == 409

        app.state.store.register_session(
            SessionRecord(
                session_id="sess-history",
                station_id="nuc-sala-04",
                turma="ES2025-T1",
                assessment="Quiz-03",
                started_at=datetime(2026, 4, 16, 18, 0, tzinfo=timezone.utc),
                status=StationStatus.COMPLETED,
            )
        )
        app.state.store.finalize_session("sess-history")
        delete_response = await client.delete("/api/stations/nuc-sala-04")
        assert delete_response.status_code == 200
        assert app.state.store.get_station("nuc-sala-04") is None
        assert app.state.store.get_session("sess-history") is not None

        rejected_heartbeat = await client.post(
            "/api/heartbeats",
            json=heartbeat,
            headers=station_headers,
        )
        assert rejected_heartbeat.status_code == 401


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("NUC Sala 04", "nuc-sala-04"),
        ("Estação São José", "estacao-sao-jose"),
        ("  Prova / Bloco A  ", "prova-bloco-a"),
    ],
)
def test_station_id_from_name(name, expected):
    assert _station_id_from_name(name) == expected


def test_parse_roster_csv_extracts_login_before_at_and_dedupes():
    raw = (
        "UID,Name,UIN,Role,Enrollment,Labels,Quiz-1\n"
        "anacmm2@al.insper.edu.br,Ana Clara Minicheli Martinelli,uin-1,Student,joined,,100\n"
        "arthursvs@al.insper.edu.br,Arthur Soria Vaz da Silva,uin-2,Student,joined,,100\n"
        "arthursvs@al.insper.edu.br,Arthur Duplicado,uin-2,Student,joined,,100\n"
        ",Sem UID,uin-3,Student,joined,,100\n"
        "semnome@al.insper.edu.br,,uin-4,Student,joined,,100\n"
    ).encode("utf-8")

    entries = _parse_roster_csv(raw)

    assert entries == [
        ("anacmm2", "Ana Clara Minicheli Martinelli"),
        ("arthursvs", "Arthur Soria Vaz da Silva"),
    ]


def test_parse_roster_csv_rejects_missing_columns():
    with pytest.raises(ValueError, match="UID.*Name"):
        _parse_roster_csv(b"Login,FullName\nfelipehl,Felipe\n")


def test_parse_roster_csv_rejects_csv_with_no_valid_rows():
    with pytest.raises(ValueError, match="Nenhum aluno"):
        _parse_roster_csv(b"UID,Name\n,\n")


@pytest.mark.asyncio
async def test_heartbeat_rejects_station_id_mismatch_between_header_and_body(tmp_path, dashboard_database_url):
    app = _make_app(tmp_path, dashboard_database_url)
    headers_for_nuc01 = _station_headers(app, "nuc-01")
    heartbeat_claiming_nuc02 = {
        "station_id": "nuc-02",
        "station_name": "NUC Sala 2",
        "status": "IDLE",
        "student": None,
        "active_session_id": None,
        "assessment": None,
        "turma": None,
        "auto_start_enabled": True,
        "seconds_remaining": None,
        "recent_events": [],
    }

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        response = await client.post(
            "/api/heartbeats", json=heartbeat_claiming_nuc02, headers=headers_for_nuc01
        )

    assert response.status_code == 403


@pytest.mark.asyncio
async def test_finalize_session_rejects_token_of_another_station(tmp_path, dashboard_database_url):
    app = _make_app(tmp_path, dashboard_database_url)
    owner_headers = _station_headers(app, "nuc-01")
    intruder_headers = _station_headers(app, "nuc-02")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        session_payload = SessionRecord(
            session_id="sess-owned",
            station_id="nuc-01",
            turma="ES2025-T1",
            assessment="Quiz-03",
            started_at=datetime(2026, 4, 16, 18, 0, tzinfo=timezone.utc),
            status=StationStatus.SESSION,
        ).model_dump(mode="json")
        create_response = await client.post("/api/sessions", json=session_payload, headers=owner_headers)
        assert create_response.status_code == 201

        intruder_response = await client.post("/api/sessions/sess-owned/finalize", headers=intruder_headers)
        assert intruder_response.status_code == 403

        owner_response = await client.post("/api/sessions/sess-owned/finalize", headers=owner_headers)
        assert owner_response.status_code == 200


@pytest.mark.asyncio
async def test_station_token_does_not_authenticate_professor_routes(tmp_path, dashboard_database_url):
    config = AppConfig(
        data_dir=tmp_path,
        dashboard={
            "admin_user": "prof",
            "admin_password": "senha-forte",
            "database_url": dashboard_database_url,
        },
    )
    app = create_app(config=config)
    station_headers = _station_headers(app, "nuc-01")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        home_response = await client.get("/", headers=station_headers)
        clear_response = await client.post("/api/sessions/clear", headers=station_headers)

    assert home_response.status_code == 401
    assert clear_response.status_code == 401


@pytest.mark.asyncio
async def test_legacy_manual_enrollment_endpoint_is_removed(tmp_path, dashboard_database_url):
    async with AsyncClient(transport=ASGITransport(app=_make_app(tmp_path, dashboard_database_url)), base_url="http://testserver") as client:
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

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_enrollment_page_lists_s3_turmas(tmp_path, dashboard_database_url):
    app = _make_app(tmp_path, dashboard_database_url)
    app.state.s3_enrollment_service = FakeS3EnrollmentService()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        response = await client.get("/enrollment")

    assert response.status_code == 200
    assert "Enrollment completo via S3" in response.text
    assert '<option value="ES2025-T1">ES2025-T1</option>' in response.text
    assert '<option value="ES2025-T2">ES2025-T2</option>' in response.text
    assert "Reprocessar" not in response.text
    assert 'name="force"' not in response.text
    assert "Novo enrollment" not in response.text
    assert "Cadastros recentes" not in response.text
    assert 'hx-post="/api/enrollment"' not in response.text


@pytest.mark.asyncio
async def test_s3_enrollment_endpoint_processes_turma_and_records_successes(tmp_path, dashboard_database_url):
    app = _make_app(tmp_path, dashboard_database_url)
    service = FakeS3EnrollmentService()
    app.state.s3_enrollment_service = service

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        response = await client.post(
            "/api/enrollment/s3",
            data={"turma": "ES2025-T1"},
        )

    assert response.status_code == 200
    assert "ES2025-T1" in response.text
    assert "1/2 aluno" in response.text
    assert "alice" in response.text
    assert "bob" in response.text
    assert service.calls == [("ES2025-T1", True)]

    enrollments = app.state.store.list_enrollments()
    assert len(enrollments) == 1
    assert enrollments[0].student_id == "alice"
    assert enrollments[0].source == "s3"
    assert enrollments[0].file_names == ["fotos/ES2025-T1/alice.jpg"]


@pytest.mark.asyncio
async def test_review_status_endpoint_updates_session_and_rejects_unknown_session(
    tmp_path, dashboard_database_url
):
    app = _make_app(tmp_path, dashboard_database_url)
    app.state.store.register_session(
        SessionRecord(
            session_id="sess-1",
            station_id="nuc-01",
            turma="ES2025-T1",
            assessment="Quiz-03",
            started_at=datetime(2026, 4, 16, 18, 0, tzinfo=timezone.utc),
        )
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        response = await client.post(
            "/api/sessions/sess-1/review-status",
            json={"review_status": "REVIEWED"},
        )
        missing_response = await client.post(
            "/api/sessions/missing/review-status",
            json={"review_status": "REVIEWED"},
        )
        invalid_response = await client.post(
            "/api/sessions/sess-1/review-status",
            json={"review_status": "NOT_A_STATUS"},
        )

    assert response.status_code == 200
    assert response.json()["review_status"] == "REVIEWED"
    assert missing_response.status_code == 404
    assert invalid_response.status_code == 422
    assert app.state.store.get_session("sess-1").review_status == SessionReviewStatus.REVIEWED


@pytest.mark.asyncio
async def test_session_review_links_to_chronologically_adjacent_sessions(
    tmp_path, dashboard_database_url
):
    app = _make_app(tmp_path, dashboard_database_url)
    started_at = datetime(2026, 4, 16, 18, 0, tzinfo=timezone.utc)
    for offset, session_id in enumerate(("older", "current", "newer")):
        app.state.store.register_session(
            SessionRecord(
                session_id=session_id,
                station_id="nuc-01",
                turma="ES2025-T1",
                assessment=session_id,
                started_at=started_at + timedelta(minutes=offset),
            )
        )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        response = await client.get("/sessions/current")

    assert response.status_code == 200
    assert 'href="/sessions/older"' in response.text
    assert 'href="/sessions/newer"' in response.text


@pytest.mark.asyncio
async def test_roster_upload_endpoint_imports_csv_and_resolves_names_in_ui(
    tmp_path, dashboard_database_url
):
    app = _make_app(tmp_path, dashboard_database_url)
    app.state.store.register_session(
        SessionRecord(
            session_id="sess-1",
            station_id="nuc-01",
            turma="ES2025-T1",
            assessment="Quiz-03",
            started_at=datetime(2026, 4, 16, 18, 0, tzinfo=timezone.utc),
            student=StudentInfo(student_id="felipehl", student_name="felipehl"),
        )
    )
    csv_bytes = (
        "UID,Name,UIN,Role,Enrollment,Labels\n"
        "felipehl@al.insper.edu.br,Felipe Henrique Lima,uin-1,Student,joined,\n"
    ).encode("utf-8")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        upload_response = await client.post(
            "/api/roster/upload",
            data={"turma": "ES2025-T1"},
            files={"roster_csv": ("roster.csv", csv_bytes, "text/csv")},
        )
        sessions_partial = await client.get("/partials/sessions")
        session_page = await client.get("/sessions/sess-1")

    assert upload_response.status_code == 200
    assert "1 aluno importado" in upload_response.text
    assert "Felipe Henrique Lima" in sessions_partial.text
    assert "(felipehl)" in sessions_partial.text
    assert "Felipe Henrique Lima" in session_page.text


@pytest.mark.asyncio
async def test_roster_upload_endpoint_rejects_csv_without_expected_columns(
    tmp_path, dashboard_database_url
):
    async with AsyncClient(transport=ASGITransport(app=_make_app(tmp_path, dashboard_database_url)), base_url="http://testserver") as client:
        response = await client.post(
            "/api/roster/upload",
            data={"turma": "ES2025-T1"},
            files={"roster_csv": ("roster.csv", b"Login,FullName\nfelipehl,Felipe\n", "text/csv")},
        )

    assert response.status_code == 200
    assert "UID" in response.text


@pytest.mark.asyncio
async def test_session_review_shows_student_photo_from_s3_enrollment(tmp_path, dashboard_database_url):
    app = _make_app(tmp_path, dashboard_database_url)
    service = FakeS3EnrollmentService()
    app.state.s3_enrollment_service = service
    app.state.store.register_session(
        SessionRecord(
            session_id="sess-1",
            station_id="nuc-01",
            turma="ES2025-T1",
            assessment="Quiz-03",
            started_at=datetime(2026, 4, 16, 18, 0, tzinfo=timezone.utc),
            student=StudentInfo(student_id="felipehl", student_name="felipehl"),
        )
    )
    app.state.store.register_session(
        SessionRecord(
            session_id="sess-2",
            station_id="nuc-02",
            turma="ES2025-T1",
            assessment="Quiz-03",
            started_at=datetime(2026, 4, 16, 18, 0, tzinfo=timezone.utc),
            student=StudentInfo(student_id="ghost", student_name="ghost"),
        )
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        with_photo = await client.get("/sessions/sess-1")
        without_photo = await client.get("/sessions/sess-2")

    assert with_photo.status_code == 200
    assert 'class="student-photo"' in with_photo.text
    assert "https://s3.example.com/fotos/ES2025-T1/felipehl.jpg?sig=1" in with_photo.text
    assert without_photo.status_code == 200
    assert 'class="student-photo"' not in without_photo.text
    assert service.photo_calls == [("ES2025-T1", "felipehl"), ("ES2025-T1", "ghost")]


@pytest.mark.asyncio
async def test_station_command_endpoints_enqueue_commands(tmp_path, dashboard_database_url):
    app = _make_app(tmp_path, dashboard_database_url)
    station_headers = _station_headers(app, "nuc-01")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
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
                "auto_start_enabled": True,
                "seconds_remaining": None,
                "recent_events": [],
            },
            headers=station_headers,
        )

    assert stop_response.status_code == 202
    assert unblock_response.status_code == 202
    commands = heartbeat_response.json()["commands"]
    assert [item["command_type"] for item in commands] == ["STOP_SESSION", "UNBLOCK_SESSION"]


@pytest.mark.asyncio
async def test_autostart_command_endpoints_enqueue_toggle(tmp_path, dashboard_database_url):
    app = _make_app(tmp_path, dashboard_database_url)
    station_headers = _station_headers(app, "nuc-01")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        enable_response = await client.post("/api/stations/nuc-01/autostart/enable")
        disable_response = await client.post("/api/stations/nuc-01/autostart/disable")
        heartbeat_response = await client.post(
            "/api/heartbeats",
            json={
                "station_id": "nuc-01",
                "station_name": "NUC Sala 1",
                "status": "IDLE",
                "student": None,
                "active_session_id": None,
                "assessment": None,
                "turma": None,
                "auto_start_enabled": False,
                "seconds_remaining": None,
                "recent_events": [],
            },
            headers=station_headers,
        )

    assert enable_response.status_code == 202
    assert disable_response.status_code == 202
    commands = heartbeat_response.json()["commands"]
    assert [item["command_type"] for item in commands] == ["SET_AUTOSTART", "SET_AUTOSTART"]
    assert [item["payload"]["auto_start"] for item in commands] == [True, False]


@pytest.mark.asyncio
async def test_config_page_lists_history_and_notification_settings(tmp_path, dashboard_database_url):
    """Config de estação fica no modal; /config reúne histórico e notificações."""
    app = _make_app(tmp_path, dashboard_database_url)
    store = app.state.store
    store.create_config(
        ExamConfigPayload(
            turma="ES2025-T1",
            assessment="Quiz-03",
            prairielearn_url="https://prairielearn.org/pl",
            target_station_ids=["nuc-01"],
        )
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        response = await client.get("/config")
        api_response = await client.get("/api/configs")

    assert response.status_code == 200
    assert api_response.status_code == 200
    assert api_response.json()[0]["target_station_ids"] == ["nuc-01"]
    assert 'id="new-config-form"' not in response.text
    assert 'id="notification-settings-form"' in response.text
    assert '<select name="turma">' not in response.text
    assert "ES2025-T1" in response.text
    assert "Quiz-03" in response.text
    assert "nuc-01" in response.text
    assert "config-card" in response.text
    assert "Ver destinos e acesso" in response.text
    assert "Editar e redistribuir" in response.text
    assert "limparConfigs" in response.text or "clearConfigs" in response.text


def test_dashboard_store_distributes_one_config_to_multiple_stations(dashboard_database_url):
    store = DashboardStore(dashboard_database_url)
    config = store.create_config(
        ExamConfigPayload(
            turma="ES2025-T1",
            assessment="Quiz-03",
            prairielearn_url="https://prairielearn.org/pl",
            target_station_ids=["nuc-01", "nuc-02", "nuc-03"],
        )
    )

    assert config.target_station_ids == ["nuc-01", "nuc-02", "nuc-03"]
    for station_id in config.target_station_ids:
        command = store.drain_commands(station_id)
        assert len(command) == 1
        assert command[0].command_type.value == "APPLY_CONFIG"
        assert command[0].payload["assessment"] == "Quiz-03"


@pytest.mark.asyncio
async def test_clear_configs_endpoint_removes_dashboard_history(tmp_path, dashboard_database_url):
    app = _make_app(tmp_path, dashboard_database_url)
    store = app.state.store
    store.create_config(
        ExamConfigPayload(
            turma="ES2025-T1",
            assessment="Quiz-03",
            prairielearn_url="https://prairielearn.org/pl",
            target_station_ids=["nuc-01"],
        )
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        response = await client.post("/api/configs/clear")
        config_page = await client.get("/config")

    assert response.status_code == 200
    assert response.json() == {"removed": 1}
    assert "ES2025-T1" not in config_page.text


@pytest.mark.asyncio
async def test_s3_turmas_endpoint_returns_list(tmp_path, dashboard_database_url):
    app = _make_app(tmp_path, dashboard_database_url)
    app.state.s3_enrollment_service = FakeS3EnrollmentService()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        response = await client.get("/api/s3-turmas")

    assert response.status_code == 200
    assert response.json() == {"turmas": ["ES2025-T1", "ES2025-T2"], "error": None}


@pytest.mark.asyncio
async def test_s3_turmas_endpoint_falls_back_to_known_turmas_on_s3_error(tmp_path, dashboard_database_url):
    class BrokenS3EnrollmentService:
        def list_turmas(self):
            raise RuntimeError("Unable to locate credentials")

    app = _make_app(tmp_path, dashboard_database_url)
    app.state.s3_enrollment_service = BrokenS3EnrollmentService()
    app.state.store.add_enrollment(
        turma="LOCAL-ONLY",
        student_id="123",
        student_name="Alice Silva",
        source="upload",
        file_names=["alice.jpg"],
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        response = await client.get("/api/s3-turmas")

    body = response.json()
    assert body["turmas"] == ["LOCAL-ONLY"]
    assert "Unable to locate credentials" in body["error"]


@pytest.mark.asyncio
async def test_run_enroll_endpoint_enqueues_command_delivered_over_heartbeat(tmp_path, dashboard_database_url):
    app = _make_app(tmp_path, dashboard_database_url)
    station_headers = _station_headers(app, "nuc-01")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        response = await client.post(
            "/api/stations/nuc-01/enroll",
            json={"turma_ids": ["ES2025-T1", "ES2025-T2"]},
        )
        heartbeat_response = await client.post(
            "/api/heartbeats",
            json={
                "station_id": "nuc-01",
                "station_name": "NUC Sala 1",
                "status": "IDLE",
                "student": None,
                "active_session_id": None,
                "assessment": None,
                "turma": None,
                "auto_start_enabled": False,
                "seconds_remaining": None,
                "recent_events": [],
            },
            headers=station_headers,
        )

    assert response.status_code == 202
    commands = heartbeat_response.json()["commands"]
    assert [item["command_type"] for item in commands] == ["RUN_ENROLL"]
    assert commands[0]["payload"]["turma_ids"] == ["ES2025-T1", "ES2025-T2"]


@pytest.mark.asyncio
async def test_run_enroll_endpoint_rejects_empty_turma_list(tmp_path, dashboard_database_url):
    app = _make_app(tmp_path, dashboard_database_url)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        response = await client.post("/api/stations/nuc-01/enroll", json={"turma_ids": []})

    assert response.status_code == 400


def test_station_status_matches_canonical_state_vocabulary():
    """StationStatus é a união de SessionState, StationMode e COMPLETED/OFFLINE.

    O dashboard não pode importar src.core.session (arrastaria cv2/dlib/boto3),
    então a consistência entre os dois enums é garantida aqui em vez de por
    herança. Falha se um estado novo surgir em qualquer um dos lados.
    """
    assert {status.value for status in StationStatus} == known_station_statuses()


def test_dashboard_store_persists_across_restarts(dashboard_database_url):
    store = DashboardStore(dashboard_database_url)
    store.create_config(
        ExamConfigPayload(
            turma="ES2025-T1",
            assessment="Quiz-03",
            timer_minutes=45,
            prairielearn_url="https://prairielearn.org/pl",
            allowlist=["prairielearn.org"],
            auto_start=True,
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

    reloaded = DashboardStore(dashboard_database_url)
    snapshot = reloaded.snapshot()

    assert snapshot["configs"][0].assessment == "Quiz-03"
    assert snapshot["enrollments"][0].student_name == "Alice Silva"
    assert snapshot["sessions"][0].session_id == "sess-1"


def test_dashboard_store_import_roster_resolves_name_and_replaces_on_reupload(dashboard_database_url):
    store = DashboardStore(dashboard_database_url)

    imported = store.import_roster(
        "ES2025-T1",
        [("felipehl", "Felipe Henrique Lima"), ("lennyw", "Lenny Watanabe")],
    )

    assert imported == 2
    assert store.roster_name("ES2025-T1", "felipehl") == "Felipe Henrique Lima"
    assert store.roster_name("ES2025-T1", "FelipeHL") == "Felipe Henrique Lima"
    assert store.roster_name("ES2025-T1", "unknown") is None
    assert store.roster_name("OUTRA-TURMA", "felipehl") is None

    store.import_roster("ES2025-T1", [("felipehl", "Felipe H. Lima Corrigido")])

    assert store.roster_name("ES2025-T1", "felipehl") == "Felipe H. Lima Corrigido"
    assert store.roster_name("ES2025-T1", "lennyw") is None

    reloaded = DashboardStore(dashboard_database_url)
    assert reloaded.roster_name("ES2025-T1", "felipehl") == "Felipe H. Lima Corrigido"


def test_dashboard_store_run_enroll_enqueues_command_and_sets_queued_status(dashboard_database_url):
    store = DashboardStore(dashboard_database_url)

    command = store.run_enroll("nuc-01", ["ES2025-T1", "ES2025-T2"])

    assert command.command_type == "RUN_ENROLL"
    assert command.payload["turma_ids"] == ["ES2025-T1", "ES2025-T2"]
    station = store.get_station("nuc-01")
    assert station.enroll_status == "queued"


def test_queued_enroll_waits_for_station_acknowledgement(dashboard_database_url):
    store = DashboardStore(dashboard_database_url)
    store.run_enroll("nuc-01", ["T2026-T2"])

    store.upsert_station_heartbeat(
        StationHeartbeat(
            station_id="nuc-01",
            station_name="NUC Sala 1",
            status=StationStatus.IDLE,
            enroll_status="idle",
            enroll_message="",
        )
    )

    station = store.get_station("nuc-01")
    assert station.enroll_status == "queued"
    assert station.enroll_message == "1 turma(s) na fila"

    store.upsert_station_heartbeat(
        StationHeartbeat(
            station_id="nuc-01",
            station_name="NUC Sala 1",
            status=StationStatus.IDLE,
            enroll_status="running",
            enroll_message="python scripts/enroll.py --turma T2026-T2 --force",
        )
    )

    station = store.get_station("nuc-01")
    assert station.enroll_status == "running"
    assert "--force" in station.enroll_message


def test_dashboard_store_heartbeat_persists_enroll_status(dashboard_database_url):
    store = DashboardStore(dashboard_database_url)

    store.upsert_station_heartbeat(
        StationHeartbeat.model_validate(
            {
                "station_id": "nuc-01",
                "station_name": "NUC Sala 1",
                "status": "IDLE",
                "student": None,
                "active_session_id": None,
                "assessment": None,
                "turma": None,
                "auto_start_enabled": False,
                "seconds_remaining": None,
                "recent_events": [],
                "enroll_status": "done",
                "enroll_message": "2 turma(s) processada(s)",
            }
        )
    )

    station = store.get_station("nuc-01")
    assert station.enroll_status == "done"
    assert station.enroll_message == "2 turma(s) processada(s)"


def test_dashboard_store_clear_configs_removes_local_history(dashboard_database_url):
    store = DashboardStore(dashboard_database_url)
    store.create_config(
        ExamConfigPayload(
            turma="ES2025-T1",
            assessment="Quiz-03",
            prairielearn_url="https://prairielearn.org/pl",
            target_station_ids=["nuc-01"],
        )
    )

    removed = store.clear_configs()
    reloaded = DashboardStore(dashboard_database_url)

    assert removed == 1
    assert reloaded.snapshot()["configs"] == []


def test_dashboard_store_clear_sessions_removes_local_history(dashboard_database_url):
    store = DashboardStore(dashboard_database_url)
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
            ended_at=datetime(2026, 4, 16, 18, 30, tzinfo=timezone.utc),
            student=StudentInfo(student_id="123", student_name="Alice Silva"),
            status=StationStatus.COMPLETED,
        )
    )
    assert store.queue_email_report(
        "sess-1",
        NotificationSettings(
            enabled=True,
            sender_email="proctor@example.edu",
            recipient_emails=["teacher@example.edu"],
            public_dashboard_url="https://dashboard.example.edu",
        ),
    )

    removed = store.clear_sessions()
    reloaded = DashboardStore(dashboard_database_url)

    assert removed == 1
    assert reloaded.list_sessions() == []
    assert reloaded.get_email_report("sess-1") is None
    assert reloaded.list_enrollments()[0].student_name == "Alice Silva"


@pytest.mark.asyncio
async def test_clear_sessions_endpoint_removes_dashboard_history(tmp_path, dashboard_database_url):
    app = _make_app(tmp_path, dashboard_database_url)
    store = app.state.store
    store.register_session(
        SessionRecord(
            session_id="sess-1",
            station_id="nuc-01",
            turma="ES2025-T1",
            assessment="Quiz-03",
            started_at=datetime(2026, 4, 16, 18, 0, tzinfo=timezone.utc),
            student=StudentInfo(student_id="123", student_name="Alice Silva"),
            status=StationStatus.COMPLETED,
        )
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        response = await client.post("/api/sessions/clear")
        sessions_response = await client.get("/api/sessions")

    assert response.status_code == 200
    assert response.json() == {"removed": 1}
    assert sessions_response.json() == []


def test_finalize_session_marks_history_completed_and_station_idle(dashboard_database_url):
    store = DashboardStore(dashboard_database_url)
    store.upsert_station_heartbeat(
        StationHeartbeat.model_validate(
            {
                "station_id": "nuc-01",
                "station_name": "NUC Sala 1",
                "status": "SESSION",
                "student": {"student_id": "123", "student_name": "Alice Silva"},
                "active_session_id": "sess-1",
                "assessment": "Quiz-03",
                "turma": "ES2025-T1",
                "auto_start_enabled": True,
                "seconds_remaining": 1200,
                "recent_events": [],
            }
        )
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

    finalized = store.finalize_session("sess-1")
    station = store.get_station("nuc-01")

    assert finalized is not None
    assert finalized.status == StationStatus.COMPLETED
    assert station is not None
    assert station.status == StationStatus.IDLE
    assert station.active_session_id is None


def test_finalize_session_preserves_block_timeout_cancellation(dashboard_database_url):
    store = DashboardStore(dashboard_database_url)
    store.register_session(
        SessionRecord(
            session_id="sess-timeout",
            station_id="nuc-01",
            turma="ES2025-T1",
            assessment="Quiz-03",
            started_at=datetime(2026, 4, 16, 18, 0, tzinfo=timezone.utc),
            status=StationStatus.TIMEOUT,
        )
    )

    finalized = store.finalize_session("sess-timeout")

    assert finalized is not None
    assert finalized.status == StationStatus.TIMEOUT


def test_session_record_defaults_to_needs_review():
    session = SessionRecord(
        session_id="sess-1",
        station_id="nuc-01",
        turma="ES2025-T1",
        assessment="Quiz-03",
        started_at=datetime(2026, 4, 16, 18, 0, tzinfo=timezone.utc),
    )

    assert session.review_status == SessionReviewStatus.NEEDS_REVIEW


def test_dashboard_store_sets_session_review_status(dashboard_database_url):
    store = DashboardStore(dashboard_database_url)
    store.register_session(
        SessionRecord(
            session_id="sess-1",
            station_id="nuc-01",
            turma="ES2025-T1",
            assessment="Quiz-03",
            started_at=datetime(2026, 4, 16, 18, 0, tzinfo=timezone.utc),
        )
    )

    updated = store.set_session_review_status("sess-1", SessionReviewStatus.VIOLATION)
    persisted = store.get_session("sess-1")

    assert updated is not None
    assert updated.review_status == SessionReviewStatus.VIOLATION
    assert persisted is not None
    assert persisted.review_status == SessionReviewStatus.VIOLATION
    assert store.set_session_review_status("missing", SessionReviewStatus.REVIEWED) is None


def test_dashboard_snapshot_does_not_sign_recording_urls(dashboard_database_url):
    class S3:
        def __init__(self):
            self.calls = 0

        def generate_presigned_url(self, *_args, **_kwargs):
            self.calls += 1
            return "https://example.test/recording"

    s3 = S3()
    store = DashboardStore(
        dashboard_database_url,
        app_config=AppConfig(),
        s3_client=s3,
    )
    store.register_session(
        SessionRecord(
            session_id="sess-1",
            station_id="nuc-01",
            turma="ES2025-T1",
            assessment="Quiz-03",
            started_at=datetime(2026, 4, 16, 18, 0, tzinfo=timezone.utc),
            recordings=[
                RecordingAsset(
                    label="Câmera principal",
                    s3_bucket="recordings",
                    s3_key="sessions/sess-1/webcam_000.mp4",
                )
            ],
        )
    )

    snapshot = store.snapshot()

    assert s3.calls == 0
    assert snapshot["sessions"][0].recordings[0].url is None


def test_session_record_migrates_legacy_cancelled_timeout_status():
    session = SessionRecord.model_validate(
        {
            "session_id": "sess-legacy-timeout",
            "station_id": "nuc-01",
            "turma": "ES2025-T1",
            "assessment": "Quiz-03",
            "started_at": "2026-04-16T18:00:00+00:00",
            "status": "CANCELLED_TIMEOUT",
        }
    )

    assert session.status == StationStatus.TIMEOUT


def test_dashboard_store_generates_presigned_url_for_s3_assets(tmp_path, dashboard_database_url):
    class FakeS3:
        def generate_presigned_url(self, _operation, Params, ExpiresIn):
            return f"https://signed.example/{Params['Bucket']}/{Params['Key']}?exp={ExpiresIn}"

    store = DashboardStore(dashboard_database_url, app_config=AppConfig(data_dir=tmp_path), s3_client=FakeS3())
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


def test_timeline_builds_virtual_clip_across_segment_boundary():
    started_at = datetime(2026, 4, 16, 18, 0, tzinfo=timezone.utc)
    session = SessionRecord(
        session_id="sess-clips",
        station_id="nuc-01",
        turma="ES2025-T1",
        assessment="Quiz-03",
        started_at=started_at,
        ended_at=datetime(2026, 4, 16, 18, 10, tzinfo=timezone.utc),
        events=[
            SessionEventPayload(
                timestamp=datetime(2026, 4, 16, 18, 4, 58, tzinfo=timezone.utc),
                event_type="GAZE_BLOCKED",
                severity=EventSeverity.CRITICAL,
            )
        ],
        recordings=[
            RecordingAsset(
                label=f"{stream.capitalize()} {index:03d}",
                url=f"https://video.test/{stream}_{index:03d}.mp4",
                stream=stream,
                segment_index=index,
                start_offset_seconds=index * 300,
                duration_seconds=300,
            )
            for stream in ("webcam", "screen")
            for index in (0, 1)
        ],
    )

    timeline = _build_timeline(session)

    assert timeline[0]["offset_seconds"] == 298
    assert {clip["stream"] for clip in timeline[0]["clips"]} == {"webcam", "screen"}
    for clip in timeline[0]["clips"]:
        assert clip["event_at"] == 5
        assert [(segment["start"], segment["end"]) for segment in clip["segments"]] == [
            (293.0, 300.0),
            (0.0, 3.0),
        ]


def test_timeline_infers_legacy_recording_metadata_from_s3_key():
    started_at = datetime(2026, 4, 16, 18, 0, tzinfo=timezone.utc)
    session = SessionRecord(
        session_id="sess-legacy",
        station_id="nuc-01",
        turma="ES2025-T1",
        assessment="Quiz-03",
        started_at=started_at,
        events=[
            SessionEventPayload(
                timestamp=datetime(2026, 4, 16, 18, 5, 5, tzinfo=timezone.utc),
                event_type="ABSENCE_BLOCKED",
                severity=EventSeverity.CRITICAL,
            )
        ],
        recordings=[
            RecordingAsset(
                label="Webcam 001",
                url="https://video.test/webcam_001.mp4",
                s3_key="gravacoes/sess-legacy/webcam_001.mp4",
            )
        ],
    )

    timeline = _build_timeline(session)

    clip = timeline[0]["clips"][0]
    assert clip["stream"] == "webcam"
    assert clip["segments"][0]["start"] == 0.0
    assert clip["segments"][0]["end"] == 10.0


def test_timeline_is_chronological_and_human_readable():
    started_at = datetime(2026, 4, 16, 18, 0, tzinfo=timezone.utc)
    session = SessionRecord(
        session_id="sess-readable",
        station_id="nuc-01",
        turma="ES2025-T1",
        assessment="Quiz-03",
        started_at=started_at,
        events=[
            SessionEventPayload(
                timestamp=datetime(2026, 4, 16, 19, 2, 3, tzinfo=timezone.utc),
                event_type="DIFFERENT_USER_BLOCKED",
                severity=EventSeverity.CRITICAL,
            ),
            SessionEventPayload(
                timestamp=datetime(2026, 4, 16, 19, 2, 4, tzinfo=timezone.utc),
                event_type="BLOCK_TIMEOUT_CANCELLED",
                severity=EventSeverity.CRITICAL,
            ),
            SessionEventPayload(
                timestamp=datetime(2026, 4, 16, 18, 0, 9, tzinfo=timezone.utc),
                event_type="GAZE_WARNING",
                severity=EventSeverity.WARNING,
            ),
        ],
    )

    timeline = _build_timeline(session)

    assert [event["relative_time"] for event in timeline] == ["00:09", "01:02:03", "01:02:04"]
    assert [event["reason"] for event in timeline] == [
        "Olhar desviado detectado",
        "Avaliação pausada: usuário diferente detectado",
        "Avaliação cancelada: bloqueio não resolvido no prazo",
    ]
    assert _event_counts(timeline) == {"ALL": 3, "INFO": 0, "WARNING": 1, "CRITICAL": 2}


def test_session_review_template_prioritizes_human_event_information():
    started_at = datetime(2026, 4, 16, 18, 0, tzinfo=timezone.utc)
    session = SessionRecord(
        session_id="sess-readable",
        station_id="nuc-01",
        turma="ES2025-T1",
        assessment="Quiz-03",
        started_at=started_at,
        ended_at=datetime(2026, 4, 16, 18, 1, 2, tzinfo=timezone.utc),
        student=StudentInfo(student_id="alice", student_name="Alice Silva"),
        status=StationStatus.COMPLETED,
        events=[
            SessionEventPayload(
                timestamp=datetime(2026, 4, 16, 18, 0, 9, tzinfo=timezone.utc),
                event_type="GAZE_WARNING",
                severity=EventSeverity.WARNING,
                details={"ratio": 0.52},
            )
        ],
    )
    timeline = _build_timeline(session)
    template_dir = Path(__file__).parents[1] / "src" / "dashboard" / "templates"
    env = Environment(loader=FileSystemLoader(template_dir))
    env.globals["integrity_score"] = compute_integrity_score
    env.globals["roster_name"] = lambda turma, login: None
    template = env.get_template("session_detail.html")

    html = template.render(
        title="Sessão sess-readable",
        session=session,
        timeline=timeline,
        event_counts=_event_counts(timeline),
        integrity=compute_integrity_score(session),
        duration_label=_format_duration(session.duration_seconds),
        status_label="Concluída",
    )

    assert "Alice Silva" in html
    assert "Olhar desviado detectado" in html
    assert "00:09" in html
    assert "Alertas <span>1</span>" in html
    assert "Ver vídeos do evento" in html
    assert "Expandir todos os vídeos" in html
    assert "Gravações completas" in html
    assert "ID da sessão" in html
    assert "Informações técnicas" not in html
    assert html.index("Gravações completas") < html.index("Timeline de eventos")
    assert "GAZE_WARNING" not in html
    assert "raw_metric_secret" not in html


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (9, "00:09"),
        (62, "01:02"),
        (3723, "01:02:03"),
    ],
)
def test_format_relative_time(seconds, expected):
    assert _format_relative_time(seconds) == expected


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0, "0s"),
        (62, "1min 2s"),
        (3723, "1h 2min 3s"),
    ],
)
def test_format_duration(seconds, expected):
    assert _format_duration(seconds) == expected
