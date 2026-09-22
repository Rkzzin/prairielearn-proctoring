from datetime import UTC, datetime, timedelta
from email import policy
from email.parser import BytesParser
from io import BytesIO

from PIL import Image

from src.dashboard.models import (
    EventSeverity,
    EventSnapshotRecord,
    SessionEmailReport,
    SessionEventPayload,
    SessionRecord,
    StationStatus,
    StudentInfo,
)
from src.dashboard.session_report_mailer import SessionReportMailer


def _snapshot(key, event_type, severity, timestamp):
    return EventSnapshotRecord(
        session_id="session-1",
        event_key=key,
        event_timestamp=timestamp,
        event_type=event_type,
        severity=severity,
        status="ready",
        s3_bucket="bucket",
        s3_key=f"snapshots/{key}.jpg",
    )


def test_snapshot_limit_prioritizes_critical_events_and_zero_selects_all():
    now = datetime.now(UTC)
    snapshots = [
        _snapshot("warning", "GAZE_WARNING", EventSeverity.WARNING, now),
        _snapshot("critical-2", "ABSENCE_ALERT", EventSeverity.CRITICAL, now + timedelta(seconds=2)),
        _snapshot("critical-1", "MULTI_FACE_ALERT", EventSeverity.CRITICAL, now + timedelta(seconds=1)),
    ]

    selected = SessionReportMailer._select_snapshots(snapshots, 2)

    assert [snapshot.event_key for snapshot in selected] == ["critical-1", "critical-2"]
    assert len(SessionReportMailer._select_snapshots(snapshots, 0)) == 3


def test_mailer_uses_gmail_smtp_for_gmail_reports():
    class SMTP:
        def __init__(self):
            self.login_args = None
            self.sent = None

        def ehlo(self):
            pass

        def starttls(self, *, context):
            assert context is not None

        def login(self, username, password):
            self.login_args = (username, password)

        def send_message(self, message):
            self.sent = message
            return {}

        def quit(self):
            pass

    report = SessionEmailReport(
        session_id="session-1",
        sender_email="corsiferrao@gmail.com",
        recipient_emails=["teacher@example.edu"],
        image_link_limit=12,
        delivery_provider="gmail",
        ses_region="sa-east-1",
        public_dashboard_url="https://dashboard.example.edu",
    )
    message = BytesParser(policy=policy.default).parsebytes(
        b"From: corsiferrao@gmail.com\nTo: teacher@example.edu\nMessage-ID: <gmail-123>\n\nTeste"
    )
    smtp = SMTP()
    mailer = SessionReportMailer(
        store=object(),
        gmail_username="corsiferrao@gmail.com",
        gmail_app_password="app-password",
        smtp_factory=lambda *_args, **_kwargs: smtp,
    )

    message_id = mailer._send_message(report, message)

    assert message_id == "<gmail-123>"
    assert smtp.login_args == ("corsiferrao@gmail.com", "app-password")
    assert smtp.sent is message


def test_mailer_explains_when_alert_images_do_not_fit_in_report():
    now = datetime.now(UTC)
    session = SessionRecord(
        session_id="session-1",
        station_id="nuc-01",
        turma="T1",
        assessment="Quiz",
        started_at=now,
        student=StudentInfo(student_id="alice1", student_name="Alice"),
    )
    report = SessionEmailReport(
        session_id=session.session_id,
        sender_email="proctor@example.edu",
        recipient_emails=["teacher@example.edu"],
        image_link_limit=1,
        ses_region="sa-east-1",
        public_dashboard_url="https://dashboard.example.edu",
    )

    message = SessionReportMailer._build_message(
        report,
        session,
        [],
        available_snapshot_count=2,
    )

    html_message = message.get_payload()[-1].get_content()
    assert "Foram mostradas 0 de 2 imagem(ns) de alerta." in html_message
    assert "Há mais 2 alerta(s) disponível(is) na revisão completa." in html_message


def test_mailer_sends_html_summary_and_permanent_dashboard_image_links():
    now = datetime(2026, 9, 15, 21, 2, 3, tzinfo=UTC)
    events = [
        SessionEventPayload(
            timestamp=now,
            event_type="SESSION_STARTED",
            severity=EventSeverity.INFO,
        ),
        SessionEventPayload(
            timestamp=now + timedelta(seconds=10),
            event_type="ELECTRONIC_DEVICE_DETECTED",
            severity=EventSeverity.CRITICAL,
        ),
    ]
    session = SessionRecord(
        session_id="session-1",
        station_id="nuc-01",
        turma="T1",
        assessment="Quiz",
        started_at=now,
        ended_at=now + timedelta(minutes=20),
        student=StudentInfo(student_id="alice1", student_name="Alice"),
        status=StationStatus.COMPLETED,
        events=events,
    )
    report = SessionEmailReport(
        session_id=session.session_id,
        sender_email="proctor@example.edu",
        recipient_emails=["teacher@example.edu"],
        image_link_limit=12,
        ses_region="sa-east-1",
        public_dashboard_url="https://dashboard.example.edu",
    )
    snapshots = [
        _snapshot(
            "event-key",
            "ELECTRONIC_DEVICE_DETECTED",
            EventSeverity.CRITICAL,
            now + timedelta(seconds=10),
        )
    ]

    class Store:
        finished = None

        def claim_email_report(self, _session_id):
            return report

        def get_session(self, _session_id):
            return session

        def list_event_snapshots(self, _session_id):
            return snapshots

        def read_event_snapshot_image(self, _snapshot):
            return b"jpeg-image"

        def read_student_photo(self, _turma, _student_id):
            output = BytesIO()
            Image.new("RGB", (1, 1), "white").save(output, format="PNG")
            return output.getvalue()

        def finish_email_report(self, value, **kwargs):
            self.finished = (value, kwargs)

    class SES:
        request = None

        def send_raw_email(self, **kwargs):
            self.request = kwargs
            return {"MessageId": "ses-123"}

    store = Store()
    ses = SES()
    mailer = SessionReportMailer(
        store=store,
        ses_client_factory=lambda _region: ses,
    )

    mailer._run(session.session_id)

    message = BytesParser(policy=policy.default).parsebytes(
        ses.request["RawMessage"]["Data"]
    )
    raw_message = "\n".join(
        part.get_content()
        for part in message.walk()
        if part.get_content_type() in {"text/plain", "text/html"}
    )
    html_message = next(
        part.get_content()
        for part in message.walk()
        if part.get_content_type() == "text/html"
    )
    assert ses.request["Destinations"] == ["teacher@example.edu"]
    assert message["To"] == "teacher@example.edu"
    assert "Resumo da avaliação" in raw_message
    assert "Nome do aluno: Alice" in raw_message
    assert "Usuário: alice1" in raw_message
    assert "Início da prova: 15/09/2026 às 18:02:03" in raw_message
    assert "ELECTRONIC_DEVICE_DETECTED" not in raw_message
    assert 'src="cid:snapshot-1@proctoring"' in raw_message
    assert 'src="cid:student-photo@proctoring"' in raw_message
    assert "Foram mostradas todas as 1 imagem(ns) de alerta disponíveis." in raw_message
    assert html_message.index("Abrir revisão completa") < html_message.index(
        "Imagens selecionadas"
    )
    assert (
        "https://dashboard.example.edu/sessions/session-1/event-snapshots/event-key"
        in raw_message
    )
    image_parts = [part for part in message.walk() if part.get_content_type() == "image/jpeg"]
    assert len(image_parts) == 2
    assert image_parts[0]["Content-ID"] == "<student-photo@proctoring>"
    assert image_parts[1].get_content() == b"jpeg-image"
    assert image_parts[1]["Content-ID"] == "<snapshot-1@proctoring>"
    assert store.finished == (report, {"message_id": "ses-123"})
