from datetime import UTC, datetime, timedelta
from email import policy
from email.parser import BytesParser

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


def test_mailer_sends_html_summary_and_permanent_dashboard_image_links():
    now = datetime.now(UTC)
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
    assert ses.request["Destinations"] == ["teacher@example.edu"]
    assert "Resumo da avaliação" in raw_message
    assert "ELECTRONIC_DEVICE_DETECTED" not in raw_message
    assert (
        "https://dashboard.example.edu/sessions/session-1/event-snapshots/event-key"
        in raw_message
    )
    assert store.finished == (report, {"message_id": "ses-123"})
