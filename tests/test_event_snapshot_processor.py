from datetime import datetime, timedelta, timezone

from src.core.config import AppConfig
from src.dashboard.event_snapshot_processor import EventSnapshotProcessor
from src.dashboard.models import (
    EventSeverity,
    EventSnapshotRecord,
    RecordingAsset,
    SessionEventPayload,
    SessionRecord,
    StationStatus,
)


class FakeStore:
    def __init__(self):
        self.finished = []

    def finish_event_snapshot(self, snapshot, **kwargs):
        self.finished.append((snapshot, kwargs))


class FakeS3:
    def __init__(self):
        self.downloads = []
        self.uploads = []

    def download_file(self, bucket, key, destination):
        self.downloads.append((bucket, key))
        with open(destination, "wb") as output:
            output.write(b"video")

    def upload_file(self, source, bucket, key, ExtraArgs=None):
        self.uploads.append((bucket, key, ExtraArgs))


def test_processor_extracts_flagged_event_from_webcam_and_uploads_jpeg(tmp_path):
    started_at = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)
    event_at = started_at + timedelta(seconds=7)
    session = SessionRecord(
        session_id="session-1",
        station_id="nuc-1",
        turma="T1",
        assessment="Quiz",
        started_at=started_at - timedelta(seconds=5),
        ended_at=started_at + timedelta(minutes=1),
        status=StationStatus.COMPLETED,
        events=[
            SessionEventPayload(
                timestamp=started_at,
                event_type="SESSION_STARTED",
                severity=EventSeverity.INFO,
            ),
            SessionEventPayload(
                timestamp=event_at,
                event_type="GAZE_ALERT",
                severity=EventSeverity.CRITICAL,
            ),
        ],
        recordings=[
            RecordingAsset(
                label="Câmera principal 000",
                stream="webcam",
                s3_bucket="recordings",
                s3_key="sessions/session-1/webcam_000.mp4",
                start_offset_seconds=0,
                duration_seconds=300,
            )
        ],
    )
    snapshot = EventSnapshotRecord(
        session_id=session.session_id,
        event_key="event-1",
        event_timestamp=event_at,
        event_type="GAZE_ALERT",
        severity=EventSeverity.CRITICAL,
    )
    store = FakeStore()
    s3 = FakeS3()
    processor = EventSnapshotProcessor(
        store=store,
        app_config=AppConfig(data_dir=tmp_path),
        s3_client=s3,
    )
    extracted_offsets = []

    def extract_frame(_source, offset, destination):
        extracted_offsets.append(offset)
        destination.write_bytes(b"\xff\xd8image")

    processor._extract_frame = extract_frame

    processor._process(session, [snapshot])

    assert extracted_offsets == [7.0]
    assert s3.downloads == [("recordings", "sessions/session-1/webcam_000.mp4")]
    assert s3.uploads == [
        (
            "proctor-station",
            "gravacoes/session-1/event-snapshots/event-1.jpg",
            {"ContentType": "image/jpeg"},
        )
    ]
    assert store.finished == [
        (
            snapshot,
            {
                "s3_bucket": "proctor-station",
                "s3_key": "gravacoes/session-1/event-snapshots/event-1.jpg",
            },
        )
    ]


def test_processor_marks_snapshot_failed_without_webcam_recording(tmp_path):
    now = datetime.now(timezone.utc)
    session = SessionRecord(
        session_id="session-1",
        station_id="nuc-1",
        turma="T1",
        assessment="Quiz",
        started_at=now,
        ended_at=now,
        status=StationStatus.COMPLETED,
    )
    snapshot = EventSnapshotRecord(
        session_id=session.session_id,
        event_key="event-1",
        event_timestamp=now,
        event_type="ABSENCE_ALERT",
        severity=EventSeverity.WARNING,
    )
    store = FakeStore()
    processor = EventSnapshotProcessor(
        store=store,
        app_config=AppConfig(data_dir=tmp_path),
        s3_client=FakeS3(),
    )

    processor._process(session, [snapshot])

    assert store.finished[0][1]["error"] == "gravação da câmera principal indisponível"
