from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from src.core.dashboard_payload import collect_session_recordings
from src.dashboard.app import _build_event_clips
from src.dashboard.models import RecordingAsset, SessionRecord


def test_recording_payload_includes_virtual_timeline_metadata():
    uploader = SimpleNamespace(
        uploaded_segments=[
            (
                SimpleNamespace(stream="webcam", index=2),
                "gravacoes/session-1/webcam_002.mp4",
            )
        ]
    )

    recordings = collect_session_recordings(uploader, "proctor-station", 300)

    assert recordings == [
        {
            "label": "Câmera principal 002",
            "s3_bucket": "proctor-station",
            "s3_key": "gravacoes/session-1/webcam_002.mp4",
            "kind": "video",
            "stream": "webcam",
            "segment_index": 2,
            "start_offset_seconds": 600,
            "duration_seconds": 300,
        }
    ]


def test_recording_payload_labels_environment_camera():
    uploader = SimpleNamespace(
        uploaded_segments=[
            (
                SimpleNamespace(stream="environment", index=0),
                "gravacoes/session-1/environment_000.mp4",
            )
        ]
    )

    recordings = collect_session_recordings(uploader, "proctor-station", 300)

    assert recordings[0]["label"] == "Câmera ambiente 000"
    assert recordings[0]["stream"] == "environment"


def test_event_clips_include_environment_camera():
    started_at = datetime.now(timezone.utc)
    session = SessionRecord(
        session_id="session-1",
        station_id="nuc-01",
        turma="T1",
        assessment="Quiz",
        started_at=started_at,
        ended_at=started_at + timedelta(minutes=5),
        recordings=[
            RecordingAsset(
                label="Câmera ambiente 000",
                stream="environment",
                segment_index=0,
                start_offset_seconds=0,
                duration_seconds=300,
                url="https://example.test/environment.mp4",
            )
        ],
    )

    clips = _build_event_clips(session, 30)

    assert clips[0]["stream"] == "environment"
    assert clips[0]["label"] == "Câmera ambiente"
