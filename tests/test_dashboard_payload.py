from types import SimpleNamespace

from src.core.dashboard_payload import collect_session_recordings


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
            "label": "Webcam 002",
            "s3_bucket": "proctor-station",
            "s3_key": "gravacoes/session-1/webcam_002.mp4",
            "kind": "video",
            "stream": "webcam",
            "segment_index": 2,
            "start_offset_seconds": 600,
            "duration_seconds": 300,
        }
    ]
