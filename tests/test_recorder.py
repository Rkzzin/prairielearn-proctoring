from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

from src.core.config import AppConfig, FaceConfig, S3Config
from src.recorder.capture import Capture, SegmentInfo
from src.recorder.uploader import Uploader


class DummyProc:
    def __init__(self):
        self.stdin = SimpleNamespace(close=lambda: None)

    def poll(self):
        return None

    def wait(self, timeout=None):
        return 0

    def kill(self):
        return None

    def send_signal(self, _signal):
        return None


def test_watch_segments_notifies_completed_segment_before_stop(tmp_path: Path):
    notified: list[SegmentInfo] = []
    capture = Capture(
        session_id="sess-short",
        s3_config=S3Config(segment_duration_sec=300),
        face_config=FaceConfig(
            models_dir=tmp_path / "models",
            encodings_dir=tmp_path / "encodings",
        ),
        app_config=AppConfig(data_dir=tmp_path),
        on_segment_ready=notified.append,
    )

    segment = tmp_path / "sessions" / "sess-short" / "recordings" / "webcam_000.mp4"
    segment.parent.mkdir(parents=True, exist_ok=True)
    segment.write_bytes(b"video-bytes")
    (segment.parent / "webcam_001.mp4").write_bytes(b"next-segment")

    capture._running = True
    watcher = threading.Thread(target=capture._watch_segments, args=("webcam",))
    watcher.start()

    capture._stop_event.set()
    watcher.join(timeout=2)
    capture._running = False

    assert [item.path.name for item in notified] == ["webcam_000.mp4"]
    assert notified[0].index == 0


def test_capture_stop_waits_for_watchers_to_flush_final_segments(tmp_path: Path):
    notified: list[SegmentInfo] = []
    capture = Capture(
        session_id="sess-stop",
        s3_config=S3Config(segment_duration_sec=300),
        face_config=FaceConfig(
            models_dir=tmp_path / "models",
            encodings_dir=tmp_path / "encodings",
        ),
        app_config=AppConfig(data_dir=tmp_path),
        on_segment_ready=notified.append,
    )

    capture._running = True
    capture._webcam_proc = DummyProc()
    capture._procs["screen"] = DummyProc()

    segment = tmp_path / "sessions" / "sess-stop" / "recordings" / "screen_000.mp4"
    segment.parent.mkdir(parents=True, exist_ok=True)
    segment.write_bytes(b"screen-bytes")

    watcher = threading.Thread(target=capture._watch_segments, args=("screen",), daemon=True)
    capture._monitor_threads["screen_segments"] = watcher
    watcher.start()

    capture.stop()

    assert [item.path.name for item in notified] == ["screen_000.mp4"]


def test_capture_stop_flushes_webcam_and_screen_final_segments(tmp_path: Path):
    notified: list[SegmentInfo] = []
    capture = Capture(
        session_id="sess-both",
        s3_config=S3Config(segment_duration_sec=300),
        face_config=FaceConfig(
            models_dir=tmp_path / "models",
            encodings_dir=tmp_path / "encodings",
        ),
        app_config=AppConfig(data_dir=tmp_path),
        on_segment_ready=notified.append,
    )

    capture._running = True
    capture._webcam_proc = DummyProc()
    capture._procs["screen"] = DummyProc()

    rec_dir = tmp_path / "sessions" / "sess-both" / "recordings"
    rec_dir.mkdir(parents=True, exist_ok=True)
    (rec_dir / "webcam_000.mp4").write_bytes(b"webcam-bytes")
    (rec_dir / "screen_000.mp4").write_bytes(b"screen-bytes")

    webcam_watcher = threading.Thread(
        target=capture._watch_segments,
        args=("webcam",),
        daemon=True,
    )
    screen_watcher = threading.Thread(
        target=capture._watch_segments,
        args=("screen",),
        daemon=True,
    )
    capture._monitor_threads["webcam_segments"] = webcam_watcher
    capture._monitor_threads["screen_segments"] = screen_watcher
    webcam_watcher.start()
    screen_watcher.start()

    capture.stop()

    assert sorted(item.path.name for item in notified) == [
        "screen_000.mp4",
        "webcam_000.mp4",
    ]


def test_capture_stop_flushes_only_after_process_wait_finalizes_file(tmp_path: Path):
    notified: list[SegmentInfo] = []

    class FinalizingProc(DummyProc):
        def __init__(self, path: Path):
            super().__init__()
            self._path = path

        def wait(self, timeout=None):
            self._path.write_bytes(b"finalized-video")
            return 0

    capture = Capture(
        session_id="sess-finalize",
        s3_config=S3Config(segment_duration_sec=300),
        face_config=FaceConfig(
            models_dir=tmp_path / "models",
            encodings_dir=tmp_path / "encodings",
        ),
        app_config=AppConfig(data_dir=tmp_path),
        on_segment_ready=notified.append,
    )

    rec_dir = tmp_path / "sessions" / "sess-finalize" / "recordings"
    rec_dir.mkdir(parents=True, exist_ok=True)
    webcam_path = rec_dir / "webcam_000.mp4"

    capture._running = True
    capture._webcam_proc = FinalizingProc(webcam_path)
    capture._procs["screen"] = DummyProc()

    capture.stop()

    assert [item.path.name for item in notified] == ["webcam_000.mp4"]


def test_uploader_stop_processes_segments_enqueued_before_sentinel(tmp_path: Path, monkeypatch):
    uploaded: list[str] = []

    class FakeS3:
        def upload_file(self, Filename, Bucket, Key, ExtraArgs):
            uploaded.append(f"{Bucket}:{Key}:{Path(Filename).name}")

    monkeypatch.setattr("src.recorder.uploader.boto3.client", lambda *_args, **_kwargs: FakeS3())

    uploader = Uploader(
        session_id="sess-upload",
        s3_config=S3Config(bucket="test-bucket", recordings_prefix="gravacoes"),
        app_config=AppConfig(data_dir=tmp_path),
        delete_after_upload=False,
    )

    file_path = tmp_path / "webcam_000.mp4"
    file_path.write_bytes(b"content")
    uploader.start()
    uploader.enqueue(
        SegmentInfo(
            stream="webcam",
            path=file_path,
            session_id="sess-upload",
            index=0,
        )
    )
    uploader.stop()

    assert uploaded == ["test-bucket:gravacoes/sess-upload/webcam_000.mp4:webcam_000.mp4"]
