from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

from src.core.config import AppConfig, FaceConfig, RecorderConfig, S3Config
from src.core.cpu_affinity import auto_split_cpu_sets, split_ffmpeg_stream_cpu_sets
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
    capture._procs["webcam"] = DummyProc()
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
    capture._procs["webcam"] = DummyProc()
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
    capture._procs["webcam"] = FinalizingProc(webcam_path)
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


def test_capture_webcam_ffmpeg_command_includes_preview_output(tmp_path: Path, monkeypatch):
    commands: list[list[str]] = []

    class FakeProc(DummyProc):
        def __init__(self, cmd):
            super().__init__()
            self.cmd = cmd
            self.stderr = []

    def fake_popen(cmd, **_kwargs):
        commands.append(cmd)
        return FakeProc(cmd)

    monkeypatch.setattr("src.recorder.capture.subprocess.Popen", fake_popen)

    capture = Capture(
        session_id="sess-preview",
        s3_config=S3Config(segment_duration_sec=300),
        face_config=FaceConfig(
            models_dir=tmp_path / "models",
            encodings_dir=tmp_path / "encodings",
        ),
        app_config=AppConfig(data_dir=tmp_path),
        recorder_config=RecorderConfig(
            preview_host="127.0.0.1",
            preview_port=19191,
            preview_width=640,
            preview_height=360,
            preview_fps=10,
        ),
    )

    monkeypatch.setattr(capture, "_start_monitor_threads", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(capture, "_ensure_process_started", lambda *_args, **_kwargs: None)

    capture._start_webcam_stream()

    assert commands, "ffmpeg da webcam não foi iniciado"
    cmd = commands[0]
    assert "-filter_complex" in cmd
    assert "udp://127.0.0.1:19191?pkt_size=1316" in cmd
    assert "-use_wallclock_as_timestamps" in cmd
    assert "-fps_mode" in cmd
    assert "passthrough" in cmd
    assert "-segment_format_options" in cmd
    assert "movflags=+faststart" in cmd
    assert "-profile:v" in cmd
    assert "libx264" in cmd
    assert "veryfast" in cmd
    assert "ultrafast" in cmd
    assert "zerolatency" in cmd
    assert cmd.count("-pix_fmt") >= 2
    assert cmd.count("yuv420p") >= 2
    assert "-g" in cmd
    assert "-keyint_min" in cmd
    assert "-sc_threshold" in cmd
    assert "repeat-headers=1:aud=1" in cmd
    assert "mpeg1video" not in cmd
    assert capture.preview_url == "udp://127.0.0.1:19191?overrun_nonfatal=1&fifo_size=5000000"


def test_capture_ensure_process_started_raises_if_ffmpeg_exits_early(tmp_path: Path, monkeypatch):
    capture = Capture(
        session_id="sess-preview-fail",
        s3_config=S3Config(segment_duration_sec=300),
        face_config=FaceConfig(
            models_dir=tmp_path / "models",
            encodings_dir=tmp_path / "encodings",
        ),
        app_config=AppConfig(data_dir=tmp_path),
    )

    class FailingProc(DummyProc):
        def poll(self):
            return 234

    monkeypatch.setattr("src.recorder.capture.time.sleep", lambda _seconds: None)

    try:
        capture._ensure_process_started("webcam", FailingProc(), timeout_sec=0.1)
    except RuntimeError as exc:
        assert "webcam" in str(exc)
        assert "234" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("esperava falha ao validar start do ffmpeg")


def test_capture_screen_ffmpeg_command_uses_browser_compatible_h264(tmp_path: Path, monkeypatch):
    commands: list[list[str]] = []

    class FakeProc(DummyProc):
        def __init__(self, cmd):
            super().__init__()
            self.cmd = cmd
            self.stderr = []

    def fake_popen(cmd, **_kwargs):
        commands.append(cmd)
        return FakeProc(cmd)

    monkeypatch.setattr("src.recorder.capture.subprocess.Popen", fake_popen)

    capture = Capture(
        session_id="sess-screen-preview",
        s3_config=S3Config(segment_duration_sec=300),
        face_config=FaceConfig(
            models_dir=tmp_path / "models",
            encodings_dir=tmp_path / "encodings",
        ),
        app_config=AppConfig(data_dir=tmp_path),
        recorder_config=RecorderConfig(),
    )

    monkeypatch.setattr(capture, "_start_monitor_threads", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(capture, "_ensure_process_started", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(capture, "_resolve_screen_capture_size", lambda: "1920x1080")

    capture._start_screen_stream()

    assert commands, "ffmpeg da tela não foi iniciado"
    cmd = commands[0]
    assert "-use_wallclock_as_timestamps" in cmd
    assert "-fps_mode" in cmd
    assert "passthrough" in cmd
    assert "-vf" in cmd
    assert "scale=1280:720:flags=fast_bilinear,setsar=1" in cmd
    assert "-profile:v" in cmd
    assert "high" in cmd
    assert "veryfast" in cmd
    assert "-pix_fmt" in cmd
    assert "yuv420p" in cmd
    assert "-segment_format_options" in cmd
    assert "movflags=+faststart" in cmd


def test_resolve_screen_capture_size_uses_display_resolution_from_xrandr(tmp_path: Path, monkeypatch):
    capture = Capture(
        session_id="sess-screen-size",
        s3_config=S3Config(segment_duration_sec=300),
        face_config=FaceConfig(
            models_dir=tmp_path / "models",
            encodings_dir=tmp_path / "encodings",
        ),
        app_config=AppConfig(data_dir=tmp_path),
        recorder_config=RecorderConfig(display=":1", screen_size="1280x720"),
    )

    def fake_run(cmd, **kwargs):
        assert cmd == ["xrandr", "--current"]
        assert kwargs["env"]["DISPLAY"] == ":1"
        return SimpleNamespace(
            returncode=0,
            stdout="Screen 0: minimum 320 x 200, current 1920 x 1080, maximum 16384 x 16384\n",
            stderr="",
        )

    monkeypatch.setattr("src.recorder.capture.subprocess.run", fake_run)

    assert capture._resolve_screen_capture_size() == "1920x1080"


def test_auto_split_cpu_sets_reserves_two_cores_for_ffmpeg_when_available():
    ffmpeg_cpus, proctor_cpus = auto_split_cpu_sets(available={0, 1, 2, 3, 4, 5, 6, 7})

    assert ffmpeg_cpus == {6, 7}
    assert proctor_cpus == {0, 1, 2, 3, 4, 5}


def test_split_ffmpeg_stream_cpu_sets_gives_screen_dedicated_core():
    cpu_sets = split_ffmpeg_stream_cpu_sets({3, 4, 5, 6, 7})

    assert cpu_sets["webcam"] == {3, 4, 5, 6}
    assert cpu_sets["screen"] == {7}
