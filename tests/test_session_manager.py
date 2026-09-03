from __future__ import annotations

import dataclasses
import threading
from collections import deque
from datetime import datetime, timezone

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.routes import ConfigUpdateRequest
from src.api.server import create_app
from src.core.autostart import SessionAutoStartWorker
from src.core.camera import CameraSource, SessionCamera
from src.core.config import AppConfig, FaceConfig, ProctorConfig, RecorderConfig, S3Config
from src.core.models import IdentifyResult, IdentifyStatus
from src.core.session import (
    DASHBOARD_CONFIG_FIELD_MAP,
    DASHBOARD_PROCTOR_FIELD_CASTS,
    DASHBOARD_ROUTING_FIELDS,
    SessionConfig,
    SessionError,
    SessionManager,
    SessionRuntime,
    SessionState,
    StationMode,
)
from src.core.states import SessionState as CanonicalSessionState
from src.core.states import StationMode as CanonicalStationMode
from src.core.states import derive_station_status
from src.core.teardown import EXIT_EXAM_MODE_REASON, ShutdownPolicy
from src.dashboard.models import ExamConfigPayload
from src.proctor.engine import BlockReason, ProctorState


def _session_config_field_names() -> set[str]:
    return {field.name for field in dataclasses.fields(SessionConfig)}


class FakeCamera:
    def __init__(self, frames):
        self.frames = deque(frames)
        self.released = False

    def isOpened(self):
        return True

    def set(self, *_args, **_kwargs):
        return True

    def read(self):
        if self.frames:
            return True, self.frames.popleft()
        return False, None

    def release(self):
        self.released = True


class RepeatingCamera(FakeCamera):
    def __init__(self, frame):
        super().__init__([])
        self.frame = frame

    def read(self):
        return True, self.frame


class FakeRecognizer:
    def __init__(self, identify_results):
        self.identify_results = deque(identify_results)
        self.loaded_turma = None
        self.identify_calls = 0

    def load_turma(self, turma_id):
        self.loaded_turma = turma_id

    def identify(self, _frame):
        self.identify_calls += 1
        if self.identify_results:
            return self.identify_results.popleft()
        return IdentifyResult(status=IdentifyStatus.NO_FACE)


class FakeEngine:
    def __init__(self, states):
        self.states = deque(states)
        self.started = False
        self.stopped = False
        self.unblocked = False
        self.external_blocks = []
        self.cancelled_timeouts: list[float] = []
        self.block_reason = type("Reason", (), {"value": "ABSENCE"})()

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def update(self, _frame):
        if self.states:
            return self.states.popleft()
        return ProctorState.NORMAL

    def unblock(self):
        self.unblocked = True

    def block(self, reason, *, details=None):
        self.block_reason = reason
        self.external_blocks.append((reason, details))

    def cancel_after_block_timeout(self, timeout_sec):
        self.cancelled_timeouts.append(timeout_sec)


class FakeCapture:
    def __init__(self):
        self.started = False
        self.stopped = False
        self.frames = 0
        self.preview_url = "udp://127.0.0.1:18181?overrun_nonfatal=1&fifo_size=5000000"

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def write_frame(self, _frame):
        self.frames += 1


class FakeUploader:
    def __init__(self):
        self.started = False
        self.stopped = False
        self.queue_size = 0
        self.uploaded_segments = []

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def enqueue(self, _segment):
        self.queue_size += 1


class FakeKiosk:
    def __init__(self):
        self.started = False
        self.start_calls = 0
        self.stopped = False
        self.blocked = False
        self.unblocked = False
        self.url = None
        self.allowlist = None

    def start(self, url, *, allowlist=None):
        self.started = True
        self.start_calls += 1
        self.stopped = False
        self.url = url
        self.allowlist = allowlist

    def stop(self):
        self.stopped = True

    def block(self):
        self.blocked = True

    def unblock(self):
        self.unblocked = True

    @property
    def is_running(self):
        return self.started and not self.stopped


class FakeLockdown:
    def __init__(self):
        self.enabled = False
        self.disabled = False

    def enable(self):
        self.enabled = True

    def disable(self):
        self.disabled = True
        self.enabled = False

    @property
    def is_enabled(self):
        return self.enabled


class FakeOverlay:
    def __init__(self):
        self.controls_started = False
        self.blocked_shown: list[str | None] = []
        self.blocked_students: list[str | None] = []
        self.blocked_timeouts: list[float] = []
        self.waiting_shown: list[str | None] = []
        self.waiting_hidden = 0
        self.confirmations: list[dict[str, object]] = []
        self.confirmations_hidden = 0
        self.hide_calls = 0
        self.stopped = False

    def start_controls(self):
        self.controls_started = True

    def show_waiting(self, message=None):
        self.waiting_shown.append(message)

    def hide_waiting(self):
        self.waiting_hidden += 1

    def show_identity_confirmation(self, **kwargs):
        self.confirmations.append(kwargs)

    def hide_identity_confirmation(self):
        self.confirmations_hidden += 1

    def show_blocked(self, reason=None, *, student_id=None, timeout_sec=20.0):
        self.blocked_shown.append(reason)
        self.blocked_students.append(student_id)
        self.blocked_timeouts.append(timeout_sec)

    def hide_blocked(self):
        self.hide_calls += 1

    def stop(self):
        self.stopped = True


class EventReidentify:
    def __init__(self):
        self.calls = 0
        self.event = threading.Event()

    def __call__(self, **_kwargs):
        self.calls += 1
        self.event.set()
        return True


def _make_manager(
    *,
    identify_results,
    engine_states,
    frames,
    reidentify_fn=None,
    video_capture_factory=None,
    confirmation_fn=lambda _student_id, _student_name, _timeout_sec: True,
):
    fake_recognizer = FakeRecognizer(identify_results)
    fake_engine = FakeEngine(engine_states)
    fake_capture = FakeCapture()
    fake_uploader = FakeUploader()
    fake_kiosk = FakeKiosk()
    fake_overlay = FakeOverlay()
    fake_lockdown = FakeLockdown()
    fake_camera = FakeCamera(frames)

    manager = SessionManager(
        app_config=AppConfig(
            data_dir="/tmp/proctor-tests",
            persist_session_config=False,
            restore_exam_mode_on_startup=False,
        ),
        face_config=FaceConfig(
            models_dir="models",
            encodings_dir="data/encodings",
            max_identification_attempts=3,
        ),
        proctor_config=ProctorConfig(),
        recorder_config=RecorderConfig(),
        s3_config=S3Config(bucket="test-bucket"),
        recognizer_factory=lambda: fake_recognizer,
        engine_factory=lambda _session_id: fake_engine,
        capture_factory=lambda _session_id: fake_capture,
        uploader_factory=lambda _session_id: fake_uploader,
        kiosk_factory=lambda: fake_kiosk,
        overlay_factory=lambda: fake_overlay,
        lockdown_factory=lambda: fake_lockdown,
        video_capture_factory=video_capture_factory or (lambda _index: fake_camera),
        reidentify_fn=reidentify_fn or (lambda **_kwargs: True),
        confirmation_fn=confirmation_fn,
        s3_probe=lambda: True,
        sleep_fn=lambda _seconds: None,
    )
    return manager, fake_recognizer, fake_engine, fake_capture, fake_uploader, fake_kiosk, fake_overlay, fake_lockdown, fake_camera


def test_session_manager_start_and_stop_manual_session():
    manager, recognizer, engine, capture, uploader, kiosk, overlay, lockdown, camera = _make_manager(
        identify_results=[
            IdentifyResult(
                status=IdentifyStatus.MATCH,
                student_id="123",
                student_name="Alice",
                confidence=0.9,
            )
        ],
        engine_states=[ProctorState.NORMAL],
        frames=["identify-frame", "loop-frame"],
    )

    manager.update_config(turma_id="ES2025-T1", prairielearn_url="https://pl.test/exam")
    started = manager.start_session()
    assert started["state"] in {SessionState.SESSION.value, SessionState.BLOCKED.value}
    assert recognizer.loaded_turma == "ES2025-T1"
    assert engine.started is True
    assert capture.started is True
    assert uploader.started is True
    assert kiosk.started is True
    assert kiosk.url == "https://pl.test/exam"
    assert kiosk.allowlist == []
    assert overlay.controls_started is True
    assert lockdown.enabled is True

    stopped = manager.stop_session(reason="test")
    assert stopped["state"] == SessionState.IDLE.value
    assert engine.stopped is True
    assert capture.stopped is True
    assert uploader.stopped is True
    assert kiosk.stopped is True
    assert overlay.stopped is True
    assert lockdown.disabled is True
    assert camera.released is True


def test_session_manager_switches_from_device_camera_to_capture_preview():
    direct_camera = FakeCamera(["identify-frame"])
    preview_camera = RepeatingCamera("loop-frame")
    opened_sources = []

    def video_capture_factory(source):
        opened_sources.append(source)
        if isinstance(source, str):
            return preview_camera
        return direct_camera

    manager, recognizer, engine, capture, uploader, kiosk, overlay, lockdown, _camera = _make_manager(
        identify_results=[
            IdentifyResult(
                status=IdentifyStatus.MATCH,
                student_id="123",
                student_name="Alice",
                confidence=0.9,
            )
        ],
        engine_states=[ProctorState.NORMAL],
        frames=[],
        video_capture_factory=video_capture_factory,
    )

    manager.update_config(turma_id="ES2025-T1", prairielearn_url="https://pl.test/exam")
    manager.start_session()

    assert opened_sources[0] == manager._face_cfg.camera_index
    assert opened_sources[1] == capture.preview_url
    assert direct_camera.released is True
    assert preview_camera.released is False

    manager.stop_session(reason="test")
    assert preview_camera.released is True


def test_camera_handoff_prevents_probe_from_reopening_physical_device():
    opened_sources = []

    def capture_factory(source):
        opened_sources.append(source)
        return RepeatingCamera("frame")

    camera = SessionCamera(
        face_config=FaceConfig(models_dir="models", encodings_dir="data/encodings", camera_index=0),
        capture_factory=capture_factory,
    )
    device = camera.open_device()

    camera.handoff_to_external()

    assert device.released is True
    assert camera.source == CameraSource.EXTERNAL
    assert camera.probe() is True
    assert opened_sources == [0]


def test_camera_moves_from_external_owner_to_preview_without_device_probe():
    opened_sources = []

    def capture_factory(source):
        opened_sources.append(source)
        return RepeatingCamera("frame")

    camera = SessionCamera(
        face_config=FaceConfig(models_dir="models", encodings_dir="data/encodings", camera_index=0),
        capture_factory=capture_factory,
    )
    camera.open_device()
    camera.handoff_to_external()

    preview = camera.open_preview("udp://preview")

    assert camera.source == CameraSource.PREVIEW
    assert camera.handle is preview
    assert opened_sources == [0, "udp://preview"]


def test_exam_checks_do_not_repair_browser_from_request_thread():
    class StoppedKiosk:
        is_running = False

        def relaunch(self):
            pytest.fail("reparo deve ficar restrito ao browser guard")

    manager, *_ = _make_manager(identify_results=[], engine_states=[], frames=[])
    manager._kiosk = StoppedKiosk()
    manager._browser_ready = False
    manager._mode = StationMode.SESSION

    checks = manager.get_exam_checks()

    chromium = next(check for check in checks["checks"] if check["key"] == "chromium")
    assert chromium["state"] == "fail"


def test_exam_checks_allow_confirmation_before_browser_starts():
    manager, *_ = _make_manager(identify_results=[], engine_states=[], frames=[])
    manager.enter_exam_mode()
    manager._camera.open_device()
    manager._identified_student_id = "alice01"

    checks = manager.get_exam_checks()

    chromium = next(check for check in checks["checks"] if check["key"] == "chromium")
    assert chromium["state"] == "pending"
    assert checks["ready"] is True


def test_session_manager_reconnects_stale_preview():
    direct_camera = FakeCamera(["identify-frame"])
    initial_preview = FakeCamera(["preview-open-frame"])
    recovered_preview = RepeatingCamera("recovered-open-frame")
    opened_sources = []

    def video_capture_factory(source):
        opened_sources.append(source)
        if isinstance(source, str):
            return initial_preview if opened_sources.count(source) == 1 else recovered_preview
        return direct_camera

    manager, _recognizer, _engine, _capture, _uploader, _kiosk, _overlay, _lockdown, _camera = _make_manager(
        identify_results=[
            IdentifyResult(
                status=IdentifyStatus.MATCH,
                student_id="123",
                student_name="Alice",
                confidence=0.9,
            )
        ],
        engine_states=[ProctorState.NORMAL],
        frames=[],
        video_capture_factory=video_capture_factory,
    )
    manager.update_config(turma_id="ES2025-T1")
    manager.start_session()
    manager._recover_preview_camera()

    assert opened_sources.count(manager._capture.preview_url) >= 2
    assert initial_preview.released is True
    assert manager._camera.is_open is True
    manager.stop_session(reason="done")


def test_session_manager_generated_session_id_includes_student_name(monkeypatch):
    monkeypatch.setattr("src.core.session.time.strftime", lambda _fmt: "20260426_160000")

    manager, _recognizer, _engine, _capture, _uploader, _kiosk, _overlay, _lockdown, _camera = _make_manager(
        identify_results=[
            IdentifyResult(
                status=IdentifyStatus.MATCH,
                student_id="123",
                student_name="Alice da Silva",
                confidence=0.9,
            )
        ],
        engine_states=[ProctorState.NORMAL],
        frames=["identify-frame", "loop-frame"],
    )

    manager.update_config(turma_id="ES2025-T1")
    started = manager.start_session()

    assert started["session_id"] == "ES2025-T1_alice_da_silva_20260426_160000"

    manager.stop_session(reason="done")


def test_session_manager_transitions_to_blocked_and_auto_unblocks():
    reidentify = EventReidentify()
    manager, _recognizer, engine, _capture, _uploader, kiosk, overlay, _lockdown, _camera = _make_manager(
        identify_results=[
            IdentifyResult(
                status=IdentifyStatus.MATCH,
                student_id="123",
                student_name="Alice",
                confidence=0.9,
            )
        ],
        engine_states=[ProctorState.BLOCKED, ProctorState.NORMAL],
        frames=["identify-frame", "blocked-frame", "resume-frame"],
        reidentify_fn=reidentify,
    )

    manager.update_config(turma_id="ES2025-T1")
    manager.start_session()
    assert reidentify.event.wait(timeout=1.0) is True

    for _ in range(50):
        if manager.state == SessionState.SESSION:
            break

    assert kiosk.blocked is True
    assert kiosk.unblocked is True
    assert overlay.blocked_shown == ["ABSENCE"]
    assert overlay.blocked_students == ["123"]
    assert overlay.blocked_timeouts == [20.0]
    assert overlay.hide_calls == 1
    assert engine.unblocked is True
    assert manager.state == SessionState.SESSION
    manager.stop_session(reason="done")


def test_periodic_identity_check_blocks_a_different_student(monkeypatch):
    manager, recognizer, engine, *_ = _make_manager(
        identify_results=[
            IdentifyResult(
                status=IdentifyStatus.MATCH,
                student_id="456",
                student_name="Bob",
                confidence=0.92,
            )
        ],
        engine_states=[],
        frames=[],
    )
    manager._recognizer = recognizer
    manager._engine = engine
    manager._runtime = SessionRuntime(
        session_id="session-1",
        turma_id="ES2025-T1",
        assessment="Quiz-01",
        timer_minutes=45,
        student_id="123",
        student_name="Alice",
        started_at=datetime.now(timezone.utc),
        state=SessionState.SESSION,
        prairielearn_url="https://pl.test/exam",
    )
    manager._last_identity_check_at = 100.0
    monkeypatch.setattr("src.core.session.time.monotonic", lambda: 110.0)

    manager._verify_session_identity("frame")

    assert engine.external_blocks == [
        (
            BlockReason.DIFFERENT_USER,
            {"expected_student_id": "123", "detected_student_id": "456"},
        )
    ]


def test_periodic_identity_check_runs_only_every_ten_seconds(monkeypatch):
    manager, recognizer, engine, *_ = _make_manager(
        identify_results=[
            IdentifyResult(
                status=IdentifyStatus.MATCH,
                student_id="123",
                student_name="Alice",
                confidence=0.9,
            )
        ],
        engine_states=[],
        frames=[],
    )
    manager._recognizer = recognizer
    manager._engine = engine
    manager._runtime = SessionRuntime(
        session_id="session-1",
        turma_id="ES2025-T1",
        assessment="Quiz-01",
        timer_minutes=45,
        student_id="123",
        student_name="Alice",
        started_at=datetime.now(timezone.utc),
        state=SessionState.SESSION,
        prairielearn_url="https://pl.test/exam",
    )
    manager._last_identity_check_at = 100.0
    now = {"value": 109.9}
    monkeypatch.setattr("src.core.session.time.monotonic", lambda: now["value"])

    manager._verify_session_identity("frame")
    assert recognizer.identify_calls == 0

    now["value"] = 110.0
    manager._verify_session_identity("frame")
    assert recognizer.identify_calls == 1
    assert engine.external_blocks == []


def test_session_manager_cancels_session_after_block_timeout():
    class TimedOutReidentify:
        def __init__(self):
            self.event = threading.Event()

        def __call__(self, **_kwargs):
            self.event.set()
            return False

    reidentify = TimedOutReidentify()
    manager, _recognizer, engine, _capture, _uploader, kiosk, overlay, _lockdown, _camera = _make_manager(
        identify_results=[
            IdentifyResult(
                status=IdentifyStatus.MATCH,
                student_id="123",
                student_name="Alice",
                confidence=0.9,
            )
        ],
        engine_states=[ProctorState.BLOCKED],
        frames=["identify-frame", "blocked-frame"],
        reidentify_fn=reidentify,
    )
    manager.update_config(turma_id="ES2025-T1")

    manager.start_session()

    assert reidentify.event.wait(timeout=1.0) is True
    for _ in range(50):
        if manager.state == SessionState.IDLE:
            break

    assert engine.cancelled_timeouts == [20.0]
    assert manager._last_session is not None
    assert manager._last_session.notes["stop_reason"] == "block_timeout"
    assert manager.dashboard_session_payload()["status"] == "CANCELLED_TIMEOUT"
    assert kiosk.stopped is True


def test_session_manager_manual_unblock():
    manager, _recognizer, engine, _capture, _uploader, kiosk, overlay, _lockdown, _camera = _make_manager(
        identify_results=[
            IdentifyResult(
                status=IdentifyStatus.MATCH,
                student_id="123",
                student_name="Alice",
                confidence=0.9,
            )
        ],
        engine_states=[ProctorState.NORMAL],
        frames=["identify-frame", "loop-frame"],
    )

    manager.update_config(turma_id="ES2025-T1")
    manager.start_session()
    manager._state = SessionState.BLOCKED
    manager._block_handled = True
    manager._runtime.block_reason = "ABSENCE"
    result = manager.unblock_session()

    assert result["state"] == SessionState.SESSION.value
    assert engine.unblocked is True
    assert kiosk.unblocked is True
    assert overlay.hide_calls == 1
    manager.stop_session(reason="done")


def test_exam_checks_expose_absence_block_and_presence_failure():
    manager, *_ = _make_manager(identify_results=[], engine_states=[], frames=[])
    manager._runtime = SessionRuntime(
        session_id="session-1",
        turma_id="ES2025-T1",
        assessment="Quiz-01",
        timer_minutes=45,
        student_id="123",
        student_name="Alice",
        started_at=datetime.now(timezone.utc),
        state=SessionState.BLOCKED,
        prairielearn_url="https://pl.test/exam",
        block_reason="ABSENCE",
    )
    manager._state = SessionState.BLOCKED
    manager._browser_ready = True
    manager._camera.open_device()

    payload = manager.get_exam_checks()
    checks = {check["key"]: check for check in payload["checks"]}

    assert payload["state"] == "BLOCKED"
    assert payload["block_reason"] == "ABSENCE"
    assert 0 < payload["seconds_remaining"] <= 45 * 60
    assert payload["ready"] is False
    assert checks["session"] == {
        "key": "session",
        "label": "Sessão bloqueada: ausência detectada",
        "state": "fail",
    }
    assert checks["presence"]["label"] == "Aluno ausente"
    assert checks["presence"]["state"] == "fail"
    assert checks["faces"]["state"] == "fail"


def test_session_manager_resets_to_idle_after_identification_failure():
    manager, _recognizer, _engine, _capture, _uploader, _kiosk, _overlay, _lockdown, camera = _make_manager(
        identify_results=[
            IdentifyResult(status=IdentifyStatus.NO_FACE),
            IdentifyResult(status=IdentifyStatus.NO_FACE),
            IdentifyResult(status=IdentifyStatus.NO_FACE),
        ],
        engine_states=[],
        frames=["frame-1", "frame-2", "frame-3"],
    )

    manager.update_config(turma_id="ES2025-T1")

    with pytest.raises(SessionError):
        manager.start_session()

    assert manager.state == SessionState.IDLE
    assert manager.get_session() is None
    assert camera.released is True


def test_session_manager_resets_to_idle_if_capture_start_fails():
    class FailingCapture(FakeCapture):
        def start(self):
            self.started = True
            raise RuntimeError("FFmpeg do stream 'webcam' encerrou no start (código 234)")

    manager, _recognizer, _engine, _capture, uploader, _kiosk, _overlay, _lockdown, camera = _make_manager(
        identify_results=[
            IdentifyResult(
                status=IdentifyStatus.MATCH,
                student_id="123",
                student_name="Alice",
                confidence=0.9,
            )
        ],
        engine_states=[],
        frames=["identify-frame"],
    )
    failing_capture = FailingCapture()
    manager._capture_factory = lambda _session_id: failing_capture

    manager.update_config(turma_id="ES2025-T1")

    with pytest.raises(RuntimeError, match="webcam"):
        manager.start_session()

    assert manager.state == SessionState.IDLE
    assert manager.get_session() is None
    assert failing_capture.started is True
    assert failing_capture.stopped is True
    assert uploader.started is True
    assert uploader.stopped is True
    assert camera.released is True


def test_session_manager_resets_to_idle_if_preview_camera_cannot_open():
    direct_camera = FakeCamera(["identify-frame"])

    class ClosedPreviewCamera:
        released = False

        def isOpened(self):
            return False

        def read(self):
            return False, None

        def release(self):
            self.released = True

    def video_capture_factory(source):
        if isinstance(source, str):
            return ClosedPreviewCamera()
        return direct_camera

    manager, _recognizer, _engine, capture, uploader, _kiosk, _overlay, _lockdown, _camera = _make_manager(
        identify_results=[
            IdentifyResult(
                status=IdentifyStatus.MATCH,
                student_id="123",
                student_name="Alice",
                confidence=0.9,
            )
        ],
        engine_states=[],
        frames=[],
        video_capture_factory=video_capture_factory,
    )

    manager.update_config(turma_id="ES2025-T1")

    with pytest.raises(SessionError, match="preview local"):
        manager.start_session()

    assert manager.state == SessionState.IDLE
    assert manager.get_session() is None
    assert capture.started is True
    assert capture.stopped is True
    assert uploader.started is True
    assert uploader.stopped is True
    assert direct_camera.released is True


def test_exam_mode_waiting_overlay_and_autostart_flow():
    manager, _recognizer, _engine, _capture, _uploader, kiosk, overlay, lockdown, _camera = _make_manager(
        identify_results=[
            IdentifyResult(
                status=IdentifyStatus.MATCH,
                student_id="123",
                student_name="Alice",
                confidence=0.9,
            )
        ],
        engine_states=[ProctorState.NORMAL],
        frames=["identify-frame", "loop-frame"],
    )

    manager.update_config(turma_id="ES2025-T1", auto_start=True)
    status = manager.enter_exam_mode()

    assert status["mode"] == StationMode.WAITING_STUDENT.value
    assert status["station_status"] == StationMode.WAITING_STUDENT.value
    assert overlay.waiting_shown == [None]
    assert lockdown.enabled is True
    assert kiosk.started is False
    chromium = next(
        check for check in manager.get_exam_checks()["checks"] if check["key"] == "chromium"
    )
    assert chromium["state"] == "pending"

    worker = SessionAutoStartWorker(session_manager=manager, enabled=True)
    worker.run_once()

    assert manager.state in {SessionState.SESSION, SessionState.BLOCKED}
    assert overlay.waiting_hidden == 1
    manager.stop_session(reason="done")
    assert manager.mode == StationMode.WAITING_STUDENT
    assert lockdown.enabled is True
    assert lockdown.disabled is False


def test_manual_stop_keeps_waiting_overlay_without_prestarting_browser():
    manager, _recognizer, _engine, _capture, _uploader, kiosk, overlay, lockdown, _camera = _make_manager(
        identify_results=[
            IdentifyResult(
                status=IdentifyStatus.MATCH,
                student_id="123",
                student_name="Alice",
                confidence=0.9,
            )
        ],
        engine_states=[ProctorState.NORMAL],
        frames=["identify-frame", "loop-frame"],
    )
    manager.update_config(turma_id="ES2025-T1", auto_start=False)
    manager.enter_exam_mode()
    manager.start_session()

    manager.stop_session(reason="manual")

    assert manager.mode == StationMode.WAITING_STUDENT
    assert lockdown.enabled is True
    assert overlay.waiting_shown == [
        None,
        "Preparando sua avaliação...",
        "Preparando a próxima sessão...",
        None,
    ]
    assert kiosk.start_calls == 1
    assert kiosk.stopped is True
    assert _camera.released is True


def test_session_manager_requires_identity_confirmation_before_starting_components():
    confirmation_calls = []

    def confirm(student_id, student_name, timeout_sec):
        confirmation_calls.append((student_id, student_name, timeout_sec))
        return True

    manager, _recognizer, engine, capture, uploader, kiosk, _overlay, _lockdown, _camera = _make_manager(
        identify_results=[
            IdentifyResult(
                status=IdentifyStatus.MATCH,
                student_id="alice01",
                student_name="Alice Silva",
                confidence=0.9,
            )
        ],
        engine_states=[ProctorState.NORMAL],
        frames=["identify-frame", "loop-frame"],
        confirmation_fn=confirm,
    )
    manager.update_config(turma_id="ES2025-T1")

    manager.start_session()

    assert confirmation_calls == [("alice01", "Alice Silva", 20.0)]
    assert engine.started is True
    assert capture.started is True
    assert uploader.started is True
    assert kiosk.started is True
    manager.stop_session(reason="done")


def test_cancelled_identity_confirmation_returns_to_waiting_screen():
    manager, _recognizer, _engine, _capture, _uploader, kiosk, overlay, lockdown, camera = _make_manager(
        identify_results=[
            IdentifyResult(
                status=IdentifyStatus.MATCH,
                student_id="alice01",
                student_name="Alice Silva",
                confidence=0.9,
            )
        ],
        engine_states=[],
        frames=["identify-frame"],
        confirmation_fn=lambda _student_id, _student_name, _timeout_sec: False,
    )
    manager.update_config(turma_id="ES2025-T1", auto_start=True)
    manager.enter_exam_mode()

    with pytest.raises(SessionError, match="cancelada ou expirada"):
        manager.start_session()

    assert manager.state == SessionState.IDLE
    assert manager.mode == StationMode.WAITING_STUDENT
    assert overlay.waiting_shown == [None]
    assert lockdown.enabled is True
    assert camera.released is True
    assert kiosk.start_calls == 0


def test_pre_exam_confirmation_response_requires_pending_confirmation():
    manager, *_ = _make_manager(identify_results=[], engine_states=[], frames=[])

    with pytest.raises(SessionError, match="pendente"):
        manager.respond_to_pre_exam_confirmation(accepted=True)


def test_camera_preview_encodes_latest_frame(monkeypatch):
    manager, *_ = _make_manager(identify_results=[], engine_states=[], frames=[])
    manager._latest_camera_frame = "camera-frame"

    class Encoded:
        def tobytes(self):
            return b"jpeg-data"

    monkeypatch.setattr("src.core.session.cv2.imencode", lambda *_args: (True, Encoded()))

    assert manager.get_camera_preview_jpeg() == b"jpeg-data"


def test_preview_recovery_keeps_last_frame_visible():
    manager, *_rest, camera = _make_manager(
        identify_results=[],
        engine_states=[],
        frames=[],
    )
    manager._latest_camera_frame = "last-preview-frame"
    manager._last_camera_frame_at = 1.0
    manager._camera.open_device()

    manager._release_camera(clear_preview=False)

    assert camera.released is True
    assert manager._latest_camera_frame == "last-preview-frame"
    assert manager._last_camera_frame_at == 1.0


def test_preview_watchdog_does_not_reconnect_during_camera_read(monkeypatch):
    class StopAfterOneIteration:
        def __init__(self):
            self.calls = 0

        def wait(self, _timeout):
            self.calls += 1
            return self.calls > 1

    manager, *_ = _make_manager(identify_results=[], engine_states=[], frames=[])
    manager._capture = FakeCapture()
    manager._camera_read_active.set()
    manager._preview_watchdog_stop = StopAfterOneIteration()

    monkeypatch.setattr(manager, "_preview_is_stale", lambda **_kwargs: True)
    monkeypatch.setattr(manager, "_recover_preview_camera", lambda: pytest.fail("não deve reconectar"))

    manager._preview_watchdog_loop()


def test_browser_guard_blocks_screen_when_browser_is_not_running():
    class StoppedKiosk:
        is_running = False

        def relaunch(self):
            return False

    manager, _recognizer, _engine, _capture, _uploader, _kiosk, overlay, _lockdown, _camera = _make_manager(
        identify_results=[],
        engine_states=[],
        frames=[],
    )
    manager._kiosk = StoppedKiosk()

    assert manager._ensure_exam_browser_fullscreen() is False
    assert overlay.blocked_shown == ["BROWSER_EXIT"]


def test_prepare_exam_mode_does_not_show_waiting_overlay_until_enter():
    manager, _recognizer, _engine, _capture, _uploader, _kiosk, overlay, lockdown, camera = _make_manager(
        identify_results=[],
        engine_states=[],
        frames=[],
    )

    prepare_status = manager.prepare_exam_mode()

    assert prepare_status["mode"] == StationMode.EXAM_READY.value
    assert overlay.waiting_shown == []

    enter_status = manager.enter_exam_mode()

    assert enter_status["mode"] == StationMode.WAITING_STUDENT.value
    assert overlay.waiting_shown == [None]
    assert manager._camera.is_open is False

    exit_status = manager.exit_exam_mode()

    assert exit_status["mode"] == StationMode.MAINTENANCE.value
    assert lockdown.disabled is True


def test_recover_exam_mode_restores_idle_maintenance_and_lockdown():
    manager, _recognizer, _engine, _capture, _uploader, _kiosk, overlay, lockdown, _camera = _make_manager(
        identify_results=[],
        engine_states=[],
        frames=[],
    )

    manager.update_config(turma_id="ES2025-T1")
    manager.enter_exam_mode()

    status = manager.recover_exam_mode()

    assert status["state"] == SessionState.IDLE.value
    assert status["mode"] == StationMode.MAINTENANCE.value
    assert overlay.waiting_hidden == 1
    assert lockdown.disabled is True


def test_autostart_worker_starts_session_when_enabled_and_idle():
    class FakeManager:
        def __init__(self):
            self.state = SessionState.IDLE
            self.mode = StationMode.WAITING_STUDENT
            self.next_config = type("Cfg", (), {"auto_start": True, "turma_id": "ES2025-T1"})()
            self.calls = 0

        def start_session(self):
            self.calls += 1

    manager = FakeManager()
    worker = SessionAutoStartWorker(session_manager=manager, enabled=True)
    worker.run_once()

    assert manager.calls == 1


def test_autostart_worker_ignores_disabled_or_busy_states():
    class FakeManager:
        def __init__(self, *, auto_start: bool, state: SessionState, mode: StationMode = StationMode.WAITING_STUDENT):
            self.state = state
            self.mode = mode
            self.next_config = type("Cfg", (), {"auto_start": auto_start, "turma_id": "ES2025-T1"})()
            self.calls = 0

        def start_session(self):
            self.calls += 1

    disabled = FakeManager(auto_start=False, state=SessionState.IDLE)
    busy = FakeManager(auto_start=True, state=SessionState.SESSION)
    maintenance = FakeManager(
        auto_start=True,
        state=SessionState.IDLE,
        mode=StationMode.MAINTENANCE,
    )

    SessionAutoStartWorker(session_manager=disabled, enabled=True).run_once()
    SessionAutoStartWorker(session_manager=busy, enabled=True).run_once()
    SessionAutoStartWorker(session_manager=maintenance, enabled=True).run_once()

    assert disabled.calls == 0
    assert busy.calls == 0
    assert maintenance.calls == 0


def test_autostart_worker_does_not_restart_same_student_after_completion():
    manager, _recognizer, _engine, _capture, _uploader, _kiosk, overlay, _lockdown, _camera = _make_manager(
        identify_results=[
            IdentifyResult(
                status=IdentifyStatus.MATCH,
                student_id="123",
                student_name="Alice",
                confidence=0.9,
            ),
        ],
        engine_states=[ProctorState.NORMAL],
        frames=["identify-frame"],
    )

    manager.update_config(
        turma_id="ES2025-T1",
        assessment="Quiz-03",
        auto_start=True,
        allow_repeat_attempts=False,
    )
    manager._last_session = SessionRuntime(
        session_id="completed",
        turma_id="ES2025-T1",
        assessment="Quiz-03",
        timer_minutes=45,
        student_id="123",
        student_name="Alice",
        started_at=datetime.now(timezone.utc),
        stopped_at=datetime.now(timezone.utc),
        state=SessionState.IDLE,
        prairielearn_url="https://prairielearn.org/pl",
    )
    manager.enter_exam_mode()

    worker = SessionAutoStartWorker(session_manager=manager, enabled=True)
    worker.run_once()

    assert manager.state == SessionState.IDLE
    assert manager.get_session()["student_id"] == "123"
    assert overlay.waiting_shown == [None]
    assert overlay.waiting_hidden == 0
    assert _camera.released is True


def test_autostart_allows_same_student_again_by_default():
    manager, _recognizer, _engine, _capture, _uploader, _kiosk, _overlay, _lockdown, _camera = _make_manager(
        identify_results=[
            IdentifyResult(
                status=IdentifyStatus.MATCH,
                student_id="123",
                student_name="Alice",
                confidence=0.9,
            ),
        ],
        engine_states=[ProctorState.NORMAL],
        frames=["identify-frame", "loop-frame"],
    )
    manager.update_config(turma_id="ES2025-T1", assessment="Quiz-03", auto_start=True)
    manager._last_session = SessionRuntime(
        session_id="completed",
        turma_id="ES2025-T1",
        assessment="Quiz-03",
        timer_minutes=45,
        student_id="123",
        student_name="Alice",
        started_at=datetime.now(timezone.utc),
        stopped_at=datetime.now(timezone.utc),
        state=SessionState.IDLE,
        prairielearn_url="https://prairielearn.org/pl",
    )
    manager.enter_exam_mode()

    SessionAutoStartWorker(session_manager=manager, enabled=True).run_once()

    assert manager.state in {SessionState.SESSION, SessionState.BLOCKED}
    manager.stop_session(reason="done")


def test_failed_identification_releases_waiting_camera_between_attempts():
    manager, _recognizer, _engine, _capture, _uploader, _kiosk, overlay, _lockdown, camera = _make_manager(
        identify_results=[],
        engine_states=[],
        frames=[],
    )

    manager.update_config(
        turma_id="ES2025-T1",
        assessment="Quiz-03",
        auto_start=True,
    )
    manager.enter_exam_mode()

    worker = SessionAutoStartWorker(session_manager=manager, enabled=True)
    worker.run_once()
    worker.run_once()

    assert manager.state == SessionState.IDLE
    assert manager.mode == StationMode.WAITING_STUDENT
    assert overlay.waiting_shown == [None]
    assert overlay.waiting_hidden == 0
    assert camera.released is True


# ── Contratos de estado e de config ─────────────────────────────────────────
#
# Estes testes não exercitam comportamento: eles travam as fronteiras onde antes
# um campo se perdia em silêncio. Falham por construção quando alguém adiciona
# um campo/estado sem decidir seu destino.


def test_session_module_reexports_canonical_states():
    """Os estados são definidos em src/core/states.py, não duplicados aqui."""
    assert SessionState is CanonicalSessionState
    assert StationMode is CanonicalStationMode


@pytest.mark.parametrize(
    ("state", "mode", "expected"),
    [
        (SessionState.IDLE, StationMode.MAINTENANCE, "IDLE"),
        (SessionState.IDLE, StationMode.EXAM_READY, "EXAM_READY"),
        (SessionState.IDLE, StationMode.WAITING_STUDENT, "WAITING_STUDENT"),
        # Sessão em andamento vence o modo da estação.
        (SessionState.SESSION, StationMode.WAITING_STUDENT, "SESSION"),
        (SessionState.BLOCKED, StationMode.SESSION, "BLOCKED"),
        (SessionState.UPLOADING, StationMode.SESSION, "UPLOADING"),
        (SessionState.IDENTIFYING, StationMode.WAITING_STUDENT, "IDENTIFYING"),
    ],
)
def test_derive_station_status_collapses_state_and_mode(state, mode, expected):
    assert derive_station_status(state, mode) == expected


def test_dashboard_config_payload_fields_are_all_classified():
    """Todo campo de ExamConfigPayload tem destino declarado.

    Sem isto, um campo novo no dashboard simplesmente não chega à estação e
    nada acusa.
    """
    classified = (
        set(DASHBOARD_CONFIG_FIELD_MAP)
        | set(DASHBOARD_PROCTOR_FIELD_CASTS)
        | set(DASHBOARD_ROUTING_FIELDS)
    )

    assert set(ExamConfigPayload.model_fields) == classified


def test_dashboard_config_field_map_targets_real_session_config_fields():
    """Os destinos do mapa existem em SessionConfig — pega typo no alvo."""
    assert set(DASHBOARD_CONFIG_FIELD_MAP.values()) <= _session_config_field_names()


def test_config_update_request_only_exposes_session_config_fields():
    """POST /config não pode oferecer campo que update_config agora rejeita."""
    assert set(ConfigUpdateRequest.model_fields) <= _session_config_field_names()


def test_update_config_rejects_unknown_field():
    manager, *_ = _make_manager(identify_results=[], engine_states=[], frames=[])

    # "turma" é o nome do lado do dashboard; o SessionConfig usa "turma_id".
    # Antes isso era descartado em silêncio e a turma nunca era aplicada.
    with pytest.raises(SessionError, match="turma"):
        manager.update_config(turma="ES2025-T1")

    assert manager.next_config.turma_id is None


def test_update_config_treats_none_as_no_change():
    manager, *_ = _make_manager(identify_results=[], engine_states=[], frames=[])

    manager.update_config(turma_id="ES2025-T1", assessment="Quiz-03")
    manager.update_config(turma_id=None, assessment="Quiz-04")

    config = manager.next_config
    assert config.turma_id == "ES2025-T1"
    assert config.assessment == "Quiz-04"


def test_apply_dashboard_config_maps_renamed_and_threshold_fields():
    manager, *_ = _make_manager(identify_results=[], engine_states=[], frames=[])

    config = manager.apply_dashboard_config(
        {
            "turma": "ES2025-T1",
            "assessment": "Quiz-03",
            "timer_minutes": 30,
            "prairielearn_url": "https://pl.test/exam",
            "allowlist": ["example.edu"],
            "auto_start": True,
            "s3_prefix": "ES2025-T1/quiz-03",
            "target_station_ids": ["nuc-01"],
            "gaze_h_threshold": 0.4,
            "gaze_duration_sec": 4.0,
            "absence_timeout_sec": 6.0,
            "multi_face_block": False,
        }
    )

    assert config.turma_id == "ES2025-T1"
    assert config.assessment == "Quiz-03"
    assert config.timer_minutes == 30
    assert config.prairielearn_url == "https://pl.test/exam"
    assert config.allowlist == ["example.edu"]
    assert config.auto_start is True
    assert config.s3_prefix == "ES2025-T1/quiz-03"

    assert manager._proctor_cfg.gaze_h_threshold == 0.4
    assert manager._proctor_cfg.gaze_duration_sec == 4.0
    assert manager._proctor_cfg.absence_timeout_sec == 6.0
    assert manager._proctor_cfg.multi_face_block is False


def test_apply_dashboard_config_ignores_missing_optional_fields():
    manager, *_ = _make_manager(identify_results=[], engine_states=[], frames=[])
    manager.update_config(turma_id="ES2025-T1", timer_minutes=45)
    before = manager._proctor_cfg.gaze_h_threshold

    config = manager.apply_dashboard_config({"assessment": "Quiz-09"})

    assert config.assessment == "Quiz-09"
    assert config.turma_id == "ES2025-T1"
    assert config.timer_minutes == 45
    assert manager._proctor_cfg.gaze_h_threshold == before


def test_session_config_is_restored_from_disk(tmp_path):
    app_config = AppConfig(
        data_dir=tmp_path,
        persist_session_config=True,
        restore_exam_mode_on_startup=False,
    )
    manager = SessionManager(app_config=app_config)
    manager.update_config(
        turma_id="ES2025-T1",
        assessment="Quiz-03",
        timer_minutes=35,
        prairielearn_url="https://pl.test/exam",
        auto_start=True,
    )

    restored = SessionManager(app_config=app_config).next_config

    assert restored.turma_id == "ES2025-T1"
    assert restored.assessment == "Quiz-03"
    assert restored.timer_minutes == 35
    assert restored.prairielearn_url == "https://pl.test/exam"
    assert restored.auto_start is True


def test_invalid_persisted_config_falls_back_to_defaults(tmp_path):
    (tmp_path / "station-config.json").write_text("not-json", encoding="utf-8")
    manager = SessionManager(
        app_config=AppConfig(
            data_dir=tmp_path,
            persist_session_config=True,
            restore_exam_mode_on_startup=False,
        )
    )

    assert manager.next_config.turma_id is None
    assert manager.next_config.assessment == "Prova"


def test_startup_restores_exam_mode_and_enables_autostart():
    manager, _recognizer, _engine, _capture, _uploader, kiosk, overlay, lockdown, _camera = _make_manager(
        identify_results=[],
        engine_states=[],
        frames=[],
    )
    manager._app_cfg.restore_exam_mode_on_startup = True
    manager.update_config(turma_id="ES2025-T1", assessment="Quiz-03", auto_start=False)

    status = manager.restore_exam_mode_on_startup()

    assert status["mode"] == StationMode.WAITING_STUDENT.value
    assert manager.next_config.auto_start is True
    assert kiosk.started is False
    assert overlay.waiting_shown == [None]
    assert lockdown.enabled is True


# ── Política de teardown ─────────────────────────────────────────────────────
#
# A regra de "o que sobrevive ao encerramento" era três booleanos calculados no
# except de start_session, cobertos só de forma indireta. Agora tem teste direto.


def test_full_teardown_keeps_nothing():
    policy = ShutdownPolicy.full_teardown()
    assert (policy.keep_lockdown, policy.keep_waiting_overlay, policy.keep_camera) == (
        False,
        False,
        False,
    )


@pytest.mark.parametrize(
    ("mode", "session_started", "expected"),
    [
        # Em WAITING_STUDENT sem sessão criada: preserva overlay e lockdown,
        # mas libera a câmera até o próximo clique do aluno.
        (StationMode.WAITING_STUDENT, False, (True, True, False)),
        # Sessão já existia quando falhou: lockdown fica (segue em modo prova),
        # mas overlay/câmera são reconstruídos.
        (StationMode.WAITING_STUDENT, True, (True, False, False)),
        # Fora do modo prova, nada sobrevive.
        (StationMode.SESSION, True, (False, False, False)),
        (StationMode.EXAM_READY, False, (False, False, False)),
        (StationMode.MAINTENANCE, False, (False, False, False)),
    ],
)
def test_failed_start_policy(mode, session_started, expected):
    policy = ShutdownPolicy.for_failed_start(mode=mode, session_started=session_started)
    assert (
        policy.keep_lockdown,
        policy.keep_waiting_overlay,
        policy.keep_camera,
    ) == expected


@pytest.mark.parametrize(
    ("mode", "auto_start", "reason", "exam_mode_active", "expected"),
    [
        # Sessão encerrada com auto-start ligado: volta a esperar aluno, mantém
        # o lockdown de pé.
        (StationMode.SESSION, True, "manual", True, (True, True, False)),
        (StationMode.SESSION, True, "dashboard_command", True, (True, True, False)),
        # Saída do modo prova nunca preserva: o operador quer manutenção.
        (StationMode.SESSION, True, EXIT_EXAM_MODE_REASON, True, (False, False, False)),
        # Mesmo sem auto-start, o stop manual mantém a estação segura.
        (StationMode.SESSION, False, "manual", True, (True, True, False)),
        (StationMode.SESSION, True, "manual", False, (False, False, False)),
        # Sem sessão ativa não há o que preservar.
        (StationMode.WAITING_STUDENT, True, "manual", False, (False, False, False)),
        (StationMode.MAINTENANCE, True, "manual", False, (False, False, False)),
    ],
)
def test_stopped_session_policy(mode, auto_start, reason, exam_mode_active, expected):
    policy = ShutdownPolicy.for_stopped_session(
        mode=mode, auto_start=auto_start, reason=reason, exam_mode_active=exam_mode_active
    )
    assert (policy.keep_lockdown, policy.keep_waiting_overlay, policy.keep_camera) == expected


@pytest.mark.asyncio
async def test_api_routes_expose_phase5_flow():
    manager, _recognizer, _engine, _capture, _uploader, _kiosk, _overlay, _lockdown, _camera = _make_manager(
        identify_results=[
            IdentifyResult(
                status=IdentifyStatus.MATCH,
                student_id="123",
                student_name="Alice",
                confidence=0.9,
            )
        ],
        engine_states=[ProctorState.NORMAL],
        frames=["identify-frame", "loop-frame"],
    )
    app = create_app(session_manager=manager)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        response = await client.post("/config", json={"turma_id": "ES2025-T1"})
        assert response.status_code == 200
        assert response.json()["config"]["turma_id"] == "ES2025-T1"

        enter = await client.post("/exam-mode/enter")
        assert enter.status_code == 200
        assert enter.json()["mode"] == StationMode.WAITING_STUDENT.value

        health = await client.get("/health")
        assert health.status_code == 200
        assert health.json()["camera_ok"] is True
        assert health.json()["s3_ok"] is True

        start = await client.post("/pre-exam/start")
        assert start.status_code == 201
        assert start.json()["student_name"] == "Alice"

        status_view = await client.get("/status")
        assert status_view.status_code == 200
        assert status_view.json()["state"] in {SessionState.SESSION.value, SessionState.BLOCKED.value}

        session_view = await client.get("/session")
        assert session_view.status_code == 200
        assert session_view.json()["session"]["turma_id"] == "ES2025-T1"

        stop = await client.post("/session/stop")
        assert stop.status_code == 200
        assert stop.json()["state"] == SessionState.IDLE.value

        exit_mode = await client.post("/exam-mode/exit")
        assert exit_mode.status_code == 200
        assert exit_mode.json()["mode"] == StationMode.MAINTENANCE.value

        recover = await client.post("/exam-mode/recover")
        assert recover.status_code == 200
        assert recover.json()["mode"] == StationMode.MAINTENANCE.value
