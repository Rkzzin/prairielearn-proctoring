from __future__ import annotations

import threading
from collections import deque

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.server import create_app
from src.core.config import AppConfig, FaceConfig, ProctorConfig, RecorderConfig, S3Config
from src.core.models import IdentifyResult, IdentifyStatus
from src.core.session import SessionManager, SessionState
from src.proctor.engine import ProctorState


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


class FakeRecognizer:
    def __init__(self, identify_results):
        self.identify_results = deque(identify_results)
        self.loaded_turma = None

    def load_turma(self, turma_id):
        self.loaded_turma = turma_id

    def identify(self, _frame):
        if self.identify_results:
            return self.identify_results.popleft()
        return IdentifyResult(status=IdentifyStatus.NO_FACE)


class FakeEngine:
    def __init__(self, states):
        self.states = deque(states)
        self.started = False
        self.stopped = False
        self.unblocked = False
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


class FakeCapture:
    def __init__(self):
        self.started = False
        self.stopped = False
        self.frames = 0

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
        self.stopped = False
        self.blocked = False
        self.unblocked = False
        self.url = None

    def start(self, url):
        self.started = True
        self.url = url

    def stop(self):
        self.stopped = True

    def block(self):
        self.blocked = True

    def unblock(self):
        self.unblocked = True


class FakeLockdown:
    def __init__(self):
        self.enabled = False
        self.disabled = False

    def enable(self):
        self.enabled = True

    def disable(self):
        self.disabled = True


class FakeOverlay:
    def __init__(self):
        self.controls_started = False
        self.blocked_shown: list[str | None] = []
        self.hide_calls = 0
        self.stopped = False

    def start_controls(self):
        self.controls_started = True

    def show_blocked(self, reason=None):
        self.blocked_shown.append(reason)

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
        app_config=AppConfig(data_dir="/tmp/proctor-tests"),
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
        video_capture_factory=lambda _index: fake_camera,
        reidentify_fn=reidentify_fn or (lambda **_kwargs: True),
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
    assert overlay.hide_calls == 1
    assert engine.unblocked is True
    assert manager.state == SessionState.SESSION
    manager.stop_session(reason="done")


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

        health = await client.get("/health")
        assert health.status_code == 200
        assert health.json()["camera_ok"] is True
        assert health.json()["s3_ok"] is True

        start = await client.post("/session/start", json={"prairielearn_url": "https://pl.test/exam"})
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
