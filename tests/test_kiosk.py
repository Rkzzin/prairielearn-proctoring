from __future__ import annotations

import signal
import subprocess
from types import SimpleNamespace

from src.core.models import IdentifyResult, IdentifyStatus
from src.kiosk.chromium import ChromiumKiosk
from src.kiosk.lockdown import Lockdown
from src.kiosk.overlay import SessionOverlay
from src.kiosk.reidentify import run_reidentify


class DummyProc:
    def __init__(self, pid: int = 4321):
        self.pid = pid
        self.signals: list[int] = []
        self.terminated = False
        self.killed = False
        self.returncode = None

    def poll(self):
        return self.returncode

    def send_signal(self, sig: int) -> None:
        self.signals.append(sig)

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0

    def wait(self, timeout: float | None = None) -> int:
        return 0

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


class FakeCapture:
    def __init__(self, frames: list[object]):
        self._frames = list(frames)

    def read(self):
        if not self._frames:
            return False, None
        return True, self._frames.pop(0)


class FakeRecognizer:
    def __init__(self, results: list[IdentifyResult]):
        self._results = list(results)

    def identify(self, frame):
        if not self._results:
            return IdentifyResult(status=IdentifyStatus.NO_FACE)
        return self._results.pop(0)


class FakeClock:
    def __init__(self, values: list[float]):
        self._values = list(values)
        self._last = values[-1] if values else 0.0

    def __call__(self) -> float:
        if self._values:
            self._last = self._values.pop(0)
        return self._last


def test_chromium_start_adds_kiosk_flags(monkeypatch):
    proc = DummyProc()
    popen_calls = []

    def fake_popen(cmd, env, stdout, stderr):
        popen_calls.append((cmd, env, stdout, stderr))
        return proc

    monkeypatch.setattr("src.kiosk.chromium._find_chromium", lambda: "/usr/bin/chromium")
    monkeypatch.setattr("src.kiosk.chromium.subprocess.Popen", fake_popen)
    monkeypatch.setattr(ChromiumKiosk, "_disable_gnome_extensions", lambda self: None)
    monkeypatch.setattr(ChromiumKiosk, "_force_fullscreen_by_pid", lambda self: None)

    kiosk = ChromiumKiosk(display=":9")
    kiosk.start("https://example.com/exam")

    assert kiosk.is_running is True
    cmd, env, *_ = popen_calls[0]
    assert cmd[:3] == ["/usr/bin/chromium", "--kiosk", "--start-fullscreen"]
    assert env["DISPLAY"] == ":9"
    assert cmd[-1] == "https://example.com/exam"


def test_chromium_block_and_unblock_send_signals(monkeypatch):
    monkeypatch.setattr(ChromiumKiosk, "_restore_gnome_extensions", lambda self: None)

    kiosk = ChromiumKiosk()
    proc = DummyProc()
    kiosk._proc = proc

    kiosk.block()
    kiosk.unblock()

    assert proc.signals == [signal.SIGSTOP, signal.SIGCONT]


def test_disable_gnome_extensions_is_best_effort_when_binary_missing(monkeypatch):
    monkeypatch.setattr("src.kiosk.chromium.shutil.which", lambda name: None)

    kiosk = ChromiumKiosk()
    kiosk._disable_gnome_extensions()

    assert kiosk._disabled_extensions == []


def test_disable_and_restore_only_extensions_seen_enabled(monkeypatch):
    commands = []

    def fake_which(name):
        return "/usr/bin/gnome-extensions"

    def fake_run(cmd, env, capture_output, timeout, check=False):
        commands.append(cmd)
        if cmd[:3] == ["gnome-extensions", "list", "--enabled"]:
            return SimpleNamespace(stdout=b"ubuntu-dock@ubuntu.com\n")
        return SimpleNamespace(stdout=b"")

    monkeypatch.setattr("src.kiosk.chromium.shutil.which", fake_which)
    monkeypatch.setattr("src.kiosk.chromium.subprocess.run", fake_run)

    kiosk = ChromiumKiosk(display=":7")
    kiosk._disable_gnome_extensions()
    assert kiosk._disabled_extensions == ["ubuntu-dock@ubuntu.com"]

    kiosk._restore_gnome_extensions()

    assert commands == [
        ["gnome-extensions", "list", "--enabled"],
        ["gnome-extensions", "disable", "ubuntu-dock@ubuntu.com"],
        ["gnome-extensions", "enable", "ubuntu-dock@ubuntu.com"],
    ]
    assert kiosk._disabled_extensions == []


def test_run_reidentify_succeeds_after_required_consecutive_matches(monkeypatch):
    monkeypatch.setattr(
        "src.kiosk.reidentify.time.time",
        FakeClock([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
    )

    recognizer = FakeRecognizer(
        [
            IdentifyResult(status=IdentifyStatus.NO_FACE),
            IdentifyResult(
                status=IdentifyStatus.MATCH,
                student_id="s1",
                student_name="Alice",
                confidence=0.91,
            ),
            IdentifyResult(
                status=IdentifyStatus.MATCH,
                student_id="s1",
                student_name="Alice",
                confidence=0.93,
            ),
        ]
    )
    cap = FakeCapture([object(), object(), object()])

    ok = run_reidentify(
        recognizer=recognizer,
        cap=cap,
        expected_student_id="s1",
        timeout_sec=5.0,
        required_matches=2,
    )

    assert ok is True


def test_run_reidentify_resets_counter_on_wrong_student(monkeypatch):
    monkeypatch.setattr(
        "src.kiosk.reidentify.time.time",
        FakeClock([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
    )

    recognizer = FakeRecognizer(
        [
            IdentifyResult(
                status=IdentifyStatus.MATCH,
                student_id="s1",
                student_name="Alice",
                confidence=0.91,
            ),
            IdentifyResult(
                status=IdentifyStatus.MATCH,
                student_id="s2",
                student_name="Bob",
                confidence=0.88,
            ),
            IdentifyResult(
                status=IdentifyStatus.MATCH,
                student_id="s1",
                student_name="Alice",
                confidence=0.92,
            ),
            IdentifyResult(
                status=IdentifyStatus.MATCH,
                student_id="s1",
                student_name="Alice",
                confidence=0.94,
            ),
        ]
    )
    cap = FakeCapture([object(), object(), object(), object()])

    ok = run_reidentify(
        recognizer=recognizer,
        cap=cap,
        expected_student_id="s1",
        timeout_sec=5.0,
        required_matches=2,
    )

    assert ok is True


def test_run_reidentify_times_out(monkeypatch):
    monkeypatch.setattr(
        "src.kiosk.reidentify.time.time",
        FakeClock([0.0, 0.6, 1.2, 1.8, 2.4, 2.4]),
    )

    recognizer = FakeRecognizer(
        [
            IdentifyResult(status=IdentifyStatus.NO_FACE),
            IdentifyResult(status=IdentifyStatus.NO_MATCH),
            IdentifyResult(status=IdentifyStatus.MULTIPLE_FACES, face_count=2),
        ]
    )
    cap = FakeCapture([object(), object(), object()])

    ok = run_reidentify(
        recognizer=recognizer,
        cap=cap,
        expected_student_id="s1",
        timeout_sec=2.0,
        required_matches=2,
    )

    assert ok is False


def test_lockdown_enable_disable_tracks_state():
    lockdown = Lockdown(display=":3")

    assert lockdown.is_enabled is False

    lockdown.enable()
    assert lockdown.is_enabled is True

    lockdown.disable()
    assert lockdown.is_enabled is False


def test_session_overlay_starts_controls_and_blocked_overlay(monkeypatch):
    calls = []
    procs = [DummyProc(pid=10), DummyProc(pid=11)]

    def fake_popen(cmd, env, stdout, stderr):
        calls.append((cmd, env))
        return procs[len(calls) - 1]

    monkeypatch.setattr("src.kiosk.overlay.subprocess.Popen", fake_popen)

    overlay = SessionOverlay(display=":5", api_port=8123)
    overlay.start_controls()
    overlay.show_blocked("ABSENCE")

    assert calls[0][0][-4:] == ["--mode", "controls", "--stop-url", "http://127.0.0.1:8123/session/stop"]
    assert calls[0][1]["DISPLAY"] == ":5"
    assert calls[1][0][-4:] == ["--mode", "blocked", "--reason", "ABSENCE"]

    overlay.stop()
    assert procs[0].terminated is True
    assert procs[1].terminated is True
