from __future__ import annotations

import json
import signal
import subprocess
from datetime import datetime
from types import SimpleNamespace

import pytest

from src.core.models import IdentifyResult, IdentifyStatus
from src.kiosk.chromium import ChromiumKiosk
from src.kiosk.lockdown import Lockdown
from src.kiosk.overlay import SessionOverlay
from src.kiosk.overlay_app import (
    _blocked_reason_message,
    _format_clock_time,
    _format_remaining_time,
    _show_preview_during_block,
    _status_summary,
    _violation_report_message,
)
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


def test_blocked_overlay_reports_absence_and_multiple_faces():
    expected = "Esta ocorrência foi registrada automaticamente para a equipe responsável."

    assert _violation_report_message("ABSENCE") == expected
    assert _violation_report_message("MULTI_FACE") == expected
    assert _violation_report_message("DIFFERENT_USER") == expected
    assert _violation_report_message("GAZE") is None


def test_blocked_overlay_hides_preview_for_proctoring_violations():
    assert _show_preview_during_block("ABSENCE") is False
    assert _show_preview_during_block("GAZE") is False
    assert _show_preview_during_block("MULTI_FACE") is False
    assert _show_preview_during_block("DIFFERENT_USER") is False
    assert _show_preview_during_block("BROWSER_EXIT") is True


def test_blocked_overlay_explains_absence():
    assert _blocked_reason_message("ABSENCE") == "Ausência detectada. Volte para a frente da câmera."


def test_blocked_overlay_explains_different_user():
    assert _blocked_reason_message("DIFFERENT_USER") == (
        "Usuário diferente detectado. O aluno autenticado deve retornar."
    )


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"state": "BLOCKED", "checks": []}, ("Avaliação pausada", "fail")),
        ({"state": "SESSION", "checks": [{"state": "fail"}]}, ("Atenção necessária", "fail")),
        ({"state": "SESSION", "checks": [{"state": "ok"}]}, ("Monitoramento ativo", "ok")),
        ({"state": "IDENTIFYING", "checks": [{"state": "pending"}]}, ("Preparando o ambiente", "pending")),
    ],
)
def test_compact_status_summary(payload, expected):
    assert _status_summary(payload) == expected


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (None, None),
        (0, "00:00"),
        (65, "01:05"),
        (3661, "01:01:01"),
    ],
)
def test_format_remaining_time(seconds, expected):
    assert _format_remaining_time(seconds) == expected


def test_format_clock_time():
    assert _format_clock_time(datetime(2026, 8, 21, 15, 42, 7)) == "15:42:07"


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


def test_chromium_start_adds_controlled_browser_flags(monkeypatch, tmp_path):
    proc = DummyProc()
    popen_calls = []

    def fake_popen(cmd, env, stdout, stderr):
        popen_calls.append((cmd, env, stdout, stderr))
        return proc

    monkeypatch.setattr("src.kiosk.chromium._find_chromium", lambda: "/usr/bin/chromium")
    monkeypatch.setattr("src.kiosk.chromium.subprocess.Popen", fake_popen)
    monkeypatch.setattr(ChromiumKiosk, "_apply_window_mode_by_pid", lambda self: None)

    extension_dir = tmp_path / "extension"
    policy_path = tmp_path / "managed-policy.json"
    policy_path.touch()
    kiosk = ChromiumKiosk(
        display=":9",
        profile_dir=tmp_path / "proctor-chromium-profile",
        extension_dir=extension_dir,
        cleanup_profile_on_stop=False,
        policy_paths=[policy_path],
        require_managed_policy=True,
    )
    kiosk.start("https://example.com/exam", allowlist=["prairielearn.org"])

    assert kiosk.is_running is True
    cmd, env, *_ = popen_calls[0]
    assert cmd[:2] == ["/usr/bin/chromium", "--start-fullscreen"]
    assert "--kiosk" not in cmd
    assert "--incognito" in cmd
    assert "--disable-extensions" not in cmd
    assert f"--load-extension={extension_dir}" in cmd
    assert f"--disable-extensions-except={extension_dir}" in cmd
    assert any(
        item.startswith(f"--user-data-dir={tmp_path / 'proctor-chromium-profile' / 'proctor-session-'}")
        for item in cmd
    )
    assert not any(item.startswith("--host-resolver-rules=") for item in cmd)
    assert env["DISPLAY"] == ":9"
    assert cmd[-1] == "https://example.com/exam"
    assert (extension_dir / "config.json").exists()
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    assert policy["URLBlocklist"] == ["*"]
    assert "https://example.com" in policy["URLAllowlist"]
    assert "https://prairielearn.org" in policy["URLAllowlist"]


def test_chromium_refuses_to_start_without_required_managed_policy(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.kiosk.chromium.subprocess.Popen",
        lambda *_args, **_kwargs: pytest.fail("Chromium nao deve iniciar sem policy"),
    )
    kiosk = ChromiumKiosk(
        profile_dir=tmp_path / "proctor-chromium-profile",
        extension_dir=tmp_path / "extension",
        policy_paths=[tmp_path / "missing-policy.json"],
        require_managed_policy=True,
    )

    with pytest.raises(RuntimeError, match="Policy gerenciada"):
        kiosk.start("https://example.com/exam")


def test_maximized_window_mode_does_not_wait_for_wmctrl(monkeypatch, tmp_path):
    kiosk = ChromiumKiosk(profile_dir=tmp_path / "profile", window_mode="maximized")
    kiosk._proc = DummyProc()

    monkeypatch.setattr("src.kiosk.chromium.shutil.which", lambda _name: "/usr/bin/wmctrl")
    monkeypatch.setattr(
        "src.kiosk.chromium.subprocess.run",
        lambda *_args, **_kwargs: pytest.fail("wmctrl não deve ser chamado em modo maximizado"),
    )

    kiosk._apply_window_mode_by_pid()


def test_fullscreen_window_lookup_falls_back_to_snap_chromium_window(monkeypatch, tmp_path):
    kiosk = ChromiumKiosk(profile_dir=tmp_path / "profile")
    kiosk._proc = DummyProc(pid=999)

    monkeypatch.setattr("src.kiosk.chromium.subprocess.run", lambda *_args, **_kwargs: SimpleNamespace(
        stdout=b"0x01200004  0 4321 chromium.profile.Chromium proctor PrairieLearn\n"
    ))

    assert kiosk._window_id_for_pid() == "0x01200004"


def _start_kiosk_capturing_cmd(monkeypatch, tmp_path, *, proxy_server):
    """Sobe o kiosk com um proxy fixo e devolve o argv passado ao Popen."""
    popen_calls = []

    def fake_popen(cmd, env, stdout, stderr):
        popen_calls.append(cmd)
        return DummyProc()

    monkeypatch.setattr("src.kiosk.chromium._find_chromium", lambda: "/usr/bin/chromium")
    monkeypatch.setattr("src.kiosk.chromium.subprocess.Popen", fake_popen)
    monkeypatch.setattr(ChromiumKiosk, "_apply_window_mode_by_pid", lambda self: None)
    monkeypatch.setattr(
        "src.kiosk.chromium.config",
        SimpleNamespace(proxy_server=proxy_server),
    )

    kiosk = ChromiumKiosk(
        display=":9",
        profile_dir=tmp_path / "proctor-chromium-profile",
        extension_dir=tmp_path / "extension",
        cleanup_profile_on_stop=False,
    )
    kiosk.start("https://example.com/exam", allowlist=["prairielearn.org"])
    return popen_calls[0]


def test_chromium_omits_proxy_flag_when_none_configured(monkeypatch, tmp_path):
    """Sem proxy configurado o flag não pode aparecer.

    Antes da guarda, um PROCTOR_APP_PROXY_SERVER ausente virava
    '--proxy-server=None' e quebrava toda a navegação da prova.
    """
    cmd = _start_kiosk_capturing_cmd(monkeypatch, tmp_path, proxy_server=None)

    assert not any(item.startswith("--proxy-server") for item in cmd)
    assert "--proxy-server=None" not in cmd
    assert cmd[-1] == "https://example.com/exam"


def test_chromium_omits_proxy_flag_when_configured_empty(monkeypatch, tmp_path):
    cmd = _start_kiosk_capturing_cmd(monkeypatch, tmp_path, proxy_server="")

    assert not any(item.startswith("--proxy-server") for item in cmd)
    assert cmd[-1] == "https://example.com/exam"


def test_chromium_passes_configured_proxy_before_url(monkeypatch, tmp_path):
    cmd = _start_kiosk_capturing_cmd(
        monkeypatch, tmp_path, proxy_server="http://proxy.test:443"
    )

    assert "--proxy-server=http://proxy.test:443" in cmd
    # A URL precisa continuar sendo o último argumento.
    assert cmd[-1] == "https://example.com/exam"
    assert cmd.index("--proxy-server=http://proxy.test:443") < len(cmd) - 1


def test_chromium_block_and_unblock_send_signals(monkeypatch):
    monkeypatch.setattr(ChromiumKiosk, "_restore_gnome_extensions", lambda self: None)

    kiosk = ChromiumKiosk()
    proc = DummyProc()
    kiosk._proc = proc

    kiosk.block()
    kiosk.unblock()

    assert proc.signals == [signal.SIGSTOP, signal.SIGCONT]


def test_chromium_relaunch_uses_last_url_and_allowlist(monkeypatch, tmp_path):
    procs = [DummyProc(pid=10), DummyProc(pid=11)]
    popen_calls = []

    def fake_popen(cmd, env, stdout, stderr):
        popen_calls.append((cmd, env, stdout, stderr))
        return procs[len(popen_calls) - 1]

    monkeypatch.setattr("src.kiosk.chromium._find_chromium", lambda: "/usr/bin/chromium")
    monkeypatch.setattr("src.kiosk.chromium.subprocess.Popen", fake_popen)
    monkeypatch.setattr(ChromiumKiosk, "_apply_window_mode_by_pid", lambda self: None)

    extension_dir = tmp_path / "extension"
    kiosk = ChromiumKiosk(
        display=":9",
        profile_dir=tmp_path / "proctor-chromium-profile",
        extension_dir=extension_dir,
        cleanup_profile_on_stop=False,
    )
    kiosk.start("https://example.com/exam", allowlist=["example.edu"])
    assert kiosk._profile_dir is not None
    cookie_file = kiosk._profile_dir / "Default" / "Cookies"
    cookie_file.parent.mkdir(parents=True)
    cookie_file.write_text("keep-login", encoding="utf-8")

    procs[0].returncode = 1
    assert kiosk.relaunch() is True

    assert len(popen_calls) == 2
    assert popen_calls[1][0][-1] == "https://example.com/exam"
    assert cookie_file.read_text(encoding="utf-8") == "keep-login"
    payload = (extension_dir / "config.json").read_text(encoding="utf-8")
    assert "example.edu" in payload


def test_chromium_stop_cleans_profile_after_formal_session_end(monkeypatch, tmp_path):
    proc = DummyProc()

    monkeypatch.setattr("src.kiosk.chromium._find_chromium", lambda: "/usr/bin/chromium")
    monkeypatch.setattr("src.kiosk.chromium.subprocess.Popen", lambda *args, **kwargs: proc)
    monkeypatch.setattr(ChromiumKiosk, "_apply_window_mode_by_pid", lambda self: None)

    profile_dir = tmp_path / "proctor-chromium-profile"
    kiosk = ChromiumKiosk(
        display=":9",
        profile_dir=profile_dir,
        extension_dir=tmp_path / "extension",
    )
    kiosk.start("https://example.com/exam", allowlist=["example.edu"])
    assert kiosk._profile_dir is not None
    active_profile = kiosk._profile_dir
    (active_profile / "Default").mkdir(parents=True)
    (active_profile / "Default" / "Cookies").write_text("cookie", encoding="utf-8")
    (active_profile / "Default" / "Local Storage").mkdir()

    kiosk.stop()

    assert proc.terminated is True
    assert not active_profile.exists()
    assert kiosk._profile_dir is None


def test_chromium_start_uses_new_profile_after_formal_stop(monkeypatch, tmp_path):
    procs = [DummyProc(pid=10), DummyProc(pid=11)]
    popen_calls = []

    def fake_popen(cmd, env, stdout, stderr):
        popen_calls.append((cmd, env, stdout, stderr))
        return procs[len(popen_calls) - 1]

    monkeypatch.setattr("src.kiosk.chromium._find_chromium", lambda: "/usr/bin/chromium")
    monkeypatch.setattr("src.kiosk.chromium.subprocess.Popen", fake_popen)
    monkeypatch.setattr(ChromiumKiosk, "_apply_window_mode_by_pid", lambda self: None)

    kiosk = ChromiumKiosk(
        display=":9",
        profile_dir=tmp_path / "proctor-chromium-profile",
        extension_dir=tmp_path / "extension",
    )
    kiosk.start("https://example.com/exam", allowlist=["example.edu"])
    first_profile = kiosk._profile_dir
    assert first_profile is not None
    kiosk.stop()

    kiosk.start("https://example.com/exam", allowlist=["example.edu"])
    second_profile = kiosk._profile_dir

    assert second_profile is not None
    assert second_profile != first_profile
    assert f"--user-data-dir={first_profile}" in popen_calls[0][0]
    assert f"--user-data-dir={second_profile}" in popen_calls[1][0]


def test_chromium_stop_terminates_remaining_processes_for_profile(monkeypatch, tmp_path):
    proc = DummyProc()
    proc.returncode = 0
    signals = []

    monkeypatch.setattr("src.kiosk.chromium._find_chromium", lambda: "/usr/bin/chromium")
    monkeypatch.setattr("src.kiosk.chromium.subprocess.Popen", lambda *args, **kwargs: proc)
    monkeypatch.setattr(ChromiumKiosk, "_apply_window_mode_by_pid", lambda self: None)
    monkeypatch.setattr(ChromiumKiosk, "_find_profile_processes", lambda self: [9876])
    monkeypatch.setattr(ChromiumKiosk, "_wait_profile_processes", lambda self, pids, timeout: None)
    monkeypatch.setattr(
        ChromiumKiosk,
        "_signal_pid",
        staticmethod(lambda pid, sig: signals.append((pid, sig))),
    )

    profile_dir = tmp_path / "proctor-chromium-profile"
    kiosk = ChromiumKiosk(
        display=":9",
        profile_dir=profile_dir,
        extension_dir=tmp_path / "extension",
        cleanup_profile_on_stop=False,
    )
    kiosk.start("https://example.com/exam", allowlist=["example.edu"])

    kiosk.stop()

    assert proc.terminated is False
    assert signals == [(9876, signal.SIGTERM), (9876, signal.SIGKILL)]


def test_chromium_start_terminates_existing_browser_processes(monkeypatch, tmp_path):
    proc = DummyProc()
    terminated = []

    monkeypatch.setattr("src.kiosk.chromium._find_chromium", lambda: "/usr/bin/chromium")
    monkeypatch.setattr("src.kiosk.chromium.subprocess.Popen", lambda *args, **kwargs: proc)
    monkeypatch.setattr(ChromiumKiosk, "_apply_window_mode_by_pid", lambda self: None)
    monkeypatch.setattr(ChromiumKiosk, "_find_chromium_processes", lambda self: [100, 200])
    monkeypatch.setattr(ChromiumKiosk, "_wait_pids_exit", lambda self, pids, timeout: None)
    monkeypatch.setattr(ChromiumKiosk, "_pid_exists", staticmethod(lambda _pid: False))
    monkeypatch.setattr(
        ChromiumKiosk,
        "_signal_pid",
        staticmethod(lambda pid, sig: terminated.append((pid, sig))),
    )

    kiosk = ChromiumKiosk(
        profile_dir=tmp_path / "proctor-chromium-profile",
        extension_dir=tmp_path / "extension",
        cleanup_profile_on_stop=False,
    )
    kiosk.start("https://example.com/exam")

    assert terminated == [(100, signal.SIGTERM), (200, signal.SIGTERM)]


def test_chromium_profile_process_match_requires_exact_user_data_dir(tmp_path):
    profile_dir = str((tmp_path / "proctor-chromium-profile").resolve())

    assert ChromiumKiosk._cmdline_uses_profile(
        ["/usr/bin/chromium", f"--user-data-dir={profile_dir}"],
        profile_dir,
    )
    assert ChromiumKiosk._cmdline_uses_profile(
        ["/usr/bin/chromium", "--user-data-dir", profile_dir],
        profile_dir,
    )
    assert not ChromiumKiosk._cmdline_uses_profile(
        ["/usr/bin/chromium", f"--user-data-dir={profile_dir}-other"],
        profile_dir,
    )


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


def test_lockdown_enable_disable_tracks_state(monkeypatch, tmp_path):
    run_calls = []
    popen_calls = []

    class RunningProc(DummyProc):
        def __init__(self):
            super().__init__(pid=99)
            self.stderr = SimpleNamespace(read=lambda: b"")

        def wait(self, timeout: float | None = None) -> int:
            if timeout == 0.25 and self.returncode is None:
                raise subprocess.TimeoutExpired(cmd="xbindkeys", timeout=timeout)
            return 0

    proc = RunningProc()

    def fake_run(cmd, **_kwargs):
        run_calls.append(cmd)
        if cmd == ["gsettings", "get", "org.gnome.mutter", "overlay-key"]:
            return SimpleNamespace(returncode=0, stdout="'Super_L'\n", stderr="")
        if cmd[:2] == ["gsettings", "get"]:
            return SimpleNamespace(returncode=0, stdout="['orig']\n", stderr="")
        if cmd == ["setxkbmap", "-query"]:
            return SimpleNamespace(
                returncode=0,
                stdout="rules: evdev\nmodel: pc105\nlayout: br\noptions: terminate:ctrl_alt_bksp\n",
                stderr="",
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def fake_popen(cmd, **_kwargs):
        popen_calls.append(cmd)
        return proc

    monkeypatch.setattr("src.kiosk.lockdown.subprocess.run", fake_run)
    monkeypatch.setattr("src.kiosk.lockdown.subprocess.Popen", fake_popen)
    monkeypatch.setattr("src.kiosk.lockdown.shutil.which", lambda name: f"/usr/bin/{name}")

    lockdown = Lockdown(display=":3", state_path=tmp_path / "lockdown.json")

    assert lockdown.is_enabled is False

    lockdown.enable()
    assert lockdown.is_enabled is True
    assert ["setxkbmap", "-option", "srvrkeys:none"] in run_calls
    assert any(call[:4] == ["gsettings", "set", "org.gnome.desktop.wm.keybindings", "close"] for call in run_calls)
    assert ["gsettings", "set", "org.gnome.mutter", "overlay-key", "''"] in run_calls
    assert popen_calls[0][:3] == ["xbindkeys", "-n", "-f"]
    assert popen_calls[0][-2:] == ["-X", ":3"]

    lockdown.disable()
    assert lockdown.is_enabled is False
    assert proc.terminated is True
    assert ["setxkbmap", "-option"] in run_calls
    assert ["setxkbmap", "-option", "terminate:ctrl_alt_bksp"] in run_calls
    assert ["gsettings", "set", "org.gnome.mutter", "overlay-key", "'Super_L'"] in run_calls
    assert not (tmp_path / "lockdown.json").exists()


def test_lockdown_can_allow_browser_navigation_shortcuts(monkeypatch, tmp_path):
    configs = []

    class RunningProc(DummyProc):
        stderr = SimpleNamespace(read=lambda: b"")

        def wait(self, timeout: float | None = None) -> int:
            if timeout == 0.25:
                raise subprocess.TimeoutExpired(cmd="xbindkeys", timeout=timeout)
            return 0

    def fake_run(cmd, **_kwargs):
        if cmd[:2] == ["gsettings", "get"]:
            return SimpleNamespace(returncode=1, stdout="", stderr="missing")
        if cmd == ["setxkbmap", "-query"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def fake_popen(cmd, **_kwargs):
        configs.append(cmd[3])
        return RunningProc()

    monkeypatch.setattr("src.kiosk.lockdown.subprocess.run", fake_run)
    monkeypatch.setattr("src.kiosk.lockdown.subprocess.Popen", fake_popen)
    monkeypatch.setattr("src.kiosk.lockdown.shutil.which", lambda name: f"/usr/bin/{name}")

    lockdown = Lockdown(
        display=":3",
        allow_browser_shortcuts=True,
        state_path=tmp_path / "lockdown.json",
    )
    lockdown.enable()

    body = open(configs[0], encoding="utf-8").read()
    assert "Control + t" not in body
    assert "Control + l" not in body
    assert "Control + q" in body

    lockdown.disable()


def test_lockdown_skips_gnome_operations_under_matchbox(monkeypatch, tmp_path):
    run_calls = []

    class RunningProc(DummyProc):
        stderr = SimpleNamespace(read=lambda: b"")

        def wait(self, timeout: float | None = None) -> int:
            if timeout == 0.25:
                raise subprocess.TimeoutExpired(cmd="xbindkeys", timeout=timeout)
            return 0

    def fake_run(cmd, **_kwargs):
        run_calls.append(cmd)
        if cmd == ["setxkbmap", "-query"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("src.kiosk.lockdown.subprocess.run", fake_run)
    monkeypatch.setattr("src.kiosk.lockdown.subprocess.Popen", lambda *_args, **_kwargs: RunningProc())
    monkeypatch.setattr("src.kiosk.lockdown.shutil.which", lambda name: f"/usr/bin/{name}")

    lockdown = Lockdown(
        display=":3",
        manage_gnome=False,
        state_path=tmp_path / "lockdown.json",
    )
    lockdown.enable()
    lockdown.disable()

    assert any(call[0] == "setxkbmap" for call in run_calls)
    assert not any(call[0] in {"gsettings", "gnome-extensions"} for call in run_calls)


def test_lockdown_can_restore_from_persisted_state(monkeypatch, tmp_path):
    run_calls = []
    state_path = tmp_path / "lockdown.json"
    state_path.write_text(
        """
{
  "display": ":3",
  "gsettings": [
    {
      "key": "overlay-key",
      "schema": "org.gnome.mutter",
      "value": "'Super_L'"
    }
  ],
  "xkb_options": "terminate:ctrl_alt_bksp"
}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    def fake_run(cmd, **_kwargs):
        run_calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("src.kiosk.lockdown.subprocess.run", fake_run)
    monkeypatch.setattr("src.kiosk.lockdown.shutil.which", lambda name: f"/usr/bin/{name}")

    lockdown = Lockdown(display=":3", state_path=state_path)
    lockdown.disable()

    assert ["setxkbmap", "-option"] in run_calls
    assert ["setxkbmap", "-option", "terminate:ctrl_alt_bksp"] in run_calls
    assert ["gsettings", "set", "org.gnome.mutter", "overlay-key", "'Super_L'"] in run_calls
    assert not state_path.exists()


def test_session_overlay_starts_controls_and_blocked_overlay(monkeypatch):
    calls = []
    procs = [DummyProc(pid=10), DummyProc(pid=11), DummyProc(pid=12), DummyProc(pid=13)]

    def fake_popen(cmd, env, stdout, stderr):
        calls.append((cmd, env))
        return procs[len(calls) - 1]

    monkeypatch.setattr("src.kiosk.overlay.subprocess.Popen", fake_popen)

    overlay = SessionOverlay(display=":5", api_port=8123)
    overlay.show_waiting()
    overlay.start_controls()
    overlay.show_blocked("ABSENCE", student_id="alice01")

    assert calls[0][0][3:5] == ["--mode", "waiting"]
    assert calls[0][0][calls[0][0].index("--start-url") + 1] == "http://127.0.0.1:8123/pre-exam/start"
    assert "--preview-url" not in calls[0][0]
    assert calls[0][1]["DISPLAY"] == ":5"
    assert calls[1][0][-2:] == ["--mode", "guard"]
    assert calls[2][0][calls[2][0].index("--stop-url") + 1] == "http://127.0.0.1:8123/session/stop"
    assert calls[2][0][calls[2][0].index("--status-url") + 1] == "http://127.0.0.1:8123/exam-checks"
    assert calls[3][0][3:5] == ["--mode", "blocked"]
    assert calls[3][0][calls[3][0].index("--reason") + 1] == "ABSENCE"
    assert calls[3][0][calls[3][0].index("--student-id") + 1] == "alice01"
    assert calls[3][0][calls[3][0].index("--stop-url") + 1] == "http://127.0.0.1:8123/session/stop"
    assert calls[3][0][calls[3][0].index("--preview-url") + 1] == "http://127.0.0.1:8123/camera-preview.jpg"
    assert calls[3][0][calls[3][0].index("--status-url") + 1] == "http://127.0.0.1:8123/exam-checks"

    overlay.stop()
    assert procs[0].terminated is True
    assert procs[1].terminated is True
    assert procs[2].terminated is True
    assert procs[3].terminated is True


def test_session_overlay_starts_identity_confirmation(monkeypatch):
    calls = []

    def fake_popen(cmd, env, stdout, stderr):
        calls.append((cmd, env))
        return DummyProc(pid=10)

    monkeypatch.setattr("src.kiosk.overlay.subprocess.Popen", fake_popen)
    overlay = SessionOverlay(display=":5", api_port=8123)

    overlay.show_identity_confirmation(
        student_id="alice01",
        student_name="Alice Silva",
        timeout_sec=60,
    )

    cmd, env = calls[0]
    assert cmd[3:5] == ["--mode", "confirmation"]
    assert "--student-id" in cmd and cmd[cmd.index("--student-id") + 1] == "alice01"
    assert "--student-name" in cmd and cmd[cmd.index("--student-name") + 1] == "Alice Silva"
    assert cmd[cmd.index("--confirm-url") + 1] == "http://127.0.0.1:8123/pre-exam/confirmation/accept"
    assert cmd[cmd.index("--cancel-url") + 1] == "http://127.0.0.1:8123/pre-exam/confirmation/cancel"
    assert cmd[cmd.index("--preview-url") + 1] == "http://127.0.0.1:8123/camera-preview.jpg"
    assert env["DISPLAY"] == ":5"

    overlay.hide_identity_confirmation()
