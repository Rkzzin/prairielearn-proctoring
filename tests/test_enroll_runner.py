from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from src.core.enroll_runner import EnrollRunner
from src.core.session import SessionState


@dataclass
class FakeSessionManager:
    state: SessionState = SessionState.IDLE


@dataclass
class FakeCompletedProcess:
    returncode: int = 0
    stderr: str = ""


def test_run_processes_each_turma_and_marks_done(monkeypatch):
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return FakeCompletedProcess(returncode=0)

    monkeypatch.setattr("src.core.enroll_runner.subprocess.run", fake_run)
    runner = EnrollRunner(session_manager=FakeSessionManager(), project_root=Path("/tmp"))

    runner._run(["ES2026-T1", "ES2026-T2"])

    assert len(calls) == 2
    assert calls[0][-3:] == ["--turma", "ES2026-T1", "--force"]
    assert calls[1][-3:] == ["--turma", "ES2026-T2", "--force"]
    status = runner.status_dict()
    assert status["enroll_status"] == "done"
    assert "2 turma" in status["enroll_message"]


def test_run_marks_error_and_lists_failed_turmas(monkeypatch):
    def fake_run(cmd, **kwargs):
        turma = cmd[cmd.index("--turma") + 1]
        return FakeCompletedProcess(returncode=1 if turma == "ES2026-T2" else 0)

    monkeypatch.setattr("src.core.enroll_runner.subprocess.run", fake_run)
    runner = EnrollRunner(session_manager=FakeSessionManager(), project_root=Path("/tmp"))

    runner._run(["ES2026-T1", "ES2026-T2"])

    status = runner.status_dict()
    assert status["enroll_status"] == "error"
    assert "ES2026-T2" in status["enroll_message"]
    assert "ES2026-T1" not in status["enroll_message"]


def test_run_marks_error_on_timeout(monkeypatch):
    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout", 0))

    monkeypatch.setattr("src.core.enroll_runner.subprocess.run", fake_run)
    runner = EnrollRunner(session_manager=FakeSessionManager(), project_root=Path("/tmp"))

    runner._run(["ES2026-T1"])

    assert runner.status_dict()["enroll_status"] == "error"


def test_start_refuses_when_station_not_idle(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "src.core.enroll_runner.subprocess.run",
        lambda cmd, **kwargs: calls.append(cmd),
    )
    runner = EnrollRunner(session_manager=FakeSessionManager(state=SessionState.SESSION))

    runner.start(["ES2026-T1"])

    assert calls == []
    status = runner.status_dict()
    assert status["enroll_status"] == "recusado"
    assert "SESSION" in status["enroll_message"]


def test_start_refuses_empty_turma_list():
    runner = EnrollRunner(session_manager=FakeSessionManager())

    runner.start([])

    assert runner.status_dict()["enroll_status"] == "error"


def test_start_ignores_second_call_while_running(monkeypatch):
    runner = EnrollRunner(session_manager=FakeSessionManager())
    with runner._lock:
        runner._status = "running"

    runner.start(["ES2026-T1"])

    assert runner._thread is None


def test_start_spawns_a_background_thread_and_returns_immediately(monkeypatch):
    import threading

    started = []
    release = threading.Event()

    def fake_run(cmd, **kwargs):
        started.append(cmd)
        release.wait(timeout=5)
        return FakeCompletedProcess(returncode=0)

    monkeypatch.setattr("src.core.enroll_runner.subprocess.run", fake_run)
    runner = EnrollRunner(session_manager=FakeSessionManager(), project_root=Path("/tmp"))

    runner.start(["ES2026-T1"])
    # start() não bloqueia: a thread de fundo já deve ter chamado fake_run e
    # ficado presa nela, mas o caller (esta linha) segue livre.
    for _ in range(50):
        if started:
            break
        threading.Event().wait(0.01)
    assert started
    assert runner.status_dict()["enroll_status"] == "running"

    release.set()
    runner._thread.join(timeout=5)
    assert runner.status_dict()["enroll_status"] == "done"
