from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path

from src.core.session import SessionState
from src.core.update_runner import UpdateRunner


class FakeSessionManager:
    def __init__(self, state=SessionState.IDLE):
        self.state = state


def test_update_runner_refuses_when_a_session_is_active(tmp_path):
    runner = UpdateRunner(session_manager=FakeSessionManager(SessionState.SESSION), project_root=tmp_path)

    runner.start()

    assert runner.status_dict()["update_status"] == "recusado"
    assert "prova ativa" in runner.status_dict()["update_message"]


def test_update_runner_restarts_service_only_after_a_successful_pull(tmp_path, monkeypatch):
    signals = []

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, "", "")

    monkeypatch.setattr("src.core.update_runner.subprocess.run", fake_run)
    monkeypatch.setattr("src.core.update_runner.os.kill", lambda pid, sig: signals.append((pid, sig)))
    runner = UpdateRunner(session_manager=FakeSessionManager(), project_root=Path(tmp_path))

    runner.start()
    runner._thread.join(timeout=1)

    assert signals == [(os.getpid(), signal.SIGTERM)]
    assert runner.status_dict()["update_status"] == "reiniciando"
