from __future__ import annotations

import logging
import os
import subprocess
import sys

logger = logging.getLogger(__name__)


class SessionOverlay:
    def __init__(self, *, display: str | None = None, api_port: int = 8000):
        self._display = display or os.environ.get("DISPLAY", ":0")
        self._api_port = api_port
        self._controls_proc: subprocess.Popen | None = None
        self._blocked_proc: subprocess.Popen | None = None
        self._waiting_proc: subprocess.Popen | None = None
        self._confirmation_proc: subprocess.Popen | None = None
        self._guard_proc: subprocess.Popen | None = None

    def show_waiting(self, message: str | None = None) -> None:
        if self._waiting_proc and self._waiting_proc.poll() is None:
            return
        self._waiting_proc = self._spawn(
            [
                sys.executable,
                "-m",
                "src.kiosk.overlay_app",
                "--mode",
                "waiting",
                "--message",
                message or "",
                "--start-url",
                f"http://127.0.0.1:{self._api_port}/pre-exam/start",
            ]
        )

    def hide_waiting(self) -> None:
        self._terminate(self._waiting_proc)
        self._waiting_proc = None

    def show_identity_confirmation(
        self,
        *,
        student_id: str,
        student_name: str,
        timeout_sec: float,
    ) -> None:
        if self._confirmation_proc and self._confirmation_proc.poll() is None:
            return
        base_url = f"http://127.0.0.1:{self._api_port}/pre-exam/confirmation"
        self._confirmation_proc = self._spawn(
            [
                sys.executable,
                "-m",
                "src.kiosk.overlay_app",
                "--mode",
                "confirmation",
                "--student-id",
                student_id,
                "--student-name",
                student_name,
                "--timeout-sec",
                str(timeout_sec),
                "--confirm-url",
                f"{base_url}/accept",
                "--cancel-url",
                f"{base_url}/cancel",
                "--preview-url",
                self._camera_preview_url,
                "--status-url",
                self._status_url,
            ]
        )

    def hide_identity_confirmation(self) -> None:
        self._terminate(self._confirmation_proc)
        self._confirmation_proc = None

    def start_controls(self) -> None:
        self.start_guard()
        if self._controls_proc and self._controls_proc.poll() is None:
            return
        stop_url = f"http://127.0.0.1:{self._api_port}/session/stop"
        self._controls_proc = self._spawn(
            [
                sys.executable,
                "-m",
                "src.kiosk.overlay_app",
                "--mode",
                "controls",
                "--stop-url",
                stop_url,
                "--status-url",
                self._status_url,
            ]
        )

    def start_guard(self) -> None:
        if self._guard_proc and self._guard_proc.poll() is None:
            return
        self._guard_proc = self._spawn(
            [
                sys.executable,
                "-m",
                "src.kiosk.overlay_app",
                "--mode",
                "guard",
            ]
        )

    def hide_guard(self) -> None:
        self._terminate(self._guard_proc)
        self._guard_proc = None

    def show_blocked(self, reason: str | None = None, *, student_id: str | None = None) -> None:
        if self._blocked_proc and self._blocked_proc.poll() is None:
            return
        self._blocked_proc = self._spawn(
            [
                sys.executable,
                "-m",
                "src.kiosk.overlay_app",
                "--mode",
                "blocked",
                "--reason",
                reason or "",
                "--student-id",
                student_id or "",
                "--stop-url",
                f"http://127.0.0.1:{self._api_port}/session/stop",
                "--preview-url",
                self._camera_preview_url,
                "--status-url",
                self._status_url,
            ]
        )

    def hide_blocked(self) -> None:
        self._terminate(self._blocked_proc)
        self._blocked_proc = None

    def stop(self) -> None:
        self.hide_waiting()
        self.hide_identity_confirmation()
        self.hide_blocked()
        self.hide_guard()
        self._terminate(self._controls_proc)
        self._controls_proc = None

    def _spawn(self, cmd: list[str]) -> subprocess.Popen:
        env = os.environ.copy()
        env["DISPLAY"] = self._display
        logger.info("Iniciando overlay: %s", " ".join(cmd))
        return subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    @property
    def _camera_preview_url(self) -> str:
        return f"http://127.0.0.1:{self._api_port}/camera-preview.jpg"

    @property
    def _status_url(self) -> str:
        return f"http://127.0.0.1:{self._api_port}/exam-checks"

    @staticmethod
    def _terminate(proc: subprocess.Popen | None) -> None:
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
