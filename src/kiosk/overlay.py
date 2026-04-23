from __future__ import annotations

import logging
import os
import subprocess
import sys

logger = logging.getLogger(__name__)


class SessionOverlay:
    def __init__(self, *, display: str | None = None, api_port: int = 8000):
        self._display = display or os.environ.get("DISPLAY", ":1")
        self._api_port = api_port
        self._controls_proc: subprocess.Popen | None = None
        self._blocked_proc: subprocess.Popen | None = None

    def start_controls(self) -> None:
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
            ]
        )

    def show_blocked(self, reason: str | None = None) -> None:
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
            ]
        )

    def hide_blocked(self) -> None:
        self._terminate(self._blocked_proc)
        self._blocked_proc = None

    def stop(self) -> None:
        self.hide_blocked()
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
