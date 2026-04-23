"""Gerenciador do Chromium em modo kiosk.

Responsabilidades:
  - Abrir o Chromium em fullscreen
  - Congelar (SIGSTOP) e retomar (SIGCONT) durante bloqueios de proctoring
  - Encerrar limpo ao fim da sessão

Lockdown de teclado (Alt+F4, Super, Ctrl+Alt+T etc.) foi movido para M7.

Requer: wmctrl (sudo apt install wmctrl)

Uso típico:
    kiosk = ChromiumKiosk()
    kiosk.start("https://prairielearn.org/pl")
    kiosk.block()    # SIGSTOP durante bloqueio
    kiosk.unblock()  # SIGCONT após re-identificação
    kiosk.stop()
"""

from __future__ import annotations

import logging
import os
import signal
import shutil
import subprocess
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_CHROMIUM_BINS = [
    "chromium-browser",
    "chromium",
    "google-chrome",
    "google-chrome-stable",
]

_GNOME_EXTENSIONS_TO_DISABLE = [
    "ubuntu-dock@ubuntu.com",
    "tiling-assistant@ubuntu.com",
]


def _find_chromium() -> str:
    for bin_name in _CHROMIUM_BINS:
        path = shutil.which(bin_name)
        if path:
            return path
    raise FileNotFoundError(
        "Chromium não encontrado. Instale com: sudo apt install chromium-browser"
    )


class ChromiumKiosk:
    """Gerencia o Chromium em modo kiosk para uma sessão de prova.

    Args:
        display: Display X11 (default: lê $DISPLAY ou ":1").
        profile_dir: Diretório de perfil temporário do Chromium.
    """

    def __init__(
        self,
        display: str | None = None,
        profile_dir: Path | str | None = None,
    ):
        self._display = display or os.environ.get("DISPLAY", ":1")
        self._profile_dir = Path(profile_dir or "/tmp/proctor-chromium-profile")
        self._proc: subprocess.Popen | None = None
        self._blocked = False
        self._disabled_extensions: list[str] = []

    # ──────────────────────────────────────────────
    #  Ciclo de vida
    # ──────────────────────────────────────────────

    def start(self, url: str) -> None:
        """Abre o Chromium em fullscreen na URL especificada."""
        if self._proc and self._proc.poll() is None:
            logger.warning("Chromium já está rodando — ignorando start()")
            return

        self._profile_dir.mkdir(parents=True, exist_ok=True)
        chromium = _find_chromium()

        # Desabilitar extensões do Gnome que impedem fullscreen
        self._disable_gnome_extensions()

        cmd = [
            chromium,
            "--kiosk",
            "--start-fullscreen",
            "--no-first-run",
            "--disable-translate",
            "--disable-extensions",
            "--disable-dev-tools",
            "--disable-default-apps",
            "--disable-background-networking",
            "--disable-sync",
            "--disable-infobars",
            "--noerrdialogs",
            "--incognito",
            f"--user-data-dir={self._profile_dir}",
            url,
        ]

        env = os.environ.copy()
        env["DISPLAY"] = self._display

        logger.info("Iniciando Chromium: %s", url)
        self._proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Forçar fullscreen pelo PID — evita pegar janela errada por nome
        self._force_fullscreen_by_pid()
        logger.info("Chromium iniciado (PID %d)", self._proc.pid)

    def stop(self) -> None:
        """Encerra o Chromium e restaura o ambiente."""
        if self._blocked:
            self.unblock()

        if self._proc and self._proc.poll() is None:
            logger.info("Encerrando Chromium (PID %d)...", self._proc.pid)
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()

        self._proc = None
        self._restore_gnome_extensions()
        logger.info("Chromium encerrado")

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    # ──────────────────────────────────────────────
    #  Bloqueio / desbloqueio
    # ──────────────────────────────────────────────

    def block(self) -> None:
        """Congela o Chromium (SIGSTOP) durante um bloqueio de proctoring."""
        if not self.is_running or self._blocked:
            return
        logger.info("Chromium congelado (SIGSTOP)")
        self._proc.send_signal(signal.SIGSTOP)
        self._blocked = True

    def unblock(self) -> None:
        """Retoma o Chromium (SIGCONT) após re-identificação bem-sucedida."""
        if not self.is_running or not self._blocked:
            return
        logger.info("Chromium retomado (SIGCONT)")
        self._proc.send_signal(signal.SIGCONT)
        self._blocked = False

    # ──────────────────────────────────────────────
    #  Fullscreen
    # ──────────────────────────────────────────────

    def _force_fullscreen_by_pid(self) -> None:
        """Força fullscreen via wmctrl buscando a janela pelo PID do processo.

        Buscar pelo PID evita pegar janelas de outros aplicativos que
        possam ter "Chromium" no nome (ex: abas do VSCode ou Firefox).
        """
        if not shutil.which("wmctrl"):
            logger.warning(
                "wmctrl não encontrado — fullscreen não será aplicado. "
                "Instale com: sudo apt install wmctrl"
            )
            return

        env = os.environ.copy()
        env["DISPLAY"] = self._display
        pid = self._proc.pid

        # Aguarda a janela aparecer — tenta por até 10s
        win_id = None
        for _ in range(20):
            try:
                # wmctrl -l -p lista janelas com PID
                result = subprocess.run(
                    ["wmctrl", "-l", "-p"],
                    env=env, capture_output=True, timeout=3,
                )
                for line in result.stdout.decode().splitlines():
                    parts = line.split()
                    # formato: WIN_ID DESKTOP PID HOST TITLE
                    if len(parts) >= 3 and parts[2] == str(pid):
                        win_id = parts[0]
                        break
                if win_id:
                    break
            except Exception:
                pass
            time.sleep(0.5)

        if not win_id:
            logger.warning(
                "wmctrl: janela do Chromium (PID %d) não encontrada após 10s", pid
            )
            return

        # Aplicar fullscreen pela window ID
        time.sleep(0.3)
        subprocess.run(
            ["wmctrl", "-i", "-r", win_id, "-b", "add,fullscreen"],
            env=env, capture_output=True, timeout=3,
        )
        logger.info("Fullscreen aplicado (PID %d, window %s)", pid, win_id)

    # ──────────────────────────────────────────────
    #  Gnome extensions
    # ──────────────────────────────────────────────

    def _disable_gnome_extensions(self) -> None:
        """Desabilita extensões do Gnome que interferem no fullscreen."""
        if not shutil.which("gnome-extensions"):
            logger.warning(
                "gnome-extensions não encontrado — extensões do Gnome não serão alteradas"
            )
            return

        env = os.environ.copy()
        env["DISPLAY"] = self._display

        try:
            result = subprocess.run(
                ["gnome-extensions", "list", "--enabled"],
                env=env,
                capture_output=True,
                timeout=3,
                check=False,
            )
        except (subprocess.SubprocessError, OSError) as exc:
            logger.warning("Falha ao listar extensões do Gnome: %s", exc)
            return

        enabled = result.stdout.decode().strip().splitlines()

        for ext in _GNOME_EXTENSIONS_TO_DISABLE:
            if ext in enabled:
                try:
                    subprocess.run(
                        ["gnome-extensions", "disable", ext],
                        env=env,
                        capture_output=True,
                        timeout=3,
                        check=False,
                    )
                except (subprocess.SubprocessError, OSError) as exc:
                    logger.warning("Falha ao desabilitar extensão %s: %s", ext, exc)
                    continue

                self._disabled_extensions.append(ext)
                logger.info("Extensão desabilitada: %s", ext)

    def _restore_gnome_extensions(self) -> None:
        """Restaura extensões do Gnome desabilitadas pelo kiosk."""
        if not self._disabled_extensions:
            return

        if not shutil.which("gnome-extensions"):
            logger.warning(
                "gnome-extensions não encontrado — não foi possível restaurar extensões"
            )
            self._disabled_extensions.clear()
            return

        env = os.environ.copy()
        env["DISPLAY"] = self._display

        for ext in self._disabled_extensions:
            try:
                subprocess.run(
                    ["gnome-extensions", "enable", ext],
                    env=env,
                    capture_output=True,
                    timeout=3,
                    check=False,
                )
            except (subprocess.SubprocessError, OSError) as exc:
                logger.warning("Falha ao restaurar extensão %s: %s", ext, exc)
                continue
            logger.info("Extensão restaurada: %s", ext)

        self._disabled_extensions.clear()
