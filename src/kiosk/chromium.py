"""Gerenciador do Chromium controlado para prova.

Responsabilidades:
  - Abrir o Chromium em modo controlado/maximizado
  - Carregar a extensão local de allowlist
  - Congelar (SIGSTOP) e retomar (SIGCONT) durante bloqueios de proctoring
  - Limpar perfil dedicado ao fim da sessão
  - Encerrar limpo ao fim da sessão

Lockdown de teclado fica em src/kiosk/lockdown.py e é acionado pelo
SessionManager junto com o browser.

Requer: wmctrl (sudo apt install wmctrl)

Uso típico:
    kiosk = ChromiumKiosk()
    kiosk.start("https://prairielearn.org/pl", allowlist=["prairielearn.org"])
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
import uuid
from pathlib import Path

from src.kiosk.allowlist import EXTENSION_DIR, build_allowlist_config, write_extension_config
from src.kiosk.profile import ChromiumProfileCleaner, ProfileCleanupError
from src.core.config import config

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
    """Gerencia o Chromium controlado para uma sessão de prova.

    Args:
        display: Display X11 (default: lê $DISPLAY ou ":1").
        profile_dir: Diretório de perfil dedicado do Chromium.
        extension_dir: Diretório da extensão allowlist estática.
        window_mode: ``maximized`` (produção), ``fullscreen`` ou ``kiosk``.
        cleanup_profile_on_stop: limpar perfil ao encerrar o browser.
        manage_gnome_extensions: compatibilidade para testes no GNOME.
    """

    def __init__(
        self,
        display: str | None = None,
        profile_dir: Path | str | None = None,
        extension_dir: Path | str | None = None,
        window_mode: str = "maximized",
        cleanup_profile_on_stop: bool = True,
        manage_gnome_extensions: bool = False,
    ):
        self._display = display or os.environ.get("DISPLAY", ":1")
        self._profile_root = Path(profile_dir or "/tmp/proctor-chromium-profile")
        self._profile_dir: Path | None = None
        self._extension_dir = Path(extension_dir or EXTENSION_DIR)
        self._window_mode = window_mode
        self._cleanup_profile_on_stop = cleanup_profile_on_stop
        self._manage_gnome_extensions = manage_gnome_extensions
        self._proc: subprocess.Popen | None = None
        self._blocked = False
        self._disabled_extensions: list[str] = []
        self._last_url: str | None = None
        self._last_allowlist: list[str] = []

    # ──────────────────────────────────────────────
    #  Ciclo de vida
    # ──────────────────────────────────────────────

    def start(self, url: str, *, allowlist: list[str] | None = None) -> None:
        """Abre o Chromium controlado na URL especificada."""
        if self._proc and self._proc.poll() is None:
            logger.warning("Chromium já está rodando — ignorando start()")
            return

        self._last_url = url
        self._last_allowlist = list(allowlist or [])
        profile_dir = self._ensure_profile_dir()
        allowlist_config = build_allowlist_config(
            start_url=url,
            allowlist=self._last_allowlist,
        )
        write_extension_config(allowlist_config, extension_dir=self._extension_dir)
        chromium = _find_chromium()

        if self._manage_gnome_extensions:
            self._disable_gnome_extensions()

        cmd = [
            chromium,
            "--no-first-run",
            "--disable-translate",
            "--disable-dev-tools",
            "--disable-default-apps",
            "--disable-background-networking",
            "--disable-sync",
            "--disable-infobars",
            "--noerrdialogs",
            "--password-store=basic",
            "--disable-features=AutofillServerCommunication",
            f"--disable-extensions-except={self._extension_dir}",
            f"--load-extension={self._extension_dir}",
            f"--user-data-dir={profile_dir}",
            f"--proxy-server={config.proxy_server}",
            url,
        ]
        if self._window_mode == "kiosk":
            cmd.insert(1, "--kiosk")
        elif self._window_mode == "fullscreen":
            cmd.insert(1, "--start-fullscreen")
        else:
            cmd.insert(1, "--start-maximized")

        env = os.environ.copy()
        env["DISPLAY"] = self._display

        logger.info("Iniciando Chromium: %s", url)
        self._proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        self._apply_window_mode_by_pid()
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

        self._terminate_profile_processes()
        self._proc = None
        if self._manage_gnome_extensions:
            self._restore_gnome_extensions()
        if self._cleanup_profile_on_stop:
            try:
                if self._profile_dir is not None:
                    ChromiumProfileCleaner(self._profile_dir).cleanup()
                    self._remove_empty_profile_root()
            except ProfileCleanupError as exc:
                logger.error("Falha na limpeza do perfil Chromium: %s", exc)
                raise
        self._profile_dir = None
        logger.info("Chromium encerrado")

    def _ensure_profile_dir(self) -> Path:
        """Cria um perfil isolado para a sessão atual do browser."""
        if self._profile_dir is not None:
            self._profile_dir.mkdir(parents=True, exist_ok=True)
            return self._profile_dir

        self._cleanup_legacy_profile_root()
        self._profile_root.mkdir(parents=True, exist_ok=True)
        self._profile_dir = self._profile_root / f"proctor-session-{os.getpid()}-{uuid.uuid4().hex}"
        self._profile_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Perfil Chromium da sessão: %s", self._profile_dir)
        return self._profile_dir

    def _cleanup_legacy_profile_root(self) -> None:
        """Remove dados de sessões anteriores antes de criar novo perfil isolado."""
        if not self._profile_root.exists() or not self._profile_root.is_dir():
            return
        try:
            ChromiumProfileCleaner(self._profile_root).cleanup()
        except ProfileCleanupError as exc:
            logger.warning("Falha ao limpar perfil Chromium anterior %s: %s", self._profile_root, exc)

    def _remove_empty_profile_root(self) -> None:
        try:
            self._profile_root.rmdir()
        except OSError:
            return

    def _terminate_profile_processes(self) -> None:
        """Encerra processos Chromium remanescentes que ainda usam o perfil da prova."""
        pids = self._find_profile_processes()
        if not pids:
            return

        logger.warning(
            "Encerrando %d processo(s) Chromium remanescente(s) do perfil %s",
            len(pids),
            self._profile_dir,
        )
        for pid in pids:
            self._signal_pid(pid, signal.SIGTERM)
        self._wait_profile_processes(pids, timeout=3)

        remaining = self._find_profile_processes()
        for pid in remaining:
            self._signal_pid(pid, signal.SIGKILL)
        if remaining:
            self._wait_profile_processes(remaining, timeout=2)

    def _find_profile_processes(self) -> list[int]:
        if self._profile_dir is None:
            return []
        profile_dir = str(self._profile_dir.resolve())
        current_pid = os.getpid()
        pids: list[int] = []
        proc_root = Path("/proc")
        if not proc_root.exists():
            return pids

        for entry in proc_root.iterdir():
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            if pid == current_pid:
                continue
            try:
                raw = (entry / "cmdline").read_bytes()
            except OSError:
                continue
            if not raw:
                continue
            args = [part.decode("utf-8", errors="ignore") for part in raw.split(b"\0") if part]
            if self._cmdline_uses_profile(args, profile_dir):
                pids.append(pid)
        return pids

    @staticmethod
    def _cmdline_uses_profile(args: list[str], profile_dir: str) -> bool:
        for index, arg in enumerate(args):
            if arg == f"--user-data-dir={profile_dir}":
                return True
            if arg == "--user-data-dir" and index + 1 < len(args):
                try:
                    if str(Path(args[index + 1]).resolve()) == profile_dir:
                        return True
                except OSError:
                    continue
        return False

    @staticmethod
    def _signal_pid(pid: int, sig: signal.Signals) -> None:
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            return
        except PermissionError as exc:
            logger.warning("Sem permissao para sinalizar Chromium remanescente PID %d: %s", pid, exc)

    def _wait_profile_processes(self, pids: list[int], *, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        pending = set(pids)
        while pending and time.monotonic() < deadline:
            pending = {pid for pid in pending if pid in self._find_profile_processes()}
            if pending:
                time.sleep(0.05)

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def relaunch(self) -> bool:
        """Reabre o Chromium com a última URL/config caso ele tenha fechado."""
        if self.is_running:
            return True
        if self._last_url is None:
            logger.error("Não há URL anterior para relançar o Chromium")
            return False
        logger.warning("Chromium não está rodando; tentando relançar")
        try:
            self.start(self._last_url, allowlist=self._last_allowlist)
        except Exception as exc:
            logger.error("Falha ao relançar Chromium: %s", exc)
            return False
        return self.is_running

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

    def _apply_window_mode_by_pid(self) -> None:
        """Aplica maximização/fullscreen via wmctrl buscando a janela pelo PID.

        Buscar pelo PID evita pegar janelas de outros aplicativos que
        possam ter "Chromium" no nome (ex: abas do VSCode ou Firefox).
        """
        if self._window_mode == "kiosk":
            return
        if not shutil.which("wmctrl"):
            logger.warning(
                "wmctrl não encontrado — modo de janela não será aplicado. "
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

        action = (
            "add,fullscreen"
            if self._window_mode == "fullscreen"
            else "add,maximized_vert,maximized_horz"
        )
        time.sleep(0.3)
        subprocess.run(
            ["wmctrl", "-i", "-r", win_id, "-b", action],
            env=env, capture_output=True, timeout=3,
        )
        logger.info("Modo de janela aplicado (PID %d, window %s, %s)", pid, win_id, action)

    def _force_fullscreen_by_pid(self) -> None:
        """Compatibilidade com testes/uso legado."""
        previous = self._window_mode
        self._window_mode = "fullscreen"
        try:
            self._apply_window_mode_by_pid()
        finally:
            self._window_mode = previous

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
