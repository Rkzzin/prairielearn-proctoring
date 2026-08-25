"""Lockdown leve de teclado durante sessão de prova.

O bloqueio é aplicado somente enquanto a sessão está ativa e é restaurado no
encerramento. A estratégia evita loops de polling: desativa atalhos do
GNOME/WM via gsettings, desativa teclas especiais do servidor X via XKB e usa
xbindkeys como segunda camada para engolir combinações comuns de escape.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _GSettingsKey:
    schema: str
    key: str
    disabled_value: str = "[]"


_WM_KEYS = [
    "close",
    "switch-applications",
    "switch-applications-backward",
    "switch-windows",
    "switch-windows-backward",
    "switch-group",
    "switch-group-backward",
    "cycle-windows",
    "cycle-windows-backward",
    "cycle-group",
    "cycle-group-backward",
    "panel-main-menu",
    "panel-run-dialog",
    "activate-window-menu",
    "show-desktop",
    "begin-move",
    "begin-resize",
    "minimize",
    "toggle-fullscreen",
    "toggle-maximized",
    "switch-to-workspace-left",
    "switch-to-workspace-right",
    "switch-to-workspace-up",
    "switch-to-workspace-down",
    "move-to-workspace-left",
    "move-to-workspace-right",
    "move-to-workspace-up",
    "move-to-workspace-down",
]
_WM_KEYS.extend(f"switch-to-workspace-{i}" for i in range(1, 13))
_WM_KEYS.extend(f"move-to-workspace-{i}" for i in range(1, 13))

_SHELL_KEYS = [
    "toggle-overview",
    "toggle-application-view",
    "toggle-message-tray",
    "toggle-quick-settings",
    "focus-active-notification",
    "show-screen-recording-ui",
    "show-screenshot-ui",
    "screenshot",
    "screenshot-window",
]
_SHELL_KEYS.extend(f"switch-to-application-{i}" for i in range(1, 10))
_SHELL_KEYS.extend(f"open-new-window-application-{i}" for i in range(1, 10))

_MEDIA_KEYS = [
    "terminal",
    "logout",
    "screensaver",
    "home",
    "www",
    "search",
    "control-center",
    "help",
]

_GSETTINGS_KEYS = tuple(
    [_GSettingsKey("org.gnome.desktop.wm.keybindings", key) for key in _WM_KEYS]
    + [_GSettingsKey("org.gnome.shell.keybindings", key) for key in _SHELL_KEYS]
    + [_GSettingsKey("org.gnome.settings-daemon.plugins.media-keys", key) for key in _MEDIA_KEYS]
    + [
        _GSettingsKey("org.gnome.mutter", "overlay-key", "''"),
        _GSettingsKey("org.gnome.desktop.interface", "enable-hot-corners", "false"),
        _GSettingsKey("org.gnome.shell.extensions.dash-to-dock", "show-show-apps-button", "false"),
        _GSettingsKey("org.gnome.shell.extensions.dash-to-dock", "dock-fixed", "false"),
        # M7 trocou --kiosk por Chromium maximizado (abas/barra de endereço
        # normais), o que deixa minimizar/maximizar/fechar da decoração da
        # janela clicáveis com o mouse — o bloqueio de atalhos de teclado
        # (Alt+F4, Ctrl+W etc., já bloqueados via xbindkeys) não cobre isso.
        # ':' remove os três botões da decoração; sem eles não tem como o
        # aluno tirar a janela de foco ou fechá-la sem um atalho já
        # bloqueado.
        _GSettingsKey("org.gnome.desktop.wm.preferences", "button-layout", "':'"),
    ]
)

_GNOME_EXTENSIONS_TO_DISABLE = [
    "ubuntu-dock@ubuntu.com",
    "ding@rastersoft.com",
    "tiling-assistant@ubuntu.com",
]

_XBINDKEYS_ALWAYS_BLOCKS = [
    "Alt + F4",
    "Alt + Tab",
    "Shift+Alt + Tab",
    "Alt + Escape",
    "Alt + F2",
    "Alt + space",
    "Mod4 + Tab",
    "Shift+Mod4 + Tab",
    "Mod4 + a",
    "Mod4 + d",
    "Mod4 + l",
    "Mod4 + m",
    "Mod4 + s",
    "Mod4 + v",
    "Control+Alt + t",
    "Control+Alt + Delete",
    "Control+Alt + BackSpace",
    "Control+Alt + F1",
    "Control+Alt + F2",
    "Control+Alt + F3",
    "Control+Alt + F4",
    "Control+Alt + F5",
    "Control+Alt + F6",
    "Control+Alt + F7",
    "Control+Alt + F8",
    "Control+Alt + F9",
    "Control+Alt + F10",
    "Control+Alt + F11",
    "Control+Alt + F12",
    "Control + n",
    "Control + q",
    "Control + w",
    "Control+Shift + n",
    "Control+Shift + q",
    "Control+Shift + t",
    "Control+Shift + w",
    "F11",
    "Print",
    "Alt + Print",
    "Shift + Print",
]

_XBINDKEYS_BROWSER_NAVIGATION_BLOCKS = [
    "Control + l",
    "Control + t",
]


class Lockdown:
    """Bloqueia atalhos de fuga do Chromium durante uma sessão.

    Args:
        display: Display X11 onde Chromium/overlay estão rodando.
    """

    def __init__(
        self,
        display: str = ":0",
        *,
        allow_browser_shortcuts: bool = False,
        manage_gnome: bool = True,
        state_path: Path | str | None = None,
    ):
        self._display = display
        self._allow_browser_shortcuts = allow_browser_shortcuts
        self._manage_gnome = manage_gnome
        self._state_path = Path(
            state_path
            or os.environ.get("PROCTOR_LOCKDOWN_STATE_PATH", f"/tmp/proctor-lockdown-{os.getuid()}.json")
        )
        self._enabled = False
        self._original_gsettings: dict[_GSettingsKey, str] = {}
        self._original_xkb_options: str | None = None
        self._disabled_gnome_extensions: list[str] = []
        self._xbindkeys_proc: subprocess.Popen | None = None
        self._xbindkeys_config: Path | None = None
        atexit.register(self.disable)

    def enable(self) -> None:
        """Ativa o lockdown de atalhos para a prova atual."""
        if self._enabled:
            return

        if self._state_path.exists():
            logger.warning("Estado de lockdown anterior encontrado; restaurando antes de reaplicar")
            self.disable()

        self._original_gsettings.clear()
        self._original_xkb_options = None
        env = self._command_env()

        try:
            if self._manage_gnome:
                self._disable_gnome_shortcuts(env)
                self._disable_gnome_extensions(env)
            self._disable_x_server_keys(env)
            self._write_state()
            self._start_xbindkeys(env)
        except Exception as exc:
            logger.warning("Falha ao ativar lockdown; restaurando estado anterior: %s", exc)
            self._stop_xbindkeys()
            self._restore_x_server_keys(env)
            if self._manage_gnome:
                self._restore_gnome_extensions(env)
                self._restore_gnome_shortcuts(env)
            self._clear_state()
            self._enabled = False
            return

        self._enabled = True
        logger.info("Lockdown de teclado ativado no display %s", self._display)

    def disable(self) -> None:
        """Restaura atalhos alterados por enable()."""
        if not self._enabled and not self._state_path.exists():
            return

        if self._state_path.exists():
            self._load_state()

        env = self._command_env()
        try:
            self._stop_xbindkeys()
        except Exception as exc:
            logger.warning("Falha ao encerrar xbindkeys temporário: %s", exc)
        try:
            self._restore_x_server_keys(env)
        except Exception as exc:
            logger.warning("Falha ao restaurar opções XKB: %s", exc)
        if self._manage_gnome:
            try:
                self._restore_gnome_extensions(env)
            except Exception as exc:
                logger.warning("Falha ao restaurar extensões GNOME: %s", exc)
            try:
                self._restore_gnome_shortcuts(env)
            except Exception as exc:
                logger.warning("Falha ao restaurar atalhos GNOME: %s", exc)
        self._enabled = False
        self._clear_state()
        logger.info("Lockdown de teclado desativado")

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def _command_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["DISPLAY"] = self._display

        runtime_dir = Path(f"/run/user/{os.getuid()}")
        if runtime_dir.exists():
            env.setdefault("XDG_RUNTIME_DIR", str(runtime_dir))
            bus = runtime_dir / "bus"
            if bus.exists():
                env.setdefault("DBUS_SESSION_BUS_ADDRESS", f"unix:path={bus}")

        return env

    def _disable_gnome_shortcuts(self, env: dict[str, str]) -> None:
        if not shutil.which("gsettings"):
            logger.warning("gsettings não encontrado; atalhos do GNOME não serão alterados")
            return

        for binding in _GSETTINGS_KEYS:
            current = self._gsettings_get(binding, env)
            if current is None:
                continue
            self._original_gsettings[binding] = current
            if self._is_already_disabled(binding, current):
                continue
            self._gsettings_set(binding, binding.disabled_value, env)

    def _restore_gnome_shortcuts(self, env: dict[str, str]) -> None:
        for binding, value in self._original_gsettings.items():
            self._gsettings_set(binding, value, env)
        self._original_gsettings.clear()

    def _disable_gnome_extensions(self, env: dict[str, str]) -> None:
        if not shutil.which("gnome-extensions"):
            logger.warning("gnome-extensions não encontrado; dock do GNOME não será alterado")
            return

        result = subprocess.run(
            ["gnome-extensions", "list", "--enabled"],
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=3,
        )
        if result.returncode != 0:
            logger.warning(
                "Falha ao listar extensões GNOME: %s",
                result.stderr.strip() or result.returncode,
            )
            return

        enabled = set(result.stdout.strip().splitlines())
        for extension in _GNOME_EXTENSIONS_TO_DISABLE:
            if extension not in enabled:
                continue
            disable = subprocess.run(
                ["gnome-extensions", "disable", extension],
                env=env,
                capture_output=True,
                text=True,
                check=False,
                timeout=3,
            )
            if disable.returncode != 0:
                logger.warning(
                    "Falha ao desabilitar extensão GNOME %s: %s",
                    extension,
                    disable.stderr.strip() or disable.returncode,
                )
                continue
            self._disabled_gnome_extensions.append(extension)
            logger.info("Extensão GNOME desabilitada durante prova: %s", extension)

    def _restore_gnome_extensions(self, env: dict[str, str]) -> None:
        if not self._disabled_gnome_extensions:
            return
        if not shutil.which("gnome-extensions"):
            logger.warning("gnome-extensions não encontrado; não foi possível restaurar extensões GNOME")
            self._disabled_gnome_extensions.clear()
            return

        for extension in self._disabled_gnome_extensions:
            enable = subprocess.run(
                ["gnome-extensions", "enable", extension],
                env=env,
                capture_output=True,
                text=True,
                check=False,
                timeout=3,
            )
            if enable.returncode != 0:
                logger.warning(
                    "Falha ao restaurar extensão GNOME %s: %s",
                    extension,
                    enable.stderr.strip() or enable.returncode,
                )
                continue
            logger.info("Extensão GNOME restaurada: %s", extension)
        self._disabled_gnome_extensions.clear()

    def _gsettings_get(self, binding: _GSettingsKey, env: dict[str, str]) -> str | None:
        result = subprocess.run(
            ["gsettings", "get", binding.schema, binding.key],
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            logger.debug(
                "Atalho GNOME ignorado (%s %s): %s",
                binding.schema,
                binding.key,
                stderr or result.returncode,
            )
            return None
        return result.stdout.strip()

    def _gsettings_set(self, binding: _GSettingsKey, value: str, env: dict[str, str]) -> None:
        result = subprocess.run(
            ["gsettings", "set", binding.schema, binding.key, value],
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
        if result.returncode != 0:
            logger.warning(
                "Falha ao alterar atalho GNOME %s/%s: %s",
                binding.schema,
                binding.key,
                result.stderr.strip() or result.returncode,
            )

    @staticmethod
    def _is_already_disabled(binding: _GSettingsKey, current: str) -> bool:
        if current == binding.disabled_value:
            return True
        return binding.disabled_value == "[]" and current == "@as []"

    def _disable_x_server_keys(self, env: dict[str, str]) -> None:
        if not shutil.which("setxkbmap"):
            logger.warning("setxkbmap não encontrado; Ctrl+Alt+Fn não será tratado")
            return

        query = subprocess.run(
            ["setxkbmap", "-query"],
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
        if query.returncode == 0:
            self._original_xkb_options = self._parse_xkb_options(query.stdout)

        result = subprocess.run(
            ["setxkbmap", "-option", "srvrkeys:none"],
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
        if result.returncode != 0:
            logger.warning(
                "Falha ao aplicar XKB srvrkeys:none: %s",
                result.stderr.strip() or result.returncode,
            )

    def _restore_x_server_keys(self, env: dict[str, str]) -> None:
        if self._original_xkb_options is None or not shutil.which("setxkbmap"):
            self._original_xkb_options = None
            return

        clear = subprocess.run(
            ["setxkbmap", "-option"],
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
        if clear.returncode != 0:
            logger.warning(
                "Falha ao limpar opções XKB: %s",
                clear.stderr.strip() or clear.returncode,
            )

        for option in self._split_xkb_options(self._original_xkb_options):
            subprocess.run(
                ["setxkbmap", "-option", option],
                env=env,
                capture_output=True,
                text=True,
                check=False,
                timeout=2,
            )

        self._original_xkb_options = None

    @staticmethod
    def _parse_xkb_options(query_output: str) -> str:
        for line in query_output.splitlines():
            if line.strip().startswith("options:"):
                return line.split(":", 1)[1].strip()
        return ""

    @staticmethod
    def _split_xkb_options(options: str) -> list[str]:
        return [option.strip() for option in options.split(",") if option.strip()]

    def _start_xbindkeys(self, env: dict[str, str]) -> None:
        if not shutil.which("xbindkeys"):
            logger.warning("xbindkeys não encontrado; usando apenas gsettings/XKB")
            return

        fd, path = tempfile.mkstemp(prefix="proctor-lockdown-", suffix=".xbindkeysrc")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write("# Gerado automaticamente pelo Proctor Station.\n")
            handle.write("# Este arquivo é removido ao encerrar a sessão.\n\n")
            for binding in self._xbindkeys_blocks():
                handle.write('"/bin/true"\n')
                handle.write(f"  {binding}\n\n")

        self._xbindkeys_config = Path(path)
        self._xbindkeys_proc = subprocess.Popen(
            ["xbindkeys", "-n", "-f", path, "-X", self._display],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        try:
            ret = self._xbindkeys_proc.wait(timeout=0.25)
        except subprocess.TimeoutExpired:
            logger.info("xbindkeys temporário iniciado para bloqueios adicionais")
            return

        stderr = ""
        if self._xbindkeys_proc.stderr is not None:
            stderr = self._xbindkeys_proc.stderr.read().decode("utf-8", errors="replace").strip()
        logger.warning("xbindkeys temporário encerrou cedo (código %s): %s", ret, stderr)
        self._xbindkeys_proc = None
        self._remove_xbindkeys_config()

    def _stop_xbindkeys(self) -> None:
        proc = self._xbindkeys_proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)
        self._xbindkeys_proc = None
        self._remove_xbindkeys_config()

    def _remove_xbindkeys_config(self) -> None:
        if self._xbindkeys_config is None:
            return
        try:
            self._xbindkeys_config.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Falha ao remover config temporário do xbindkeys: %s", exc)
        self._xbindkeys_config = None

    def _xbindkeys_blocks(self) -> list[str]:
        blocks = list(_XBINDKEYS_ALWAYS_BLOCKS)
        if not self._allow_browser_shortcuts:
            blocks.extend(_XBINDKEYS_BROWSER_NAVIGATION_BLOCKS)
        return blocks

    def _write_state(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "display": self._display,
            "xkb_options": self._original_xkb_options,
            "gnome_extensions": list(self._disabled_gnome_extensions),
            "gsettings": [
                {
                    "schema": binding.schema,
                    "key": binding.key,
                    "value": value,
                }
                for binding, value in self._original_gsettings.items()
            ],
        }
        tmp_path = self._state_path.with_suffix(f"{self._state_path.suffix}.tmp")
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tmp_path.replace(self._state_path)

    def _load_state(self) -> None:
        try:
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Falha ao ler estado de lockdown %s: %s", self._state_path, exc)
            return

        if not self._original_gsettings:
            for item in payload.get("gsettings", []):
                schema = item.get("schema")
                key = item.get("key")
                value = item.get("value")
                if isinstance(schema, str) and isinstance(key, str) and isinstance(value, str):
                    self._original_gsettings[_GSettingsKey(schema=schema, key=key)] = value

        if self._original_xkb_options is None:
            xkb_options = payload.get("xkb_options")
            self._original_xkb_options = xkb_options if isinstance(xkb_options, str) else ""
        if not self._disabled_gnome_extensions:
            extensions = payload.get("gnome_extensions")
            if isinstance(extensions, list):
                self._disabled_gnome_extensions = [
                    extension for extension in extensions if isinstance(extension, str)
                ]

    def _clear_state(self) -> None:
        try:
            self._state_path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Falha ao remover estado de lockdown %s: %s", self._state_path, exc)
