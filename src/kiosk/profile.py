"""Limpeza do perfil do Chromium usado pela prova."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

_SAFE_NAME_MARKERS = ("proctor", "chromium", "chrome")


class ProfileCleanupError(RuntimeError):
    """Falha operacional ao limpar o perfil de browser."""


class ChromiumProfileCleaner:
    def __init__(self, profile_dir: Path | str):
        self.profile_dir = Path(profile_dir)

    def cleanup(self) -> None:
        """Remove dados persistentes do browser.

        A limpeza apaga o diretorio inteiro do perfil dedicado. Isso e mais
        confiavel do que tentar remover cookies/storage seletivamente.
        """
        if not self.profile_dir.exists():
            return
        self._validate_target()
        try:
            shutil.rmtree(self.profile_dir)
        except OSError as exc:
            raise ProfileCleanupError(
                f"Falha ao limpar perfil Chromium em {self.profile_dir}: {exc}"
            ) from exc
        logger.info("Perfil Chromium limpo: %s", self.profile_dir)

    def _validate_target(self) -> None:
        resolved = self.profile_dir.resolve()
        if resolved == Path("/") or len(resolved.parts) < 3:
            raise ProfileCleanupError(f"Diretorio de perfil inseguro: {resolved}")
        lowered = resolved.name.lower()
        if not any(marker in lowered for marker in _SAFE_NAME_MARKERS):
            raise ProfileCleanupError(
                f"Diretorio de perfil nao parece ser dedicado ao Proctor: {resolved}"
            )
