"""Lockdown de teclado durante sessão de prova.

Placeholder — implementação completa na M7 (hardening).

Por ora não faz nada, mas mantém a interface para que o
test_integration.py não precise mudar quando o lockdown for implementado.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class Lockdown:
    """Interface de lockdown de teclado — implementação completa na M7.

    Args:
        display: Display X11 (reservado para M7).
    """

    def __init__(self, display: str = ":1"):
        self._display = display
        self._enabled = False

    def enable(self) -> None:
        """Ativa o lockdown (no-op até M7)."""
        self._enabled = True
        logger.info("Lockdown de teclado: pendente (M7)")

    def disable(self) -> None:
        """Desativa o lockdown (no-op até M7)."""
        self._enabled = False

    @property
    def is_enabled(self) -> bool:
        return self._enabled