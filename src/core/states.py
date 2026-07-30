"""Estados canônicos da sessão e da estação.

Fonte de verdade única para "em que estado está a estação". Antes, o mesmo
vocabulário existia em três lugares — ``SessionState`` e ``StationMode`` aqui,
a colapsagem dos dois em ``SessionManager._station_status`` e o enum
``StationStatus`` do dashboard — mantidos consistentes à mão.

Este módulo é deliberadamente **só stdlib**: o dashboard precisa dos mesmos
estados sem arrastar ``cv2``/``dlib``/``boto3`` junto, o que aconteceria se
importasse de ``src.core.session``.
"""

from __future__ import annotations

from enum import Enum


class SessionState(str, Enum):
    """Estado da FSM de sessão de prova."""

    IDLE = "IDLE"
    IDENTIFYING = "IDENTIFYING"
    SESSION = "SESSION"
    BLOCKED = "BLOCKED"
    UPLOADING = "UPLOADING"


class StationMode(str, Enum):
    """Modo operacional da estação, independente do estado da sessão."""

    MAINTENANCE = "MAINTENANCE"
    EXAM_READY = "EXAM_READY"
    WAITING_STUDENT = "WAITING_STUDENT"
    SESSION = "SESSION"


#: Status que só existem na visão do dashboard, sem contrapartida na NUC.
#: ``COMPLETED`` é atribuído ao finalizar a sessão; ``OFFLINE`` quando o
#: heartbeat expira.
DASHBOARD_ONLY_STATUSES = frozenset({"COMPLETED", "OFFLINE"})


def derive_station_status(state: SessionState, mode: StationMode) -> str:
    """Colapsa (estado de sessão, modo da estação) no status único do heartbeat.

    Regra: uma sessão em andamento sempre vence; sem sessão, um modo de prova
    ativo é o que interessa; fora disso a estação está ociosa.
    """
    if state != SessionState.IDLE:
        return state.value
    if mode in {StationMode.EXAM_READY, StationMode.WAITING_STUDENT}:
        return mode.value
    return SessionState.IDLE.value


def known_station_statuses() -> frozenset[str]:
    """Vocabulário completo que o dashboard precisa aceitar.

    ``src.dashboard.models.StationStatus`` é mantido explícito por
    legibilidade (templates e store referenciam membros pelo nome), mas
    ``tests/test_dashboard.py`` compara os dois conjuntos para que um estado
    novo aqui não passe silenciosamente sem representação lá.
    """
    return frozenset(
        {member.value for member in SessionState}
        | {member.value for member in StationMode}
        | DASHBOARD_ONLY_STATUSES
    )
