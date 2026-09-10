"""Índice de Integridade da Sessão — métrica derivada dos eventos de proctoring.

Pedido do professor Corsi (WhatsApp, 04/09/2026): "criar uma métrica (usando
os eventos) que gera uma nota para aquela prova [...] ai facilita o que o
professor deve ir revisar".

IMPORTANTE — isto não é a nota acadêmica da prova (essa é do
PrairieLearn/PrairieTest). É um score auxiliar 0-100 (+ faixa A-E) que sinaliza
o quanto vale a pena o professor gastar tempo revisando o vídeo daquela sessão
manualmente. Design completo, pesos e justificativa em:
    Obsidian: Ninja/Pendentes/Proctor Station - Índice de Integridade da
    Sessão (Nota Auxiliar).md

Função pura: consome `SessionRecord.events` (já carregado do Postgres/memória
pelo DashboardStore), não faz I/O nem toca em nada do lado da NUC.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

from src.dashboard.models import SessionEventPayload, SessionRecord

#: Peso de cada event_type na penalidade por contagem (seção "Contagem
#: ponderada de eventos" da nota). Tipos fora deste dict não penalizam
#: (ex.: SESSION_STARTED/SESSION_ENDED/SESSION_RESUMED são informativos).
EVENT_WEIGHTS: dict[str, float] = {
    "GAZE_WARNING": 1.0,
    "ABSENCE_WARNING": 1.0,
    "GAZE_BLOCKED": 8.0,
    "ABSENCE_BLOCKED": 8.0,
    "MULTI_FACE_BLOCKED": 12.0,
    "DIFFERENT_USER_BLOCKED": 25.0,
    "GAZE_ALERT": 8.0,
    "ABSENCE_ALERT": 8.0,
    "MULTI_FACE_ALERT": 12.0,
    "DIFFERENT_USER_ALERT": 25.0,
    "BROWSER_EXIT_ALERT": 12.0,
    "ELECTRONIC_DEVICE_DETECTED": 8.0,
}

#: event_types que representam início de um bloqueio (abrem um intervalo de
#: "tempo bloqueado" até o próximo SESSION_RESUMED ou o fim da sessão).
_BLOCK_START_TYPES = frozenset(
    {"GAZE_BLOCKED", "ABSENCE_BLOCKED", "MULTI_FACE_BLOCKED", "DIFFERENT_USER_BLOCKED"}
)
_SESSION_RESUMED = "SESSION_RESUMED"
_BLOCK_TIMEOUT_CANCELLED = "BLOCK_TIMEOUT_CANCELLED"

#: Teto da penalidade por tempo bloqueado (pontos) — evita que uma sessão
#: inteira bloqueada por um único incidente já zere o score sozinha, deixando
#: espaço pra penalidade por contagem também pesar.
_MAX_TIME_PENALTY = 40.0

#: Score máximo permitido quando a sessão terminou por BLOCK_TIMEOUT_CANCELLED
#: (prova encerrada à força) — não é uma sessão "quase perfeita com um evento",
#: é uma sessão que não terminou de forma normal.
_TIMEOUT_CANCELLED_SCORE_CAP = 20

#: Limites (inclusive) de cada faixa, do pior pro melhor.
_BANDS: tuple[tuple[int, str, str], ...] = (
    (90, "A", "Sem irregularidades relevantes"),
    (75, "B", "Ruído leve, provavelmente falso-positivo"),
    (55, "C", "Vale espiar o vídeo"),
    (30, "D", "Revisão recomendada"),
    (0, "E", "Revisão obrigatória — indício forte de irregularidade"),
)


@dataclass
class IntegrityScore:
    """Resultado do cálculo, com breakdown pra exibir "por que esse score"."""

    score: int
    band: str
    band_label: str
    event_penalty: float
    time_penalty: float
    blocked_seconds: int
    duration_seconds: int
    event_counts: dict[str, int] = field(default_factory=dict)
    capped_by_timeout: bool = False


def _band_for(score: int) -> tuple[str, str]:
    for threshold, band, label in _BANDS:
        if score >= threshold:
            return band, label
    return _BANDS[-1][1], _BANDS[-1][2]  # pragma: no cover - _BANDS cobre 0..100


def _blocked_seconds(events: list[SessionEventPayload], session_end: datetime) -> int:
    """Soma o tempo entre cada evento de bloqueio e o próximo SESSION_RESUMED.

    Se um bloqueio nunca é seguido de SESSION_RESUMED (sessão terminou
    bloqueada, ex. BLOCK_TIMEOUT_CANCELLED), conta até o fim da sessão.
    Bloqueios sobrepostos (não deveria acontecer na FSM real, que é um estado
    absorvente único, mas a timeline é só uma lista) não são somados em
    dobro — um novo bloqueio enquanto já há um aberto não abre outro
    intervalo.
    """
    ordered = sorted(events, key=lambda e: e.timestamp)
    total = 0.0
    open_since: datetime | None = None
    for event in ordered:
        ts = event.timestamp.astimezone(timezone.utc)
        if event.event_type in _BLOCK_START_TYPES:
            if open_since is None:
                open_since = ts
        elif event.event_type == _SESSION_RESUMED and open_since is not None:
            total += (ts - open_since).total_seconds()
            open_since = None
    if open_since is not None:
        total += (session_end.astimezone(timezone.utc) - open_since).total_seconds()
    return max(0, int(total))


def compute_integrity_score(session: SessionRecord) -> IntegrityScore:
    """Calcula o Índice de Integridade de uma sessão a partir dos eventos.

    Não modifica `session`. Seguro de chamar em sessões sem eventos (score
    100/A) ou sem `ended_at` (usa `datetime.now()` como fim, mesma convenção
    de `SessionRecord.duration_seconds`).
    """
    events = session.events
    duration = session.duration_seconds or 0
    session_end = session.ended_at or datetime.now(timezone.utc)

    counts: dict[str, int] = {}
    for event in events:
        counts[event.event_type] = counts.get(event.event_type, 0) + 1

    event_penalty = sum(
        EVENT_WEIGHTS[event_type] * math.sqrt(count)
        for event_type, count in counts.items()
        if event_type in EVENT_WEIGHTS
    )

    blocked = _blocked_seconds(events, session_end)
    time_penalty = min(_MAX_TIME_PENALTY, (blocked / duration) * 100) if duration > 0 else 0.0

    raw_score = 100.0 - event_penalty - time_penalty
    score = max(0, min(100, round(raw_score)))

    capped = any(event.event_type == _BLOCK_TIMEOUT_CANCELLED for event in events)
    if capped:
        score = min(score, _TIMEOUT_CANCELLED_SCORE_CAP)

    band, band_label = _band_for(score)

    return IntegrityScore(
        score=score,
        band=band,
        band_label=band_label,
        event_penalty=round(event_penalty, 1),
        time_penalty=round(time_penalty, 1),
        blocked_seconds=blocked,
        duration_seconds=duration,
        event_counts=counts,
        capped_by_timeout=capped,
    )
