"""Testes do Índice de Integridade da Sessão (src/dashboard/integrity_score.py).

Ver design em Obsidian: Ninja/Pendentes/Proctor Station - Índice de
Integridade da Sessão (Nota Auxiliar).md
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.dashboard.integrity_score import compute_integrity_score
from src.dashboard.models import EventSeverity, SessionEventPayload, SessionRecord, StationStatus

_START = datetime(2026, 9, 7, 10, 0, 0, tzinfo=timezone.utc)


def _event(event_type: str, offset_seconds: float, severity: EventSeverity = EventSeverity.WARNING):
    return SessionEventPayload(
        timestamp=_START + timedelta(seconds=offset_seconds),
        event_type=event_type,
        severity=severity,
    )


def _session(events: list[SessionEventPayload], *, duration_seconds: float = 2400) -> SessionRecord:
    return SessionRecord(
        session_id="s1",
        station_id="nuc-1",
        turma="ES2025-T1",
        assessment="P1",
        started_at=_START,
        ended_at=_START + timedelta(seconds=duration_seconds),
        status=StationStatus.COMPLETED,
        events=events,
    )


def test_sessao_sem_eventos_e_score_maximo():
    result = compute_integrity_score(_session([]))
    assert result.score == 100
    assert result.band == "A"
    assert result.blocked_seconds == 0


def test_poucos_warnings_nao_derrubam_muito_o_score():
    events = [_event("GAZE_WARNING", 10), _event("GAZE_WARNING", 20)]
    result = compute_integrity_score(_session(events))
    assert result.score >= 95
    assert result.band == "A"


def test_muitos_warnings_picados_penalizam_com_raiz_quadrada_nao_linear():
    # 25 GAZE_WARNING: penalidade = 1 * sqrt(25) = 5, não 25.
    events = [_event("GAZE_WARNING", i) for i in range(25)]
    result = compute_integrity_score(_session(events))
    assert result.event_penalty == pytest.approx(5.0, abs=0.1)
    assert result.score == 95


def test_bloqueio_resolvido_penaliza_por_contagem_e_tempo():
    events = [
        _event("GAZE_BLOCKED", 100, EventSeverity.CRITICAL),
        _event("SESSION_RESUMED", 160, EventSeverity.INFO),  # 60s bloqueado
    ]
    result = compute_integrity_score(_session(events, duration_seconds=2400))
    assert result.blocked_seconds == 60
    # penalidade evento: 8*sqrt(1)=8; penalidade tempo: 60/2400*100=2.5
    assert result.event_penalty == pytest.approx(8.0)
    assert result.time_penalty == pytest.approx(2.5)
    assert result.score == 90  # round(100 - 8 - 2.5) == round(89.5) == 90


def test_different_user_pesa_mais_que_gaze():
    gaze_result = compute_integrity_score(_session([_event("GAZE_BLOCKED", 10, EventSeverity.CRITICAL)]))
    different_user_result = compute_integrity_score(
        _session([_event("DIFFERENT_USER_BLOCKED", 10, EventSeverity.CRITICAL)])
    )
    assert different_user_result.score < gaze_result.score


def test_bloqueio_nunca_resolvido_conta_ate_o_fim_da_sessao():
    events = [_event("ABSENCE_BLOCKED", 2340, EventSeverity.CRITICAL)]  # sessão de 2400s, bloqueia 60s antes do fim
    result = compute_integrity_score(_session(events, duration_seconds=2400))
    assert result.blocked_seconds == 60


def test_block_timeout_cancelled_capa_o_score_em_20():
    events = [
        _event("ABSENCE_BLOCKED", 10, EventSeverity.CRITICAL),
        _event("BLOCK_TIMEOUT_CANCELLED", 40, EventSeverity.CRITICAL),
    ]
    result = compute_integrity_score(_session(events, duration_seconds=60))
    assert result.capped_by_timeout is True
    assert result.score <= 20
    assert result.band == "E"


def test_faixas_cobrem_toda_a_escala_0_a_100():
    from src.dashboard.integrity_score import _band_for

    assert _band_for(100) == ("A", "Sem irregularidades relevantes")
    assert _band_for(90) == ("A", "Sem irregularidades relevantes")
    assert _band_for(89) == ("B", "Ruído leve, provavelmente falso-positivo")
    assert _band_for(75) == ("B", "Ruído leve, provavelmente falso-positivo")
    assert _band_for(74) == ("C", "Vale espiar o vídeo")
    assert _band_for(55) == ("C", "Vale espiar o vídeo")
    assert _band_for(54) == ("D", "Revisão recomendada")
    assert _band_for(30) == ("D", "Revisão recomendada")
    assert _band_for(29) == ("E", "Revisão obrigatória — indício forte de irregularidade")
    assert _band_for(0) == ("E", "Revisão obrigatória — indício forte de irregularidade")


def test_score_nunca_sai_do_intervalo_0_100():
    # Sessão curtíssima com muitos eventos críticos — garante clamp em 0, não negativo.
    events = [_event("DIFFERENT_USER_BLOCKED", i, EventSeverity.CRITICAL) for i in range(20)]
    result = compute_integrity_score(_session(events, duration_seconds=30))
    assert 0 <= result.score <= 100


def test_sessao_sem_ended_at_usa_now_sem_lancar_excecao():
    session = SessionRecord(
        session_id="s2",
        station_id="nuc-1",
        turma="ES2025-T1",
        assessment="P1",
        started_at=_START,
        ended_at=None,
        status=StationStatus.SESSION,
        events=[_event("GAZE_WARNING", 5)],
    )
    result = compute_integrity_score(session)
    assert 0 <= result.score <= 100
