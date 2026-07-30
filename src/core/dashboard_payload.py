"""Montagem dos payloads que a NUC envia ao dashboard.

Tradução pura de estado interno (``SessionRuntime``/``SessionConfig``) para os
formatos de wire do dashboard. Fica fora do ``SessionManager`` porque não toca
FSM, lock nem hardware — é só mapeamento.

Motivo de existir como módulo próprio: o mesmo dicionário de evento era montado
em dois lugares (``SessionManager._collect_dashboard_events`` e
``DashboardHeartbeatWorker._read_recent_events``), então uma mudança de formato
precisava ser lembrada duas vezes. Agora ``event_to_payload`` é a única fonte.

Só stdlib + ``src.proctor.events``: nada de cv2/dlib/boto3 aqui.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.proctor.events import EventLogger, ProctorEvent

if TYPE_CHECKING:  # evita import circular com src.core.session em runtime
    from src.core.session import SessionConfig, SessionRuntime

#: Severidades que contam como "flag" na contagem exibida pelo dashboard.
FLAGGED_SEVERITIES = frozenset({"WARNING", "CRITICAL"})


def session_events_path(data_dir: Path, session_id: str) -> Path:
    """Caminho do JSONL de eventos de uma sessão."""
    return Path(data_dir) / "sessions" / session_id / "events.jsonl"


def event_to_payload(event: ProctorEvent) -> dict[str, Any]:
    """Converte um ``ProctorEvent`` no formato ``SessionEventPayload``.

    Fonte única do formato: usado tanto na coleta final da sessão quanto na
    leitura incremental do heartbeat.
    """
    return {
        "timestamp": datetime.fromtimestamp(event.timestamp, tz=timezone.utc).isoformat(),
        "frame_number": event.frame,
        "event_type": event.type,
        "severity": event.severity,
        "details": event.details,
    }


def collect_session_events(data_dir: Path, session_id: str) -> list[dict[str, Any]]:
    """Lê o JSONL completo da sessão e devolve os eventos no formato do dashboard."""
    log_path = session_events_path(data_dir, session_id)
    if not log_path.exists():
        return []
    return [event_to_payload(event) for event in EventLogger.read_session(log_path)]


def collect_session_recordings(uploader: Any, bucket: str) -> list[dict[str, Any]]:
    """Descreve os segmentos já enviados ao S3 como assets do dashboard.

    Só entra o que o uploader confirmou como enviado — segmentos que falharam
    ficam de fora de propósito, para o dashboard não oferecer vídeo inexistente.
    """
    if uploader is None:
        return []
    return [
        {
            "label": f"{segment.stream.capitalize()} {segment.index:03d}",
            "s3_bucket": bucket,
            "s3_key": s3_key,
            "kind": "video",
        }
        for segment, s3_key in uploader.uploaded_segments
    ]


def build_station_snapshot(
    *,
    status: dict[str, Any],
    config: SessionConfig,
    runtime: SessionRuntime | None,
) -> dict[str, Any]:
    """Monta o corpo do heartbeat (``StationHeartbeat``) da estação."""
    student = None
    if runtime is not None:
        student = {
            "student_id": runtime.student_id,
            "student_name": runtime.student_name,
        }

    return {
        "station_id": config.station_id,
        "station_name": config.station_name,
        "status": status["station_status"],
        "mode": status["mode"],
        "student": student,
        "active_session_id": status["session_id"],
        "assessment": status["assessment"],
        "turma": status["turma_id"],
        "auto_start_enabled": config.auto_start,
        "seconds_remaining": status["seconds_remaining"],
        "recent_events": [],
    }


def build_session_payload(
    *,
    target: SessionRuntime,
    station_id: str,
) -> dict[str, Any]:
    """Monta o registro de sessão (``SessionRecord``) enviado ao dashboard.

    ``events``/``recordings`` vêm de ``target.notes``, preenchidos no
    encerramento da sessão por ``collect_session_events``/
    ``collect_session_recordings``. Note o rename ``turma_id`` → ``turma``: o
    dashboard usa o nome curto.
    """
    events = target.notes.get("dashboard_events", [])
    return {
        "session_id": target.session_id,
        "station_id": station_id,
        "turma": target.turma_id,
        "assessment": target.assessment,
        "started_at": target.started_at.isoformat(),
        "ended_at": target.stopped_at.isoformat() if target.stopped_at else None,
        "timer_minutes": target.timer_minutes,
        "student": {
            "student_id": target.student_id,
            "student_name": target.student_name,
        },
        "status": target.state.value,
        "flags_count": sum(
            1 for event in events if event["severity"] in FLAGGED_SEVERITIES
        ),
        "events": events,
        "recordings": target.notes.get("dashboard_recordings", []),
    }
