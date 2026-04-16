"""Event bus e log de timestamps do proctoring engine.

Cada sessão de prova gera um arquivo JSONL separado:
    {data_dir}/sessions/{session_id}/events.jsonl

Formato de cada linha:
    {"timestamp": 1719000000.0, "frame": 4520, "type": "GAZE_WARNING",
     "severity": "WARNING", "details": {"yaw": 35.2, "pitch": -2.1}}
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from src.core.config import AppConfig


class EventType(str, Enum):
    """Tipos de evento gerados pelo proctoring engine."""

    # Olhar desviado (warning inicial)
    GAZE_WARNING = "GAZE_WARNING"
    # Olhar desviado por tempo suficiente → bloqueio
    GAZE_BLOCKED = "GAZE_BLOCKED"
    # Nenhum rosto detectado por tempo suficiente
    ABSENCE_WARNING = "ABSENCE_WARNING"
    ABSENCE_BLOCKED = "ABSENCE_BLOCKED"
    # Múltiplos rostos detectados
    MULTI_FACE_BLOCKED = "MULTI_FACE_BLOCKED"
    # Aluno retornou ao normal após bloqueio
    SESSION_RESUMED = "SESSION_RESUMED"
    # Marcadores de início/fim de sessão
    SESSION_STARTED = "SESSION_STARTED"
    SESSION_ENDED = "SESSION_ENDED"


class Severity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class ProctorEvent:
    """Representa um evento de proctoring."""

    timestamp: float
    frame: int
    type: str        # EventType value
    severity: str    # Severity value
    details: dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> "ProctorEvent":
        return cls(**json.loads(line))


class EventLogger:
    """Escreve eventos em JSONL para uma sessão específica.

    O arquivo de log é criado em:
        {app_config.data_dir}/sessions/{session_id}/events.jsonl

    Args:
        session_id: Identificador único da sessão (ex: "ES2025-T1_20240601_143000").
        app_config: Configuração da aplicação. Se None, usa o singleton padrão.
    """

    def __init__(self, session_id: str, app_config: AppConfig | None = None):
        cfg = app_config or AppConfig()
        self.session_id = session_id

        session_dir = Path(cfg.data_dir) / "sessions" / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = session_dir / "events.jsonl"

        self._file = open(self.log_path, "a", encoding="utf-8", buffering=1)  # line-buffered

    def log(self, event: ProctorEvent) -> None:
        """Escreve um evento no arquivo JSONL."""
        self._file.write(event.to_json() + "\n")

    def log_event(
        self,
        frame: int,
        event_type: EventType,
        severity: Severity,
        details: dict[str, Any] | None = None,
    ) -> ProctorEvent:
        """Cria e persiste um evento em uma única chamada."""
        event = ProctorEvent(
            timestamp=time.time(),
            frame=frame,
            type=event_type.value,
            severity=severity.value,
            details=details or {},
        )
        self.log(event)
        return event

    def close(self) -> None:
        """Fecha o arquivo de log."""
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> "EventLogger":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ── Leitura (para replay e testes) ──────────────

    def read_all(self) -> list[ProctorEvent]:
        """Lê todos os eventos do arquivo de log da sessão atual."""
        events = []
        if not self.log_path.exists():
            return events
        with open(self.log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(ProctorEvent.from_json(line))
                    except (json.JSONDecodeError, TypeError):
                        continue
        return events

    @staticmethod
    def read_session(log_path: Path) -> list[ProctorEvent]:
        """Lê eventos de qualquer arquivo JSONL (para revisão pós-prova)."""
        events = []
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(ProctorEvent.from_json(line))
                    except (json.JSONDecodeError, TypeError):
                        continue
        return events