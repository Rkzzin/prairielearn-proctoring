"""Modelos do dashboard do professor."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum

from pydantic import BaseModel, Field


class StationStatus(str, Enum):
    IDLE = "IDLE"
    SESSION = "SESSION"
    BLOCKED = "BLOCKED"
    UPLOADING = "UPLOADING"
    OFFLINE = "OFFLINE"


class CommandType(str, Enum):
    APPLY_CONFIG = "APPLY_CONFIG"
    STOP_SESSION = "STOP_SESSION"
    UNBLOCK_SESSION = "UNBLOCK_SESSION"


class EventSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class StudentInfo(BaseModel):
    student_id: str
    student_name: str


class SessionEventPayload(BaseModel):
    timestamp: datetime
    frame_number: int = 0
    event_type: str
    severity: EventSeverity = EventSeverity.INFO
    details: dict[str, object] = Field(default_factory=dict)


class RecordingAsset(BaseModel):
    label: str
    url: str | None = None
    s3_bucket: str | None = None
    s3_key: str | None = None
    kind: str = "video"


class SessionRecord(BaseModel):
    session_id: str
    station_id: str
    turma: str
    assessment: str
    started_at: datetime
    ended_at: datetime | None = None
    timer_minutes: int = 45
    student: StudentInfo | None = None
    status: StationStatus = StationStatus.SESSION
    flags_count: int = 0
    events: list[SessionEventPayload] = Field(default_factory=list)
    recordings: list[RecordingAsset] = Field(default_factory=list)

    @property
    def duration_seconds(self) -> int | None:
        end = self.ended_at or datetime.now(timezone.utc)
        if self.started_at is None:
            return None
        return max(0, int((end - self.started_at).total_seconds()))


class ExamConfigPayload(BaseModel):
    turma: str
    assessment: str
    timer_minutes: int = 45
    prairielearn_url: str
    allowlist: list[str] = Field(default_factory=list)
    target_station_ids: list[str] = Field(default_factory=list)
    gaze_h_threshold: float = 0.35
    gaze_duration_sec: float = 3.0
    absence_timeout_sec: float = 5.0
    multi_face_block: bool = True
    s3_prefix: str = ""


class CommandRecord(BaseModel):
    command_id: str
    station_id: str
    command_type: CommandType
    issued_at: datetime
    payload: dict[str, object] = Field(default_factory=dict)
    delivered: bool = False


class StationHeartbeat(BaseModel):
    station_id: str
    station_name: str | None = None
    status: StationStatus
    student: StudentInfo | None = None
    active_session_id: str | None = None
    assessment: str | None = None
    turma: str | None = None
    seconds_remaining: int | None = None
    last_event: SessionEventPayload | None = None
    recent_events: list[SessionEventPayload] = Field(default_factory=list)


class StationRecord(BaseModel):
    station_id: str
    station_name: str
    status: StationStatus = StationStatus.IDLE
    student: StudentInfo | None = None
    active_session_id: str | None = None
    assessment: str | None = None
    turma: str | None = None
    seconds_remaining: int | None = None
    last_seen_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_event: SessionEventPayload | None = None
    recent_events: list[SessionEventPayload] = Field(default_factory=list)
    pending_commands: list[CommandRecord] = Field(default_factory=list)
    assigned_config: ExamConfigPayload | None = None

    def effective_status(self, now: datetime | None = None, offline_after_sec: int = 15) -> StationStatus:
        reference = now or datetime.now(timezone.utc)
        if reference - self.last_seen_at > timedelta(seconds=offline_after_sec):
            return StationStatus.OFFLINE
        return self.status


class EnrollmentRecord(BaseModel):
    enrollment_id: str
    turma: str
    student_id: str
    student_name: str
    created_at: datetime
    source: str
    file_names: list[str] = Field(default_factory=list)
