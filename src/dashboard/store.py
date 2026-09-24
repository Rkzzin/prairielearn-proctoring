"""Armazenamento do dashboard com cache em memória e persistência Postgres."""

from __future__ import annotations

import asyncio
import hashlib
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from uuid import uuid4

import boto3
import psycopg
from botocore.exceptions import ClientError
from psycopg.rows import dict_row

from src.dashboard.models import (
    CameraSnapshotRecord,
    CommandRecord,
    CommandType,
    EnrollmentRecord,
    EventSeverity,
    EventSnapshotRecord,
    ExamConfigPayload,
    NotificationSettings,
    SessionEmailReport,
    SessionEventPayload,
    SessionRecord,
    SessionReviewStatus,
    StationHeartbeat,
    StationRecord,
    StationStatus,
)


class DashboardStore:
    def __init__(self, database_url: str, *, app_config=None, s3_client=None):
        if not database_url:
            raise ValueError(
                "PROCTOR_DASHBOARD_DATABASE_URL não configurado — obrigatório para "
                "o dashboard (ver docs/setup_dashboard.md)."
            )
        self._db = psycopg.connect(database_url, row_factory=dict_row)
        self._app_cfg = app_config
        self._s3 = s3_client or self._default_s3_client()
        self._stations: dict[str, StationRecord] = {}
        self._sessions: dict[str, SessionRecord] = {}
        self._enrollments: dict[str, EnrollmentRecord] = {}
        self._configs: list[ExamConfigPayload] = []
        self._roster: dict[tuple[str, str], str] = {}
        self._subscribers: set[asyncio.Queue[dict[str, object]]] = set()
        self._lock = Lock()
        self._init_db()
        self._load_from_db()

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            now = datetime.now(timezone.utc)
            stations = [
                station.model_copy(update={"status": station.effective_status(now)})
                for station in sorted(self._stations.values(), key=lambda item: item.station_id)
            ]
            sessions = sorted(
                # The dashboard overview never renders recording URLs. Signing every
                # recording here makes each heartbeat and status change depend on S3.
                (session.model_copy(deep=True) for session in self._sessions.values()),
                key=lambda item: item.started_at,
                reverse=True,
            )
            enrollments = sorted(
                self._enrollments.values(),
                key=lambda item: item.created_at,
                reverse=True,
            )
            configs = list(self._configs)

        return {
            "stations": stations,
            "sessions": sessions,
            "enrollments": enrollments,
            "configs": configs,
        }

    def list_stations(self) -> list[StationRecord]:
        return self.snapshot()["stations"]  # type: ignore[return-value]

    def list_sessions(self) -> list[SessionRecord]:
        return self.snapshot()["sessions"]  # type: ignore[return-value]

    def clear_sessions(self) -> int:
        """Remove o histórico local de sessões conhecidas do dashboard."""
        with self._lock:
            removed = len(self._sessions)
            self._sessions.clear()
            self._db.execute("DELETE FROM session_email_reports")
            self._db.execute("DELETE FROM event_snapshots")
            self._db.execute("DELETE FROM sessions")
            self._db.commit()

        self._broadcast()
        return removed

    def list_enrollments(self) -> list[EnrollmentRecord]:
        return self.snapshot()["enrollments"]  # type: ignore[return-value]

    def list_known_turmas(self) -> list[str]:
        with self._lock:
            turmas = {
                enrollment.turma
                for enrollment in self._enrollments.values()
                if enrollment.turma
            }
            turmas.update(config.turma for config in self._configs if config.turma)
            turmas.update(session.turma for session in self._sessions.values() if session.turma)
            turmas.update(station.turma for station in self._stations.values() if station.turma)
        return sorted(turmas)

    def get_session(self, session_id: str) -> SessionRecord | None:
        with self._lock:
            session = self._sessions.get(session_id)
            return self._hydrate_session(session) if session else None

    def has_session(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._sessions

    def adjacent_sessions(
        self,
        session_id: str,
    ) -> tuple[SessionRecord | None, SessionRecord | None]:
        """Return the chronologically adjacent sessions without hydrating recordings."""
        with self._lock:
            sessions = sorted(self._sessions.values(), key=lambda session: session.started_at)
            for index, session in enumerate(sessions):
                if session.session_id != session_id:
                    continue
                previous = sessions[index - 1] if index else None
                following = sessions[index + 1] if index + 1 < len(sessions) else None
                return (
                    previous.model_copy(deep=True) if previous else None,
                    following.model_copy(deep=True) if following else None,
                )
        return None, None

    def queue_event_snapshots(self, session_id: str) -> int:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or session.ended_at is None:
                return 0
            queued = 0
            snapshot_events = [
                SessionEventPayload(
                    timestamp=session.started_at,
                    event_type="AUTHENTICATION_FRAME",
                    severity=EventSeverity.INFO,
                ),
                *(
                    SessionEventPayload(
                        timestamp=session.started_at + timedelta(seconds=offset),
                        event_type="PERIODIC_FRAME",
                        severity=EventSeverity.INFO,
                    )
                    for offset in range(60, (session.duration_seconds or 0) + 1, 60)
                ),
                *(
                    event
                    for event in session.events
                    if event.severity in {EventSeverity.WARNING, EventSeverity.CRITICAL}
                ),
            ]
            for event in snapshot_events:
                identity = (
                    f"{session_id}|{event.timestamp.isoformat()}|"
                    f"{event.event_type}|{event.frame_number}"
                )
                event_key = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
                payload = EventSnapshotRecord(
                    session_id=session_id,
                    event_key=event_key,
                    event_timestamp=event.timestamp,
                    event_type=event.event_type,
                    severity=event.severity,
                    frame_number=event.frame_number,
                )
                result = self._db.execute(
                    "INSERT INTO event_snapshots (session_id, event_key, payload) "
                    "VALUES (%s, %s, %s::jsonb) ON CONFLICT DO NOTHING",
                    (session_id, event_key, payload.model_dump_json()),
                )
                queued += result.rowcount
            self._db.commit()
            return queued

    def retry_event_snapshots(self, session_id: str) -> int:
        self.queue_event_snapshots(session_id)
        with self._lock:
            session = self._sessions.get(session_id)
            has_environment_recording = bool(
                session
                and any(self._is_environment(asset) for asset in session.recordings)
            )
            rows = self._db.execute(
                "SELECT event_key, payload FROM event_snapshots WHERE session_id = %s",
                (session_id,),
            ).fetchall()
            queued = 0
            for row in rows:
                snapshot = EventSnapshotRecord.model_validate(row["payload"])
                if snapshot.status == "ready" and (
                    not has_environment_recording or snapshot.environment_s3_key
                ):
                    continue
                snapshot.status = "queued"
                snapshot.error = None
                self._save_event_snapshot(snapshot)
                queued += 1
            self._db.commit()
            return queued

    def claim_event_snapshots(self, session_id: str) -> list[EventSnapshotRecord]:
        with self._lock:
            rows = self._db.execute(
                "SELECT payload FROM event_snapshots "
                "WHERE session_id = %s AND payload->>'status' = 'queued'",
                (session_id,),
            ).fetchall()
            snapshots = [EventSnapshotRecord.model_validate(row["payload"]) for row in rows]
            for snapshot in snapshots:
                snapshot.status = "processing"
                self._save_event_snapshot(snapshot)
            self._db.commit()
            return snapshots

    def recover_event_snapshot_session_ids(self) -> list[str]:
        """Recoloca trabalhos interrompidos na fila após reinício do dashboard."""
        with self._lock:
            rows = self._db.execute(
                "SELECT session_id, payload FROM event_snapshots "
                "WHERE payload->>'status' IN ('queued', 'processing')"
            ).fetchall()
            for row in rows:
                snapshot = EventSnapshotRecord.model_validate(row["payload"])
                if snapshot.status == "processing":
                    snapshot.status = "queued"
                    self._save_event_snapshot(snapshot)
            self._db.commit()
            return list(dict.fromkeys(row["session_id"] for row in rows))

    def finish_event_snapshot(
        self,
        snapshot: EventSnapshotRecord,
        *,
        s3_bucket: str | None = None,
        s3_key: str | None = None,
        environment_s3_bucket: str | None = None,
        environment_s3_key: str | None = None,
        environment_error: str | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            snapshot.status = "failed" if error else "ready"
            snapshot.s3_bucket = s3_bucket
            snapshot.s3_key = s3_key
            snapshot.environment_s3_bucket = environment_s3_bucket
            snapshot.environment_s3_key = environment_s3_key
            snapshot.environment_error = environment_error
            snapshot.error = error
            self._save_event_snapshot(snapshot)
            self._db.commit()

    def list_event_snapshots(self, session_id: str) -> list[EventSnapshotRecord]:
        with self._lock:
            rows = self._db.execute(
                "SELECT payload FROM event_snapshots WHERE session_id = %s "
                "ORDER BY (payload->>'event_timestamp')::timestamptz",
                (session_id,),
            ).fetchall()
            snapshots = [EventSnapshotRecord.model_validate(row["payload"]) for row in rows]
            return [self._hydrate_event_snapshot(snapshot) for snapshot in snapshots]

    def get_event_snapshot(
        self,
        session_id: str,
        event_key: str,
    ) -> EventSnapshotRecord | None:
        with self._lock:
            row = self._db.execute(
                "SELECT payload FROM event_snapshots WHERE session_id = %s AND event_key = %s",
                (session_id, event_key),
            ).fetchone()
            if row is None:
                return None
            return self._hydrate_event_snapshot(
                EventSnapshotRecord.model_validate(row["payload"])
            )

    def read_event_snapshot_image(self, snapshot: EventSnapshotRecord) -> bytes | None:
        if snapshot.local_path and self._app_cfg is not None:
            path = Path(self._app_cfg.data_dir) / snapshot.local_path
            if path.is_file():
                return path.read_bytes()
        if not snapshot.s3_bucket or not snapshot.s3_key or self._s3 is None:
            return None
        response = self._s3.get_object(Bucket=snapshot.s3_bucket, Key=snapshot.s3_key)
        return response["Body"].read()

    def read_student_photo(self, turma: str, student_id: str) -> bytes | None:
        if self._app_cfg is None or self._s3 is None:
            return None
        prefix = self._app_cfg.s3.photos_prefix_for_turma(turma)
        for extension in (".png", ".jpg", ".jpeg"):
            try:
                response = self._s3.get_object(
                    Bucket=self._app_cfg.s3.bucket,
                    Key=f"{prefix}{student_id}{extension}",
                )
                return response["Body"].read()
            except ClientError:
                continue
        return None

    def get_notification_settings(self) -> NotificationSettings:
        with self._lock:
            row = self._db.execute(
                "SELECT payload FROM notification_settings WHERE id = 1"
            ).fetchone()
            if row is not None:
                return NotificationSettings.model_validate(row["payload"])
        return NotificationSettings(
            ses_region=(self._app_cfg.s3.region if self._app_cfg is not None else "sa-east-1"),
            public_dashboard_url=(
                self._app_cfg.dashboard.base_url if self._app_cfg is not None else ""
            ),
        )

    def save_notification_settings(
        self,
        settings: NotificationSettings,
    ) -> NotificationSettings:
        with self._lock:
            self._db.execute(
                "INSERT INTO notification_settings (id, payload) VALUES (1, %s::jsonb) "
                "ON CONFLICT (id) DO UPDATE SET payload = EXCLUDED.payload",
                (settings.model_dump_json(),),
            )
            self._db.commit()
        return settings.model_copy(deep=True)

    def queue_email_report(
        self,
        session_id: str,
        settings: NotificationSettings,
        *,
        force: bool = False,
        status: str = "queued",
    ) -> bool:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or session.ended_at is None:
                return False
            existing_row = self._db.execute(
                "SELECT payload FROM session_email_reports WHERE session_id = %s",
                (session_id,),
            ).fetchone()
            existing = (
                SessionEmailReport.model_validate(existing_row["payload"])
                if existing_row
                else None
            )
            if existing is not None and existing.status in {"queued", "sending", "sent"} and not force:
                return False
            report = SessionEmailReport(
                session_id=session_id,
                status=status,
                sender_email=settings.sender_email,
                recipient_emails=settings.recipient_emails,
                image_link_limit=settings.image_link_limit,
                delivery_provider=settings.delivery_provider,
                ses_region=settings.ses_region,
                public_dashboard_url=settings.public_dashboard_url,
                attempts=existing.attempts if existing else 0,
            )
            self._save_email_report(report)
            self._db.commit()
            return True

    def claim_email_report(self, session_id: str) -> SessionEmailReport | None:
        with self._lock:
            row = self._db.execute(
                "SELECT payload FROM session_email_reports WHERE session_id = %s "
                "FOR UPDATE SKIP LOCKED",
                (session_id,),
            ).fetchone()
            if row is None:
                self._db.rollback()
                return None
            report = SessionEmailReport.model_validate(row["payload"])
            if report.status != "queued":
                self._db.rollback()
                return None
            report.status = "sending"
            report.attempts += 1
            report.claim_id = uuid4().hex
            report.claimed_at = datetime.now(timezone.utc)
            report.error = None
            self._save_email_report(report)
            self._db.commit()
            return report

    def finish_email_report(
        self,
        report: SessionEmailReport,
        *,
        message_id: str | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            claim_id = report.claim_id
            report.status = "failed" if error else "sent"
            report.error = error
            report.ses_message_id = message_id
            report.claim_id = None
            report.claimed_at = None
            report.sent_at = None if error else datetime.now(timezone.utc)
            self._db.execute(
                "UPDATE session_email_reports SET payload = %s::jsonb "
                "WHERE session_id = %s AND payload->>'status' = 'sending' "
                "AND payload->>'claim_id' = %s",
                (report.model_dump_json(), report.session_id, claim_id),
            )
            self._db.commit()

    def get_email_report(self, session_id: str) -> SessionEmailReport | None:
        with self._lock:
            row = self._db.execute(
                "SELECT payload FROM session_email_reports WHERE session_id = %s",
                (session_id,),
            ).fetchone()
            return SessionEmailReport.model_validate(row["payload"]) if row else None

    def pending_email_report_ids(self) -> list[str]:
        with self._lock:
            rows = self._db.execute(
                "SELECT session_id, payload FROM session_email_reports "
                "WHERE payload->>'status' IN ('queued', 'sending')"
            ).fetchall()
            pending = []
            stale_before = datetime.now(timezone.utc) - timedelta(minutes=15)
            for row in rows:
                report = SessionEmailReport.model_validate(row["payload"])
                if report.status == "sending":
                    if report.claimed_at is not None and report.claimed_at > stale_before:
                        continue
                    report.status = "queued"
                    report.claim_id = None
                    report.claimed_at = None
                    self._save_email_report(report)
                pending.append(row["session_id"])
            self._db.commit()
            return pending

    def activate_waiting_email_report(self, session_id: str) -> bool:
        with self._lock:
            row = self._db.execute(
                "SELECT payload FROM session_email_reports WHERE session_id = %s "
                "FOR UPDATE SKIP LOCKED",
                (session_id,),
            ).fetchone()
            if row is None:
                self._db.rollback()
                return False
            report = SessionEmailReport.model_validate(row["payload"])
            if report.status != "waiting_snapshots":
                self._db.rollback()
                return False
            report.status = "queued"
            self._save_email_report(report)
            self._db.commit()
            return True

    def waiting_email_report_ids(self) -> list[str]:
        with self._lock:
            rows = self._db.execute(
                "SELECT session_id FROM session_email_reports "
                "WHERE payload->>'status' = 'waiting_snapshots'"
            ).fetchall()
            return [row["session_id"] for row in rows]

    def get_station(self, station_id: str) -> StationRecord | None:
        with self._lock:
            station = self._stations.get(station_id)
            return station.model_copy(deep=True) if station else None

    def create_station(
        self,
        station_id: str,
        station_name: str,
        token_hash: str,
    ) -> StationRecord:
        """Cadastra uma estação offline e sua credencial sem tocar no histórico."""
        with self._lock:
            token_exists = self._db.execute(
                "SELECT 1 FROM station_tokens WHERE station_id = %s",
                (station_id,),
            ).fetchone()
            if station_id in self._stations or token_exists:
                raise ValueError(f"A estação '{station_id}' já existe.")

            station = StationRecord(
                station_id=station_id,
                station_name=station_name,
                status=StationStatus.OFFLINE,
                last_seen_at=datetime.fromtimestamp(0, tz=timezone.utc),
            )
            try:
                self._db.execute(
                    "INSERT INTO stations (station_id, payload) VALUES (%s, %s::jsonb)",
                    (station.station_id, station.model_dump_json()),
                )
                self._db.execute(
                    "INSERT INTO station_tokens (station_id, token_hash, label) "
                    "VALUES (%s, %s, %s)",
                    (station_id, token_hash, station_name),
                )
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise
            self._stations[station_id] = station
            result = station.model_copy(deep=True)

        self._broadcast()
        return result

    def delete_station(self, station_id: str) -> bool:
        """Remove estação e token; sessões anteriores permanecem para auditoria."""
        with self._lock:
            if station_id not in self._stations:
                return False
            try:
                self._db.execute("DELETE FROM station_tokens WHERE station_id = %s", (station_id,))
                self._db.execute("DELETE FROM stations WHERE station_id = %s", (station_id,))
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise
            del self._stations[station_id]

        self._broadcast()
        return True

    def upsert_station_heartbeat(self, payload: StationHeartbeat) -> StationRecord:
        with self._lock:
            station = self._stations.get(payload.station_id)
            if station is None:
                station = StationRecord(
                    station_id=payload.station_id,
                    station_name=payload.station_name or payload.station_id,
                )
                self._stations[payload.station_id] = station

            if payload.station_name:
                station.station_name = payload.station_name
            station.status = payload.status
            station.mode = payload.mode
            station.student = payload.student
            station.active_session_id = payload.active_session_id
            station.assessment = payload.assessment
            station.turma = payload.turma
            if payload.auto_start_enabled is not None:
                station.auto_start_enabled = payload.auto_start_enabled
            if payload.enroll_status is not None:
                # The heartbeat delivering RUN_ENROLL was built before the NUC
                # processed the response. Keep queued until the next heartbeat
                # acknowledges running/done/error.
                if not (
                    station.enroll_status == "queued"
                    and payload.enroll_status == "idle"
                ):
                    station.enroll_status = payload.enroll_status
                    station.enroll_message = payload.enroll_message
            if payload.update_status is not None:
                station.update_status = payload.update_status
                station.update_message = payload.update_message
            if payload.available_cameras is not None:
                station.available_cameras = payload.available_cameras
            station.electronic_device_calibration_supported = bool(
                payload.electronic_device_calibration_supported
            )
            if (
                payload.camera_capture_status is not None
                and (
                    station.camera_capture_batch_id is None
                    or payload.camera_capture_batch_id == station.camera_capture_batch_id
                )
            ):
                station.camera_capture_status = payload.camera_capture_status
                station.camera_capture_message = payload.camera_capture_message or ""
                station.camera_capture_batch_id = payload.camera_capture_batch_id
            station.seconds_remaining = payload.seconds_remaining
            station.last_seen_at = datetime.now(timezone.utc)
            recent_events = [
                event
                for event in payload.recent_events
                if not self._is_unconfirmed_different_user(event)
            ]
            station.last_event = (
                payload.last_event
                if payload.last_event is not None
                and not self._is_unconfirmed_different_user(payload.last_event)
                else (recent_events[-1] if recent_events else None)
            )
            station.recent_events = recent_events[-10:]
            if (
                payload.status == StationStatus.BLOCKED
                and station.assigned_config is not None
                and station.assigned_config.flexible_mode
                and not any(
                    command.command_type == CommandType.UNBLOCK_SESSION
                    for command in station.pending_commands
                )
            ):
                station.pending_commands.append(
                    CommandRecord(
                        command_id=str(uuid4()),
                        station_id=payload.station_id,
                        command_type=CommandType.UNBLOCK_SESSION,
                        issued_at=datetime.now(timezone.utc),
                    )
                )

            if payload.active_session_id:
                session = self._sessions.get(payload.active_session_id)
                if session is None:
                    session = SessionRecord(
                        session_id=payload.active_session_id,
                        station_id=payload.station_id,
                        turma=payload.turma or "unknown",
                        assessment=payload.assessment or "unknown",
                        started_at=datetime.now(timezone.utc),
                        student=payload.student,
                        timer_minutes=max(1, int((payload.seconds_remaining or 0) / 60) or 45),
                        status=payload.status,
                    )
                    self._sessions[session.session_id] = session
                else:
                    session.status = payload.status
                    session.student = payload.student or session.student
                    session.assessment = payload.assessment or session.assessment
                    session.turma = payload.turma or session.turma
                if payload.recent_events:
                    self._merge_events(session, payload.recent_events)

            result = station.model_copy(deep=True)
            self._save_station(station)
            if payload.active_session_id:
                self._save_session(self._sessions[payload.active_session_id])

        self._broadcast()
        return result

    def register_session(self, payload: SessionRecord) -> SessionRecord:
        with self._lock:
            self._sessions[payload.session_id] = payload.model_copy(deep=True)
            station = self._stations.get(payload.station_id)
            if station:
                station.active_session_id = payload.session_id
                station.student = payload.student
                station.assessment = payload.assessment
                station.turma = payload.turma
                station.status = payload.status
            result = payload.model_copy(deep=True)
            self._save_session(self._sessions[payload.session_id])
            if station:
                self._save_station(station)

        self._broadcast()
        return result

    def register_unrecognized_authentication(
        self,
        *,
        station_id: str,
        turma: str,
        assessment: str,
        attempted_at: datetime,
        local_path: str,
    ) -> SessionRecord:
        session_id = f"auth-alert-{uuid4().hex}"
        event_key = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:24]
        event = SessionEventPayload(
            timestamp=attempted_at,
            event_type="UNRECOGNIZED_AUTHENTICATION",
            severity=EventSeverity.CRITICAL,
            details={"category": "AUTHENTICATION_ALERT"},
        )
        session = SessionRecord(
            session_id=session_id,
            station_id=station_id,
            turma=turma,
            assessment=assessment,
            started_at=attempted_at,
            ended_at=attempted_at,
            status=StationStatus.TIMEOUT,
            flags_count=1,
            events=[event],
            category="AUTHENTICATION_ALERT",
        )
        snapshot = EventSnapshotRecord(
            session_id=session_id,
            event_key=event_key,
            event_timestamp=attempted_at,
            event_type=event.event_type,
            severity=event.severity,
            status="ready",
            local_path=local_path,
        )
        with self._lock:
            self._sessions[session_id] = session
            self._save_session(session)
            self._save_event_snapshot(snapshot)
            self._db.commit()
        self._broadcast()
        return session.model_copy(deep=True)

    def finalize_session(self, session_id: str, ended_at: datetime | None = None) -> SessionRecord | None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            session.ended_at = ended_at or datetime.now(timezone.utc)
            if session.status != StationStatus.TIMEOUT:
                session.status = StationStatus.COMPLETED
            station = self._stations.get(session.station_id)
            if station:
                station.status = StationStatus.IDLE
                station.mode = "MAINTENANCE"
                station.active_session_id = None
                station.student = None
                station.seconds_remaining = None
            result = session.model_copy(deep=True)
            self._save_session(session)
            if station:
                self._save_station(station)

        self._broadcast()
        return result

    def set_session_review_status(
        self, session_id: str, review_status: SessionReviewStatus
    ) -> SessionRecord | None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            session.review_status = review_status
            result = session.model_copy(deep=True)
            self._save_session(session)

        self._broadcast()
        return result

    def append_events(self, session_id: str, events: list[SessionEventPayload]) -> SessionRecord | None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            self._merge_events(session, events)
            result = session.model_copy(deep=True)
            self._save_session(session)

        self._broadcast()
        return result

    def create_config(self, payload: ExamConfigPayload) -> ExamConfigPayload:
        with self._lock:
            stored = payload.model_copy(deep=True)
            self._configs.insert(0, stored)
            for station_id in payload.target_station_ids:
                station = self._stations.get(station_id)
                if station is None:
                    station = StationRecord(station_id=station_id, station_name=station_id)
                    self._stations[station_id] = station
                station.assigned_config = stored
                station.pending_commands.append(
                    CommandRecord(
                        command_id=str(uuid4()),
                        station_id=station_id,
                        command_type=CommandType.APPLY_CONFIG,
                        issued_at=datetime.now(timezone.utc),
                        payload=stored.model_dump(mode="json"),
                    )
                )
            result = stored.model_copy(deep=True)
            self._insert_config(stored)
            for station_id in payload.target_station_ids:
                self._save_station(self._stations[station_id])

        self._broadcast()
        return result

    def enqueue_command(self, station_id: str, command_type: CommandType) -> CommandRecord:
        with self._lock:
            station = self._stations.get(station_id)
            if station is None:
                station = StationRecord(station_id=station_id, station_name=station_id)
                self._stations[station_id] = station
            command = CommandRecord(
                command_id=str(uuid4()),
                station_id=station_id,
                command_type=command_type,
                issued_at=datetime.now(timezone.utc),
            )
            station.pending_commands.append(command)
            result = command.model_copy(deep=True)
            self._save_station(station)

        self._broadcast()
        return result

    def queue_camera_snapshots(
        self,
        *,
        calibrate_electronics: bool = False,
    ) -> dict[str, object]:
        """Enfileira um lote para todas as estações online em manutenção."""
        batch_id = str(uuid4())
        queued: list[str] = []
        skipped: list[dict[str, str]] = []
        now = datetime.now(timezone.utc)
        with self._lock:
            for station in self._stations.values():
                status = station.effective_status(now)
                capture_in_progress = (
                    station.camera_capture_status in {"queued", "running"}
                    and station.camera_capture_requested_at is not None
                    and now - station.camera_capture_requested_at <= timedelta(seconds=30)
                )
                ready = (
                    status in {
                        StationStatus.IDLE,
                        StationStatus.EXAM_READY,
                        StationStatus.WAITING_STUDENT,
                    }
                    and station.active_session_id is None
                    and bool(station.available_cameras)
                    and station.enroll_status != "running"
                    and station.update_status != "running"
                    and not capture_in_progress
                    and (
                        not calibrate_electronics
                        or station.electronic_device_calibration_supported
                    )
                )
                if not ready:
                    if capture_in_progress:
                        skipped.append({"station_id": station.station_id, "reason": "captura em andamento"})
                        continue
                    if (
                        calibrate_electronics
                        and not station.electronic_device_calibration_supported
                    ):
                        reason = "sem suporte à calibração"
                    elif status == StationStatus.OFFLINE:
                        reason = "offline"
                    elif not station.available_cameras:
                        reason = "sem suporte à captura"
                    else:
                        reason = "com avaliação ativa"
                    station.camera_snapshots = []
                    station.camera_capture_batch_id = batch_id
                    station.camera_capture_requested_at = now
                    station.camera_capture_status = "skipped"
                    station.camera_capture_message = f"Ignorada: estação {reason}"
                    skipped.append({"station_id": station.station_id, "reason": reason})
                else:
                    station.camera_snapshots = []
                    station.camera_capture_batch_id = batch_id
                    station.camera_capture_requested_at = now
                    station.camera_capture_status = "queued"
                    station.camera_capture_message = (
                        "Aguardando calibração da estação..."
                        if calibrate_electronics
                        else "Aguardando a estação..."
                    )
                    station.pending_commands = [
                        command
                        for command in station.pending_commands
                        if command.command_type != CommandType.CAPTURE_CAMERA_SNAPSHOTS
                    ]
                    station.pending_commands.append(
                        CommandRecord(
                            command_id=str(uuid4()),
                            station_id=station.station_id,
                            command_type=CommandType.CAPTURE_CAMERA_SNAPSHOTS,
                            issued_at=now,
                            payload={
                                "batch_id": batch_id,
                                "calibrate_electronics": calibrate_electronics,
                            },
                        )
                    )
                    queued.append(station.station_id)
                self._save_station(station)
        self._broadcast()
        return {"batch_id": batch_id, "queued_station_ids": queued, "skipped": skipped}

    def store_camera_snapshot(self, station_id: str, snapshot: CameraSnapshotRecord) -> StationRecord:
        with self._lock:
            station = self._validate_camera_snapshot_locked(
                station_id,
                snapshot.batch_id,
                snapshot.camera_index,
            )
            station.camera_snapshots = [
                item for item in station.camera_snapshots if item.camera_index != snapshot.camera_index
            ]
            if len(station.camera_snapshots) >= 8:
                raise ValueError("Limite de fotos do lote atingido")
            station.camera_snapshots.append(snapshot)
            station.camera_snapshots.sort(key=lambda item: item.camera_index)
            station.camera_capture_status = "running"
            station.camera_capture_message = f"{len(station.camera_snapshots)} foto(s) recebida(s)"
            self._save_station(station)
            result = station.model_copy(deep=True)
        self._broadcast()
        return result

    def validate_camera_snapshot(self, station_id: str, batch_id: str, camera_index: int) -> None:
        with self._lock:
            self._validate_camera_snapshot_locked(station_id, batch_id, camera_index)

    def _validate_camera_snapshot_locked(
        self,
        station_id: str,
        batch_id: str,
        camera_index: int,
    ) -> StationRecord:
        station = self._stations.get(station_id)
        if station is None:
            raise KeyError(station_id)
        if station.camera_capture_batch_id != batch_id:
            raise ValueError("Lote de captura expirado")
        if station.camera_capture_status not in {"queued", "running"}:
            raise ValueError("Lote de captura já encerrado")
        if (
            station.camera_capture_requested_at is None
            or datetime.now(timezone.utc) - station.camera_capture_requested_at > timedelta(minutes=2)
        ):
            raise ValueError("Lote de captura expirado")
        if camera_index not in {camera.index for camera in station.available_cameras}:
            raise ValueError("Câmera não reportada pela estação")
        return station

    def set_station_autostart(self, station_id: str, enabled: bool) -> CommandRecord:
        with self._lock:
            station = self._stations.get(station_id)
            if station is None:
                station = StationRecord(station_id=station_id, station_name=station_id)
                self._stations[station_id] = station
            station.auto_start_enabled = enabled
            if station.assigned_config is not None:
                station.assigned_config = station.assigned_config.model_copy(
                    update={"auto_start": enabled}
                )
            command = CommandRecord(
                command_id=str(uuid4()),
                station_id=station_id,
                command_type=CommandType.SET_AUTOSTART,
                issued_at=datetime.now(timezone.utc),
                payload={"auto_start": enabled},
            )
            station.pending_commands.append(command)
            result = command.model_copy(deep=True)
            self._save_station(station)

        self._broadcast()
        return result

    def run_enroll(self, station_id: str, turma_ids: list[str]) -> CommandRecord:
        """Manda a NUC rodar `scripts/enroll.py --force` para cada turma listada.

        Roda de verdade na estação (dlib local, .pkl local) — o enroll via S3
        do próprio dashboard (`/enrollment`) só registra o enrollment no banco
        do dashboard, não gera o .pkl que a NUC usa para identificar aluno.
        """
        with self._lock:
            station = self._stations.get(station_id)
            if station is None:
                station = StationRecord(station_id=station_id, station_name=station_id)
                self._stations[station_id] = station
            station.enroll_status = "queued"
            station.enroll_message = f"{len(turma_ids)} turma(s) na fila"
            command = CommandRecord(
                command_id=str(uuid4()),
                station_id=station_id,
                command_type=CommandType.RUN_ENROLL,
                issued_at=datetime.now(timezone.utc),
                payload={"turma_ids": turma_ids},
            )
            station.pending_commands.append(command)
            result = command.model_copy(deep=True)
            self._save_station(station)

        self._broadcast()
        return result

    def clear_configs(self) -> int:
        """Remove o histórico local de configs distribuídas — não afeta as NUCs."""
        with self._lock:
            removed = len(self._configs)
            self._configs.clear()
            self._db.execute("DELETE FROM configs")
            self._db.commit()

        self._broadcast()
        return removed

    def drain_commands(self, station_id: str) -> list[CommandRecord]:
        with self._lock:
            station = self._stations.get(station_id)
            if station is None:
                return []
            busy = bool(station.active_session_id) or station.status in {
                StationStatus.IDENTIFYING,
                StationStatus.SESSION,
                StationStatus.BLOCKED,
                StationStatus.UPLOADING,
            }
            commands = [
                command.model_copy(deep=True)
                for command in station.pending_commands
                if not (busy and command.command_type == CommandType.UPDATE_AND_REBOOT)
            ]
            station.pending_commands = [
                command
                for command in station.pending_commands
                if busy and command.command_type == CommandType.UPDATE_AND_REBOOT
            ]
            self._save_station(station)

        self._broadcast()
        return commands

    def add_enrollment(
        self,
        turma: str,
        student_id: str,
        student_name: str,
        source: str,
        file_names: list[str],
    ) -> EnrollmentRecord:
        with self._lock:
            record = EnrollmentRecord(
                enrollment_id=str(uuid4()),
                turma=turma,
                student_id=student_id,
                student_name=student_name,
                created_at=datetime.now(timezone.utc),
                source=source,
                file_names=file_names,
            )
            self._enrollments[record.enrollment_id] = record
            result = record.model_copy(deep=True)
            self._save_enrollment(record)

        self._broadcast()
        return result

    def import_roster(self, turma: str, entries: list[tuple[str, str]]) -> int:
        """Substitui o roster (login → nome) de uma turma pelo CSV importado."""
        with self._lock:
            self._db.execute("DELETE FROM roster_entries WHERE turma = %s", (turma,))
            for login, student_name in entries:
                self._db.execute(
                    "INSERT INTO roster_entries (turma, login, student_name) VALUES (%s, %s, %s) "
                    "ON CONFLICT (turma, login) DO UPDATE SET student_name = EXCLUDED.student_name",
                    (turma, login, student_name),
                )
            self._db.commit()
            self._roster = {key: value for key, value in self._roster.items() if key[0] != turma}
            self._roster.update({(turma, login): student_name for login, student_name in entries})
        self._broadcast()
        return len(entries)

    def roster_name(self, turma: str, login: str) -> str | None:
        if not login:
            return None
        with self._lock:
            return self._roster.get((turma, login.strip().lower()))

    def subscribe(self) -> asyncio.Queue[dict[str, object]]:
        queue: asyncio.Queue[dict[str, object]] = asyncio.Queue()
        self._subscribers.add(queue)
        queue.put_nowait(self.snapshot())
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, object]]) -> None:
        self._subscribers.discard(queue)

    def _broadcast(self) -> None:
        if not self._subscribers:
            return
        payload = self.snapshot()
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(deepcopy(payload))
            except asyncio.QueueFull:
                pass

    def _init_db(self) -> None:
        for statement in (
            """
            CREATE TABLE IF NOT EXISTS stations (
              station_id TEXT PRIMARY KEY,
              payload JSONB NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS sessions (
              session_id TEXT PRIMARY KEY,
              payload JSONB NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS event_snapshots (
              session_id TEXT NOT NULL,
              event_key TEXT NOT NULL,
              payload JSONB NOT NULL,
              PRIMARY KEY (session_id, event_key)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS notification_settings (
              id SMALLINT PRIMARY KEY CHECK (id = 1),
              payload JSONB NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS session_email_reports (
              session_id TEXT PRIMARY KEY,
              payload JSONB NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS enrollments (
              enrollment_id TEXT PRIMARY KEY,
              payload JSONB NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS configs (
              id BIGSERIAL PRIMARY KEY,
              payload JSONB NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS credentials (
              username TEXT PRIMARY KEY,
              password_hash TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS station_tokens (
              station_id TEXT PRIMARY KEY,
              token_hash TEXT NOT NULL,
              label TEXT,
              created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS roster_entries (
              turma TEXT NOT NULL,
              login TEXT NOT NULL,
              student_name TEXT NOT NULL,
              PRIMARY KEY (turma, login)
            )
            """,
        ):
            self._db.execute(statement)
        self._db.commit()

    def get_credential_hash(self, username: str) -> str | None:
        with self._lock:
            row = self._db.execute(
                "SELECT password_hash FROM credentials WHERE username = %s",
                (username,),
            ).fetchone()
        return row["password_hash"] if row else None

    def ensure_credential(self, username: str, password_hash: str) -> None:
        """Insere a credencial só se `username` ainda não existir (não sobrescreve)."""
        with self._lock:
            self._db.execute(
                "INSERT INTO credentials (username, password_hash) VALUES (%s, %s) "
                "ON CONFLICT (username) DO NOTHING",
                (username, password_hash),
            )
            self._db.commit()

    def get_station_token_hash(self, station_id: str) -> str | None:
        with self._lock:
            row = self._db.execute(
                "SELECT token_hash FROM station_tokens WHERE station_id = %s",
                (station_id,),
            ).fetchone()
        return row["token_hash"] if row else None

    def set_station_token_hash(self, station_id: str, token_hash: str, *, label: str | None = None) -> None:
        """Grava (ou sobrescreve) o hash do token de uma estação — reemitir revoga o anterior."""
        with self._lock:
            self._db.execute(
                "INSERT INTO station_tokens (station_id, token_hash, label) VALUES (%s, %s, %s) "
                "ON CONFLICT (station_id) DO UPDATE SET token_hash = EXCLUDED.token_hash, "
                "label = EXCLUDED.label, created_at = now()",
                (station_id, token_hash, label),
            )
            self._db.commit()

    def _load_from_db(self) -> None:
        self._stations = {
            row["station_id"]: StationRecord.model_validate(row["payload"])
            for row in self._db.execute("SELECT station_id, payload FROM stations")
        }
        self._sessions = {
            row["session_id"]: SessionRecord.model_validate(row["payload"])
            for row in self._db.execute("SELECT session_id, payload FROM sessions")
        }
        self._enrollments = {
            row["enrollment_id"]: EnrollmentRecord.model_validate(row["payload"])
            for row in self._db.execute("SELECT enrollment_id, payload FROM enrollments")
        }
        self._configs = [
            ExamConfigPayload.model_validate(row["payload"])
            for row in self._db.execute("SELECT payload FROM configs ORDER BY id DESC")
        ]
        self._roster = {
            (row["turma"], row["login"]): row["student_name"]
            for row in self._db.execute("SELECT turma, login, student_name FROM roster_entries")
        }

    def _default_s3_client(self):
        if self._app_cfg is None:
            return None
        try:
            return boto3.client("s3", region_name=self._app_cfg.s3.region)
        except Exception:
            return None

    def _save_station(self, station: StationRecord) -> None:
        self._db.execute(
            "INSERT INTO stations (station_id, payload) VALUES (%s, %s::jsonb) "
            "ON CONFLICT (station_id) DO UPDATE SET payload = EXCLUDED.payload",
            (station.station_id, station.model_dump_json()),
        )
        self._db.commit()

    def _save_session(self, session: SessionRecord) -> None:
        self._db.execute(
            "INSERT INTO sessions (session_id, payload) VALUES (%s, %s::jsonb) "
            "ON CONFLICT (session_id) DO UPDATE SET payload = EXCLUDED.payload",
            (session.session_id, session.model_dump_json()),
        )
        self._db.commit()

    def _save_event_snapshot(self, snapshot: EventSnapshotRecord) -> None:
        self._db.execute(
            "INSERT INTO event_snapshots (session_id, event_key, payload) VALUES (%s, %s, %s::jsonb) "
            "ON CONFLICT (session_id, event_key) DO UPDATE SET payload = EXCLUDED.payload",
            (snapshot.session_id, snapshot.event_key, snapshot.model_dump_json()),
        )

    def _save_email_report(self, report: SessionEmailReport) -> None:
        self._db.execute(
            "INSERT INTO session_email_reports (session_id, payload) VALUES (%s, %s::jsonb) "
            "ON CONFLICT (session_id) DO UPDATE SET payload = EXCLUDED.payload",
            (report.session_id, report.model_dump_json()),
        )

    def _save_enrollment(self, enrollment: EnrollmentRecord) -> None:
        self._db.execute(
            "INSERT INTO enrollments (enrollment_id, payload) VALUES (%s, %s::jsonb) "
            "ON CONFLICT (enrollment_id) DO UPDATE SET payload = EXCLUDED.payload",
            (enrollment.enrollment_id, enrollment.model_dump_json()),
        )
        self._db.commit()

    def _insert_config(self, config: ExamConfigPayload) -> None:
        self._db.execute(
            "INSERT INTO configs (payload) VALUES (%s::jsonb)",
            (config.model_dump_json(),),
        )
        self._db.commit()

    def _hydrate_session(self, session: SessionRecord) -> SessionRecord:
        hydrated = session.model_copy(deep=True)
        hydrated.recordings = [self._hydrate_asset(asset) for asset in hydrated.recordings]
        return hydrated

    def _hydrate_event_snapshot(self, snapshot: EventSnapshotRecord) -> EventSnapshotRecord:
        hydrated = snapshot.model_copy(deep=True)
        if hydrated.s3_bucket and hydrated.s3_key and self._s3 is not None:
            try:
                hydrated.url = self._s3.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": hydrated.s3_bucket, "Key": hydrated.s3_key},
                    ExpiresIn=3600,
                )
            except Exception:
                pass
        if (
            hydrated.environment_s3_bucket
            and hydrated.environment_s3_key
            and self._s3 is not None
        ):
            try:
                hydrated.environment_url = self._s3.generate_presigned_url(
                    "get_object",
                    Params={
                        "Bucket": hydrated.environment_s3_bucket,
                        "Key": hydrated.environment_s3_key,
                    },
                    ExpiresIn=3600,
                )
            except Exception:
                pass
        return hydrated

    @staticmethod
    def _is_environment(asset) -> bool:
        return asset.stream == "environment" or (
            asset.stream is None
            and bool(asset.s3_key)
            and Path(asset.s3_key or "").name.startswith("environment_")
        )

    def _hydrate_asset(self, asset):
        if asset.url or not asset.s3_bucket or not asset.s3_key or self._s3 is None:
            return asset
        try:
            signed = self._s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": asset.s3_bucket, "Key": asset.s3_key},
                ExpiresIn=3600,
            )
            return asset.model_copy(update={"url": signed})
        except Exception:
            return asset

    @staticmethod
    def _merge_events(session: SessionRecord, events: list[SessionEventPayload]) -> None:
        known_keys = {
            (
                event.timestamp.isoformat(),
                event.event_type,
                event.frame_number,
            )
            for event in session.events
        }
        for event in events:
            if DashboardStore._is_unconfirmed_different_user(event):
                continue
            key = (event.timestamp.isoformat(), event.event_type, event.frame_number)
            if key in known_keys:
                continue
            session.events.append(event)
            if event.severity in {EventSeverity.WARNING, EventSeverity.CRITICAL}:
                session.flags_count += 1
            known_keys.add(key)
        session.events.sort(key=lambda item: item.timestamp)

    @staticmethod
    def _is_unconfirmed_different_user(event: SessionEventPayload) -> bool:
        return (
            event.event_type in {"DIFFERENT_USER_ALERT", "DIFFERENT_USER_BLOCKED"}
            and event.details.get("detected_status") == "NO_MATCH"
        )
