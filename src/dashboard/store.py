"""Armazenamento do dashboard com cache em memória e persistência Postgres."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from datetime import datetime, timezone
from threading import Lock
from uuid import uuid4

import boto3
import psycopg
from psycopg.rows import dict_row

from src.dashboard.models import (
    CommandRecord,
    CommandType,
    EnrollmentRecord,
    EventSeverity,
    ExamConfigPayload,
    SessionEventPayload,
    SessionRecord,
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
                (self._hydrate_session(session) for session in self._sessions.values()),
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
                station.enroll_status = payload.enroll_status
                station.enroll_message = payload.enroll_message
            if payload.update_status is not None:
                station.update_status = payload.update_status
                station.update_message = payload.update_message
            station.seconds_remaining = payload.seconds_remaining
            station.last_seen_at = datetime.now(timezone.utc)
            station.last_event = payload.last_event
            station.recent_events = payload.recent_events[-10:]

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

    def finalize_session(self, session_id: str, ended_at: datetime | None = None) -> SessionRecord | None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            session.ended_at = ended_at or datetime.now(timezone.utc)
            if session.status != StationStatus.CANCELLED_TIMEOUT:
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
            commands = [command.model_copy(deep=True) for command in station.pending_commands]
            station.pending_commands.clear()
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

    def subscribe(self) -> asyncio.Queue[dict[str, object]]:
        queue: asyncio.Queue[dict[str, object]] = asyncio.Queue()
        self._subscribers.add(queue)
        queue.put_nowait(self.snapshot())
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, object]]) -> None:
        self._subscribers.discard(queue)

    def _broadcast(self) -> None:
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
            key = (event.timestamp.isoformat(), event.event_type, event.frame_number)
            if key in known_keys:
                continue
            session.events.append(event)
            if event.severity in {EventSeverity.WARNING, EventSeverity.CRITICAL}:
                session.flags_count += 1
            known_keys.add(key)
        session.events.sort(key=lambda item: item.timestamp)
