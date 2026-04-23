"""Armazenamento do dashboard com cache em memória e persistência SQLite."""

from __future__ import annotations

import asyncio
import os
import sqlite3
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from uuid import uuid4

import boto3

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
    def __init__(self, data_dir: Path, app_config=None, s3_client=None):
        self.data_dir = self._prepare_data_dir(data_dir)
        self.upload_dir = self.data_dir / "enrollment_uploads"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "dashboard.sqlite3"
        self._db = sqlite3.connect(self.db_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
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

    @staticmethod
    def _prepare_data_dir(data_dir: Path) -> Path:
        try:
            data_dir.mkdir(parents=True, exist_ok=True)
            return data_dir
        except OSError:
            fallback = Path(os.getcwd()) / ".localdata" / "dashboard"
            fallback.mkdir(parents=True, exist_ok=True)
            return fallback

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

    def list_enrollments(self) -> list[EnrollmentRecord]:
        return self.snapshot()["enrollments"]  # type: ignore[return-value]

    def get_session(self, session_id: str) -> SessionRecord | None:
        with self._lock:
            session = self._sessions.get(session_id)
            return self._hydrate_session(session) if session else None

    def get_station(self, station_id: str) -> StationRecord | None:
        with self._lock:
            station = self._stations.get(station_id)
            return station.model_copy(deep=True) if station else None

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
            station.student = payload.student
            station.active_session_id = payload.active_session_id
            station.assessment = payload.assessment
            station.turma = payload.turma
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
            session.status = StationStatus.UPLOADING
            station = self._stations.get(session.station_id)
            if station:
                station.status = StationStatus.UPLOADING
                station.active_session_id = None
                station.student = None
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
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS stations (
              station_id TEXT PRIMARY KEY,
              payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
              session_id TEXT PRIMARY KEY,
              payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS enrollments (
              enrollment_id TEXT PRIMARY KEY,
              payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS configs (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              payload TEXT NOT NULL
            );
            """
        )
        self._db.commit()

    def _load_from_db(self) -> None:
        self._stations = {
            row["station_id"]: StationRecord.model_validate_json(row["payload"])
            for row in self._db.execute("SELECT station_id, payload FROM stations")
        }
        self._sessions = {
            row["session_id"]: SessionRecord.model_validate_json(row["payload"])
            for row in self._db.execute("SELECT session_id, payload FROM sessions")
        }
        self._enrollments = {
            row["enrollment_id"]: EnrollmentRecord.model_validate_json(row["payload"])
            for row in self._db.execute("SELECT enrollment_id, payload FROM enrollments")
        }
        self._configs = [
            ExamConfigPayload.model_validate_json(row["payload"])
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
            "INSERT OR REPLACE INTO stations (station_id, payload) VALUES (?, ?)",
            (station.station_id, station.model_dump_json()),
        )
        self._db.commit()

    def _save_session(self, session: SessionRecord) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO sessions (session_id, payload) VALUES (?, ?)",
            (session.session_id, session.model_dump_json()),
        )
        self._db.commit()

    def _save_enrollment(self, enrollment: EnrollmentRecord) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO enrollments (enrollment_id, payload) VALUES (?, ?)",
            (enrollment.enrollment_id, enrollment.model_dump_json()),
        )
        self._db.commit()

    def _insert_config(self, config: ExamConfigPayload) -> None:
        self._db.execute(
            "INSERT INTO configs (payload) VALUES (?)",
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
