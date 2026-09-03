"""Aplicação FastAPI do dashboard do professor."""

from __future__ import annotations

import csv
import hmac
import json
import re
import secrets
import unicodedata
from datetime import timezone
from io import StringIO
from pathlib import Path

import anyio
from fastapi import (
    Depends,
    FastAPI,
    Form,
    HTTPException,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.core.config import AppConfig
from src.dashboard.auth import hash_password, parse_basic_auth, verify_password
from src.dashboard.enrollment_service import S3EnrollmentError, S3EnrollmentService
from src.dashboard.models import (
    CommandType,
    ExamConfigPayload,
    SessionEventPayload,
    SessionRecord,
    StationCreatePayload,
    StationHeartbeat,
)
from src.dashboard.store import DashboardStore

#: Rotas que só a NUC chama — autenticadas por token de estação
#: (`require_station_token`), não pela senha do professor. O middleware de
#: Basic Auth abaixo pula exatamente estas.
_STATION_EXACT_ROUTES = {("POST", "/api/heartbeats"), ("POST", "/api/sessions")}
_STATION_SESSION_ACTION_RE = re.compile(r"^/api/sessions/[^/]+/(finalize|events)$")
_EVENT_CLIP_CONTEXT_SECONDS = 5
_LEGACY_SEGMENT_DURATION_SECONDS = 300
_EVENT_REASON_LABELS = {
    "SESSION_STARTED": "Avaliação iniciada",
    "SESSION_ENDED": "Avaliação finalizada",
    "SESSION_RESUMED": "Identidade confirmada; avaliação retomada",
    "GAZE_WARNING": "Olhar desviado detectado",
    "GAZE_BLOCKED": "Avaliação pausada por olhar desviado",
    "ABSENCE_WARNING": "Aluno não foi detectado pela câmera",
    "ABSENCE_BLOCKED": "Avaliação pausada por ausência",
    "MULTI_FACE_BLOCKED": "Avaliação pausada: mais de uma pessoa detectada",
    "DIFFERENT_USER_BLOCKED": "Avaliação pausada: usuário diferente detectado",
    "BLOCK_TIMEOUT_CANCELLED": "Avaliação cancelada: bloqueio não resolvido no prazo",
    "BROWSER_EXIT": "Avaliação pausada: navegador protegido encerrado",
}
_SESSION_STATUS_LABELS = {
    "COMPLETED": "Concluída",
    "TIMEOUT": "Timeout de bloqueio",
    "SESSION": "Em andamento",
    "BLOCKED": "Pausada",
    "UPLOADING": "Finalizando",
}


def _is_station_route(method: str, path: str) -> bool:
    if (method, path) in _STATION_EXACT_ROUTES:
        return True
    return method == "POST" and bool(_STATION_SESSION_ACTION_RE.match(path))


def _station_id_from_name(station_name: str) -> str:
    normalized = unicodedata.normalize("NFKD", station_name)
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii").lower()
    station_id = re.sub(r"[^a-z0-9]+", "-", ascii_name).strip("-")
    return (station_id or "estacao")[:64].rstrip("-")


def create_app(config: AppConfig | None = None, store: DashboardStore | None = None) -> FastAPI:
    app_config = config or AppConfig()
    dashboard_dir = Path(__file__).parent
    templates = Jinja2Templates(directory=str(dashboard_dir / "templates"))
    dashboard_store = store or DashboardStore(
        app_config.dashboard.database_url,
        app_config=app_config,
    )

    app = FastAPI(title="Proctor Station Dashboard")
    app.mount(
        "/static",
        StaticFiles(directory=str(dashboard_dir / "static")),
        name="static",
    )
    app.state.store = dashboard_store
    app.state.templates = templates
    app.state.s3_enrollment_service = S3EnrollmentService(app_config)

    auth_username = app_config.dashboard.admin_user
    if auth_username:
        admin_password = app_config.dashboard.admin_password
        if admin_password and dashboard_store.get_credential_hash(auth_username) is None:
            dashboard_store.ensure_credential(auth_username, hash_password(admin_password))

        @app.middleware("http")
        async def require_basic_auth(request: Request, call_next):
            if _is_station_route(request.method, request.url.path):
                return await call_next(request)

            stored_hash = dashboard_store.get_credential_hash(auth_username)
            credentials = parse_basic_auth(request.headers.get("authorization"))
            authenticated = (
                stored_hash is not None
                and credentials is not None
                and hmac.compare_digest(credentials[0], auth_username)
                and verify_password(credentials[1], stored_hash)
            )
            if not authenticated:
                return Response(
                    status_code=401,
                    headers={"WWW-Authenticate": 'Basic realm="proctor-dashboard"'},
                )
            return await call_next(request)

    async def require_station_token(request: Request) -> str:
        """Autentica a NUC por `X-Station-Id`/`X-Station-Token` — não pela senha do professor.

        Retorna o `station_id` autenticado, pra as rotas conferirem que ele bate
        com o que o corpo/sessão declara (uma estação não pode falar por outra).
        """
        station_id = request.headers.get("x-station-id")
        token = request.headers.get("x-station-token")
        token_hash = dashboard_store.get_station_token_hash(station_id) if station_id else None
        authenticated = (
            station_id is not None
            and token is not None
            and token_hash is not None
            and verify_password(token, token_hash)
        )
        if not authenticated:
            raise HTTPException(status_code=401, detail="Token de estação inválido.")
        return station_id

    def _ensure_station_owns_session(session_id: str, authenticated_station_id: str) -> None:
        session = dashboard_store.get_session(session_id)
        if session is not None and session.station_id != authenticated_station_id:
            raise HTTPException(status_code=403, detail="Sessão pertence a outra estação.")

    def render_template(request: Request, template_name: str, **context: object) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name=template_name,
            context=context,
        )

    @app.get("/", response_class=HTMLResponse)
    async def dashboard_home(request: Request) -> HTMLResponse:
        snapshot = dashboard_store.snapshot()
        return render_template(
            request,
            "dashboard.html",
            title="Dashboard",
            **snapshot,
        )

    @app.get("/config", response_class=HTMLResponse)
    async def config_page(request: Request) -> HTMLResponse:
        # Distribuir config agora é por estação, no modal do painel principal
        # (ver _stations.html/dashboard.html) — esta página só lista o
        # histórico do que já foi distribuído.
        snapshot = dashboard_store.snapshot()
        return render_template(
            request,
            "config.html",
            title="Configurações distribuídas",
            **snapshot,
        )

    @app.get("/enrollment", response_class=HTMLResponse)
    async def enrollment_page(request: Request) -> HTMLResponse:
        snapshot = dashboard_store.snapshot()
        s3_turmas: list[str] = []
        s3_error = None
        try:
            s3_turmas = request.app.state.s3_enrollment_service.list_turmas()
        except Exception as exc:
            s3_error = str(exc)
        return render_template(
            request,
            "enrollment.html",
            title="Enrollment",
            s3_turmas=s3_turmas,
            s3_error=s3_error,
            **snapshot,
        )

    @app.get("/sessions/{session_id}", response_class=HTMLResponse)
    async def session_review(request: Request, session_id: str) -> HTMLResponse:
        session = dashboard_store.get_session(session_id)
        if session is None:
            return HTMLResponse("Sessão não encontrada.", status_code=404)
        timeline = _build_timeline(session)
        return render_template(
            request,
            "session_detail.html",
            title=f"Sessão {session_id}",
            session=session,
            timeline=timeline,
            event_counts=_event_counts(timeline),
            duration_label=_format_duration(session.duration_seconds),
            status_label=_SESSION_STATUS_LABELS.get(
                session.status.value,
                session.status.value.replace("_", " ").title(),
            ),
        )

    @app.get("/partials/stations", response_class=HTMLResponse)
    async def stations_partial(request: Request) -> HTMLResponse:
        return render_template(
            request,
            "_stations.html",
            stations=dashboard_store.list_stations(),
        )

    @app.get("/partials/sessions", response_class=HTMLResponse)
    async def sessions_partial(request: Request) -> HTMLResponse:
        return render_template(
            request,
            "_sessions.html",
            sessions=dashboard_store.list_sessions(),
        )

    @app.get("/api/stations")
    async def list_stations() -> JSONResponse:
        return JSONResponse(
            [station.model_dump(mode="json") for station in dashboard_store.list_stations()]
        )

    @app.post("/api/stations")
    async def create_station(payload: StationCreatePayload) -> JSONResponse:
        station_name = payload.station_name.strip()
        if not station_name:
            raise HTTPException(status_code=400, detail="Informe o nome da estação.")
        if "\n" in station_name or "\r" in station_name:
            raise HTTPException(status_code=400, detail="O nome deve ter apenas uma linha.")
        station_id = _station_id_from_name(station_name)
        station_token = secrets.token_urlsafe(32)
        try:
            station = dashboard_store.create_station(
                station_id,
                station_name,
                hash_password(station_token),
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=409,
                detail="Já existe uma estação com esse nome. Use um nome diferente.",
            ) from exc
        return JSONResponse(
            {
                "station": station.model_dump(mode="json"),
                "station_token": station_token,
            },
            status_code=201,
        )

    @app.delete("/api/stations/{station_id}")
    async def delete_station(station_id: str) -> JSONResponse:
        station = dashboard_store.get_station(station_id)
        if station is not None and station.active_session_id:
            raise HTTPException(
                status_code=409,
                detail="Encerre a avaliação ativa antes de excluir a estação.",
            )
        if not dashboard_store.delete_station(station_id):
            raise HTTPException(status_code=404, detail="Estação não encontrada.")
        return JSONResponse({"deleted": station_id})

    @app.get("/api/sessions")
    async def list_sessions() -> JSONResponse:
        return JSONResponse(
            [session.model_dump(mode="json") for session in dashboard_store.list_sessions()]
        )

    @app.post("/api/sessions/clear")
    async def clear_sessions() -> JSONResponse:
        removed = dashboard_store.clear_sessions()
        return JSONResponse({"removed": removed})

    @app.post("/api/heartbeats")
    async def upsert_heartbeat(
        payload: StationHeartbeat,
        authenticated_station_id: str = Depends(require_station_token),
    ) -> JSONResponse:
        if payload.station_id != authenticated_station_id:
            raise HTTPException(status_code=403, detail="station_id não bate com o token.")
        station = dashboard_store.upsert_station_heartbeat(payload)
        commands = dashboard_store.drain_commands(payload.station_id)
        return JSONResponse(
            {
                "station": station.model_dump(mode="json"),
                "commands": [command.model_dump(mode="json") for command in commands],
            }
        )

    @app.post("/api/configs")
    async def create_config(payload: ExamConfigPayload) -> JSONResponse:
        config_record = dashboard_store.create_config(payload)
        return JSONResponse(config_record.model_dump(mode="json"), status_code=201)

    @app.get("/api/configs")
    async def list_configs() -> JSONResponse:
        return JSONResponse(
            [config.model_dump(mode="json") for config in dashboard_store.snapshot()["configs"]]
        )

    @app.post("/api/stations/{station_id}/session/stop")
    async def stop_session(station_id: str) -> JSONResponse:
        command = dashboard_store.enqueue_command(station_id, CommandType.STOP_SESSION)
        return JSONResponse(command.model_dump(mode="json"), status_code=202)

    @app.post("/api/stations/{station_id}/session/unblock")
    async def unblock_session(station_id: str) -> JSONResponse:
        command = dashboard_store.enqueue_command(station_id, CommandType.UNBLOCK_SESSION)
        return JSONResponse(command.model_dump(mode="json"), status_code=202)

    @app.post("/api/stations/{station_id}/autostart/enable")
    async def enable_autostart(station_id: str) -> JSONResponse:
        command = dashboard_store.set_station_autostart(station_id, True)
        return JSONResponse(command.model_dump(mode="json"), status_code=202)

    @app.post("/api/stations/{station_id}/autostart/disable")
    async def disable_autostart(station_id: str) -> JSONResponse:
        command = dashboard_store.set_station_autostart(station_id, False)
        return JSONResponse(command.model_dump(mode="json"), status_code=202)

    @app.post("/api/stations/{station_id}/enroll")
    async def run_enroll(station_id: str, payload: dict) -> JSONResponse:
        turma_ids = [str(t) for t in payload.get("turma_ids") or []]
        if not turma_ids:
            raise HTTPException(status_code=400, detail="turma_ids não pode ser vazio.")
        command = dashboard_store.run_enroll(station_id, turma_ids)
        return JSONResponse(command.model_dump(mode="json"), status_code=202)

    @app.post("/api/stations/{station_id}/update-and-reboot")
    async def update_and_reboot_station(station_id: str) -> JSONResponse:
        station = dashboard_store.get_station(station_id)
        if station is None:
            raise HTTPException(status_code=404, detail="Estação não encontrada.")
        if station.active_session_id or station.status.value in {
            "IDENTIFYING",
            "SESSION",
            "BLOCKED",
            "UPLOADING",
        }:
            raise HTTPException(status_code=409, detail="Encerre a avaliação ativa antes de atualizar a estação.")
        command = dashboard_store.enqueue_command(station_id, CommandType.UPDATE_AND_REBOOT)
        return JSONResponse(command.model_dump(mode="json"), status_code=202)

    @app.get("/api/s3-turmas")
    async def list_s3_turmas(request: Request) -> JSONResponse:
        try:
            turmas = request.app.state.s3_enrollment_service.list_turmas()
        except Exception as exc:
            # S3 fora do ar: cai pro que o dashboard já viu antes (enrollments,
            # configs, heartbeats), pra não deixar o professor sem opção nenhuma.
            return JSONResponse({"turmas": dashboard_store.list_known_turmas(), "error": str(exc)})
        return JSONResponse({"turmas": turmas, "error": None})

    @app.post("/api/configs/clear")
    async def clear_configs() -> JSONResponse:
        removed = dashboard_store.clear_configs()
        return JSONResponse({"removed": removed})

    @app.get("/api/reports/events.csv")
    async def export_events_csv(turma: str | None = None) -> StreamingResponse:
        csv_body = _build_events_csv(dashboard_store.list_sessions(), turma=turma)
        filename = f"eventos_{turma or 'todas_as_turmas'}.csv"
        return StreamingResponse(
            iter([csv_body]),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.post("/api/sessions")
    async def register_session(
        payload: SessionRecord,
        authenticated_station_id: str = Depends(require_station_token),
    ) -> JSONResponse:
        if payload.station_id != authenticated_station_id:
            raise HTTPException(status_code=403, detail="station_id não bate com o token.")
        session = dashboard_store.register_session(payload)
        return JSONResponse(session.model_dump(mode="json"), status_code=201)

    @app.post("/api/sessions/{session_id}/finalize")
    async def finalize_session(
        session_id: str,
        authenticated_station_id: str = Depends(require_station_token),
    ) -> JSONResponse:
        _ensure_station_owns_session(session_id, authenticated_station_id)
        session = dashboard_store.finalize_session(session_id)
        if session is None:
            return JSONResponse({"detail": "Sessão não encontrada."}, status_code=404)
        return JSONResponse(session.model_dump(mode="json"))

    @app.post("/api/sessions/{session_id}/events")
    async def append_session_events(
        session_id: str,
        payload: list[SessionEventPayload],
        authenticated_station_id: str = Depends(require_station_token),
    ) -> JSONResponse:
        _ensure_station_owns_session(session_id, authenticated_station_id)
        session = dashboard_store.append_events(session_id, payload)
        if session is None:
            return JSONResponse({"detail": "Sessão não encontrada."}, status_code=404)
        return JSONResponse(session.model_dump(mode="json"))

    @app.post("/api/enrollment/s3")
    async def create_s3_enrollment(
        request: Request,
        turma: str = Form(...),
    ) -> HTMLResponse:
        try:
            summary = await anyio.to_thread.run_sync(
                lambda: request.app.state.s3_enrollment_service.enroll_turma(
                    turma,
                    force=True,
                )
            )
        except S3EnrollmentError as exc:
            return render_template(
                request,
                "_s3_enrollment_result.html",
                summary=None,
                error=str(exc),
            )

        for student in summary.students:
            if not student.success:
                continue
            dashboard_store.add_enrollment(
                turma=summary.turma,
                student_id=student.student_id,
                student_name=student.student_name,
                source="s3",
                file_names=[student.s3_key],
            )

        return render_template(
            request,
            "_s3_enrollment_result.html",
            summary=summary,
            error=None,
        )

    @app.websocket("/ws/stations")
    async def stations_websocket(websocket: WebSocket) -> None:
        await websocket.accept()
        queue = dashboard_store.subscribe()
        try:
            while True:
                snapshot = await queue.get()
                await websocket.send_text(json.dumps(_json_ready_snapshot(snapshot)))
        except WebSocketDisconnect:
            dashboard_store.unsubscribe(queue)
        else:
            dashboard_store.unsubscribe(queue)

    return app


def _json_ready_snapshot(snapshot: dict[str, object]) -> dict[str, object]:
    def dump_items(items: object) -> object:
        if isinstance(items, list):
            return [
                item.model_dump(mode="json") if hasattr(item, "model_dump") else item
                for item in items
            ]
        return items

    return {key: dump_items(value) for key, value in snapshot.items()}


def _build_timeline(session: SessionRecord) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    started_at = session.started_at.astimezone(timezone.utc)
    for event in sorted(session.events, key=lambda item: item.timestamp):
        event_time = event.timestamp.astimezone(timezone.utc)
        offset_seconds = max(0, int((event_time - started_at).total_seconds()))
        entries.append(
            {
                "reason": _EVENT_REASON_LABELS.get(
                    event.event_type,
                    "Evento de monitoramento registrado",
                ),
                "severity": event.severity.value,
                "offset_seconds": offset_seconds,
                "relative_time": _format_relative_time(offset_seconds),
                "clips": _build_event_clips(session, offset_seconds),
            }
        )
    return entries


def _event_counts(timeline: list[dict[str, object]]) -> dict[str, int]:
    counts = {"ALL": len(timeline), "INFO": 0, "WARNING": 0, "CRITICAL": 0}
    for event in timeline:
        severity = str(event["severity"])
        if severity in counts:
            counts[severity] += 1
    return counts


def _format_relative_time(seconds: int) -> str:
    hours, remainder = divmod(max(0, seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _format_duration(seconds: int | None) -> str:
    if seconds is None:
        return "—"
    hours, remainder = divmod(max(0, seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    parts = []
    if hours:
        parts.append(f"{hours}h")
    if minutes or hours:
        parts.append(f"{minutes}min")
    parts.append(f"{seconds}s")
    return " ".join(parts)


def _build_event_clips(session: SessionRecord, event_offset: int) -> list[dict[str, object]]:
    clip_start = max(0, event_offset - _EVENT_CLIP_CONTEXT_SECONDS)
    clip_end = event_offset + _EVENT_CLIP_CONTEXT_SECONDS
    if session.ended_at is not None:
        clip_end = min(clip_end, session.duration_seconds or clip_end)

    indexed: dict[str, list[tuple[object, int, float, float]]] = {
        "webcam": [],
        "screen": [],
    }
    for asset in session.recordings:
        stream, index = _recording_identity(asset)
        if stream not in indexed or index is None:
            continue
        duration = float(asset.duration_seconds or _LEGACY_SEGMENT_DURATION_SECONDS)
        start = float(
            asset.start_offset_seconds
            if asset.start_offset_seconds is not None
            else index * duration
        )
        indexed[stream].append((asset, index, start, duration))

    clips: list[dict[str, object]] = []
    for stream, label in (("webcam", "Câmera"), ("screen", "Tela")):
        segments = []
        for asset, index, segment_start, duration in sorted(
            indexed[stream], key=lambda item: item[1]
        ):
            segment_end = segment_start + duration
            if segment_end <= clip_start or segment_start >= clip_end:
                continue
            segments.append(
                {
                    "index": index,
                    "label": asset.label,
                    "url": asset.url,
                    "start": max(0.0, clip_start - segment_start),
                    "end": min(duration, clip_end - segment_start),
                }
            )
        if segments:
            clips.append(
                {
                    "stream": stream,
                    "label": label,
                    "segments": segments,
                    "event_at": event_offset - clip_start,
                    "duration": max(0, clip_end - clip_start),
                }
            )
    return clips


def _recording_identity(asset) -> tuple[str | None, int | None]:
    stream = asset.stream.lower() if asset.stream else None
    index = asset.segment_index
    source = f"{asset.label} {asset.s3_key or ''}".lower()
    match = re.search(r"(webcam|screen)[ _-]?(\d+)", source)
    if match:
        stream = stream or match.group(1)
        index = index if index is not None else int(match.group(2))
    elif stream in {"webcam", "screen"} and index is None:
        index = 0
    elif stream is None:
        if "webcam" in source:
            stream, index = "webcam", index if index is not None else 0
        elif "screen" in source or "tela" in source:
            stream, index = "screen", index if index is not None else 0
    return stream, index


def _build_events_csv(sessions: list[SessionRecord], turma: str | None = None) -> str:
    buffer = StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "session_id",
            "station_id",
            "turma",
            "assessment",
            "student_id",
            "student_name",
            "timestamp",
            "offset_seconds",
            "event_type",
            "severity",
            "frame_number",
            "details_json",
        ]
    )

    for session in sessions:
        if turma and session.turma != turma:
            continue
        student_id = session.student.student_id if session.student else ""
        student_name = session.student.student_name if session.student else ""
        for event in session.events:
            offset_seconds = max(
                0,
                int(
                    (
                        event.timestamp.astimezone(timezone.utc)
                        - session.started_at.astimezone(timezone.utc)
                    ).total_seconds()
                ),
            )
            writer.writerow(
                [
                    session.session_id,
                    session.station_id,
                    session.turma,
                    session.assessment,
                    student_id,
                    student_name,
                    event.timestamp.isoformat(),
                    offset_seconds,
                    event.event_type,
                    event.severity.value,
                    event.frame_number,
                    json.dumps(event.details, ensure_ascii=True, sort_keys=True),
                ]
            )

    return buffer.getvalue()


# Sem instância `app` de módulo de propósito: create_app() abre uma conexão
# Postgres de verdade (DashboardStore), então instanciar no import quebraria
# qualquer import deste módulo sem PROCTOR_DASHBOARD_DATABASE_URL configurado
# (testes incluídos). Rodar via `uvicorn src.dashboard.app:create_app --factory`.
