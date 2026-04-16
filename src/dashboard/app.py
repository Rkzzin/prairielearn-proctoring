"""Aplicação FastAPI do dashboard do professor."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.core.config import AppConfig
from src.dashboard.models import (
    CommandType,
    ExamConfigPayload,
    SessionEventPayload,
    SessionRecord,
    StationHeartbeat,
)
from src.dashboard.store import DashboardStore


def create_app(config: AppConfig | None = None, store: DashboardStore | None = None) -> FastAPI:
    app_config = config or AppConfig()
    dashboard_dir = Path(__file__).parent
    templates = Jinja2Templates(directory=str(dashboard_dir / "templates"))
    dashboard_store = store or DashboardStore(app_config.data_dir / "dashboard")

    app = FastAPI(title="Proctor Station Dashboard")
    app.mount(
        "/static",
        StaticFiles(directory=str(dashboard_dir / "static")),
        name="static",
    )
    app.state.store = dashboard_store
    app.state.templates = templates

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
        snapshot = dashboard_store.snapshot()
        return render_template(
            request,
            "config.html",
            title="Configuração de prova",
            **snapshot,
        )

    @app.get("/enrollment", response_class=HTMLResponse)
    async def enrollment_page(request: Request) -> HTMLResponse:
        snapshot = dashboard_store.snapshot()
        return render_template(
            request,
            "enrollment.html",
            title="Enrollment",
            **snapshot,
        )

    @app.get("/sessions/{session_id}", response_class=HTMLResponse)
    async def session_review(request: Request, session_id: str) -> HTMLResponse:
        session = dashboard_store.get_session(session_id)
        if session is None:
            return HTMLResponse("Sessão não encontrada.", status_code=404)
        return render_template(
            request,
            "session_detail.html",
            title=f"Sessão {session_id}",
            session=session,
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

    @app.get("/partials/enrollments", response_class=HTMLResponse)
    async def enrollments_partial(request: Request) -> HTMLResponse:
        return render_template(
            request,
            "_enrollments.html",
            enrollments=dashboard_store.list_enrollments(),
        )

    @app.get("/api/stations")
    async def list_stations() -> JSONResponse:
        return JSONResponse(
            [station.model_dump(mode="json") for station in dashboard_store.list_stations()]
        )

    @app.get("/api/sessions")
    async def list_sessions() -> JSONResponse:
        return JSONResponse(
            [session.model_dump(mode="json") for session in dashboard_store.list_sessions()]
        )

    @app.post("/api/heartbeats")
    async def upsert_heartbeat(payload: StationHeartbeat) -> JSONResponse:
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

    @app.post("/api/stations/{station_id}/session/stop")
    async def stop_session(station_id: str) -> JSONResponse:
        command = dashboard_store.enqueue_command(station_id, CommandType.STOP_SESSION)
        return JSONResponse(command.model_dump(mode="json"), status_code=202)

    @app.post("/api/stations/{station_id}/session/unblock")
    async def unblock_session(station_id: str) -> JSONResponse:
        command = dashboard_store.enqueue_command(station_id, CommandType.UNBLOCK_SESSION)
        return JSONResponse(command.model_dump(mode="json"), status_code=202)

    @app.post("/api/sessions")
    async def register_session(payload: SessionRecord) -> JSONResponse:
        session = dashboard_store.register_session(payload)
        return JSONResponse(session.model_dump(mode="json"), status_code=201)

    @app.post("/api/sessions/{session_id}/finalize")
    async def finalize_session(session_id: str) -> JSONResponse:
        session = dashboard_store.finalize_session(session_id)
        if session is None:
            return JSONResponse({"detail": "Sessão não encontrada."}, status_code=404)
        return JSONResponse(session.model_dump(mode="json"))

    @app.post("/api/sessions/{session_id}/events")
    async def append_session_events(session_id: str, payload: list[SessionEventPayload]) -> JSONResponse:
        session = dashboard_store.append_events(session_id, payload)
        if session is None:
            return JSONResponse({"detail": "Sessão não encontrada."}, status_code=404)
        return JSONResponse(session.model_dump(mode="json"))

    @app.post("/api/enrollment")
    async def create_enrollment(
        request: Request,
        turma: str = Form(...),
        student_id: str = Form(...),
        student_name: str = Form(...),
        source: str = Form("upload"),
        files: list[UploadFile] | None = File(None),
    ) -> HTMLResponse:
        file_names: list[str] = []
        for upload in files or []:
            if not upload.filename:
                continue
            destination = dashboard_store.upload_dir / upload.filename
            contents = await upload.read()
            destination.write_bytes(contents)
            file_names.append(upload.filename)

        dashboard_store.add_enrollment(
            turma=turma,
            student_id=student_id,
            student_name=student_name,
            source=source,
            file_names=file_names,
        )
        return render_template(
            request,
            "_enrollments.html",
            enrollments=dashboard_store.list_enrollments(),
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


app = create_app()
