from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from src.core.session import SessionError, SessionManager


class ConfigUpdateRequest(BaseModel):
    turma_id: str | None = None
    prairielearn_url: str | None = None
    session_id: str | None = None
    station_id: str | None = None
    auto_start: bool | None = None
    no_record: bool | None = None
    no_kiosk: bool | None = None
    reidentify_timeout_sec: float | None = Field(default=None, ge=1.0)
    reidentify_matches: int | None = Field(default=None, ge=1)


class StartSessionRequest(BaseModel):
    turma_id: str | None = None
    prairielearn_url: str | None = None
    session_id: str | None = None
    student_id: str | None = None
    student_name: str | None = None
    no_record: bool | None = None
    no_kiosk: bool | None = None


def build_router(manager: SessionManager) -> APIRouter:
    router = APIRouter()

    @router.get("/health")
    def health() -> dict[str, Any]:
        return manager.get_health()

    @router.get("/status")
    def status_view() -> dict[str, Any]:
        return manager.get_status()

    @router.get("/session")
    def session_view() -> dict[str, Any]:
        session = manager.get_session()
        return {"session": session}

    @router.post("/session/start", status_code=status.HTTP_201_CREATED)
    def start_session(payload: StartSessionRequest) -> dict[str, Any]:
        try:
            return manager.start_session(**payload.model_dump())
        except SessionError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/session/stop")
    def stop_session() -> dict[str, Any]:
        return manager.stop_session(reason="api")

    @router.post("/session/unblock")
    def unblock_session() -> dict[str, Any]:
        try:
            return manager.unblock_session()
        except SessionError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/config")
    def update_config(payload: ConfigUpdateRequest) -> dict[str, Any]:
        config = manager.update_config(**payload.model_dump())
        return {"config": config.__dict__}

    return router
