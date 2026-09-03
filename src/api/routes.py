from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response, status
from pydantic import BaseModel, Field

from src.core.session import SessionError, SessionManager


class ConfigUpdateRequest(BaseModel):
    turma_id: str | None = None
    assessment: str | None = None
    timer_minutes: int | None = None
    local_timer_enabled: bool | None = None
    prairielearn_url: str | None = None
    allowlist: list[str] | None = None
    s3_prefix: str | None = None
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

    def require_local_overlay(request: Request) -> None:
        if request.client and request.client.host not in {"127.0.0.1", "::1"}:
            raise HTTPException(status_code=403, detail="Confirmação disponível somente no overlay local.")

    @router.get("/health")
    def health() -> dict[str, Any]:
        return manager.get_health()

    @router.get("/exam-checks")
    def exam_checks() -> dict[str, Any]:
        return manager.get_exam_checks()

    @router.get("/status")
    def status_view() -> dict[str, Any]:
        return manager.get_status()

    @router.get("/session")
    def session_view() -> dict[str, Any]:
        session = manager.get_session()
        return {"session": session}

    @router.get("/camera-preview.jpg")
    def camera_preview(request: Request) -> Response:
        require_local_overlay(request)
        image = manager.get_camera_preview_jpeg()
        if image is None:
            raise HTTPException(status_code=404, detail="Preview da câmera indisponível")
        return Response(content=image, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @router.post("/session/start", status_code=status.HTTP_201_CREATED)
    def start_session(payload: StartSessionRequest) -> dict[str, Any]:
        try:
            return manager.start_session(**payload.model_dump())
        except SessionError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/pre-exam/start", status_code=status.HTTP_201_CREATED)
    def start_session_from_overlay(request: Request) -> dict[str, Any]:
        require_local_overlay(request)
        try:
            return manager.start_session()
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

    @router.post("/pre-exam/confirmation/accept")
    def accept_pre_exam_confirmation(request: Request) -> dict[str, str]:
        require_local_overlay(request)
        try:
            manager.respond_to_pre_exam_confirmation(accepted=True)
        except SessionError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"status": "accepted"}

    @router.post("/pre-exam/confirmation/cancel")
    def cancel_pre_exam_confirmation(request: Request) -> dict[str, str]:
        require_local_overlay(request)
        try:
            manager.respond_to_pre_exam_confirmation(accepted=False)
        except SessionError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"status": "cancelled"}

    @router.post("/exam-mode/prepare")
    def prepare_exam_mode() -> dict[str, Any]:
        try:
            return manager.prepare_exam_mode()
        except SessionError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/exam-mode/enter")
    def enter_exam_mode() -> dict[str, Any]:
        try:
            return manager.enter_exam_mode()
        except SessionError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/exam-mode/exit")
    def exit_exam_mode() -> dict[str, Any]:
        try:
            return manager.exit_exam_mode()
        except SessionError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/exam-mode/recover")
    def recover_exam_mode() -> dict[str, Any]:
        return manager.recover_exam_mode()

    @router.post("/config")
    def update_config(payload: ConfigUpdateRequest) -> dict[str, Any]:
        config = manager.update_config(**payload.model_dump())
        return {"config": config.__dict__}

    return router
