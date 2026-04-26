from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.routes import build_router
from src.core.autostart import SessionAutoStartWorker
from src.core.config import AppConfig
from src.core.dashboard_sync import DashboardHeartbeatWorker
from src.core.session import SessionManager


def create_app(
    *,
    config: AppConfig | None = None,
    session_manager: SessionManager | None = None,
    heartbeat_worker: DashboardHeartbeatWorker | None = None,
    auto_start_worker: SessionAutoStartWorker | None = None,
) -> FastAPI:
    app_config = config or AppConfig()
    manager = session_manager or SessionManager(app_config=app_config)
    worker = heartbeat_worker or DashboardHeartbeatWorker(
        config=app_config.dashboard,
        session_manager=manager,
    )
    starter = auto_start_worker or SessionAutoStartWorker(
        session_manager=manager,
        interval_sec=app_config.auto_start_poll_sec,
        enabled=app_config.auto_start_enabled,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        worker.start()
        starter.start()
        try:
            yield
        finally:
            starter.stop()
            worker.stop()

    app = FastAPI(title="Proctor Station API", version="0.1.0", lifespan=lifespan)
    app.state.session_manager = manager
    app.state.heartbeat_worker = worker
    app.state.auto_start_worker = starter
    app.include_router(build_router(manager))
    return app


app = create_app()
