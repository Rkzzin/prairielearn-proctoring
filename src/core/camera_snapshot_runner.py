"""Captura e envia fotos diagnósticas das câmeras em background."""

from __future__ import annotations

import base64
import logging
import threading
from collections.abc import Callable
from typing import Any

import httpx

from src.core.config import DashboardConfig
from src.core.session import SessionManager

logger = logging.getLogger(__name__)


class CameraSnapshotRunner:
    def __init__(
        self,
        *,
        config: DashboardConfig,
        session_manager: SessionManager,
        client_factory: Callable[[], httpx.Client] | None = None,
    ):
        self._config = config
        self._session_manager = session_manager
        self._client_factory = client_factory or self._default_client_factory
        self._lock = threading.Lock()
        self._status = "idle"
        self._message = ""
        self._batch_id: str | None = None
        self._pending_batch_id: str | None = None

    def status_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "camera_capture_status": self._status,
                "camera_capture_message": self._message,
                "camera_capture_batch_id": self._batch_id,
            }

    def start(self, batch_id: str) -> None:
        with self._lock:
            if self._status == "running":
                if batch_id != self._batch_id:
                    self._pending_batch_id = batch_id
                return
            self._status = "running"
            self._message = "Capturando câmeras..."
            self._batch_id = batch_id
        threading.Thread(
            target=self._run,
            args=(batch_id,),
            name="camera-snapshot-runner",
            daemon=True,
        ).start()

    def _run(self, batch_id: str) -> None:
        status = "error"
        message = "Falha inesperada na captura"
        try:
            snapshots, errors = self._session_manager.capture_camera_snapshots()
            with self._client_factory() as client:
                for snapshot in snapshots:
                    upload_error: Exception | None = None
                    for _attempt in range(2):
                        try:
                            client.post(
                                "/api/camera-snapshots",
                                json={
                                    "batch_id": batch_id,
                                    "camera_index": snapshot["index"],
                                    "camera_name": snapshot["name"],
                                    "device": snapshot["device"],
                                    "image_base64": base64.b64encode(snapshot["jpeg"]).decode("ascii"),
                                },
                            ).raise_for_status()
                            upload_error = None
                            break
                        except (httpx.HTTPError, OSError) as exc:
                            upload_error = exc
                    if upload_error is not None:
                        errors.append(f"{snapshot['name']}: falha no envio")
            status = "done" if not errors else ("partial" if snapshots else "error")
            message = f"{len(snapshots)} câmera(s) fotografada(s)"
            if errors:
                message += f"; {len(errors)} falha(s)"
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - proteção da thread
            logger.warning("Captura diagnóstica das câmeras falhou: %s", exc)
            message = str(exc)
        next_batch = None
        with self._lock:
            if self._pending_batch_id is not None:
                next_batch = self._pending_batch_id
                self._pending_batch_id = None
                self._batch_id = next_batch
                self._status = "running"
                self._message = "Capturando câmeras..."
            else:
                self._status = status
                self._message = message
        if next_batch is not None:
            threading.Thread(
                target=self._run,
                args=(next_batch,),
                name="camera-snapshot-runner",
                daemon=True,
            ).start()

    def _default_client_factory(self) -> httpx.Client:
        headers = {
            "X-Station-Id": self._config.station_id,
            "X-Station-Token": self._config.station_token or "",
        }
        return httpx.Client(
            base_url=self._config.base_url,
            timeout=max(10.0, self._config.timeout_sec),
            headers=headers,
        )
