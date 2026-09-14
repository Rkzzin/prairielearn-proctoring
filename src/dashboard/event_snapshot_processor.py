"""Gera imagens dos eventos sinalizados a partir das gravações finalizadas."""

from __future__ import annotations

import logging
import tempfile
import threading
from pathlib import Path

import boto3
import cv2

from src.core.config import AppConfig
from src.dashboard.models import EventSnapshotRecord, RecordingAsset, SessionRecord
from src.dashboard.store import DashboardStore

logger = logging.getLogger(__name__)


class EventSnapshotProcessor:
    def __init__(
        self,
        *,
        store: DashboardStore,
        app_config: AppConfig,
        s3_client=None,
    ):
        self._store = store
        self._app_config = app_config
        self._s3 = s3_client or boto3.client("s3", region_name=app_config.s3.region)
        self._lock = threading.Lock()
        self._running: set[str] = set()

    def enqueue(self, session_id: str, *, retry_failed: bool = False) -> int:
        queued = (
            self._store.retry_event_snapshots(session_id)
            if retry_failed
            else self._store.queue_event_snapshots(session_id)
        )
        with self._lock:
            if session_id in self._running:
                return queued
            self._running.add(session_id)
        threading.Thread(
            target=self._run,
            args=(session_id,),
            name=f"event-snapshots-{session_id[:24]}",
            daemon=True,
        ).start()
        return queued

    def _run(self, session_id: str) -> None:
        try:
            session = self._store.get_session(session_id)
            snapshots = self._store.claim_event_snapshots(session_id)
            if session is None:
                return
            self._process(session, snapshots)
        except Exception:  # noqa: BLE001
            logger.exception("Falha ao processar imagens da sessão %s", session_id)
        finally:
            with self._lock:
                self._running.discard(session_id)

    def _process(
        self,
        session: SessionRecord,
        snapshots: list[EventSnapshotRecord],
    ) -> None:
        webcam_assets = sorted(
            (asset for asset in session.recordings if self._is_webcam(asset)),
            key=lambda asset: asset.start_offset_seconds or 0.0,
        )
        session_started_event = next(
            (event for event in session.events if event.event_type == "SESSION_STARTED"),
            None,
        )
        recording_started_at = (
            session_started_event.timestamp if session_started_event else session.started_at
        )

        with tempfile.TemporaryDirectory(prefix="proctor-event-snapshots-") as temp_dir:
            downloaded: dict[str, Path] = {}
            for snapshot in snapshots:
                try:
                    offset = max(
                        0.0,
                        (snapshot.event_timestamp - recording_started_at).total_seconds(),
                    )
                    asset = self._asset_for_offset(webcam_assets, offset)
                    if asset is None or not asset.s3_bucket or not asset.s3_key:
                        raise RuntimeError("gravação da câmera principal indisponível")
                    source = downloaded.get(asset.s3_key)
                    if source is None:
                        source = Path(temp_dir) / f"segment-{len(downloaded):04d}.mp4"
                        self._s3.download_file(asset.s3_bucket, asset.s3_key, str(source))
                        downloaded[asset.s3_key] = source
                    relative_offset = max(0.0, offset - (asset.start_offset_seconds or 0.0))
                    image_path = Path(temp_dir) / f"{snapshot.event_key}.jpg"
                    self._extract_frame(source, relative_offset, image_path)
                    output_key = (
                        f"{self._app_config.s3.recordings_prefix}/{session.session_id}/"
                        f"event-snapshots/{snapshot.event_key}.jpg"
                    )
                    self._s3.upload_file(
                        str(image_path),
                        self._app_config.s3.bucket,
                        output_key,
                        ExtraArgs={"ContentType": "image/jpeg"},
                    )
                    self._store.finish_event_snapshot(
                        snapshot,
                        s3_bucket=self._app_config.s3.bucket,
                        s3_key=output_key,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Imagem do evento %s não foi gerada: %s",
                        snapshot.event_key,
                        exc,
                    )
                    self._store.finish_event_snapshot(snapshot, error=str(exc)[:500])

    @staticmethod
    def _is_webcam(asset: RecordingAsset) -> bool:
        return asset.stream == "webcam" or (
            asset.stream is None
            and bool(asset.s3_key)
            and Path(asset.s3_key or "").name.startswith("webcam_")
        )

    @staticmethod
    def _asset_for_offset(
        assets: list[RecordingAsset],
        offset: float,
    ) -> RecordingAsset | None:
        matching = [
            asset
            for asset in assets
            if (asset.start_offset_seconds or 0.0) <= offset
            < (asset.start_offset_seconds or 0.0) + (asset.duration_seconds or 300.0)
        ]
        return matching[-1] if matching else (assets[-1] if assets else None)

    @staticmethod
    def _extract_frame(source: Path, offset: float, destination: Path) -> None:
        capture = cv2.VideoCapture(str(source))
        try:
            if not capture.isOpened():
                raise RuntimeError("não foi possível abrir o segmento de vídeo")
            capture.set(cv2.CAP_PROP_POS_MSEC, offset * 1000.0)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError("não foi possível extrair o frame do evento")
            if not cv2.imwrite(str(destination), frame, [cv2.IMWRITE_JPEG_QUALITY, 88]):
                raise RuntimeError("não foi possível salvar a imagem do evento")
        finally:
            capture.release()
