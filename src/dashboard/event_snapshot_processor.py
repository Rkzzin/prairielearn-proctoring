"""Gera imagens dos eventos sinalizados a partir das gravações finalizadas."""

from __future__ import annotations

import logging
import shutil
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
        on_complete=None,
    ):
        self._store = store
        self._app_config = app_config
        self._s3 = s3_client or boto3.client("s3", region_name=app_config.s3.region)
        self._on_complete = on_complete
        self._temp_root = Path(app_config.data_dir) / "dashboard-event-snapshots"
        try:
            self._temp_root.mkdir(parents=True, exist_ok=True)
        except OSError:
            self._temp_root = Path(tempfile.gettempdir()) / "proctor-dashboard-event-snapshots"
            self._temp_root.mkdir(parents=True, exist_ok=True)
        self._clear_stale_directories(self._temp_root)
        self._clear_stale_directories(Path(tempfile.gettempdir()))
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

    def resume_pending(self) -> None:
        for session_id in self._store.recover_event_snapshot_session_ids():
            self.enqueue(session_id)

    def _run(self, session_id: str) -> None:
        try:
            session = self._store.get_session(session_id)
            snapshots = self._store.claim_event_snapshots(session_id)
            if session is None:
                return
            self._process(session, snapshots)
            if self._on_complete is not None:
                self._on_complete(session_id)
        except Exception:
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
        environment_assets = sorted(
            (asset for asset in session.recordings if self._is_environment(asset)),
            key=lambda asset: asset.start_offset_seconds or 0.0,
        )
        session_started_event = next(
            (event for event in session.events if event.event_type == "SESSION_STARTED"),
            None,
        )
        recording_started_at = (
            session_started_event.timestamp if session_started_event else session.started_at
        )

        with tempfile.TemporaryDirectory(
            prefix="proctor-event-snapshots-",
            dir=self._temp_root,
        ) as temp_dir:
            source = Path(temp_dir) / "webcam-segment.mp4"
            environment_source = Path(temp_dir) / "environment-segment.mp4"
            current_s3_key = None
            current_environment_s3_key = None
            for snapshot in sorted(snapshots, key=lambda item: item.event_timestamp):
                try:
                    offset = max(
                        0.0,
                        (snapshot.event_timestamp - recording_started_at).total_seconds(),
                    )
                    asset = self._asset_for_offset(webcam_assets, offset)
                    if asset is None or not asset.s3_bucket or not asset.s3_key:
                        raise RuntimeError("gravação da câmera principal indisponível")
                    if asset.s3_key != current_s3_key:
                        source.unlink(missing_ok=True)
                        current_s3_key = None
                        self._s3.download_file(asset.s3_bucket, asset.s3_key, str(source))
                        current_s3_key = asset.s3_key
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
                    environment_fields = {}
                    environment_asset = self._asset_for_offset(environment_assets, offset)
                    if (
                        environment_asset is not None
                        and environment_asset.s3_bucket
                        and environment_asset.s3_key
                    ):
                        try:
                            if environment_asset.s3_key != current_environment_s3_key:
                                environment_source.unlink(missing_ok=True)
                                current_environment_s3_key = None
                                self._s3.download_file(
                                    environment_asset.s3_bucket,
                                    environment_asset.s3_key,
                                    str(environment_source),
                                )
                                current_environment_s3_key = environment_asset.s3_key
                            environment_image_path = (
                                Path(temp_dir) / f"{snapshot.event_key}-environment.jpg"
                            )
                            environment_offset = max(
                                0.0,
                                offset - (environment_asset.start_offset_seconds or 0.0),
                            )
                            self._extract_frame(
                                environment_source,
                                environment_offset,
                                environment_image_path,
                            )
                            environment_output_key = (
                                f"{self._app_config.s3.recordings_prefix}/{session.session_id}/"
                                f"event-snapshots/{snapshot.event_key}-environment.jpg"
                            )
                            self._s3.upload_file(
                                str(environment_image_path),
                                self._app_config.s3.bucket,
                                environment_output_key,
                                ExtraArgs={"ContentType": "image/jpeg"},
                            )
                            environment_fields = {
                                "environment_s3_bucket": self._app_config.s3.bucket,
                                "environment_s3_key": environment_output_key,
                            }
                            environment_image_path.unlink(missing_ok=True)
                        except Exception as exc:  # noqa: BLE001
                            logger.warning(
                                "Imagem ambiente do evento %s não foi gerada: %s",
                                snapshot.event_key,
                                exc,
                            )
                            environment_fields = {"environment_error": str(exc)[:500]}
                    self._store.finish_event_snapshot(
                        snapshot,
                        s3_bucket=self._app_config.s3.bucket,
                        s3_key=output_key,
                        **environment_fields,
                    )
                    image_path.unlink(missing_ok=True)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Imagem do evento %s não foi gerada: %s",
                        snapshot.event_key,
                        exc,
                    )
                    self._store.finish_event_snapshot(snapshot, error=str(exc)[:500])

    @staticmethod
    def _clear_stale_directories(root: Path) -> None:
        for path in root.glob("proctor-event-snapshots-*"):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)

    @staticmethod
    def _is_webcam(asset: RecordingAsset) -> bool:
        return asset.stream == "webcam" or (
            asset.stream is None
            and bool(asset.s3_key)
            and Path(asset.s3_key or "").name.startswith("webcam_")
        )

    @staticmethod
    def _is_environment(asset: RecordingAsset) -> bool:
        return asset.stream == "environment" or (
            asset.stream is None
            and bool(asset.s3_key)
            and Path(asset.s3_key or "").name.startswith("environment_")
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
            frame = None
            ok = False
            for rewind_seconds in (0.0, 2.0, 5.0, 10.0):
                capture.set(
                    cv2.CAP_PROP_POS_MSEC,
                    max(0.0, offset - rewind_seconds) * 1000.0,
                )
                ok, frame = capture.read()
                if ok and frame is not None:
                    break
            if not ok or frame is None:
                raise RuntimeError("não foi possível extrair o frame do evento")
            if not cv2.imwrite(str(destination), frame, [cv2.IMWRITE_JPEG_QUALITY, 88]):
                raise RuntimeError("não foi possível salvar a imagem do evento")
        finally:
            capture.release()
