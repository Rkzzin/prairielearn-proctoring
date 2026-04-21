"""Upload incremental de segmentos de gravação para o S3.

Recebe segmentos prontos do Capture via fila thread-safe e faz
upload para o S3 em background. Após upload bem-sucedido, o
arquivo local é removido para não encher o disco da NUC.

Layout no S3:
    s3://{bucket}/{recordings_prefix}/{session_id}/
        webcam_000.mp4
        webcam_001.mp4
        screen_000.mp4
        screen_001.mp4

Retry: 3 tentativas com backoff exponencial (2s, 4s, 8s).
Se todas falharem, o arquivo é mantido localmente e logado para
reprocessamento manual.

Uso típico:
    uploader = Uploader(session_id="ES2025-T1_20240601")
    uploader.start()

    # Capture chama isso via callback:
    uploader.enqueue(segment_info)

    uploader.stop()  # aguarda fila esvaziar antes de parar
"""

from __future__ import annotations

import logging
import queue
import threading
import time

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from src.core.config import AppConfig, S3Config
from src.recorder.capture import SegmentInfo

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_BASE_SEC = 2.0


class Uploader:
    """Faz upload de segmentos para o S3 em background.

    Thread-safe: pode receber segmentos de qualquer thread via enqueue().

    Args:
        session_id: ID da sessão — usado como prefixo no S3.
        s3_config: Configuração S3.
        app_config: Configuração geral.
        delete_after_upload: Se True (default), remove o arquivo local
            após upload bem-sucedido.
    """

    def __init__(
        self,
        session_id: str,
        s3_config: S3Config | None = None,
        app_config: AppConfig | None = None,
        delete_after_upload: bool = True,
    ):
        self.session_id = session_id
        self._cfg = s3_config or S3Config()
        self._app_cfg = app_config or AppConfig()
        self._delete = delete_after_upload

        self._queue: queue.Queue[SegmentInfo | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._failed: list[SegmentInfo] = []  # segmentos que falharam após retries

        self._s3 = boto3.client("s3", region_name=self._cfg.region)

    # ──────────────────────────────────────────────
    #  Ciclo de vida
    # ──────────────────────────────────────────────

    def start(self) -> None:
        """Inicia a thread de upload em background."""
        self._thread = threading.Thread(
            target=self._worker,
            name=f"uploader-{self.session_id}",
            daemon=True,
        )
        self._thread.start()
        logger.info("Uploader iniciado — sessão '%s'", self.session_id)

    def stop(self) -> None:
        """Para a thread após esvaziar a fila."""
        self._queue.put(None)  # sentinela de parada
        if self._thread:
            self._thread.join(timeout=60)
            if self._thread.is_alive():
                logger.warning("Uploader não encerrou em 60s — forçando parada")

        if self._failed:
            logger.error(
                "%d segmento(s) não foram upados. Arquivos mantidos em:\n%s",
                len(self._failed),
                "\n".join(str(s.path) for s in self._failed),
            )

        logger.info("Uploader encerrado — sessão '%s'", self.session_id)

    def enqueue(self, segment: SegmentInfo) -> None:
        """Adiciona um segmento à fila de upload. Thread-safe."""
        self._queue.put(segment)
        logger.debug("Segmento enfileirado: %s", segment.path.name)

    @property
    def failed_segments(self) -> list[SegmentInfo]:
        """Segmentos que falharam após todas as tentativas de retry."""
        return list(self._failed)

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    # ──────────────────────────────────────────────
    #  Worker
    # ──────────────────────────────────────────────

    def _worker(self) -> None:
        """Loop principal da thread de upload."""
        while True:
            segment = self._queue.get()
            if segment is None:  # sentinela de parada
                break

            self._upload_with_retry(segment)
            self._queue.task_done()

    def _upload_with_retry(self, segment: SegmentInfo) -> None:
        """Tenta fazer upload com retry e backoff exponencial."""
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                self._upload(segment)
                return
            except (ClientError, BotoCoreError, OSError) as e:
                wait = _RETRY_BASE_SEC * (2 ** (attempt - 1))
                logger.warning(
                    "Upload falhou (tentativa %d/%d): %s — aguardando %.0fs",
                    attempt, _MAX_RETRIES, e, wait,
                )
                if attempt < _MAX_RETRIES:
                    time.sleep(wait)

        logger.error(
            "Upload falhou após %d tentativas: %s — arquivo mantido em %s",
            _MAX_RETRIES, segment.path.name, segment.path,
        )
        self._failed.append(segment)

    def _upload(self, segment: SegmentInfo) -> None:
        """Faz o upload de um segmento para o S3."""
        if not segment.path.exists():
            logger.warning("Arquivo não encontrado, pulando: %s", segment.path)
            return

        s3_key = self._s3_key(segment)
        size_mb = segment.path.stat().st_size / 1_048_576

        logger.info(
            "Upando %s → s3://%s/%s (%.1f MB)",
            segment.path.name, self._cfg.bucket, s3_key, size_mb,
        )

        self._s3.upload_file(
            Filename=str(segment.path),
            Bucket=self._cfg.bucket,
            Key=s3_key,
            ExtraArgs={"ContentType": "video/mp4"},
        )

        logger.info("Upload OK: %s", segment.path.name)

        if self._delete:
            segment.path.unlink()
            logger.debug("Arquivo local removido: %s", segment.path)

    # ──────────────────────────────────────────────
    #  Helpers
    # ──────────────────────────────────────────────

    def _s3_key(self, segment: SegmentInfo) -> str:
        """Monta a chave S3 para um segmento.

        Ex: gravacoes/ES2025-T1_20240601/webcam_000.mp4
        """
        return (
            f"{self._cfg.recordings_prefix}"
            f"/{segment.session_id}"
            f"/{segment.path.name}"
        )
