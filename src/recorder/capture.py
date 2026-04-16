"""Gravação de webcam e tela via FFmpeg.

Gerencia dois processos FFmpeg simultâneos:
  - Stream webcam: /dev/videoN + áudio → webcam_000.mp4, webcam_001.mp4, ...
  - Stream tela:   x11grab            → screen_000.mp4, screen_001.mp4, ...

Cada segmento dura `segment_duration_sec` segundos (default: 300s / 5min).
Quando um segmento fecha, o uploader é notificado via callback para
fazer o upload ao S3 em background.

Uso típico (chamado pelo session manager):
    capture = Capture(session_id="ES2025-T1_20240601", config=s3_cfg)
    capture.start()
    # ... prova acontece ...
    capture.stop()

Layout dos arquivos locais:
    {data_dir}/sessions/{session_id}/recordings/
        webcam_000.mp4
        webcam_001.mp4
        screen_000.mp4
        screen_001.mp4
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.core.config import AppConfig, FaceConfig, S3Config

logger = logging.getLogger(__name__)


@dataclass
class SegmentInfo:
    """Metadados de um segmento de gravação finalizado."""
    stream: str        # "webcam" ou "screen"
    path: Path         # caminho local do arquivo
    session_id: str
    index: int         # número sequencial do segmento


class Capture:
    """Gerencia dois processos FFmpeg para gravação contínua segmentada.

    Args:
        session_id: ID da sessão — determina o subdiretório de gravação.
        s3_config: Configuração S3 (para segment_duration_sec).
        face_config: Configuração de câmera (índice, resolução, fps).
        app_config: Configuração geral (data_dir).
        on_segment_ready: Callback chamado quando um segmento fecha.
            Assinatura: (segment: SegmentInfo) -> None
            Chamado em thread separada — deve ser thread-safe.
        display: Display X11 para captura de tela (default: ":0.0").
        screen_size: Resolução da tela (default: "1920x1080").
    """

    def __init__(
        self,
        session_id: str,
        s3_config: S3Config | None = None,
        face_config: FaceConfig | None = None,
        app_config: AppConfig | None = None,
        on_segment_ready: Callable[[SegmentInfo], None] | None = None,
        display: str = ":0.0",
        screen_size: str = "1920x1080",
    ):
        self.session_id = session_id
        self._s3_cfg = s3_config or S3Config()
        self._face_cfg = face_config or FaceConfig()
        self._app_cfg = app_config or AppConfig()
        self._on_segment_ready = on_segment_ready
        self._display = display
        self._screen_size = screen_size

        # Diretório de gravação da sessão
        self._rec_dir = (
            Path(self._app_cfg.data_dir)
            / "sessions"
            / session_id
            / "recordings"
        )
        self._rec_dir.mkdir(parents=True, exist_ok=True)

        self._procs: dict[str, subprocess.Popen] = {}
        self._monitor_threads: dict[str, threading.Thread] = {}
        self._running = False

    # ──────────────────────────────────────────────
    #  Ciclo de vida
    # ──────────────────────────────────────────────

    def start(self) -> None:
        """Inicia os dois streams de gravação."""
        if self._running:
            logger.warning("Capture já está rodando para sessão '%s'", self.session_id)
            return

        self._running = True
        self._start_webcam_stream()
        self._start_screen_stream()
        logger.info("Gravação iniciada — sessão '%s'", self.session_id)

    def stop(self) -> None:
        """Para os dois streams e aguarda os processos finalizarem."""
        if not self._running:
            return

        self._running = False

        for name, proc in self._procs.items():
            if proc.poll() is None:  # ainda rodando
                logger.info("Encerrando stream '%s'...", name)
                try:
                    proc.send_signal(signal.SIGINT)  # FFmpeg finaliza o segmento limpo
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()

        for thread in self._monitor_threads.values():
            thread.join(timeout=5)

        self._procs.clear()
        self._monitor_threads.clear()
        logger.info("Gravação encerrada — sessão '%s'", self.session_id)

    @property
    def is_running(self) -> bool:
        return self._running

    # ──────────────────────────────────────────────
    #  Streams FFmpeg
    # ──────────────────────────────────────────────

    def _start_webcam_stream(self) -> None:
        """Inicia gravação de webcam + áudio em segmentos."""
        cam = self._face_cfg.camera_index
        w = self._face_cfg.camera_width
        h = self._face_cfg.camera_height
        fps = self._face_cfg.camera_fps
        seg = self._s3_cfg.segment_duration_sec
        pattern = str(self._rec_dir / "webcam_%03d.mp4")

        cmd = [
            "ffmpeg",
            # Webcam
            "-f", "v4l2",
            "-video_size", f"{w}x{h}",
            "-framerate", str(fps),
            "-i", f"/dev/video{cam}",
            # Áudio (PulseAudio)
            "-f", "pulse",
            "-i", "default",
            # Codec vídeo
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            # Codec áudio
            "-c:a", "aac",
            "-b:a", "128k",
            # Segmentação
            "-f", "segment",
            "-segment_time", str(seg),
            "-reset_timestamps", "1",
            "-strftime", "0",
            # Sem confirmação interativa
            "-y",
            pattern,
        ]

        self._launch_stream("webcam", cmd)

    def _start_screen_stream(self) -> None:
        """Inicia captura de tela em segmentos."""
        seg = self._s3_cfg.segment_duration_sec
        pattern = str(self._rec_dir / "screen_%03d.mp4")

        cmd = [
            "ffmpeg",
            "-f", "x11grab",
            "-video_size", self._screen_size,
            "-framerate", "15",
            "-i", self._display,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "28",
            "-f", "segment",
            "-segment_time", str(seg),
            "-reset_timestamps", "1",
            "-strftime", "0",
            "-y",
            pattern,
        ]

        self._launch_stream("screen", cmd)

    def _launch_stream(self, name: str, cmd: list[str]) -> None:
        """Lança um processo FFmpeg e inicia o monitor de segmentos."""
        logger.info("Iniciando stream '%s': %s", name, " ".join(cmd))

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,  # capturar stderr para log de erros
        )
        self._procs[name] = proc

        # Thread que monitora o stderr do FFmpeg para detectar erros
        t = threading.Thread(
            target=self._monitor_stream,
            args=(name, proc),
            daemon=True,
        )
        t.start()
        self._monitor_threads[name] = t

        # Thread que detecta segmentos prontos
        t2 = threading.Thread(
            target=self._watch_segments,
            args=(name,),
            daemon=True,
        )
        t2.start()

    # ──────────────────────────────────────────────
    #  Monitoramento
    # ──────────────────────────────────────────────

    def _monitor_stream(self, name: str, proc: subprocess.Popen) -> None:
        """Lê stderr do FFmpeg e loga erros relevantes."""
        for line in proc.stderr:
            line = line.decode("utf-8", errors="replace").strip()
            if any(kw in line.lower() for kw in ("error", "failed", "invalid")):
                logger.error("[ffmpeg/%s] %s", name, line)
            elif "opening" in line.lower() or "segment" in line.lower():
                logger.debug("[ffmpeg/%s] %s", name, line)

        ret = proc.wait()
        if ret != 0 and self._running:
            logger.error("Stream '%s' encerrou inesperadamente (código %d)", name, ret)

    def _watch_segments(self, stream: str) -> None:
        """Detecta segmentos finalizados e notifica o callback.

        Um segmento é considerado pronto quando um arquivo mais novo
        aparece no diretório — significa que o FFmpeg rotacionou para
        o próximo segmento e o anterior está completo.
        """
        seg = self._s3_cfg.segment_duration_sec
        seen: set[Path] = set()
        last_ready: Path | None = None

        while self._running:
            files = sorted(self._rec_dir.glob(f"{stream}_*.mp4"))

            # Novos arquivos que ainda não vimos
            new_files = [f for f in files if f not in seen]
            for f in new_files:
                seen.add(f)

            # O segmento anterior ao mais recente está pronto
            # (FFmpeg só fecha um segmento quando abre o próximo)
            if len(files) >= 2:
                ready = files[-2]  # penúltimo = fechado
                if ready != last_ready:
                    last_ready = ready
                    idx = int(ready.stem.split("_")[-1])
                    seg_info = SegmentInfo(
                        stream=stream,
                        path=ready,
                        session_id=self.session_id,
                        index=idx,
                    )
                    logger.info(
                        "Segmento pronto: %s (stream=%s, idx=%d)",
                        ready.name, stream, idx,
                    )
                    if self._on_segment_ready:
                        try:
                            self._on_segment_ready(seg_info)
                        except Exception as e:
                            logger.error("Erro no callback de segmento: %s", e)

            time.sleep(5)  # checar a cada 5s (bem abaixo dos 5min de segmento)

        # Ao parar, notificar o último segmento de cada stream
        files = sorted(self._rec_dir.glob(f"{stream}_*.mp4"))
        if files:
            last = files[-1]
            if last != last_ready and last.stat().st_size > 0:
                idx = int(last.stem.split("_")[-1])
                seg_info = SegmentInfo(
                    stream=stream,
                    path=last,
                    session_id=self.session_id,
                    index=idx,
                )
                logger.info("Último segmento: %s", last.name)
                if self._on_segment_ready:
                    try:
                        self._on_segment_ready(seg_info)
                    except Exception as e:
                        logger.error("Erro no callback do último segmento: %s", e)