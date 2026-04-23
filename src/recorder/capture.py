"""Gravação de webcam e tela via FFmpeg.

Arquitetura:
  - Stream webcam: recebe frames do OpenCV via stdin (pipe) — nunca abre
    /dev/videoN diretamente, eliminando conflito com o proctoring.
  - Stream tela:   x11grab independente, sem tocar na câmera.

O loop principal chama write_frame(frame) a cada frame capturado pelo
OpenCV. O FFmpeg grava esses frames em segmentos de 5 minutos e o
Uploader faz o upload ao S3 em background.

Layout dos arquivos locais:
    {data_dir}/sessions/{session_id}/recordings/
        webcam_000.mp4
        webcam_001.mp4
        screen_000.mp4
        screen_001.mp4

Uso típico:
    capture = Capture(session_id="ES2025-T1_20240601")
    capture.start()

    while prova_ativa:
        ret, frame = cap.read()
        engine.update(frame)
        capture.write_frame(frame)   # mesmo frame do proctoring

    capture.stop()
"""

from __future__ import annotations

import logging
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

import os

from src.core.config import AppConfig, FaceConfig, S3Config

logger = logging.getLogger(__name__)


@dataclass
class SegmentInfo:
    """Metadados de um segmento de gravação finalizado."""

    stream: str       # "webcam" ou "screen"
    path: Path        # caminho local do arquivo
    session_id: str
    index: int        # número sequencial do segmento


class Capture:
    """Gerencia gravação de webcam (via pipe) e tela (via x11grab).

    Args:
        session_id: ID da sessão — determina o subdiretório de gravação.
        s3_config: Configuração S3 (para segment_duration_sec).
        face_config: Configuração de câmera (resolução, fps).
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
        self._stop_event = threading.Event()
        self._notified_segments: set[Path] = set()

        # Processo FFmpeg de webcam — lê frames BGR do stdin
        self._webcam_proc: subprocess.Popen | None = None
        self._write_lock = threading.Lock()

    # ──────────────────────────────────────────────
    #  Ciclo de vida
    # ──────────────────────────────────────────────

    def start(self) -> None:
        """Inicia os dois streams de gravação."""
        if self._running:
            logger.warning("Capture já está rodando — sessão '%s'", self.session_id)
            return

        self._stop_event.clear()
        self._notified_segments.clear()
        self._running = True
        self._start_webcam_stream()
        self._start_screen_stream()
        logger.info("Gravação iniciada — sessão '%s'", self.session_id)

    def stop(self) -> None:
        """Para os dois streams e aguarda os processos finalizarem.

        Ordem de encerramento:
          1. Fecha stdin do pipe → FFmpeg de webcam recebe EOF e finaliza
             o arquivo limpo (sem SIGINT que corrompe o trailer).
          2. SIGINT no FFmpeg de tela → finaliza segmento atual.
          3. Aguarda todos os processos.
        """
        if not self._running:
            return

        self._stop_event.set()

        # 1. Webcam: fechar stdin → EOF → FFmpeg finaliza arquivo limpo
        if self._webcam_proc and self._webcam_proc.poll() is None:
            logger.info("Encerrando stream 'webcam' (EOF no pipe)...")
            try:
                self._webcam_proc.stdin.close()
            except OSError:
                pass
            try:
                self._webcam_proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                logger.warning("FFmpeg webcam não encerrou — forçando kill")
                self._webcam_proc.kill()
                self._webcam_proc.wait()

        # 2. Tela: SIGINT para encerrar segmento atual
        screen_proc = self._procs.get("screen")
        if screen_proc and screen_proc.poll() is None:
            logger.info("Encerrando stream 'screen' (SIGINT)...")
            try:
                screen_proc.send_signal(signal.SIGINT)
                screen_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                screen_proc.kill()
                screen_proc.wait()

        self._flush_pending_segments("webcam")
        self._flush_pending_segments("screen")
        self._running = False

        for name, thread in self._monitor_threads.items():
            thread.join(timeout=10)
            if thread.is_alive():
                logger.warning("Thread de monitoramento '%s' não encerrou em 10s", name)

        self._procs.clear()
        self._monitor_threads.clear()
        self._webcam_proc = None
        logger.info("Gravação encerrada — sessão '%s'", self.session_id)

    @property
    def is_running(self) -> bool:
        return self._running

    # ──────────────────────────────────────────────
    #  API pública — escrita de frames
    # ──────────────────────────────────────────────

    def write_frame(self, frame: np.ndarray) -> None:
        """Envia um frame BGR para o FFmpeg de webcam via pipe.

        Deve ser chamado a cada frame no loop de proctoring.
        Thread-safe.

        Args:
            frame: Imagem BGR (OpenCV), mesma resolução de FaceConfig.
        """
        if not self._running or self._webcam_proc is None:
            return
        if self._webcam_proc.poll() is not None:
            logger.warning("Processo webcam FFmpeg encerrou inesperadamente")
            return

        try:
            with self._write_lock:
                self._webcam_proc.stdin.write(frame.tobytes())
        except (BrokenPipeError, OSError):
            pass  # FFmpeg encerrou — stop() já cuida da limpeza

    # ──────────────────────────────────────────────
    #  Streams FFmpeg
    # ──────────────────────────────────────────────

    def _start_webcam_stream(self) -> None:
        """Inicia FFmpeg de webcam lendo frames BGR do stdin.

        O framerate declarado ao FFmpeg deve refletir o FPS real entregue
        pelo OpenCV com o dlib rodando — tipicamente 8-10fps, não os 30fps
        nominais da câmera. Usar o FPS errado acelera o vídeo.
        """
        w = self._face_cfg.camera_width
        h = self._face_cfg.camera_height
        fps = self._face_cfg.camera_fps
        seg = self._s3_cfg.segment_duration_sec
        pattern = str(self._rec_dir / "webcam_%03d.mp4")

        # FPS real do pipe — deve refletir quantos frames/s o OpenCV
        # realmente entrega com o dlib rodando. Configurável via
        # PROCTOR_FACE_PIPE_FPS no .env (default: fps/3 ≈ 10fps).
        # Medir com: python3 scripts/measure_fps.py
        pipe_fps = int(os.getenv("PROCTOR_FACE_PIPE_FPS", str(max(1, round(fps / 3)))))

        cmd = [
            "ffmpeg",
            # Entrada: rawvideo BGR via stdin (pipe do OpenCV)
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-video_size", f"{w}x{h}",
            "-framerate", str(pipe_fps),
            "-i", "pipe:0",
            # Codec — qualidade reduzida (frames são do proctoring, não da câmera nativa)
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "28",
            # Segmentação
            "-f", "segment",
            "-segment_time", str(seg),
            "-reset_timestamps", "1",
            "-strftime", "0",
            "-y",
            pattern,
        ]

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self._webcam_proc = proc
        self._procs["webcam"] = proc
        self._start_monitor_threads("webcam", proc)
        logger.info("Stream webcam iniciado (pipe stdin → %s)", pattern)

    def _start_screen_stream(self) -> None:
        """Inicia captura de tela em segmentos via x11grab."""
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

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self._procs["screen"] = proc
        self._start_monitor_threads("screen", proc)
        logger.info("Stream tela iniciado (x11grab → %s)", pattern)

    def _start_monitor_threads(self, name: str, proc: subprocess.Popen) -> None:
        """Inicia threads de monitoramento de stderr e de segmentos."""
        t_err = threading.Thread(
            target=self._monitor_stderr,
            args=(name, proc),
            daemon=True,
        )
        t_err.start()
        self._monitor_threads[f"{name}_stderr"] = t_err

        t_seg = threading.Thread(
            target=self._watch_segments,
            args=(name,),
            daemon=True,
        )
        t_seg.start()
        self._monitor_threads[f"{name}_segments"] = t_seg

    # ──────────────────────────────────────────────
    #  Monitoramento
    # ──────────────────────────────────────────────

    def _monitor_stderr(self, name: str, proc: subprocess.Popen) -> None:
        """Lê stderr do FFmpeg e loga erros relevantes."""
        for line in proc.stderr:
            line = line.decode("utf-8", errors="replace").strip()
            if any(kw in line.lower() for kw in ("error", "failed", "invalid")):
                logger.error("[ffmpeg/%s] %s", name, line)
            elif "segment" in line.lower():
                logger.debug("[ffmpeg/%s] %s", name, line)

        ret = proc.wait()
        if ret not in (0, 255) and self._running:
            logger.error("Stream '%s' encerrou inesperadamente (código %d)", name, ret)

    def _watch_segments(self, stream: str) -> None:
        """Detecta segmentos finalizados e notifica o callback.

        Um segmento é considerado pronto quando um arquivo mais novo
        aparece — o FFmpeg só fecha um segmento ao abrir o próximo.
        """
        last_ready: Path | None = None

        while not self._stop_event.is_set():
            files = sorted(self._rec_dir.glob(f"{stream}_*.mp4"))
            if len(files) >= 2:
                ready = files[-2]
                if ready != last_ready:
                    last_ready = ready
                    self._notify_segment(stream, ready)

            if self._stop_event.wait(5):
                break

    def _notify_segment(self, stream: str, path: Path) -> None:
        """Monta SegmentInfo e chama o callback."""
        if path in self._notified_segments:
            return
        if not path.exists() or path.stat().st_size <= 0:
            return

        try:
            idx = int(path.stem.split("_")[-1])
        except (ValueError, IndexError):
            idx = 0

        seg_info = SegmentInfo(
            stream=stream,
            path=path,
            session_id=self.session_id,
            index=idx,
        )
        logger.info("Segmento pronto: %s", path.name)
        self._notified_segments.add(path)

        if self._on_segment_ready:
            try:
                self._on_segment_ready(seg_info)
            except Exception as e:
                logger.error("Erro no callback de segmento '%s': %s", path.name, e)

    def _flush_pending_segments(self, stream: str) -> None:
        """Notifica segmentos já finalizados que ainda não foram enfileirados."""
        for path in sorted(self._rec_dir.glob(f"{stream}_*.mp4")):
            self._notify_segment(stream, path)
