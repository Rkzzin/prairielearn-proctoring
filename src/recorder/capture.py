"""Gravação de webcam e tela via FFmpeg.

Arquitetura:
  - Stream webcam: FFmpeg captura /dev/videoN diretamente e publica
    um preview local para o proctoring.
  - Stream tela:   x11grab independente.

O proctoring consome o preview local, então a câmera física fica
aberta só no processo do FFmpeg. Isso evita:
  - FPS degradado no vídeo gravado por causa do detector
  - conflito de /dev/videoN ocupado por OpenCV e FFmpeg ao mesmo tempo

Layout dos arquivos locais:
    {data_dir}/sessions/{session_id}/recordings/
        webcam_000.mp4
        webcam_001.mp4
        screen_000.mp4
        screen_001.mp4

Uso típico:
    capture = Capture(session_id="ES2025-T1_20240601")
    capture.start()
    ...
    capture.stop()
"""

from __future__ import annotations

import logging
import re
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import os

from src.core.config import AppConfig, FaceConfig, RecorderConfig, S3Config
from src.core.cpu_affinity import parse_cpu_set, split_ffmpeg_stream_cpu_sets

logger = logging.getLogger(__name__)


@dataclass
class SegmentInfo:
    """Metadados de um segmento de gravação finalizado."""

    stream: str       # "webcam" ou "screen"
    path: Path        # caminho local do arquivo
    session_id: str
    index: int        # número sequencial do segmento


class Capture:
    """Gerencia gravação de webcam e tela via FFmpeg.

    Args:
        session_id: ID da sessão — determina o subdiretório de gravação.
        s3_config: Configuração S3 (para segment_duration_sec).
        face_config: Configuração de câmera (resolução, fps).
        app_config: Configuração geral (data_dir).
        on_segment_ready: Callback chamado quando um segmento fecha.
            Assinatura: (segment: SegmentInfo) -> None
            Chamado em thread separada — deve ser thread-safe.
        display: Display X11 para captura de tela (default: ":0.0").
        screen_size: Resolução final do vídeo da tela (default: "1280x720").
    """

    def __init__(
        self,
        session_id: str,
        s3_config: S3Config | None = None,
        face_config: FaceConfig | None = None,
        app_config: AppConfig | None = None,
        recorder_config: RecorderConfig | None = None,
        on_segment_ready: Callable[[SegmentInfo], None] | None = None,
        display: str | None = None,
        screen_size: str | None = None,
    ):
        self.session_id = session_id
        self._s3_cfg = s3_config or S3Config()
        self._face_cfg = face_config or FaceConfig()
        self._app_cfg = app_config or AppConfig()
        self._rec_cfg = recorder_config or self._app_cfg.recorder
        self._on_segment_ready = on_segment_ready
        self._display = display or self._rec_cfg.display
        self._screen_size = screen_size or self._rec_cfg.screen_size

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
        self._preview_url = (
            f"udp://{self._rec_cfg.preview_host}:{self._rec_cfg.preview_port}"
            "?overrun_nonfatal=1&fifo_size=5000000"
        )
        self._stream_cpu_sets = split_ffmpeg_stream_cpu_sets(
            parse_cpu_set(self._rec_cfg.ffmpeg_cpu_cores)
        )

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
          1. SIGINT nos FFmpeg de webcam/tela → finaliza segmento atual.
          2. Aguarda todos os processos.
        """
        if not self._running:
            return

        self._stop_event.set()

        for stream in ("webcam", "screen"):
            proc = self._procs.get(stream)
            if proc is None or proc.poll() is not None:
                continue
            logger.info("Encerrando stream '%s' (SIGINT)...", stream)
            try:
                proc.send_signal(signal.SIGINT)
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                logger.warning("FFmpeg %s não encerrou — forçando kill", stream)
                proc.kill()
                proc.wait()

        self._flush_pending_segments("webcam")
        self._flush_pending_segments("screen")
        self._running = False

        for name, thread in self._monitor_threads.items():
            thread.join(timeout=10)
            if thread.is_alive():
                logger.warning("Thread de monitoramento '%s' não encerrou em 10s", name)

        self._procs.clear()
        self._monitor_threads.clear()
        logger.info("Gravação encerrada — sessão '%s'", self.session_id)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def preview_url(self) -> str:
        return self._preview_url

    # ──────────────────────────────────────────────
    #  Streams FFmpeg
    # ──────────────────────────────────────────────

    def _start_webcam_stream(self) -> None:
        """Inicia FFmpeg de webcam capturando /dev/videoN diretamente."""
        w = self._face_cfg.camera_width
        h = self._face_cfg.camera_height
        fps = self._face_cfg.camera_fps
        seg = self._s3_cfg.segment_duration_sec
        pattern = str(self._rec_dir / "webcam_%03d.mp4")
        video_device = f"/dev/video{self._face_cfg.camera_index}"
        input_format = self._rec_cfg.webcam_input_format.strip()
        ffmpeg_threads = max(1, self._rec_cfg.ffmpeg_threads)
        preview_fps = max(1, self._rec_cfg.preview_fps)
        preview_width = max(160, self._rec_cfg.preview_width)
        preview_height = max(120, self._rec_cfg.preview_height)
        preview_sink = (
            f"udp://{self._rec_cfg.preview_host}:{self._rec_cfg.preview_port}"
            "?pkt_size=1316"
        )

        cmd = [
            "ffmpeg",
            "-f", "v4l2",
            "-thread_queue_size", "512",
        ]
        if input_format:
            cmd.extend(["-input_format", input_format])
        cmd.extend([
            "-use_wallclock_as_timestamps", "1",
            "-framerate", str(fps),
            "-video_size", f"{w}x{h}",
            "-i", video_device,
            "-filter_complex",
            (
                "[0:v]split=2[record][preview];"
                f"[preview]fps={preview_fps},scale={preview_width}:{preview_height}[preview_out]"
            ),
            "-map", "[record]",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            "-profile:v", "high",
            "-pix_fmt", "yuv420p",
            "-threads", str(ffmpeg_threads),
            "-fps_mode", "passthrough",
            "-f", "segment",
            "-segment_time", str(seg),
            "-segment_format_options", "movflags=+faststart",
            "-reset_timestamps", "1",
            "-strftime", "0",
            "-y",
            pattern,
            "-map", "[preview_out]",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-crf", "35",
            "-pix_fmt", "yuv420p",
            "-g", str(preview_fps),
            "-keyint_min", str(preview_fps),
            "-sc_threshold", "0",
            "-x264-params", "repeat-headers=1:aud=1",
            "-threads", "1",
            "-f", "mpegts",
            preview_sink,
        ])

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            preexec_fn=self._build_affinity_preexec_fn("webcam"),
        )
        self._procs["webcam"] = proc
        self._start_monitor_threads("webcam", proc)
        self._ensure_process_started("webcam", proc)
        logger.info("Stream webcam iniciado (v4l2 %s → %s)", video_device, pattern)

    def _start_screen_stream(self) -> None:
        """Inicia captura da tela inteira e gera saída em resolução configurada."""
        seg = self._s3_cfg.segment_duration_sec
        pattern = str(self._rec_dir / "screen_%03d.mp4")
        capture_size = self._resolve_screen_capture_size()
        output_width, output_height = self._parse_size(self._screen_size)
        scale_filter = f"scale={output_width}:{output_height}:flags=fast_bilinear,setsar=1"

        cmd = [
            "ffmpeg",
            "-f", "x11grab",
            "-thread_queue_size", "512",
            "-use_wallclock_as_timestamps", "1",
            "-video_size", capture_size,
            "-framerate", "15",
            "-i", self._display,
            "-vf", scale_filter,
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "28",
            "-profile:v", "high",
            "-pix_fmt", "yuv420p",
            "-threads", str(max(1, self._rec_cfg.ffmpeg_threads)),
            "-fps_mode", "passthrough",
            "-f", "segment",
            "-segment_time", str(seg),
            "-segment_format_options", "movflags=+faststart",
            "-reset_timestamps", "1",
            "-strftime", "0",
            "-y",
            pattern,
        ]

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            preexec_fn=self._build_affinity_preexec_fn("screen"),
        )
        self._procs["screen"] = proc
        self._start_monitor_threads("screen", proc)
        self._ensure_process_started("screen", proc)
        logger.info("Stream tela iniciado (x11grab → %s)", pattern)

    def _resolve_screen_capture_size(self) -> str:
        display_size = self._detect_display_size()
        if display_size:
            return display_size
        logger.warning(
            "Não foi possível detectar a resolução real do display %s; usando '%s' como fallback",
            self._display,
            self._screen_size,
        )
        return self._screen_size

    def _detect_display_size(self) -> str | None:
        env = dict(os.environ)
        env["DISPLAY"] = self._display
        try:
            result = subprocess.run(
                ["xrandr", "--current"],
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )
        except FileNotFoundError:
            logger.warning("xrandr não encontrado; não foi possível detectar a resolução real do display")
            return None

        if result.returncode != 0:
            stderr = result.stderr.strip()
            logger.warning("xrandr falhou ao detectar display %s: %s", self._display, stderr or result.returncode)
            return None

        match = re.search(r"current\s+(\d+)\s+x\s+(\d+)", result.stdout)
        if match is None:
            logger.warning("xrandr não retornou resolução reconhecível para %s", self._display)
            return None
        return f"{match.group(1)}x{match.group(2)}"

    @staticmethod
    def _parse_size(size: str) -> tuple[int, int]:
        match = re.fullmatch(r"(\d+)x(\d+)", size.strip())
        if match is None:
            raise ValueError(f"Resolução inválida: '{size}'")
        return int(match.group(1)), int(match.group(2))

    def _build_affinity_preexec_fn(self, stream: str):
        cpus = self._stream_cpu_sets.get(stream)
        if not cpus or not hasattr(os, "sched_setaffinity"):
            return None

        def _set_affinity() -> None:
            os.sched_setaffinity(0, cpus)

        return _set_affinity

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

    def _ensure_process_started(self, name: str, proc: subprocess.Popen, timeout_sec: float = 0.5) -> None:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            ret = proc.poll()
            if ret is not None:
                raise RuntimeError(f"FFmpeg do stream '{name}' encerrou no start (código {ret})")
            time.sleep(0.05)

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
