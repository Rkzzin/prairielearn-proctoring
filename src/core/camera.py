"""Posse da câmera durante o ciclo de prova.

Invariante central do projeto, e o que esta classe existe para tornar
explícito: o **OpenCV abre `/dev/videoN` apenas na identificação inicial** e
enquanto a estação espera aluno. Quando a gravação começa, o **FFmpeg passa a
ser o único dono do dispositivo físico** e o proctoring passa a consumir o
preview MPEG-TS que o FFmpeg publica em UDP.

Antes isso eram cinco métodos soltos no ``SessionManager`` mexendo num
``self._camera`` compartilhado; a troca de dono ficava implícita na ordem das
chamadas. Aqui as duas fontes são métodos distintos (``open_device`` e
``open_preview``), e há um único lugar que fecha o handle.

A fábrica de captura é injetável, o que é o que permite testar sem hardware.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable

import cv2

logger = logging.getLogger(__name__)

#: Teto para a **abertura** de uma fonte de rede (o preview UDP do FFmpeg).
#:
#: Sem isto, ``cv2.VideoCapture`` de uma URL que não responde bloqueia ~30 s
#: dentro de uma única tentativa — medido — e o laço de retry de
#: ``open_preview`` só checa o deadline *entre* tentativas. O efeito era
#: ``start_session`` pendurado ~30 s **segurando o RLock** do SessionManager,
#: com ``/status`` e ``/health`` bloqueados junto.
#:
#: **Não reduza sem medir.** Esse timeout também limita o tempo que o OpenCV tem
#: para sondar o stream e achar SPS/PPS do H.264. Medido contra o preview real:
#:
#:   =========  ==========  ==================================================
#:   timeout    abertura    frames em 4 s
#:   =========  ==========  ==================================================
#:   800 ms     0,90 s      **0** — abre, mas nunca decodifica ("non-existing
#:                          PPS 0"); o handle fica inútil
#:   2000 ms    2,04 s      15 (degradado)
#:   3000 ms    2,63 s      44 (saudável)
#:   5000 ms    2,85 s      41 (saudável)
#:   =========  ==========  ==================================================
#:
#: A abertura saudável custa ~2,85 s, então 5000 ms dá ~2x de margem e ainda
#: corta o pior caso de 30 s para ~5 s (verificado: URL morta bloqueia 5,06 s).
PREVIEW_OPEN_TIMEOUT_MS = 5000
PREVIEW_READ_TIMEOUT_MS = 2000


def open_video_capture(source: int | str) -> Any:
    """Fábrica default de captura.

    Fonte inteira é dispositivo local (``/dev/videoN``) e abre direto. Fonte
    string é rede (preview UDP) e ganha teto de abertura, para não bloquear a
    thread que estiver chamando.

    O preview também recebe timeout de leitura. Sem isso, um UDP congelado
    bloqueia o loop de proctoring indefinidamente e impede detectar ausência.
    """
    if isinstance(source, str):
        return cv2.VideoCapture(
            source,
            cv2.CAP_FFMPEG,
            [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                PREVIEW_OPEN_TIMEOUT_MS,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                PREVIEW_READ_TIMEOUT_MS,
            ],
        )
    return cv2.VideoCapture(source)


class CameraError(RuntimeError):
    """Falha ao obter uma fonte de vídeo utilizável."""


class SessionCamera:
    """Dono do handle de vídeo usado por identificação e proctoring.

    Args:
        face_config: resolução/fps/índice do dispositivo físico.
        capture_factory: constrói o objeto de captura a partir de um índice
            (dispositivo) ou de uma URL (preview). Injetável para testes;
            default ``open_video_capture``, que limita a abertura de rede.
        sleep_fn: usado no retry de abertura do preview.
    """

    def __init__(
        self,
        *,
        face_config: Any,
        capture_factory: Callable[[Any], Any] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
    ):
        self._cfg = face_config
        self._factory = capture_factory or open_video_capture
        self._sleep = sleep_fn or time.sleep
        self._handle: Any | None = None
        self._lock = threading.RLock()

    # ── estado ────────────────────────────────────────────────

    @property
    def handle(self) -> Any | None:
        """Handle atual, ou ``None``. Passado direto para a re-identificação."""
        with self._lock:
            return self._handle

    @property
    def is_open(self) -> bool:
        with self._lock:
            if self._handle is None:
                return False
            return _is_opened(self._handle)

    def read(self) -> tuple[bool, Any]:
        """Lê um frame da fonte atual."""
        with self._lock:
            handle = self._handle
        if handle is None:
            return False, None
        ret, frame = handle.read()
        # Um recovery pode ter trocado o handle enquanto a leitura bloqueava.
        # Nunca entregue um frame da fonte antiga à engine.
        with self._lock:
            if handle is not self._handle:
                return False, None
        return ret, frame

    # ── aquisição ─────────────────────────────────────────────

    def open_device(self) -> Any:
        """Garante o dispositivo físico aberto (identificação / espera de aluno).

        Idempotente: se já há um handle aberto, reaproveita — é isso que evita
        o liga/desliga da webcam a cada tentativa de auto-start.
        """
        with self._lock:
            if self._handle is not None:
                if _is_opened(self._handle):
                    return self._handle
                self.release()
            self._handle = self._open(self._cfg.camera_index)
            return self._handle

    def open_preview(self, source: str, timeout_sec: float = 5.0) -> Any:
        """Abre o preview publicado pelo FFmpeg, com retry até ``timeout_sec``.

        O retry existe porque o FFmpeg leva um instante para começar a emitir
        no UDP depois de subir; abrir de primeira falha com frequência.
        """
        deadline = time.monotonic() + timeout_sec
        last_error: CameraError | None = None
        while time.monotonic() < deadline:
            try:
                candidate = self._open(source)
                ret, frame = candidate.read()
                if ret and frame is not None:
                    with self._lock:
                        previous = self._handle
                        self._handle = candidate
                    _release_quietly(previous)
                    return candidate
                _release_quietly(candidate)
            except CameraError as exc:
                last_error = exc
            self._sleep(0.1)
        raise CameraError(
            f"Não foi possível abrir o preview local da webcam em {source}"
        ) from last_error

    def release(self) -> None:
        """Fecha o handle atual, se houver. Único ponto de liberação."""
        with self._lock:
            if self._handle is not None:
                _release_quietly(self._handle)
            self._handle = None

    # ── diagnóstico ───────────────────────────────────────────

    def probe(self) -> bool:
        """Diz se há câmera utilizável, para o ``/health``.

        Com handle aberto, responde sobre ele. Sem handle, abre e fecha um
        descartável — de propósito, para não roubar o dispositivo do FFmpeg
        durante uma sessão gravada.
        """
        if self._handle is not None:
            try:
                return _is_opened(self._handle)
            except Exception:
                return False
        try:
            candidate = self._factory(self._cfg.camera_index)
            ok = _is_opened(candidate)
            _release_quietly(candidate)
            return bool(ok)
        except Exception:
            return False

    # ── interno ───────────────────────────────────────────────

    def _open(self, source: int | str) -> Any:
        cap = self._factory(source)
        if isinstance(source, int):
            self._tune_device(cap)
        if not _is_opened(cap):
            raise CameraError(f"Não foi possível abrir a câmera {source}")
        return cap

    def _tune_device(self, cap: Any) -> None:
        """Aplica MJPG/resolução/fps ao dispositivo físico.

        Só faz sentido no dispositivo — no preview a codificação já vem do
        FFmpeg. Silenciosamente ignorado se o handle não suportar ``set``.
        """
        if not hasattr(cap, "set"):
            return
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._cfg.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cfg.camera_height)
        cap.set(cv2.CAP_PROP_FPS, self._cfg.camera_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def _is_opened(cap: Any) -> bool:
    """``isOpened()`` quando existe; caso contrário assume aberto.

    Os dublês de teste não implementam a interface completa do cv2.
    """
    return bool(cap.isOpened()) if hasattr(cap, "isOpened") else True


def _release_quietly(cap: Any) -> None:
    if not hasattr(cap, "release"):
        return
    try:
        cap.release()
    except Exception as exc:  # pragma: no cover - cleanup best effort
        logger.debug("Falha ao liberar handle de câmera: %s", exc)
