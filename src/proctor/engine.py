"""Proctoring engine — loop principal e máquina de estados.

FSM:
                    ┌──────────┐
                    │  NORMAL  │ ← estado padrão durante a prova
                    └────┬─────┘
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
        ┌──────────┐ ┌────────┐ ┌──────────┐
        │GAZE_WARN │ │ABSENCE │ │MULTI_FACE│  (imediato se multi_face_block=True)
        └────┬─────┘ └───┬────┘ └────┬─────┘
             │           │           │
        duração >    duração >    imediato
        gaze_dur     absence_to
             │           │           │
             └───────────┴───────────┘
                         ▼
                  ┌─────────────┐
                  │   BLOCKED   │
                  └──────┬──────┘
                         │
                  rosto retorna
                  (face re-match
                   é responsab.
                   do session.py)
                         │
                         ▼
                    ┌──────────┐
                    │  NORMAL  │
                    └──────────┘

Uso típico:
    engine = ProctorEngine(session_id="ES2025-T1_20240601_143000")
    engine.start()

    while prova_em_andamento:
        ret, frame = cap.read()
        state = engine.update(frame)
        if state == ProctorState.BLOCKED:
            show_blocked_screen()

    engine.stop()
"""

from __future__ import annotations

import logging
import time
from collections import deque
from enum import Enum

import numpy as np

from src.core.config import AppConfig, FaceConfig, ProctorConfig
from src.proctor.events import EventLogger, EventType, Severity
from src.proctor.gaze import GazeData, GazeEstimator

logger = logging.getLogger(__name__)


class ProctorState(str, Enum):
    """Estados da máquina de estados do proctoring."""
    NORMAL     = "NORMAL"
    GAZE_WARN  = "GAZE_WARN"    # desvio detectado, aguardando duração
    ABSENCE    = "ABSENCE"      # sem rosto, aguardando timeout
    BLOCKED    = "BLOCKED"      # sessão bloqueada


class BlockReason(str, Enum):
    """Motivo do bloqueio."""
    GAZE       = "GAZE"
    ABSENCE    = "ABSENCE"
    MULTI_FACE = "MULTI_FACE"


class ProctorEngine:
    """Orquestra gaze estimation, FSM e logging para uma sessão de prova.

    Args:
        session_id: Identificador único da sessão. Determina o diretório
            do log de eventos.
        proctor_config: Parâmetros de thresholds e timeouts.
        face_config: Configuração dos modelos dlib.
        app_config: Configuração geral (data_dir para logs).
        enable_eye_gaze: Passa para GazeEstimator. Ativa ratio ocular
            como sinal secundário de desvio.
    """

    def __init__(
        self,
        session_id: str,
        proctor_config: ProctorConfig | None = None,
        face_config: FaceConfig | None = None,
        app_config: AppConfig | None = None,
        enable_eye_gaze: bool = False,
    ):
        self._cfg = proctor_config or ProctorConfig()
        self._app_cfg = app_config or AppConfig()

        self.session_id = session_id
        self.state = ProctorState.NORMAL
        self.block_reason: BlockReason | None = None

        # Gaze estimator
        self._gaze = GazeEstimator(
            face_config=face_config or FaceConfig(),
            enable_eye_gaze=enable_eye_gaze,
        )

        # Logger de eventos
        self._logger = EventLogger(
            session_id=session_id,
            app_config=self._app_cfg,
        )

        # Suavização: janela deslizante de yaw
        self._yaw_window: deque[float] = deque(maxlen=10)

        # Timers
        self._warn_start: float = 0.0    # quando entrou em GAZE_WARN
        self._absence_start: float = 0.0  # quando o rosto sumiu

        self._frame_count: int = 0

    # ──────────────────────────────────────────────
    #  Ciclo de vida
    # ──────────────────────────────────────────────

    def start(self) -> None:
        """Registra início da sessão no log."""
        self._logger.log_event(
            frame=0,
            event_type=EventType.SESSION_STARTED,
            severity=Severity.INFO,
            details={"session_id": self.session_id},
        )
        logger.info("Sessão '%s' iniciada", self.session_id)

    def stop(self) -> None:
        """Registra fim da sessão e fecha o log."""
        self._logger.log_event(
            frame=self._frame_count,
            event_type=EventType.SESSION_ENDED,
            severity=Severity.INFO,
            details={"total_frames": self._frame_count},
        )
        self._logger.close()
        logger.info(
            "Sessão '%s' encerrada (%d frames)", self.session_id, self._frame_count
        )

    # ──────────────────────────────────────────────
    #  Loop principal
    # ──────────────────────────────────────────────

    def update(self, frame: np.ndarray) -> ProctorState:
        """Processa um frame BGR e retorna o estado atual da FSM.

        Deve ser chamado a cada frame capturado da webcam.

        Args:
            frame: Imagem BGR (OpenCV).

        Returns:
            ProctorState atual após processar o frame.
        """
        self._frame_count += 1
        gaze_data = self._gaze.process_frame(frame)
        self._transition(gaze_data)
        return self.state

    def unblock(self) -> None:
        """Desbloqueia a sessão após re-identificação bem-sucedida.

        Deve ser chamado pelo session manager após confirmar que o
        aluno correto está presente (face re-match OK).
        """
        if self.state == ProctorState.BLOCKED:
            prev_reason = self.block_reason
            self.state = ProctorState.NORMAL
            self.block_reason = None
            self._yaw_window.clear()

            self._logger.log_event(
                frame=self._frame_count,
                event_type=EventType.SESSION_RESUMED,
                severity=Severity.INFO,
                details={"unblocked_after": prev_reason},
            )
            logger.info("Sessão '%s' desbloqueada", self.session_id)

    # ──────────────────────────────────────────────
    #  FSM
    # ──────────────────────────────────────────────

    def _transition(self, gaze_data: GazeData | None) -> None:
        """Aplica as transições da FSM com base nos dados de gaze."""

        # ── BLOCKED: só sai via unblock() ──────────
        if self.state == ProctorState.BLOCKED:
            return

        # ── Sem rosto ───────────────────────────────
        if gaze_data is None:
            self._handle_no_face()
            return

        # Rosto voltou — resetar timer de ausência
        self._absence_start = 0.0

        # ── Múltiplos rostos (bloqueio imediato) ────
        if gaze_data.face_count > 1 and self._cfg.multi_face_block:
            self._block(BlockReason.MULTI_FACE)
            return

        # ── Análise de gaze ─────────────────────────
        if self.state == ProctorState.ABSENCE:
            # Rosto retornou de uma ausência curta (não bloqueada)
            self.state = ProctorState.NORMAL

        self._handle_gaze(gaze_data)

    def _handle_no_face(self) -> None:
        """Gerencia a ausência de rosto."""
        now = time.time()

        if self.state == ProctorState.ABSENCE:
            elapsed = now - self._absence_start
            if elapsed >= self._cfg.absence_timeout_sec:
                self._block(BlockReason.ABSENCE)
            return

        # Primeira detecção de ausência
        if self.state == ProctorState.NORMAL:
            self.state = ProctorState.ABSENCE
            self._absence_start = now
            self._logger.log_event(
                frame=self._frame_count,
                event_type=EventType.ABSENCE_WARNING,
                severity=Severity.WARNING,
                details={"timeout_sec": self._cfg.absence_timeout_sec},
            )

    def _handle_gaze(self, data: GazeData) -> None:
        """Gerencia desvio de olhar com suavização."""
        # Suavizar yaw
        self._yaw_window.append(data.yaw)
        smooth_yaw = sum(self._yaw_window) / len(self._yaw_window)

        # Desvio horizontal: yaw varia ±90° → ratio 0.0–1.0
        yaw_ratio = abs(smooth_yaw) / 90.0

        # Desvio vertical: solvePnP retorna pitch ~180° quando cabeça ereta
        # (ambiguidade de decomposição de Euler). Normalizamos subtraindo 180°
        # para que cabeça ereta → ~0°, olhar para baixo → positivo.
        pitch_centered = abs(data.pitch) - 180.0
        pitch_ratio = abs(pitch_centered) / 90.0

        is_deviated = (
            yaw_ratio > self._cfg.gaze_h_threshold
            or pitch_ratio > self._cfg.gaze_v_threshold
        )

        # Sinal secundário de olho (se ativo)
        if data.eye_ratio is not None:
            # eye_ratio ~1.0 = centralizado; >1.5 ou <0.5 = desviado
            is_deviated = is_deviated or not (0.5 <= data.eye_ratio <= 1.5)

        now = time.time()

        if self.state == ProctorState.NORMAL:
            if is_deviated:
                self.state = ProctorState.GAZE_WARN
                self._warn_start = now
                self._logger.log_event(
                    frame=self._frame_count,
                    event_type=EventType.GAZE_WARNING,
                    severity=Severity.WARNING,
                    details={
                        "yaw": round(data.yaw, 2),
                        "smooth_yaw": round(smooth_yaw, 2),
                        "pitch": round(data.pitch, 2),
                        "eye_ratio": round(data.eye_ratio, 3) if data.eye_ratio else None,
                    },
                )

        elif self.state == ProctorState.GAZE_WARN:
            if not is_deviated:
                # Olhar voltou antes do timeout
                self.state = ProctorState.NORMAL
                self._warn_start = 0.0
            elif (now - self._warn_start) >= self._cfg.gaze_block_sec:
                self._block(BlockReason.GAZE)

    def _block(self, reason: BlockReason) -> None:
        """Transita para BLOCKED e registra o evento."""
        self.state = ProctorState.BLOCKED
        self.block_reason = reason

        event_type_map = {
            BlockReason.GAZE:       EventType.GAZE_BLOCKED,
            BlockReason.ABSENCE:    EventType.ABSENCE_BLOCKED,
            BlockReason.MULTI_FACE: EventType.MULTI_FACE_BLOCKED,
        }

        self._logger.log_event(
            frame=self._frame_count,
            event_type=event_type_map[reason],
            severity=Severity.CRITICAL,
            details={"reason": reason.value},
        )
        logger.warning(
            "Sessão '%s' BLOQUEADA — motivo: %s (frame %d)",
            self.session_id,
            reason.value,
            self._frame_count,
        )