"""Testes automatizados do proctoring engine (Fase 2).

Roda sem câmera — injeta GazeData sintético diretamente na FSM.
GazeEstimator é mockado para que os testes não dependam de
modelos dlib instalados.

Cobertura:
  - FSM: NORMAL → GAZE_WARN → BLOCKED
  - FSM: desvio curto não dispara bloqueio
  - FSM: olhar volta antes do timeout → retorna a NORMAL
  - FSM: NORMAL → ABSENCE → BLOCKED
  - FSM: BLOCKED → NORMAL via unblock()
  - FSM: multi-face → BLOCKED imediato
  - FSM: multi_face_block=False não bloqueia
  - EventLogger: escreve e relê JSONL corretamente
  - EventLogger: session_id isolado por sessão
  - ProctorEngine: start() e stop() registram eventos
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.core.config import AppConfig, FaceConfig, ProctorConfig
from src.proctor.engine import BlockReason, ProctorEngine, ProctorState
from src.proctor.events import EventLogger, EventType, ProctorEvent, Severity
from src.proctor.gaze import GazeData


# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_config(
    gaze_h: float = 0.35,
    gaze_v: float = 0.30,
    gaze_dur: float = 3.0,
    absence_timeout: float = 5.0,
    multi_face_block: bool = True,
) -> ProctorConfig:
    return ProctorConfig(
        gaze_h_threshold=gaze_h,
        gaze_v_threshold=gaze_v,
        gaze_duration_sec=gaze_dur,
        absence_timeout_sec=absence_timeout,
        multi_face_block=multi_face_block,
    )


def _make_app_config(tmp_path: Path) -> AppConfig:
    return AppConfig(data_dir=tmp_path)


def _gaze(yaw: float = 0.0, pitch: float = 0.0, eye_ratio: float | None = None,
          face_count: int = 1) -> GazeData:
    """Cria GazeData sintético."""
    return GazeData(yaw=yaw, pitch=pitch, roll=0.0,
                    eye_ratio=eye_ratio, face_count=face_count)


def _make_engine(
    tmp_path: Path,
    proctor_config: ProctorConfig | None = None,
    enable_eye_gaze: bool = False,
) -> ProctorEngine:
    """Cria ProctorEngine com GazeEstimator mockado."""
    with patch("src.proctor.engine.GazeEstimator"):
        engine = ProctorEngine(
            session_id="TEST-001",
            proctor_config=proctor_config or _make_config(),
            face_config=FaceConfig(),
            app_config=_make_app_config(tmp_path),
            enable_eye_gaze=enable_eye_gaze,
        )
    return engine


def _feed(engine: ProctorEngine, gaze_data: GazeData | None) -> ProctorState:
    """Injeta GazeData diretamente na FSM, sem passar pelo GazeEstimator."""
    engine._frame_count += 1
    engine._transition(gaze_data)
    return engine.state


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_app(tmp_path: Path) -> AppConfig:
    return _make_app_config(tmp_path)


@pytest.fixture
def engine(tmp_path: Path) -> ProctorEngine:
    return _make_engine(tmp_path)


# ── Testes de FSM: gaze ──────────────────────────────────────────────────────


class TestGazeFSM:
    def test_normal_when_no_deviation(self, engine: ProctorEngine):
        state = _feed(engine, _gaze(yaw=0.0))
        assert state == ProctorState.NORMAL

    def test_enters_gaze_warn_on_deviation(self, engine: ProctorEngine):
        # yaw=45° → ratio=0.5 > threshold 0.35
        state = _feed(engine, _gaze(yaw=45.0))
        assert state == ProctorState.GAZE_WARN

    def test_short_deviation_does_not_block(self, tmp_path: Path):
        """Desvio que desaparece antes do timeout não bloqueia."""
        cfg = _make_config(gaze_dur=5.0)
        engine = _make_engine(tmp_path, proctor_config=cfg)

        _feed(engine, _gaze(yaw=45.0))   # → GAZE_WARN
        assert engine.state == ProctorState.GAZE_WARN

        _feed(engine, _gaze(yaw=0.0))    # olhar voltou
        assert engine.state == ProctorState.NORMAL

    def test_sustained_deviation_blocks(self, tmp_path: Path):
        """Desvio por tempo >= gaze_duration_sec bloqueia."""
        cfg = _make_config(gaze_dur=0.0)  # timeout imediato
        engine = _make_engine(tmp_path, proctor_config=cfg)

        _feed(engine, _gaze(yaw=45.0))   # → GAZE_WARN com warn_start no passado
        engine._warn_start = time.time() - 1.0  # simula 1s já passado

        _feed(engine, _gaze(yaw=45.0))   # → BLOCKED
        assert engine.state == ProctorState.BLOCKED
        assert engine.block_reason == BlockReason.GAZE

    def test_blocked_ignores_further_frames(self, tmp_path: Path):
        """Após BLOCKED, novos frames não mudam o estado."""
        cfg = _make_config(gaze_dur=0.0)
        engine = _make_engine(tmp_path, proctor_config=cfg)

        _feed(engine, _gaze(yaw=45.0))
        engine._warn_start = time.time() - 1.0
        _feed(engine, _gaze(yaw=45.0))
        assert engine.state == ProctorState.BLOCKED

        # frames normais não desbloqueiam
        for _ in range(5):
            _feed(engine, _gaze(yaw=0.0))
        assert engine.state == ProctorState.BLOCKED

    def test_vertical_deviation_triggers_warn(self, engine: ProctorEngine):
        # pitch=45° → ratio=0.5 > threshold 0.30
        state = _feed(engine, _gaze(pitch=45.0))
        assert state == ProctorState.GAZE_WARN


# ── Testes de FSM: ausência ──────────────────────────────────────────────────


class TestAbsenceFSM:
    def test_no_face_enters_absence(self, engine: ProctorEngine):
        state = _feed(engine, None)
        assert state == ProctorState.ABSENCE

    def test_short_absence_does_not_block(self, tmp_path: Path):
        cfg = _make_config(absence_timeout=10.0)
        engine = _make_engine(tmp_path, proctor_config=cfg)

        _feed(engine, None)
        assert engine.state == ProctorState.ABSENCE

        # rosto volta antes do timeout
        _feed(engine, _gaze())
        assert engine.state == ProctorState.NORMAL

    def test_prolonged_absence_blocks(self, tmp_path: Path):
        cfg = _make_config(absence_timeout=0.0)  # timeout imediato
        engine = _make_engine(tmp_path, proctor_config=cfg)

        _feed(engine, None)  # → ABSENCE com absence_start no passado
        engine._absence_start = time.time() - 1.0

        _feed(engine, None)  # → BLOCKED
        assert engine.state == ProctorState.BLOCKED
        assert engine.block_reason == BlockReason.ABSENCE

    def test_face_returns_after_absence_warn(self, engine: ProctorEngine):
        """Rosto retorna enquanto ainda em ABSENCE (não bloqueado)."""
        _feed(engine, None)
        assert engine.state == ProctorState.ABSENCE

        _feed(engine, _gaze())
        assert engine.state == ProctorState.NORMAL


# ── Testes de FSM: multi-face ────────────────────────────────────────────────


class TestMultiFaceFSM:
    def test_multi_face_blocks_immediately(self, engine: ProctorEngine):
        state = _feed(engine, _gaze(face_count=2))
        assert state == ProctorState.BLOCKED
        assert engine.block_reason == BlockReason.MULTI_FACE

    def test_multi_face_no_block_when_disabled(self, tmp_path: Path):
        cfg = _make_config(multi_face_block=False)
        engine = _make_engine(tmp_path, proctor_config=cfg)

        state = _feed(engine, _gaze(face_count=2))
        assert state != ProctorState.BLOCKED


# ── Testes de unblock ────────────────────────────────────────────────────────


class TestUnblock:
    def test_unblock_resets_to_normal(self, tmp_path: Path):
        cfg = _make_config(gaze_dur=0.0)
        engine = _make_engine(tmp_path, proctor_config=cfg)

        _feed(engine, _gaze(yaw=45.0))
        engine._warn_start = time.time() - 1.0
        _feed(engine, _gaze(yaw=45.0))
        assert engine.state == ProctorState.BLOCKED

        engine.unblock()
        assert engine.state == ProctorState.NORMAL
        assert engine.block_reason is None

    def test_unblock_clears_yaw_window(self, tmp_path: Path):
        cfg = _make_config(gaze_dur=0.0)
        engine = _make_engine(tmp_path, proctor_config=cfg)

        # Popular janela de suavização com valores altos
        for _ in range(10):
            _feed(engine, _gaze(yaw=45.0))

        engine._warn_start = time.time() - 1.0
        _feed(engine, _gaze(yaw=45.0))
        engine.unblock()

        assert len(engine._yaw_window) == 0

    def test_unblock_on_normal_is_noop(self, engine: ProctorEngine):
        engine.unblock()
        assert engine.state == ProctorState.NORMAL


# ── Testes de EventLogger ────────────────────────────────────────────────────


class TestEventLogger:
    def test_log_creates_file(self, tmp_path: Path):
        app = _make_app_config(tmp_path)
        with EventLogger("SES-001", app_config=app) as log:
            log.log_event(1, EventType.GAZE_WARNING, Severity.WARNING,
                          {"yaw": 35.0})

        log_file = tmp_path / "sessions" / "SES-001" / "events.jsonl"
        assert log_file.exists()

    def test_log_event_content(self, tmp_path: Path):
        app = _make_app_config(tmp_path)
        with EventLogger("SES-002", app_config=app) as log:
            event = log.log_event(42, EventType.GAZE_BLOCKED, Severity.CRITICAL,
                                  {"reason": "GAZE"})

        assert event.frame == 42
        assert event.type == EventType.GAZE_BLOCKED.value
        assert event.severity == Severity.CRITICAL.value
        assert event.details["reason"] == "GAZE"

    def test_read_all_roundtrip(self, tmp_path: Path):
        app = _make_app_config(tmp_path)
        with EventLogger("SES-003", app_config=app) as log:
            log.log_event(1, EventType.SESSION_STARTED, Severity.INFO)
            log.log_event(100, EventType.GAZE_WARNING, Severity.WARNING,
                          {"yaw": 40.0})
            log.log_event(200, EventType.GAZE_BLOCKED, Severity.CRITICAL)

        with EventLogger("SES-003", app_config=app) as log:
            events = log.read_all()

        assert len(events) == 3
        assert events[0].type == EventType.SESSION_STARTED.value
        assert events[1].type == EventType.GAZE_WARNING.value
        assert events[2].type == EventType.GAZE_BLOCKED.value

    def test_sessions_are_isolated(self, tmp_path: Path):
        """Duas sessões não compartilham o mesmo arquivo."""
        app = _make_app_config(tmp_path)

        with EventLogger("SES-A", app_config=app) as log_a:
            log_a.log_event(1, EventType.GAZE_WARNING, Severity.WARNING)

        with EventLogger("SES-B", app_config=app) as log_b:
            log_b.log_event(1, EventType.ABSENCE_WARNING, Severity.WARNING)
            log_b.log_event(2, EventType.ABSENCE_BLOCKED, Severity.CRITICAL)

        with EventLogger("SES-A", app_config=app) as log_a:
            assert len(log_a.read_all()) == 1

        with EventLogger("SES-B", app_config=app) as log_b:
            assert len(log_b.read_all()) == 2

    def test_jsonl_is_valid_json_per_line(self, tmp_path: Path):
        app = _make_app_config(tmp_path)
        with EventLogger("SES-004", app_config=app) as log:
            log.log_event(1, EventType.SESSION_STARTED, Severity.INFO)
            log.log_event(2, EventType.GAZE_WARNING, Severity.WARNING,
                          {"yaw": 35.2, "pitch": -1.0})

        log_file = tmp_path / "sessions" / "SES-004" / "events.jsonl"
        lines = log_file.read_text().strip().splitlines()

        assert len(lines) == 2
        for line in lines:
            parsed = json.loads(line)
            assert "timestamp" in parsed
            assert "frame" in parsed
            assert "type" in parsed
            assert "severity" in parsed
            assert "details" in parsed


# ── Testes de ciclo de vida do engine ────────────────────────────────────────


class TestEngineLifecycle:
    def test_start_logs_session_started(self, tmp_path: Path):
        engine = _make_engine(tmp_path)
        engine.start()
        engine._logger.close()

        events = EventLogger.read_session(
            tmp_path / "sessions" / "TEST-001" / "events.jsonl"
        )
        assert any(e.type == EventType.SESSION_STARTED.value for e in events)

    def test_stop_logs_session_ended(self, tmp_path: Path):
        engine = _make_engine(tmp_path)
        engine.start()
        engine.stop()

        events = EventLogger.read_session(
            tmp_path / "sessions" / "TEST-001" / "events.jsonl"
        )
        assert any(e.type == EventType.SESSION_ENDED.value for e in events)

    def test_unblock_logs_session_resumed(self, tmp_path: Path):
        cfg = _make_config(gaze_dur=0.0)
        engine = _make_engine(tmp_path, proctor_config=cfg)

        _feed(engine, _gaze(yaw=45.0))
        engine._warn_start = time.time() - 1.0
        _feed(engine, _gaze(yaw=45.0))
        engine.unblock()
        engine._logger.close()

        events = EventLogger.read_session(
            tmp_path / "sessions" / "TEST-001" / "events.jsonl"
        )
        assert any(e.type == EventType.SESSION_RESUMED.value for e in events)