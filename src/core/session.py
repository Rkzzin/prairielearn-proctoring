"""Session manager — orquestra o ciclo de vida completo de uma prova.

FSM de alto nível:

    IDLE → IDENTIFYING → SESSION → BLOCKED → SESSION → UPLOADING → IDLE

Integra:
  - reconhecimento facial inicial
  - proctoring contínuo
  - gravação + upload
  - Chromium controlado
  - desbloqueio por re-identificação ou via API
"""

from __future__ import annotations

import logging
import threading
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import boto3
import cv2
from botocore.exceptions import BotoCoreError, ClientError

from src.core.config import AppConfig, FaceConfig, ProctorConfig, RecorderConfig, S3Config
from src.core.camera import CameraError, SessionCamera
from src.core.cpu_affinity import auto_split_cpu_sets, get_process_cpu_set, parse_cpu_set, set_process_cpu_set
from src.core.dashboard_payload import (
    build_session_payload,
    build_station_snapshot,
    collect_session_events,
    collect_session_recordings,
)
from src.core.states import SessionState, StationMode, derive_station_status
from src.core.teardown import EXIT_EXAM_MODE_REASON, ShutdownPolicy
from src.face.recognizer import FaceRecognizer
from src.kiosk.chromium import ChromiumKiosk
from src.kiosk.allowlist import build_allowlist_config, write_extension_config
from src.kiosk.lockdown import Lockdown
from src.kiosk.overlay import SessionOverlay
from src.kiosk.reidentify import run_reidentify
from src.proctor.engine import ProctorEngine, ProctorState
from src.recorder.capture import Capture
from src.recorder.uploader import Uploader

logger = logging.getLogger(__name__)


# ``SessionState``/``StationMode`` vivem em src/core/states.py (só stdlib) para
# que o dashboard use o mesmo vocabulário sem importar cv2/dlib/boto3. São
# reexportados aqui porque o resto do código já os importa deste módulo.
__all__ = [
    "DASHBOARD_CONFIG_FIELD_MAP",
    "DASHBOARD_PROCTOR_FIELD_CASTS",
    "DASHBOARD_ROUTING_FIELDS",
    "SessionConfig",
    "SessionError",
    "SessionManager",
    "SessionRuntime",
    "SessionState",
    "StationMode",
]


#: Tradução explícita ``ExamConfigPayload`` (dashboard) → campo de
#: ``SessionConfig``. É uma tabela, e não um `payload.get()` por campo espalhado
#: no código, porque o rename ``turma`` → ``turma_id`` na fronteira já era um
#: convite a perder campo em silêncio. `tests/test_session_manager.py` garante
#: que todo campo do payload está classificado em um dos três conjuntos abaixo.
DASHBOARD_CONFIG_FIELD_MAP = {
    "turma": "turma_id",
    "assessment": "assessment",
    "timer_minutes": "timer_minutes",
    "prairielearn_url": "prairielearn_url",
    "allowlist": "allowlist",
    "auto_start": "auto_start",
    "s3_prefix": "s3_prefix",
}

#: Campos do payload que ajustam thresholds do proctoring em memória, não o
#: ``SessionConfig``. O valor é o cast aplicado.
#: Nota: o payload do dashboard não expõe ``gaze_v_threshold`` — o limiar
#: vertical só é configurável por `.env`.
DASHBOARD_PROCTOR_FIELD_CASTS = {
    "gaze_h_threshold": float,
    "gaze_duration_sec": float,
    "absence_timeout_sec": float,
    "multi_face_block": bool,
}

#: Campos que são roteamento interno do dashboard e nunca chegam à estação.
DASHBOARD_ROUTING_FIELDS = frozenset({"target_station_ids"})


@dataclass
class SessionConfig:
    turma_id: str | None = None
    assessment: str = "Prova"
    timer_minutes: int = 45
    allowlist: list[str] = field(default_factory=list)
    s3_prefix: str = ""
    prairielearn_url: str = "https://prairielearn.org/pl"
    session_id: str | None = None
    station_id: str = "nuc-local"
    station_name: str = "NUC Local"
    auto_start: bool = False
    no_record: bool = False
    no_kiosk: bool = False
    reidentify_timeout_sec: float = 60.0
    reidentify_matches: int = 3


@dataclass
class SessionRuntime:
    session_id: str
    turma_id: str
    assessment: str
    timer_minutes: int
    student_id: str
    student_name: str
    started_at: datetime
    state: SessionState
    prairielearn_url: str
    block_reason: str | None = None
    stopped_at: datetime | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["started_at"] = self.started_at.isoformat()
        payload["stopped_at"] = self.stopped_at.isoformat() if self.stopped_at else None
        payload["state"] = self.state.value
        return payload


class SessionError(RuntimeError):
    """Erro operacional do session manager."""


class SessionManager:
    """FSM principal da estação de prova.

    O loop contínuo roda em thread própria após start_session().
    Dependências são injetáveis para permitir testes sem câmera real.
    """

    def __init__(
        self,
        *,
        app_config: AppConfig | None = None,
        face_config: FaceConfig | None = None,
        proctor_config: ProctorConfig | None = None,
        recorder_config: RecorderConfig | None = None,
        s3_config: S3Config | None = None,
        recognizer_factory: Callable[..., Any] | None = None,
        engine_factory: Callable[..., Any] | None = None,
        capture_factory: Callable[..., Any] | None = None,
        uploader_factory: Callable[..., Any] | None = None,
        kiosk_factory: Callable[..., Any] | None = None,
        overlay_factory: Callable[..., Any] | None = None,
        lockdown_factory: Callable[..., Any] | None = None,
        video_capture_factory: Callable[[int], Any] | None = None,
        reidentify_fn: Callable[..., bool] | None = None,
        s3_probe: Callable[[], bool] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
    ):
        self._app_cfg = app_config or AppConfig()
        self._face_cfg = face_config or FaceConfig()
        self._proctor_cfg = proctor_config or ProctorConfig()
        self._rec_cfg = recorder_config or self._app_cfg.recorder
        self._s3_cfg = s3_config or self._app_cfg.s3

        self._recognizer_factory = recognizer_factory or (lambda: FaceRecognizer(self._face_cfg))
        self._engine_factory = engine_factory or self._default_engine_factory
        self._capture_factory = capture_factory or self._default_capture_factory
        self._uploader_factory = uploader_factory or self._default_uploader_factory
        self._kiosk_factory = kiosk_factory or (
            lambda: ChromiumKiosk(display=self._rec_cfg.display, manage_gnome_extensions=True)
        )
        self._overlay_factory = overlay_factory or (
            lambda: SessionOverlay(display=self._rec_cfg.display, api_port=self._app_cfg.api_port)
        )
        self._lockdown_factory = lockdown_factory or (
            lambda: Lockdown(display=self._rec_cfg.display, allow_browser_shortcuts=True)
        )
        self._reidentify_fn = reidentify_fn or run_reidentify
        self._s3_probe = s3_probe or self._default_s3_probe
        self._sleep = sleep_fn or time.sleep

        # A posse da câmera (dispositivo físico vs. preview do FFmpeg) vive em
        # SessionCamera; ver src/core/camera.py para o invariante.
        self._camera = SessionCamera(
            face_config=self._face_cfg,
            capture_factory=video_capture_factory or cv2.VideoCapture,
            sleep_fn=self._sleep,
            cv2_module=cv2,
        )

        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._state = SessionState.IDLE
        self._mode = StationMode.MAINTENANCE
        self._next_config = SessionConfig(
            station_id=self._app_cfg.dashboard.station_id,
            station_name=self._app_cfg.dashboard.station_name,
        )
        self._runtime: SessionRuntime | None = None
        self._last_session: SessionRuntime | None = None

        self._recognizer = None
        self._engine = None
        self._capture = None
        self._uploader = None
        self._kiosk = None
        self._overlay = None
        self._waiting_overlay = None
        self._lockdown = None
        self._block_handled = False
        self._original_cpu_set: set[int] | None = None
        self._runtime_ffmpeg_cpu_cores: str | None = None
        self._runtime_proctor_cpu_set: set[int] | None = None

    @property
    def state(self) -> SessionState:
        with self._lock:
            return self._state

    @property
    def mode(self) -> StationMode:
        with self._lock:
            return self._mode

    @property
    def next_config(self) -> SessionConfig:
        with self._lock:
            return SessionConfig(**asdict(self._next_config))

    def update_config(self, **kwargs: Any) -> SessionConfig:
        """Aplica um patch parcial na config da próxima sessão.

        ``None`` significa "não mexer neste campo" — é o que dá semântica de
        patch ao ``POST /config``. Um campo **desconhecido**, por outro lado, é
        erro: antes era descartado em silêncio, então um nome errado (ou o
        ``turma`` do dashboard em vez de ``turma_id``) virava um no-op invisível.
        """
        with self._lock:
            current = asdict(self._next_config)
            unknown = sorted(set(kwargs) - set(current))
            if unknown:
                raise SessionError(
                    "Campos desconhecidos em update_config: "
                    f"{', '.join(unknown)}. Campos válidos: {', '.join(sorted(current))}"
                )
            for key, value in kwargs.items():
                if value is not None:
                    current[key] = value
            self._next_config = SessionConfig(**current)
            return self.next_config

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": self._state.value,
                "mode": self._mode.value,
                "station_status": self._station_status(),
                "session_id": self._runtime.session_id if self._runtime else None,
                "assessment": self._runtime.assessment if self._runtime else self._next_config.assessment,
                "turma_id": self._runtime.turma_id if self._runtime else self._next_config.turma_id,
                "student_id": self._runtime.student_id if self._runtime else None,
                "student_name": self._runtime.student_name if self._runtime else None,
                "seconds_remaining": self._seconds_remaining(),
                "block_reason": self._runtime.block_reason if self._runtime else None,
            }

    def get_session(self) -> dict[str, Any] | None:
        with self._lock:
            if self._runtime is not None:
                return self._runtime.to_dict()
            if self._last_session is not None:
                return self._last_session.to_dict()
            return None

    def get_health(self) -> dict[str, Any]:
        # Cada probe abre a câmera / bate no S3, então avalia uma vez só.
        camera_ok = self._camera_ok()
        s3_ok = self._s3_probe()
        return {
            "status": "ok" if camera_ok and s3_ok else "degraded",
            "state": self.state.value,
            "mode": self.mode.value,
            "camera_ok": camera_ok,
            "s3_ok": s3_ok,
        }

    def apply_dashboard_config(self, payload: dict[str, Any]) -> SessionConfig:
        """Traduz um ``ExamConfigPayload`` do dashboard para a config local.

        O mapeamento é dirigido por ``DASHBOARD_CONFIG_FIELD_MAP`` e pelos dois
        conjuntos irmãos, de modo que todo campo do payload tem destino
        declarado — config da sessão, threshold de proctoring ou roteamento do
        dashboard.
        """
        config = self.update_config(
            **{
                session_field: payload.get(payload_field)
                for payload_field, session_field in DASHBOARD_CONFIG_FIELD_MAP.items()
            }
        )

        for payload_field, cast in DASHBOARD_PROCTOR_FIELD_CASTS.items():
            value = payload.get(payload_field)
            if value is not None:
                setattr(self._proctor_cfg, payload_field, cast(value))

        self._write_browser_allowlist_config(config)
        return config

    @property
    def data_dir(self) -> Path:
        """Diretório de dados desta estação (sessões, eventos, gravações).

        Público porque o heartbeat precisa localizar o ``events.jsonl`` da
        sessão corrente; antes ele alcançava ``_app_cfg`` por dentro.
        """
        return Path(self._app_cfg.data_dir)

    def dashboard_snapshot(self) -> dict[str, Any]:
        return build_station_snapshot(
            status=self.get_status(),
            config=self._next_config,
            runtime=self._runtime,
        )

    def prepare_exam_mode(self) -> dict[str, Any]:
        with self._lock:
            if self._state != SessionState.IDLE:
                raise SessionError(f"Não é possível preparar modo prova em {self._state.value}")
            self._mode = StationMode.EXAM_READY
            return self.get_status()

    def enter_exam_mode(self) -> dict[str, Any]:
        with self._lock:
            if self._state != SessionState.IDLE:
                raise SessionError(f"Não é possível entrar em modo prova em {self._state.value}")
            self._write_browser_allowlist_config(self._next_config)
            self._ensure_exam_lockdown_enabled()
            if self._next_config.auto_start and self._next_config.turma_id:
                self._ensure_identification_camera_open()
            self._mode = StationMode.WAITING_STUDENT
            self._show_waiting_overlay()
            return self.get_status()

    def exit_exam_mode(self) -> dict[str, Any]:
        stop_error: Exception | None = None
        if self.state != SessionState.IDLE:
            try:
                self.stop_session(reason=EXIT_EXAM_MODE_REASON)
            except Exception as exc:
                logger.warning("Falha ao parar sessão durante saída do modo prova: %s", exc)
                stop_error = exc
        with self._lock:
            self._hide_waiting_overlay()
            self._disable_exam_lockdown()
            self._release_camera()
            self._mode = StationMode.MAINTENANCE
            status = self.get_status()
        if stop_error is not None:
            raise SessionError("Modo prova restaurado, mas houve erro ao encerrar a sessão") from stop_error
        return status

    def recover_exam_mode(self) -> dict[str, Any]:
        """Recuperação manual: encerra componentes e restaura o GNOME para manutenção."""
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=3)

        with self._lock:
            if self._runtime is not None:
                self._runtime.notes["stop_reason"] = "recover_exam_mode"
                self._runtime.stopped_at = datetime.now(timezone.utc)
                self._last_session = self._runtime
            self._shutdown_components(ShutdownPolicy.full_teardown())
            self._restore_runtime_cpu_affinity()
            self._runtime = None
            self._thread = None
            self._block_handled = False
            self._set_state(SessionState.IDLE)
            self._mode = StationMode.MAINTENANCE
            return self.get_status()

    def dashboard_session_payload(self, *, include_completed: bool = True) -> dict[str, Any] | None:
        target = self._runtime if self._runtime is not None else (self._last_session if include_completed else None)
        if target is None:
            return None
        return build_session_payload(target=target, station_id=self._next_config.station_id)

    def start_session(
        self,
        *,
        turma_id: str | None = None,
        prairielearn_url: str | None = None,
        session_id: str | None = None,
        student_id: str | None = None,
        student_name: str | None = None,
        no_record: bool | None = None,
        no_kiosk: bool | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            if self._state != SessionState.IDLE:
                raise SessionError(f"Sessão já está em andamento: {self._state.value}")

            try:
                cfg = self._merged_config(
                    turma_id=turma_id,
                    prairielearn_url=prairielearn_url,
                    session_id=session_id,
                    no_record=no_record,
                    no_kiosk=no_kiosk,
                )
                if not cfg.turma_id:
                    raise SessionError("turma_id é obrigatório para iniciar a sessão")

                self._set_state(SessionState.IDENTIFYING)
                self._recognizer = self._recognizer_factory()
                self._recognizer.load_turma(cfg.turma_id)
                self._ensure_identification_camera_open()

                identified_id, identified_name = self._identify_student(student_id, student_name)
                if self._is_repeated_autostart_student(
                    turma_id=cfg.turma_id,
                    assessment=cfg.assessment,
                    student_id=identified_id,
                    auto_start=cfg.auto_start,
                ):
                    raise SessionError(
                        "Aluno já concluiu esta prova nesta estação; aguardando outro aluno ou nova configuração"
                    )
                self._hide_waiting_overlay()
                runtime_session_id = cfg.session_id or self._make_session_id(
                    cfg.turma_id,
                    identified_name,
                )

                self._runtime = SessionRuntime(
                    session_id=runtime_session_id,
                    turma_id=cfg.turma_id,
                    assessment=cfg.assessment,
                    timer_minutes=cfg.timer_minutes,
                    student_id=identified_id,
                    student_name=identified_name,
                    started_at=datetime.now(timezone.utc),
                    state=SessionState.SESSION,
                    prairielearn_url=cfg.prairielearn_url,
                )

                self._uploader = None if cfg.no_record else self._uploader_factory(runtime_session_id)
                self._prepare_runtime_cpu_affinity()
                self._capture = None if cfg.no_record else self._capture_factory(runtime_session_id)
                self._kiosk = None if cfg.no_kiosk else self._kiosk_factory()
                self._overlay = self._overlay_factory()
                if self._lockdown is None:
                    self._lockdown = self._lockdown_factory()
                self._engine = self._engine_factory(runtime_session_id)

                if self._uploader is not None:
                    self._uploader.start()
                if self._capture is not None:
                    self._release_camera()
                    self._capture.start()
                    self._open_preview_camera(self._capture.preview_url)
                    self._apply_runtime_cpu_affinity()
                self._lockdown.enable()
                if self._kiosk is not None:
                    self._kiosk.start(cfg.prairielearn_url, allowlist=cfg.allowlist)
                if self._overlay is not None:
                    self._overlay.start_controls()

                self._engine.start()

                self._stop_event.clear()
                self._block_handled = False
                self._set_state(SessionState.SESSION)
                self._mode = StationMode.SESSION
                self._thread = threading.Thread(
                    target=self._session_loop,
                    name=f"session-{runtime_session_id}",
                    daemon=True,
                )
                self._thread.start()

                logger.info(
                    "Sessão iniciada: %s (%s / %s)",
                    runtime_session_id,
                    identified_id,
                    identified_name,
                )
                return self.get_status()
            except Exception:
                self._shutdown_components(
                    ShutdownPolicy.for_failed_start(
                        mode=self._mode,
                        session_started=self._runtime is not None,
                    )
                )
                self._restore_runtime_cpu_affinity()
                self._runtime = None
                self._thread = None
                self._block_handled = False
                self._set_state(SessionState.IDLE)
                if self._mode == StationMode.SESSION:
                    self._mode = StationMode.EXAM_READY
                elif self._mode == StationMode.WAITING_STUDENT:
                    self._show_waiting_overlay()
                raise

    def stop_session(self, *, reason: str = "manual") -> dict[str, Any]:
        with self._lock:
            if self._state == SessionState.IDLE:
                return self.get_status()
            self._set_state(SessionState.UPLOADING)
            if self._runtime is not None:
                self._runtime.notes["stop_reason"] = reason

        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=10)

        with self._lock:
            runtime = self._runtime
            uploader = self._uploader
            policy = ShutdownPolicy.for_stopped_session(
                mode=self._mode,
                auto_start=self._next_config.auto_start,
                reason=reason,
            )
            # keep_lockdown é, por construção, o mesmo predicado de "volta para
            # WAITING_STUDENT" — usado nas duas decisões abaixo de propósito.
            keep_waiting = policy.keep_lockdown
            self._shutdown_components(policy)
            self._restore_runtime_cpu_affinity()
            if runtime is not None:
                runtime.notes["dashboard_events"] = self._collect_dashboard_events(runtime.session_id)
                runtime.notes["dashboard_recordings"] = self._collect_dashboard_recordings(uploader)
                runtime.stopped_at = datetime.now(timezone.utc)
                self._last_session = runtime
            self._runtime = None
            self._thread = None
            self._block_handled = False
            self._set_state(SessionState.IDLE)
            if self._mode == StationMode.SESSION:
                self._mode = StationMode.WAITING_STUDENT if keep_waiting else StationMode.EXAM_READY
                if self._mode == StationMode.WAITING_STUDENT:
                    self._ensure_identification_camera_open()
                    self._show_waiting_overlay()
            return self.get_status()

    def unblock_session(self) -> dict[str, Any]:
        with self._lock:
            if self._state != SessionState.BLOCKED:
                raise SessionError("Sessão não está bloqueada")
            if self._engine is not None:
                self._engine.unblock()
            if self._kiosk is not None:
                self._kiosk.unblock()
            if self._overlay is not None:
                self._overlay.hide_blocked()
            if self._runtime is not None:
                self._runtime.block_reason = None
            self._block_handled = False
            self._set_state(SessionState.SESSION)
            return self.get_status()

    def _session_loop(self) -> None:
        while not self._stop_event.is_set():
            if not self._ensure_browser_running():
                break
            try:
                ret, frame = self._camera.read()
            except Exception as exc:  # pragma: no cover - hardware/driver path
                logger.error("Falha ao ler câmera: %s", exc)
                break

            if not ret or frame is None:
                self._sleep(0.01)
                continue

            state = self._engine.update(frame)
            if state == ProctorState.BLOCKED:
                self._handle_blocked(frame)
            else:
                with self._lock:
                    if self._state != SessionState.UPLOADING:
                        self._set_state(SessionState.SESSION)
                        self._block_handled = False

        logger.info("Loop da sessão encerrado")

    def _handle_blocked(self, frame: Any) -> None:
        with self._lock:
            if self._block_handled:
                return
            self._block_handled = True
            reason = self._engine.block_reason.value if self._engine.block_reason else None
            if self._runtime is not None:
                self._runtime.block_reason = reason
            self._set_state(SessionState.BLOCKED)
            if self._kiosk is not None:
                self._kiosk.block()
            if self._overlay is not None:
                self._overlay.show_blocked(reason)

        ok = self._reidentify_fn(
            recognizer=self._recognizer,
            cap=self._camera.handle,
            expected_student_id=self._runtime.student_id,
            timeout_sec=self._next_config.reidentify_timeout_sec,
            required_matches=self._next_config.reidentify_matches,
        )

        if ok:
            with self._lock:
                if self._engine is not None:
                    self._engine.unblock()
                if self._kiosk is not None:
                    self._kiosk.unblock()
                if self._overlay is not None:
                    self._overlay.hide_blocked()
                if self._runtime is not None:
                    self._runtime.block_reason = None
                self._block_handled = False
                if self._state != SessionState.UPLOADING:
                    self._set_state(SessionState.SESSION)

    def _identify_student(
        self,
        student_id: str | None,
        student_name: str | None,
    ) -> tuple[str, str]:
        if student_id and student_name:
            return student_id, student_name

        max_attempts = self._face_cfg.max_identification_attempts
        for _ in range(max_attempts):
            ret, frame = self._camera.read()
            if not ret or frame is None:
                continue
            result = self._recognizer.identify(frame)
            if result.is_match:
                return result.student_id, result.student_name

        raise SessionError(
            f"Aluno não identificado após {max_attempts} tentativas"
        )

    def _ensure_identification_camera_open(self) -> None:
        """Abre o dispositivo físico para identificar aluno / esperar aluno."""
        try:
            self._camera.open_device()
        except CameraError as exc:
            raise SessionError(str(exc)) from exc

    def _open_preview_camera(self, source: str, timeout_sec: float = 5.0):
        """Troca a fonte de vídeo para o preview publicado pelo FFmpeg."""
        try:
            return self._camera.open_preview(source, timeout_sec=timeout_sec)
        except CameraError as exc:
            raise SessionError(str(exc)) from exc

    def _release_camera(self) -> None:
        self._camera.release()

    def _camera_ok(self) -> bool:
        return self._camera.probe()

    def _default_s3_probe(self) -> bool:
        try:
            client = boto3.client("s3", region_name=self._s3_cfg.region)
            client.head_bucket(Bucket=self._s3_cfg.bucket)
            return True
        except (ClientError, BotoCoreError, OSError):
            return False

    def _default_engine_factory(self, session_id: str):
        return ProctorEngine(
            session_id=session_id,
            proctor_config=self._proctor_cfg,
            face_config=self._face_cfg,
            app_config=self._app_cfg,
            enable_eye_gaze=False,
        )

    def _default_capture_factory(self, session_id: str):
        recorder_cfg = self._rec_cfg
        if self._runtime_ffmpeg_cpu_cores is not None:
            recorder_cfg = recorder_cfg.model_copy(
                update={"ffmpeg_cpu_cores": self._runtime_ffmpeg_cpu_cores}
            )
        return Capture(
            session_id=session_id,
            s3_config=self._s3_cfg,
            face_config=self._face_cfg,
            app_config=self._app_cfg,
            recorder_config=recorder_cfg,
            on_segment_ready=None if self._uploader is None else self._uploader.enqueue,
            display=self._rec_cfg.display,
            screen_size=self._rec_cfg.screen_size,
        )

    def _default_uploader_factory(self, session_id: str):
        return Uploader(
            session_id=session_id,
            s3_config=self._s3_cfg,
            app_config=self._app_cfg,
            delete_after_upload=self._rec_cfg.delete_after_upload,
        )

    def _merged_config(
        self,
        *,
        turma_id: str | None,
        assessment: str | None = None,
        timer_minutes: int | None = None,
        allowlist: list[str] | None = None,
        s3_prefix: str | None = None,
        prairielearn_url: str | None,
        session_id: str | None,
        auto_start: bool | None = None,
        no_record: bool | None,
        no_kiosk: bool | None,
    ) -> SessionConfig:
        base = asdict(self._next_config)
        if turma_id is not None:
            base["turma_id"] = turma_id
        if assessment is not None:
            base["assessment"] = assessment
        if timer_minutes is not None:
            base["timer_minutes"] = timer_minutes
        if allowlist is not None:
            base["allowlist"] = allowlist
        if s3_prefix is not None:
            base["s3_prefix"] = s3_prefix
        if prairielearn_url is not None:
            base["prairielearn_url"] = prairielearn_url
        if session_id is not None:
            base["session_id"] = session_id
        if auto_start is not None:
            base["auto_start"] = auto_start
        if no_record is not None:
            base["no_record"] = no_record
        if no_kiosk is not None:
            base["no_kiosk"] = no_kiosk
        return SessionConfig(**base)

    def _make_session_id(self, turma_id: str, student_name: str) -> str:
        student_slug = self._slugify_student_name(student_name)
        return f"{turma_id}_{student_slug}_{time.strftime('%Y%m%d_%H%M%S')}"

    def _is_repeated_autostart_student(
        self,
        *,
        turma_id: str,
        assessment: str,
        student_id: str,
        auto_start: bool,
    ) -> bool:
        if not auto_start or self._last_session is None:
            return False
        return (
            self._last_session.student_id == student_id
            and self._last_session.turma_id == turma_id
            and self._last_session.assessment == assessment
        )

    @staticmethod
    def _slugify_student_name(student_name: str) -> str:
        normalized = unicodedata.normalize("NFKD", student_name)
        ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
        cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in ascii_only)
        collapsed = "_".join(part for part in cleaned.split("_") if part)
        return collapsed or "aluno"

    def _set_state(self, state: SessionState) -> None:
        self._state = state
        if self._runtime is not None:
            self._runtime.state = state

    def _station_status(self) -> str:
        return derive_station_status(self._state, self._mode)

    def _hide_waiting_overlay(self) -> None:
        if self._waiting_overlay is None:
            return
        try:
            self._waiting_overlay.hide_waiting()
            self._waiting_overlay.stop()
        except Exception as exc:
            logger.warning("Falha ao fechar overlay de espera: %s", exc)
        self._waiting_overlay = None

    def _show_waiting_overlay(self, message: str | None = None) -> None:
        if self._waiting_overlay is None:
            self._waiting_overlay = self._overlay_factory()
        else:
            if message is None:
                return
        self._waiting_overlay.show_waiting(message)

    def _write_browser_allowlist_config(self, config: SessionConfig) -> None:
        allowlist_config = build_allowlist_config(
            start_url=config.prairielearn_url,
            allowlist=config.allowlist,
        )
        write_extension_config(allowlist_config)

    def _ensure_exam_lockdown_enabled(self) -> None:
        if self._lockdown is None:
            self._lockdown = self._lockdown_factory()
        self._lockdown.enable()
        is_enabled = getattr(self._lockdown, "is_enabled", None)
        if is_enabled is False:
            self._lockdown = None
            raise SessionError("Lockdown do modo prova não foi ativado")

    def _disable_exam_lockdown(self) -> None:
        if self._lockdown is None:
            return
        try:
            self._lockdown.disable()
        finally:
            self._lockdown = None

    def _seconds_remaining(self) -> int | None:
        if self._runtime is None:
            return None
        elapsed = int((datetime.now(timezone.utc) - self._runtime.started_at).total_seconds())
        total = max(1, self._runtime.timer_minutes) * 60
        return max(0, total - elapsed)

    def _prepare_runtime_cpu_affinity(self) -> None:
        self._runtime_ffmpeg_cpu_cores = None
        self._runtime_proctor_cpu_set = None

        current = get_process_cpu_set()
        if current is None:
            return

        ffmpeg_cpus, proctor_cpus = auto_split_cpu_sets(
            available=current,
            ffmpeg_override=parse_cpu_set(self._rec_cfg.ffmpeg_cpu_cores),
            proctor_override=parse_cpu_set(self._rec_cfg.proctor_cpu_cores),
        )
        if ffmpeg_cpus:
            self._runtime_ffmpeg_cpu_cores = ",".join(str(cpu) for cpu in sorted(ffmpeg_cpus))
        self._runtime_proctor_cpu_set = proctor_cpus

    def _apply_runtime_cpu_affinity(self) -> None:
        current = get_process_cpu_set()
        if current is None or self._original_cpu_set is not None:
            return
        if not self._runtime_proctor_cpu_set or self._runtime_proctor_cpu_set == current:
            return
        if set_process_cpu_set(self._runtime_proctor_cpu_set):
            self._original_cpu_set = current
            logger.info(
                "Afinidade de CPU aplicada ao processo principal: %s (ffmpeg reservado em %s)",
                sorted(self._runtime_proctor_cpu_set),
                self._runtime_ffmpeg_cpu_cores or "default",
            )

    def _restore_runtime_cpu_affinity(self) -> None:
        if self._original_cpu_set is None:
            self._runtime_ffmpeg_cpu_cores = None
            self._runtime_proctor_cpu_set = None
            return
        if set_process_cpu_set(self._original_cpu_set):
            logger.info("Afinidade de CPU restaurada: %s", sorted(self._original_cpu_set))
        self._original_cpu_set = None
        self._runtime_ffmpeg_cpu_cores = None
        self._runtime_proctor_cpu_set = None

    def _ensure_browser_running(self) -> bool:
        kiosk = self._kiosk
        if kiosk is None:
            return True
        is_running = getattr(kiosk, "is_running", True)
        if is_running:
            return True
        relaunch = getattr(kiosk, "relaunch", None)
        if callable(relaunch):
            try:
                if relaunch():
                    if self._runtime is not None:
                        self._runtime.notes.setdefault("operational_events", []).append(
                            {
                                "type": "browser_relaunch",
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            }
                        )
                    return True
            except Exception as exc:
                logger.warning("Falha ao relançar Chromium: %s", exc)

        with self._lock:
            if self._runtime is not None:
                self._runtime.block_reason = "BROWSER_EXIT"
                self._runtime.notes.setdefault("operational_events", []).append(
                    {
                        "type": "browser_exit",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
            if self._overlay is not None:
                self._overlay.show_blocked("BROWSER_EXIT")
            if self._state != SessionState.UPLOADING:
                self._set_state(SessionState.BLOCKED)
        self._stop_event.set()
        return False

    def _shutdown_components(self, policy: ShutdownPolicy | None = None) -> None:
        """Encerra os componentes da sessão conforme a política de teardown.

        O que sobrevive é decidido por ``ShutdownPolicy`` (src/core/teardown.py),
        não por booleanos montados no ponto de chamada.
        """
        policy = policy or ShutdownPolicy.full_teardown()

        components = [
            self._kiosk,
            self._overlay,
            self._capture,
            self._engine,
            self._uploader,
        ]
        if not policy.keep_lockdown:
            components.append(self._lockdown)
        for component in components:
            if component is None:
                continue
            try:
                if component is self._lockdown:
                    component.disable()
                else:
                    component.stop()
            except Exception as exc:  # pragma: no cover - cleanup best effort
                logger.warning("Falha ao encerrar componente %s: %s", type(component).__name__, exc)

        if not policy.keep_camera:
            self._release_camera()
        if not policy.keep_waiting_overlay:
            self._hide_waiting_overlay()
        self._recognizer = None
        self._engine = None
        self._capture = None
        self._uploader = None
        self._kiosk = None
        self._overlay = None
        if not policy.keep_waiting_overlay:
            self._waiting_overlay = None
        if not policy.keep_lockdown:
            self._lockdown = None

    def _collect_dashboard_events(self, session_id: str) -> list[dict[str, Any]]:
        return collect_session_events(self.data_dir, session_id)

    def _collect_dashboard_recordings(self, uploader: Any) -> list[dict[str, Any]]:
        return collect_session_recordings(uploader, self._s3_cfg.bucket)
