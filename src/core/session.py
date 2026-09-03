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

import json
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
from src.proctor.engine import BlockReason, ProctorEngine, ProctorState
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
    "local_timer_enabled": "local_timer_enabled",
    "prairielearn_url": "prairielearn_url",
    "allowlist": "allowlist",
    "auto_start": "auto_start",
    "allow_repeat_attempts": "allow_repeat_attempts",
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
PRE_EXAM_CONFIRMATION_TIMEOUT_SEC = 20.0
SESSION_IDENTITY_CHECK_INTERVAL_SEC = 10.0


@dataclass
class SessionConfig:
    turma_id: str | None = None
    assessment: str = "Prova"
    timer_minutes: int = 45
    local_timer_enabled: bool = True
    allowlist: list[str] = field(default_factory=list)
    s3_prefix: str = ""
    prairielearn_url: str = "https://prairielearn.org/pl"
    session_id: str | None = None
    station_id: str = "nuc-local"
    station_name: str = "NUC Local"
    auto_start: bool = False
    allow_repeat_attempts: bool = True
    no_record: bool = False
    no_kiosk: bool = False
    reidentify_timeout_sec: float = 20.0
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
    local_timer_enabled: bool = True
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

    #: Componentes cujo ciclo de vida é de **uma sessão**: sempre encerrados por
    #: inteiro no teardown, nunca preservados. A ordem importa — o browser sai
    #: antes da gravação (para o vídeo registrar o fechamento) e o uploader sai
    #: por último (para drenar a fila de segmentos).
    #:
    #: Lockdown, overlay de espera e câmera **não** entram aqui: são escopo de
    #: estação e podem sobreviver entre sessões. Ver ``_shutdown_components``.
    _SESSION_SCOPED_COMPONENTS = ("_kiosk", "_overlay", "_capture", "_engine", "_uploader")

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
        confirmation_fn: Callable[[str, str, float], bool] | None = None,
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
            lambda: ChromiumKiosk(
                display=self._rec_cfg.display,
                manage_gnome_extensions=False,
                require_managed_policy=True,
                policy_helper="/usr/local/sbin/proctor-apply-chromium-policy",
            )
        )
        self._overlay_factory = overlay_factory or (
            lambda: SessionOverlay(display=self._rec_cfg.display, api_port=self._app_cfg.api_port)
        )
        self._lockdown_factory = lockdown_factory or (
            lambda: Lockdown(
                display=self._rec_cfg.display,
                allow_browser_shortcuts=True,
                manage_gnome=False,
            )
        )
        self._reidentify_fn = reidentify_fn or run_reidentify
        self._confirmation_fn = confirmation_fn or self._wait_for_student_confirmation
        self._s3_probe = s3_probe or self._default_s3_probe
        self._sleep = sleep_fn or time.sleep

        # A posse da câmera (dispositivo físico vs. preview do FFmpeg) vive em
        # SessionCamera; ver src/core/camera.py para o invariante.
        self._camera = SessionCamera(
            face_config=self._face_cfg,
            capture_factory=video_capture_factory,
            sleep_fn=self._sleep,
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
        self._config_store_path = Path(self._app_cfg.data_dir) / "station-config.json"
        self._load_persisted_config()
        self._runtime: SessionRuntime | None = None
        self._last_session: SessionRuntime | None = None
        self._last_identity_check_at = 0.0

        self._recognizer = None
        self._engine = None
        self._capture = None
        self._uploader = None
        self._kiosk = None
        self._overlay = None
        self._waiting_overlay = None
        self._confirmation_overlay = None
        self._confirmation_response: bool | None = None
        self._confirmation_event: threading.Event | None = None
        self._identified_student_id: str | None = None
        self._latest_camera_frame: Any | None = None
        self._camera_preview_lock = threading.Lock()
        self._camera_read_active = threading.Event()
        self._last_camera_frame_at: float | None = None
        self._preview_recovery_lock = threading.Lock()
        self._camera_recovering = threading.Event()
        self._preview_watchdog_stop = threading.Event()
        self._preview_watchdog: threading.Thread | None = None
        self._browser_guard_stop = threading.Event()
        self._browser_guard: threading.Thread | None = None
        self._browser_guard_overlay = None
        self._browser_ready = False
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
            self._persist_config()
            return self.next_config

    def _load_persisted_config(self) -> None:
        if not self._app_cfg.persist_session_config or not self._config_store_path.exists():
            return
        try:
            payload = json.loads(self._config_store_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("configuração persistida não é um objeto JSON")
            self._next_config = SessionConfig(**payload)
            logger.info(
                "Configuração restaurada: %s / %s",
                self._next_config.turma_id,
                self._next_config.assessment,
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("Configuração persistida inválida em %s: %s", self._config_store_path, exc)

    def _persist_config(self) -> None:
        if not self._app_cfg.persist_session_config:
            return
        try:
            self._config_store_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self._config_store_path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(asdict(self._next_config), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary.replace(self._config_store_path)
        except (OSError, TypeError, ValueError) as exc:
            logger.warning("Não foi possível persistir configuração em %s: %s", self._config_store_path, exc)

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

    def get_exam_checks(self) -> dict[str, Any]:
        with self._lock:
            student_identified = self._identified_student_id is not None or self._runtime is not None
            block_reason = self._runtime.block_reason if self._runtime else None
            no_kiosk = bool(
                self._runtime.notes.get("no_kiosk") if self._runtime else self._next_config.no_kiosk
            )
            browser_required = self._mode == StationMode.SESSION and not no_kiosk
            browser_ok = self._browser_ready
            webcam_ok = self._camera.is_open
            blocked = self._state == SessionState.BLOCKED
            reason_label = {
                "ABSENCE": "ausência detectada",
                "MULTI_FACE": "múltiplos rostos",
                "GAZE": "olhar fora do permitido",
                "BROWSER_EXIT": "Chromium encerrado",
                "DIFFERENT_USER": "usuário diferente detectado",
            }.get(block_reason, block_reason or "")
            student_ok = student_identified and block_reason != "DIFFERENT_USER"
            presence_ok = student_identified and block_reason != "ABSENCE"
            faces_ok = student_identified and block_reason not in {"ABSENCE", "MULTI_FACE"}
            gaze_ok = student_identified and block_reason != "GAZE"
            checks = [
                {
                    "key": "session",
                    "label": f"Sessão bloqueada: {reason_label}" if blocked else "Sessão liberada",
                    "state": "fail" if blocked else "ok",
                },
                {"key": "webcam", "label": "Webcam detectada", "state": "ok" if webcam_ok else "fail"},
                {
                    "key": "student",
                    "label": "Usuário diferente detectado" if block_reason == "DIFFERENT_USER" else "Aluno identificado",
                    "state": "ok" if student_ok else ("fail" if block_reason == "DIFFERENT_USER" else "pending"),
                },
                {
                    "key": "presence",
                    "label": "Aluno ausente" if block_reason == "ABSENCE" else "Aluno presente",
                    "state": "ok" if presence_ok else ("fail" if block_reason == "ABSENCE" else "pending"),
                },
                {
                    "key": "faces",
                    "label": "Rosto não detectado" if block_reason == "ABSENCE" else "Rosto único",
                    "state": "ok" if faces_ok else ("fail" if block_reason in {"ABSENCE", "MULTI_FACE"} else "pending"),
                },
                {"key": "gaze", "label": "Olhar dentro do permitido", "state": "ok" if gaze_ok else ("fail" if block_reason == "GAZE" else "pending")},
                {
                    "key": "chromium",
                    "label": (
                        "Chromium protegido"
                        if browser_required
                        else ("Chromium desativado" if no_kiosk else "Chromium será iniciado após a confirmação")
                    ),
                    "state": "ok" if no_kiosk or browser_ok else ("fail" if browser_required else "pending"),
                },
            ]
            return {
                "ready": all(
                    check["state"] == "ok"
                    for check in checks
                    if check["key"] != "chromium" or browser_required
                ),
                "state": self._state.value,
                "block_reason": block_reason,
                "seconds_remaining": self._seconds_remaining(),
                "checks": checks,
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

    def restore_exam_mode_on_startup(self) -> dict[str, Any]:
        """Restaura a última prova configurada após reinício da estação."""
        if not self._app_cfg.restore_exam_mode_on_startup:
            return self.get_status()
        with self._lock:
            if self._state != SessionState.IDLE or self._mode != StationMode.MAINTENANCE:
                return self.get_status()
            if not self._next_config.turma_id:
                logger.info("Nenhuma configuração de prova persistida para restaurar")
                return self.get_status()
            if not self._next_config.auto_start:
                self._next_config.auto_start = True
                self._persist_config()
        try:
            status = self.enter_exam_mode()
            logger.info(
                "Modo prova restaurado no startup: %s / %s",
                self._next_config.turma_id,
                self._next_config.assessment,
            )
            return status
        except Exception as exc:
            logger.exception("Falha ao restaurar modo prova no startup: %s", exc)
            return self.get_status()

    def enter_exam_mode(self) -> dict[str, Any]:
        with self._lock:
            if self._state != SessionState.IDLE:
                raise SessionError(f"Não é possível entrar em modo prova em {self._state.value}")
            self._write_browser_allowlist_config(self._next_config)
            self._ensure_exam_lockdown_enabled()
            self._mode = StationMode.WAITING_STUDENT
            self._start_browser_guard()
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
            self._stop_browser_guard()
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
            self._stop_browser_guard()
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

    def respond_to_pre_exam_confirmation(self, *, accepted: bool) -> None:
        """Recebe a decisão do overlay local sem bloquear a thread de início."""
        event = self._confirmation_event
        if event is None:
            raise SessionError("Não há confirmação de aluno pendente")
        self._confirmation_response = accepted
        event.set()

    def get_camera_preview_jpeg(self) -> bytes | None:
        """Retorna um JPEG recente para os overlays locais, sem abrir outra webcam."""
        if self.state == SessionState.IDLE:
            self._read_camera_frame()
        with self._camera_preview_lock:
            frame = self._latest_camera_frame
            if frame is None:
                return None
            frame = self._copy_camera_frame(frame)
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return encoded.tobytes() if ok else None

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
                    allow_repeat_attempts=cfg.allow_repeat_attempts,
                ):
                    raise SessionError(
                        "Aluno já concluiu esta prova nesta estação; aguardando outro aluno ou nova configuração"
                    )
                self._identified_student_id = identified_id
                # A resposta é humana e pode levar até um minuto. Não retenha o
                # lock da estação nesse período: health/status e a API local
                # precisam continuar disponíveis para o operador e o dashboard.
                self._lock.release()
                try:
                    confirmed = self._confirmation_fn(
                        identified_id,
                        identified_name,
                        PRE_EXAM_CONFIRMATION_TIMEOUT_SEC,
                    )
                finally:
                    self._lock.acquire()
                if not confirmed:
                    raise SessionError("Confirmação do aluno cancelada ou expirada")
                self._show_waiting_overlay("Preparando sua avaliação...")
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
                    local_timer_enabled=cfg.local_timer_enabled,
                    notes={
                        "exam_mode_active": self._mode == StationMode.WAITING_STUDENT,
                        "no_kiosk": cfg.no_kiosk,
                    },
                )

                self._uploader = None if cfg.no_record else self._uploader_factory(runtime_session_id)
                self._prepare_runtime_cpu_affinity()
                self._capture = None if cfg.no_record else self._capture_factory(runtime_session_id)
                if cfg.no_kiosk:
                    if self._kiosk is not None:
                        self._kiosk.stop()
                    self._kiosk = None
                elif self._kiosk is None:
                    self._kiosk = self._kiosk_factory()
                self._overlay = self._overlay_factory()
                if self._lockdown is None:
                    self._lockdown = self._lockdown_factory()
                self._engine = self._engine_factory(runtime_session_id)

                if self._uploader is not None:
                    self._uploader.start()
                if self._capture is not None:
                    self._camera.handoff_to_external()
                    self._capture.start()
                    self._open_preview_camera(self._capture.preview_url)
                    self._start_preview_watchdog()
                    self._apply_runtime_cpu_affinity()
                self._lockdown.enable()
                if self._kiosk is not None and not getattr(self._kiosk, "is_running", False):
                    self._kiosk.start(cfg.prairielearn_url, allowlist=cfg.allowlist)
                if self._kiosk is not None:
                    self._browser_ready = self._ensure_exam_browser_fullscreen()
                    if not self._browser_ready:
                        raise SessionError("Chromium não ficou pronto para iniciar a avaliação")
                self._hide_waiting_overlay()
                if self._overlay is not None:
                    self._overlay.start_controls()

                self._engine.start()

                self._stop_event.clear()
                self._last_identity_check_at = time.monotonic()
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
                logger.exception("Falha ao iniciar sessão")
                self._shutdown_components(
                    ShutdownPolicy.for_failed_start(
                        mode=self._mode,
                        session_started=self._runtime is not None,
                    )
                )
                self._restore_runtime_cpu_affinity()
                self._runtime = None
                self._identified_student_id = None
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
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=10)

        with self._lock:
            runtime = self._runtime
            uploader = self._uploader
            policy = ShutdownPolicy.for_stopped_session(
                mode=self._mode,
                auto_start=self._next_config.auto_start,
                reason=reason,
                exam_mode_active=bool(runtime and runtime.notes.get("exam_mode_active")),
            )
            # keep_lockdown é, por construção, o mesmo predicado de "volta para
            # WAITING_STUDENT" — usado nas duas decisões abaixo de propósito.
            keep_waiting = policy.keep_lockdown
            if keep_waiting:
                self._show_waiting_overlay("Preparando a próxima sessão...")
            self._shutdown_components(policy)
            self._restore_runtime_cpu_affinity()
            if runtime is not None:
                runtime.notes["dashboard_events"] = self._collect_dashboard_events(runtime.session_id)
                runtime.notes["dashboard_recordings"] = self._collect_dashboard_recordings(uploader)
                runtime.stopped_at = datetime.now(timezone.utc)
                self._last_session = runtime
            self._runtime = None
            self._identified_student_id = None
            self._thread = None
            self._block_handled = False
            self._set_state(SessionState.IDLE)
            if self._mode == StationMode.SESSION:
                self._mode = StationMode.WAITING_STUDENT if keep_waiting else StationMode.EXAM_READY
                if self._mode == StationMode.WAITING_STUDENT:
                    self._hide_waiting_overlay()
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
                ret, frame = self._read_camera_frame()
            except Exception as exc:  # pragma: no cover - hardware/driver path
                logger.error("Falha ao ler câmera: %s", exc)
                break

            if not ret or frame is None:
                if self._camera_recovering.is_set():
                    self._sleep(0.1)
                    continue
                state = self._engine.update(None)
                if state == ProctorState.BLOCKED:
                    self._handle_blocked()
                self._sleep(0.1)
                continue

            self._verify_session_identity(frame)
            state = self._engine.update(frame)
            if state == ProctorState.BLOCKED:
                self._handle_blocked()
            else:
                with self._lock:
                    if self._state != SessionState.UPLOADING:
                        self._set_state(SessionState.SESSION)
                        self._block_handled = False

        logger.info("Loop da sessão encerrado")

    def _verify_session_identity(self, frame: Any) -> None:
        """Compara periodicamente o rosto presente com o aluno autenticado."""
        now = time.monotonic()
        if now - self._last_identity_check_at < SESSION_IDENTITY_CHECK_INTERVAL_SEC:
            return
        self._last_identity_check_at = now
        if self._recognizer is None or self._runtime is None or self._engine is None:
            return

        try:
            result = self._recognizer.identify(frame)
        except Exception as exc:
            logger.warning("Falha na verificação periódica de identidade: %s", exc)
            return

        if result.is_match and result.student_id != self._runtime.student_id:
            logger.warning(
                "Usuário diferente detectado: esperado=%s detectado=%s",
                self._runtime.student_id,
                result.student_id,
            )
            self._engine.block(
                BlockReason.DIFFERENT_USER,
                details={
                    "expected_student_id": self._runtime.student_id,
                    "detected_student_id": result.student_id,
                },
            )

    def _handle_blocked(self) -> None:
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
                self._overlay.show_blocked(
                    reason,
                    student_id=self._runtime.student_id if self._runtime is not None else None,
                    timeout_sec=self._next_config.reidentify_timeout_sec,
                )

        ok = self._reidentify_fn(
            recognizer=self._recognizer,
            cap=self._camera.handle,
            read_frame=self._read_camera_frame,
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
        else:
            with self._lock:
                should_cancel = self._state == SessionState.BLOCKED and self._runtime is not None
            if should_cancel:
                if self._engine is not None:
                    self._engine.cancel_after_block_timeout(self._next_config.reidentify_timeout_sec)
                self.stop_session(reason="block_timeout")

    def _identify_student(
        self,
        student_id: str | None,
        student_name: str | None,
    ) -> tuple[str, str]:
        if student_id and student_name:
            return student_id, student_name

        max_attempts = self._face_cfg.max_identification_attempts
        for _ in range(max_attempts):
            ret, frame = self._read_camera_frame()
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
            handle = self._camera.open_preview(source, timeout_sec=timeout_sec)
            with self._camera_preview_lock:
                self._last_camera_frame_at = time.monotonic()
            return handle
        except CameraError as exc:
            raise SessionError(str(exc)) from exc

    def _start_preview_watchdog(self) -> None:
        if self._preview_watchdog and self._preview_watchdog.is_alive():
            return
        self._preview_watchdog_stop.clear()
        self._preview_watchdog = threading.Thread(
            target=self._preview_watchdog_loop,
            name="preview-watchdog",
            daemon=True,
        )
        self._preview_watchdog.start()

    def _start_browser_guard(self) -> None:
        if self._browser_guard and self._browser_guard.is_alive():
            return
        self._browser_guard_stop.clear()
        self._browser_guard = threading.Thread(
            target=self._browser_guard_loop,
            name="browser-fullscreen-guard",
            daemon=True,
        )
        self._browser_guard.start()

    def _stop_browser_guard(self) -> None:
        self._browser_guard_stop.set()
        thread = self._browser_guard
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2)
        self._browser_guard = None
        self._browser_ready = False
        self._hide_browser_guard_overlay()

    def _browser_guard_loop(self) -> None:
        while not self._browser_guard_stop.wait(1.0):
            with self._lock:
                no_kiosk = bool(
                    self._runtime.notes.get("no_kiosk")
                    if self._runtime
                    else self._next_config.no_kiosk
                )
                active_exam = self._mode == StationMode.SESSION and not no_kiosk
            if not active_exam:
                continue
            browser_ok = self._ensure_exam_browser_fullscreen()
            with self._lock:
                self._browser_ready = browser_ok
            if browser_ok:
                self._hide_browser_guard_overlay()
            else:
                self._show_browser_guard_overlay("BROWSER_NOT_FULLSCREEN")

    def _ensure_exam_browser_fullscreen(self) -> bool:
        kiosk = self._kiosk
        if kiosk is None:
            return False
        if not getattr(kiosk, "is_running", True):
            self._show_browser_guard_overlay("BROWSER_EXIT")
            relaunch = getattr(kiosk, "relaunch", None)
            if not callable(relaunch) or not relaunch():
                return False
        ensure_fullscreen = getattr(kiosk, "ensure_fullscreen", None)
        return True if not callable(ensure_fullscreen) else bool(ensure_fullscreen())

    def _show_browser_guard_overlay(self, reason: str) -> None:
        with self._lock:
            if self._browser_guard_overlay is None:
                self._browser_guard_overlay = self._overlay_factory()
            self._browser_guard_overlay.show_blocked(reason)

    def _hide_browser_guard_overlay(self) -> None:
        with self._lock:
            if self._browser_guard_overlay is None:
                return
            self._browser_guard_overlay.hide_blocked()
            self._browser_guard_overlay.stop()
            self._browser_guard_overlay = None

    def _stop_preview_watchdog(self) -> None:
        self._preview_watchdog_stop.set()
        thread = self._preview_watchdog
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2)
        self._preview_watchdog = None

    def _preview_watchdog_loop(self) -> None:
        while not self._preview_watchdog_stop.wait(0.5):
            if (
                self._capture is None
                or self._camera_read_active.is_set()
                or not self._preview_is_stale(timeout_sec=3.0)
            ):
                continue
            self._recover_preview_camera()

    def _recover_preview_camera(self) -> None:
        if self._capture is None:
            return
        if not self._preview_recovery_lock.acquire(blocking=False):
            return
        self._camera_recovering.set()
        try:
            logger.warning("Preview UDP sem frames; reconectando o proctoring")
            self._camera.handoff_to_external()
            ensure_webcam = getattr(self._capture, "ensure_webcam_running", None)
            if callable(ensure_webcam) and not ensure_webcam():
                logger.warning("Stream de webcam ainda indisponível; mantendo sessão ativa")
                return
            self._open_preview_camera(self._capture.preview_url)
        except SessionError as exc:
            logger.warning("Falha ao reconectar preview UDP: %s", exc)
        finally:
            self._camera_recovering.clear()
            self._preview_recovery_lock.release()

    def _release_camera(self, *, clear_preview: bool = True) -> None:
        self._camera.release()
        if not clear_preview:
            return
        with self._camera_preview_lock:
            self._latest_camera_frame = None
            self._last_camera_frame_at = None

    def _read_camera_frame(self) -> tuple[bool, Any]:
        self._camera_read_active.set()
        try:
            ret, frame = self._camera.read()
        finally:
            self._camera_read_active.clear()
        if ret and frame is not None:
            with self._camera_preview_lock:
                self._latest_camera_frame = self._copy_camera_frame(frame)
                self._last_camera_frame_at = time.monotonic()
        return ret, frame

    def _preview_is_stale(self, *, timeout_sec: float = 1.0) -> bool:
        with self._camera_preview_lock:
            last_frame_at = self._last_camera_frame_at
        return last_frame_at is None or time.monotonic() - last_frame_at >= timeout_sec

    @staticmethod
    def _copy_camera_frame(frame: Any) -> Any:
        return frame.copy() if hasattr(frame, "copy") else frame

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
        allow_repeat_attempts: bool,
    ) -> bool:
        if allow_repeat_attempts or not auto_start or self._last_session is None:
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

    def _wait_for_student_confirmation(
        self,
        student_id: str,
        student_name: str,
        timeout_sec: float,
    ) -> bool:
        """Exibe a confirmação e aguarda a resposta do overlay por no máximo um minuto."""
        event = threading.Event()
        self._confirmation_response = None
        self._confirmation_event = event
        self._hide_waiting_overlay()
        self._confirmation_overlay = self._overlay_factory()
        accepted = False
        try:
            self._confirmation_overlay.show_identity_confirmation(
                student_id=student_id,
                student_name=student_name,
                timeout_sec=timeout_sec,
            )
            event.wait(timeout=timeout_sec)
            accepted = self._confirmation_response is True
            return accepted
        finally:
            if self._confirmation_overlay is not None:
                self._confirmation_overlay.hide_identity_confirmation()
            self._confirmation_overlay = None
            if (
                not accepted
                and self._mode == StationMode.WAITING_STUDENT
                and self._state == SessionState.IDENTIFYING
            ):
                self._show_waiting_overlay()
            self._confirmation_response = None
            self._confirmation_event = None

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
        if self._runtime is None or not self._runtime.local_timer_enabled:
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
            self._browser_ready = False
            if self._runtime is not None:
                self._runtime.block_reason = "BROWSER_EXIT"
                self._runtime.notes.setdefault("operational_events", []).append(
                    {
                        "type": "browser_exit",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
            if self._overlay is not None:
                self._overlay.show_blocked(
                    "BROWSER_EXIT",
                    student_id=self._runtime.student_id if self._runtime is not None else None,
                )
            if self._state != SessionState.UPLOADING:
                self._set_state(SessionState.BLOCKED)
        self._stop_event.set()
        return False

    def _shutdown_components(self, policy: ShutdownPolicy | None = None) -> None:
        """Encerra os componentes conforme a política de teardown.

        Há dois escopos de vida distintos aqui, e é a diferença entre eles que
        justifica a ``ShutdownPolicy``:

        * **Escopo de sessão** (``_SESSION_SCOPED_COMPONENTS`` + recognizer):
          pertencem a *uma* prova e são sempre encerrados por inteiro.
        * **Escopo de estação** (lockdown, overlay de espera, câmera): podem
          sobreviver, porque a estação pode continuar em modo prova esperando o
          próximo aluno. Só estes consultam a política.

        Antes os dois escopos estavam na mesma lista, e o lockdown precisava ser
        detectado com ``component is self._lockdown`` para chamar ``disable()``
        em vez de ``stop()``.
        """
        policy = policy or ShutdownPolicy.full_teardown()
        self._stop_preview_watchdog()
        if not policy.keep_lockdown:
            self._stop_browser_guard()

        for name in self._SESSION_SCOPED_COMPONENTS:
            component = getattr(self, name)
            if component is None:
                continue
            try:
                component.stop()
            except Exception as exc:  # pragma: no cover - cleanup best effort
                logger.warning("Falha ao encerrar componente %s: %s", type(component).__name__, exc)
        for name in self._SESSION_SCOPED_COMPONENTS:
            setattr(self, name, None)
        # O recognizer não tem stop(); basta soltar a referência.
        self._recognizer = None

        if not policy.keep_lockdown:
            self._disable_exam_lockdown()
        if not policy.keep_camera:
            self._release_camera()
        if not policy.keep_waiting_overlay:
            self._hide_waiting_overlay()

    def _collect_dashboard_events(self, session_id: str) -> list[dict[str, Any]]:
        return collect_session_events(self.data_dir, session_id)

    def _collect_dashboard_recordings(self, uploader: Any) -> list[dict[str, Any]]:
        return collect_session_recordings(
            uploader,
            self._s3_cfg.bucket,
            self._s3_cfg.segment_duration_sec,
        )
