"""Configuração central da proctoring station.

Valores podem ser definidos via:
  1. Variáveis de ambiente (prioridade maior)
  2. Arquivo .env na raiz do projeto (prioridade menor)

Veja .env.example para referência completa.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Raiz do projeto (dois níveis acima deste arquivo: src/core/config.py)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"


class FaceConfig(BaseSettings):
    """Parâmetros do módulo de reconhecimento facial."""

    # Model paths (dlib) — ficam na NUC
    models_dir: Path = Field(
        default=Path("models"),
        description="Diretório com os .dat do dlib",
    )
    shape_predictor: str = Field(
        default="shape_predictor_68_face_landmarks.dat",
        description="Arquivo do shape predictor (dentro de models_dir)",
    )
    recognition_model: str = Field(
        default="dlib_face_recognition_resnet_model_v1.dat",
        description="Arquivo do modelo de reconhecimento (dentro de models_dir)",
    )

    # Encodings — ficam na NUC, gerados via enroll
    encodings_dir: Path = Field(
        default=Path("data/encodings"),
        description="Diretório local com arquivos .pkl de encodings por turma",
    )

    # Identificação
    match_threshold: float = Field(
        default=0.45,
        description="Distância máxima para considerar match (menor = mais restritivo)",
    )
    use_cnn_detector: bool = Field(
        default=False,
        description="Usar CNN detector (GPU) em vez de HOG (CPU)",
    )
    max_identification_attempts: int = Field(
        default=3,
        description="Tentativas antes de exigir intervenção manual",
    )

    # Camera
    camera_index: int = Field(default=0, description="Índice da câmera (/dev/videoN)")
    camera_width: int = Field(default=1280)
    camera_height: int = Field(default=720)
    camera_fps: int = Field(default=30)

    # Performance
    detection_scale: float = Field(
        default=0.5,
        description="Fator de escala do frame para detecção (0.5 = metade da resolução)",
    )
    num_jitters: int = Field(
        default=1,
        description="Jitters para encoding (mais = mais preciso, mais lento). Use 3-5 no enrollment.",
    )

    @property
    def shape_predictor_path(self) -> Path:
        return self.models_dir / self.shape_predictor

    @property
    def recognition_model_path(self) -> Path:
        return self.models_dir / self.recognition_model

    @property
    def cnn_detector_path(self) -> Path:
        return self.models_dir / "mmod_human_face_detector.dat"

    def validate_models(self) -> list[str]:
        """Retorna lista de modelos dlib faltantes."""
        missing = []
        if not self.shape_predictor_path.exists():
            missing.append(str(self.shape_predictor_path))
        if not self.recognition_model_path.exists():
            missing.append(str(self.recognition_model_path))
        if self.use_cnn_detector and not self.cnn_detector_path.exists():
            missing.append(str(self.cnn_detector_path))
        return missing

    model_config = SettingsConfigDict(
        env_prefix="PROCTOR_FACE_",
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )


class ProctorConfig(BaseSettings):
    """Parâmetros do proctoring engine."""

    gaze_h_threshold: float = 0.35
    gaze_v_threshold: float = 0.30
    gaze_duration_sec: float = 5.0   # segundos em GAZE_WARN antes de BLOCKED
    absence_timeout_sec: float = 5.0
    multi_face_block: bool = True

    model_config = SettingsConfigDict(
        env_prefix="PROCTOR_",
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )


class RecorderConfig(BaseSettings):
    """Parâmetros do módulo de gravação."""

    webcam_input_format: str = Field(
        default="mjpeg",
        description="Formato de entrada V4L2 da webcam para o ffmpeg (ex: mjpeg, yuyv422).",
    )
    display: str = Field(
        default=":0.0",
        description="Display X11 para captura de tela",
    )
    screen_size: str = Field(
        default="1280x720",
        description="Resolução final do vídeo de tela; a captura usa a resolução real do display e faz downscale se necessário.",
    )
    delete_after_upload: bool = Field(
        default=True,
        description="Remover arquivo local após upload S3 bem-sucedido",
    )
    ffmpeg_threads: int = Field(
        default=1,
        description="Número de threads do ffmpeg/x264 por stream.",
    )
    preview_host: str = Field(
        default="127.0.0.1",
        description="Host local usado pelo relay de preview da webcam para o proctoring.",
    )
    preview_port: int = Field(
        default=18181,
        description="Porta UDP local usada pelo relay de preview da webcam.",
    )
    preview_width: int = Field(
        default=640,
        description="Largura do preview local consumido pelo proctoring.",
    )
    preview_height: int = Field(
        default=360,
        description="Altura do preview local consumido pelo proctoring.",
    )
    preview_fps: int = Field(
        default=10,
        description="FPS do preview local consumido pelo proctoring.",
    )
    ffmpeg_cpu_cores: str | None = Field(
        default=None,
        description="Afinidade de CPU para processos ffmpeg, ex: '3' ou '2-3'.",
    )
    proctor_cpu_cores: str | None = Field(
        default=None,
        description="Afinidade de CPU para o processo principal/proctoring, ex: '0-2'.",
    )

    model_config = SettingsConfigDict(
        env_prefix="PROCTOR_REC_",
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )


class S3Config(BaseSettings):
    """Configuração do bucket S3.

    Dois prefixos dentro do mesmo bucket:
      - photos_prefix/  → fotos de cadastro (nome_do_aluno.png por turma)
      - recordings_prefix/ → gravações de prova (gerenciado pelo recorder)

    Estrutura das fotos:
      s3://{bucket}/{photos_prefix}/{turma_id}/{nome_do_aluno}.png
    """

    bucket: str = Field(
        default="proctor-station",
        description="Nome do bucket S3",
    )
    region: str = Field(default="us-east-1")
    photos_prefix: str = Field(
        default="fotos",
        description="Prefixo S3 onde ficam as fotos de cadastro dos alunos",
    )
    recordings_prefix: str = Field(
        default="gravacoes",
        description="Prefixo S3 onde ficam as gravações de prova",
    )
    segment_duration_sec: int = Field(
        default=300,
        description="Duração de cada segmento de gravação em segundos",
    )

    def photos_prefix_for_turma(self, turma_id: str) -> str:
        """Retorna o prefixo S3 das fotos de uma turma específica.

        Ex: 'fotos/ES2025-T1/'
        """
        return f"{self.photos_prefix}/{turma_id}/"

    model_config = SettingsConfigDict(
        env_prefix="PROCTOR_S3_",
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )


class DashboardConfig(BaseSettings):
    """Configuração da integração NUC → dashboard do professor."""

    enabled: bool = Field(
        default=False,
        description="Ativa heartbeat periódico e consumo de comandos do dashboard.",
    )
    base_url: str = Field(
        default="http://127.0.0.1:8010",
        description="URL base do dashboard central.",
    )
    heartbeat_interval_sec: float = Field(
        default=5.0,
        description="Intervalo entre heartbeats enviados ao dashboard.",
    )
    timeout_sec: float = Field(
        default=5.0,
        description="Timeout HTTP das chamadas ao dashboard.",
    )
    station_id: str = Field(
        default="nuc-local",
        description="Identificador estável da estação.",
    )
    station_name: str = Field(
        default="NUC Local",
        description="Nome amigável exibido no dashboard.",
    )

    model_config = SettingsConfigDict(
        env_prefix="PROCTOR_DASHBOARD_",
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )


class AppConfig(BaseSettings):
    """Configuração geral da aplicação."""

    face: FaceConfig = Field(default_factory=FaceConfig)
    proctor: ProctorConfig = Field(default_factory=ProctorConfig)
    recorder: RecorderConfig = Field(default_factory=RecorderConfig)
    s3: S3Config = Field(default_factory=S3Config)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)

    api_port: int = 8000
    log_level: str = "INFO"
    data_dir: Path = Path("/opt/proctor/data")
    auto_start_poll_sec: float = 2.0
    auto_start_enabled: bool = True

    model_config = SettingsConfigDict(
        env_prefix="PROCTOR_APP_",
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )


# Singleton
config = AppConfig()
