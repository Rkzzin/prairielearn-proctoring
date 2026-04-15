"""Configuração central da proctoring station."""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class FaceConfig(BaseSettings):
    """Parâmetros do módulo de reconhecimento facial."""

    # Enrollment
    encodings_dir: Path = Field(
        default=Path("src/face/encodings"),
        description="Diretório com arquivos .pkl de encodings por turma",
    )
    samples_per_student: int = Field(
        default=5,
        description="Quantidade de frames capturados no enrollment",
    )

    # Identificação
    match_threshold: float = Field(
        default=0.45,
        description="Distância máxima para considerar match (menor = mais restritivo)",
    )
    detection_model: str = Field(
        default="hog",
        description="Modelo de detecção: 'hog' (CPU) ou 'cnn' (GPU)",
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

    model_config = {"env_prefix": "PROCTOR_FACE_"}


class ProctorConfig(BaseSettings):
    """Parâmetros do proctoring engine."""

    gaze_h_threshold: float = 0.35
    gaze_v_threshold: float = 0.30
    gaze_duration_sec: float = 3.0
    absence_timeout_sec: float = 5.0
    multi_face_block: bool = True

    model_config = {"env_prefix": "PROCTOR_"}


class S3Config(BaseSettings):
    """Configuração de upload S3."""

    bucket: str = "proctor-recordings"
    region: str = "us-east-1"
    prefix: str = ""
    segment_duration_sec: int = 300

    model_config = {"env_prefix": "PROCTOR_S3_"}


class AppConfig(BaseSettings):
    """Configuração geral da aplicação."""

    face: FaceConfig = Field(default_factory=FaceConfig)
    proctor: ProctorConfig = Field(default_factory=ProctorConfig)
    s3: S3Config = Field(default_factory=S3Config)

    api_port: int = 8000
    log_level: str = "INFO"
    data_dir: Path = Path("/opt/proctor/data")

    model_config = {"env_prefix": "PROCTOR_APP_"}


# Singleton
config = AppConfig()
