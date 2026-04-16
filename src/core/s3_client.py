"""Cliente S3 para a proctoring station.

Responsabilidades:
  - Listar e baixar fotos de cadastro de alunos por turma
  - Upload de gravações de prova (usado pelo recorder, futuro)

Layout esperado no bucket:
  s3://{bucket}/fotos/{turma_id}/{nome_do_aluno}.png
  s3://{bucket}/gravacoes/{sessao_id}/...
"""

from __future__ import annotations

import io
from pathlib import Path
import logging
import os
from dataclasses import dataclass

import boto3
import numpy as np
from botocore.exceptions import ClientError
from PIL import Image

from src.core.config import S3Config

logger = logging.getLogger(__name__)


@dataclass
class StudentPhoto:
    """Foto de um aluno baixada do S3."""

    student_name: str   # stem do arquivo, ex: "joao_silva"
    s3_key: str         # chave completa no bucket
    image: np.ndarray   # BGR (OpenCV-compatible)


class S3Client:
    """Wrapper boto3 para operações da proctoring station."""

    def __init__(self, config: S3Config | None = None):
        self.config = config or S3Config()
        self._s3 = boto3.client("s3", region_name=self.config.region)

    # ──────────────────────────────────────────────
    #  Fotos de cadastro
    # ──────────────────────────────────────────────

    def list_student_photos(self, turma_id: str) -> list[str]:
        """Lista as chaves S3 de todas as fotos de uma turma.

        Retorna somente arquivos .png/.jpg/.jpeg.
        """
        prefix = self.config.photos_prefix_for_turma(turma_id)
        keys: list[str] = []

        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.config.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key: str = obj["Key"]
                if key.lower().endswith((".png", ".jpg", ".jpeg")):
                    keys.append(key)

        logger.info("Turma '%s': %d fotos encontradas no S3", turma_id, len(keys))
        return keys

    def download_student_photo(self, s3_key: str) -> StudentPhoto:
        """Baixa uma foto do S3 e retorna como array BGR (OpenCV).

        O nome do aluno é extraído do stem do arquivo:
          fotos/ES2025-T1/joao_silva.png  →  student_name = "joao_silva"
        """
        stem = s3_key.rsplit("/", 1)[-1].rsplit(".", 1)[0]

        response = self._s3.get_object(Bucket=self.config.bucket, Key=s3_key)
        data = response["Body"].read()

        pil_img = Image.open(io.BytesIO(data)).convert("RGB")
        rgb = np.array(pil_img, dtype=np.uint8)
        bgr = rgb[:, :, ::-1].copy()  # PIL é RGB, OpenCV é BGR

        logger.debug("Baixado: %s (%dx%d)", s3_key, bgr.shape[1], bgr.shape[0])
        return StudentPhoto(student_name=stem, s3_key=s3_key, image=bgr)

    def download_all_photos(
        self,
        turma_id: str,
        *,
        on_progress: callable | None = None,
    ) -> list[StudentPhoto]:
        """Baixa todas as fotos de uma turma do S3.

        Args:
            turma_id: ID da turma (ex: "ES2025-T1").
            on_progress: Callback chamado a cada foto baixada com
                         (current: int, total: int, student_name: str).

        Returns:
            Lista de StudentPhoto com imagens em BGR.

        Raises:
            ValueError: Se nenhuma foto for encontrada para a turma.
        """
        keys = self.list_student_photos(turma_id)
        if not keys:
            raise ValueError(
                f"Nenhuma foto encontrada para a turma '{turma_id}' "
                f"no prefixo s3://{self.config.bucket}/"
                f"{self.config.photos_prefix_for_turma(turma_id)}"
            )

        photos: list[StudentPhoto] = []
        for i, key in enumerate(keys, start=1):
            try:
                photo = self.download_student_photo(key)
                photos.append(photo)
                if on_progress:
                    on_progress(i, len(keys), photo.student_name)
            except ClientError as e:
                logger.warning("Falha ao baixar '%s': %s", key, e)

        return photos

    def photo_exists(self, turma_id: str, student_name: str) -> bool:
        """Verifica se a foto de um aluno existe no S3 (qualquer extensão)."""
        prefix = self.config.photos_prefix_for_turma(turma_id)
        for ext in (".png", ".jpg", ".jpeg"):
            key = f"{prefix}{student_name}{ext}"
            try:
                self._s3.head_object(Bucket=self.config.bucket, Key=key)
                return True
            except ClientError:
                continue
        return False

# ──────────────────────────────────────────────
#  Factory — escolhe S3Client ou LocalS3Client
# ──────────────────────────────────────────────

def get_s3_client(config: S3Config | None = None) -> "S3Client | LocalS3Client":
    """Retorna o cliente S3 correto com base na variável PROCTOR_S3_MOCK.

    Se PROCTOR_S3_MOCK=true no .env ou no ambiente, retorna LocalS3Client
    (lê de mock_s3/ local). Caso contrário, retorna S3Client (AWS real).

    Uso:
        from src.core.s3_client import get_s3_client
        s3 = get_s3_client()
        photos = s3.download_all_photos("ES2025-T1")

    Args:
        config: S3Config opcional. Se None, usa defaults / .env.
    """
    # Carregar .env explicitamente — os.getenv() sozinho não lê o arquivo
    _env_file = Path(__file__).resolve().parent.parent.parent / ".env"
    if _env_file.exists():
        for line in _env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())

    use_mock = os.getenv("PROCTOR_S3_MOCK", "false").lower() in ("true", "1", "yes")

    if use_mock:
        from src.core.local_s3_client import LocalS3Client
        mock_dir = os.getenv("PROCTOR_S3_MOCK_DIR", "mock_s3")
        logger.info("Modo mock S3 ativo — usando pasta local: %s", mock_dir)
        return LocalS3Client(config=config, mock_dir=mock_dir)

    return S3Client(config=config)