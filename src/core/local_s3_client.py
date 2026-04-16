"""Mock local do S3 para desenvolvimento sem credenciais AWS.

Implementa exatamente a mesma interface pública de S3Client,
lendo de uma pasta local em vez do bucket real.

Estrutura de pastas esperada (espelha o layout do S3):
    mock_s3/
    └── fotos/
        └── ES2025-T1/
            ├── joao_silva.png
            └── maria_santos.jpeg

Ativação — adicione no .env:
    PROCTOR_S3_MOCK=true
    PROCTOR_S3_MOCK_DIR=mock_s3   # opcional, default: mock_s3

O código que usa S3Client não precisa mudar nada — basta importar
via get_s3_client() em vez de instanciar S3Client diretamente.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from src.core.config import S3Config
from src.core.s3_client import StudentPhoto

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


class LocalS3Client:
    """Substituto local do S3Client para desenvolvimento.

    Lê fotos de uma pasta local com a mesma estrutura de prefixos
    que o bucket S3 real usaria.

    Args:
        config: S3Config com photos_prefix etc. Usado para montar
                os caminhos da mesma forma que o cliente real.
        mock_dir: Raiz da pasta local que simula o bucket.
                  Default: mock_s3/ na raiz do projeto.
    """

    def __init__(
        self,
        config: S3Config | None = None,
        mock_dir: Path | str | None = None,
    ):
        self.config = config or S3Config()

        if mock_dir is None:
            # Raiz do projeto: src/core/local_s3_client.py → ../../..
            project_root = Path(__file__).resolve().parent.parent.parent
            mock_dir = project_root / "mock_s3"

        self._root = Path(mock_dir)

        if not self._root.exists():
            raise FileNotFoundError(
                f"Pasta mock S3 não encontrada: {self._root}\n"
                f"Crie a estrutura:\n"
                f"  {self._root}/fotos/<turma_id>/<nome_aluno>.png"
            )

        logger.info("LocalS3Client iniciado — lendo de: %s", self._root)

    # ──────────────────────────────────────────────
    #  Interface pública (idêntica ao S3Client real)
    # ──────────────────────────────────────────────

    def list_student_photos(self, turma_id: str) -> list[str]:
        """Lista 'chaves' locais de todas as fotos de uma turma.

        As chaves retornadas seguem o mesmo formato do S3:
            fotos/ES2025-T1/joao_silva.png
        """
        prefix = self.config.photos_prefix_for_turma(turma_id)
        turma_dir = self._root / prefix

        if not turma_dir.exists():
            logger.warning(
                "Pasta da turma não encontrada: %s", turma_dir
            )
            return []

        keys = [
            f"{prefix}{f.name}"
            for f in sorted(turma_dir.iterdir())
            if f.is_file() and f.suffix.lower() in _IMAGE_EXTENSIONS
        ]

        logger.info(
            "Turma '%s': %d fotos encontradas em %s",
            turma_id, len(keys), turma_dir,
        )
        return keys

    def download_student_photo(self, s3_key: str) -> StudentPhoto:
        """Lê uma foto local e retorna como array BGR.

        Aceita a mesma chave que o S3Client real retornaria:
            fotos/ES2025-T1/joao_silva.png
        """
        stem = s3_key.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        local_path = self._root / s3_key

        if not local_path.exists():
            raise FileNotFoundError(
                f"Foto não encontrada localmente: {local_path}"
            )

        bgr = cv2.imread(str(local_path))
        if bgr is None:
            raise ValueError(
                f"Não foi possível ler a imagem: {local_path}"
            )

        logger.debug(
            "Lido local: %s (%dx%d)", s3_key, bgr.shape[1], bgr.shape[0]
        )
        return StudentPhoto(student_name=stem, s3_key=s3_key, image=bgr)

    def download_all_photos(
        self,
        turma_id: str,
        *,
        on_progress: callable | None = None,
    ) -> list[StudentPhoto]:
        """Lê todas as fotos de uma turma da pasta local."""
        keys = self.list_student_photos(turma_id)
        if not keys:
            raise ValueError(
                f"Nenhuma foto encontrada para a turma '{turma_id}' "
                f"em {self._root / self.config.photos_prefix_for_turma(turma_id)}"
            )

        photos: list[StudentPhoto] = []
        for i, key in enumerate(keys, start=1):
            try:
                photo = self.download_student_photo(key)
                photos.append(photo)
                if on_progress:
                    on_progress(i, len(keys), photo.student_name)
            except (FileNotFoundError, ValueError) as e:
                logger.warning("Falha ao ler '%s': %s", key, e)

        return photos

    def photo_exists(self, turma_id: str, student_name: str) -> bool:
        """Verifica se a foto de um aluno existe na pasta local."""
        prefix = self.config.photos_prefix_for_turma(turma_id)
        for ext in _IMAGE_EXTENSIONS:
            if (self._root / f"{prefix}{student_name}{ext}").exists():
                return True
        return False