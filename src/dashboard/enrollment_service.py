"""Enrollment facial de turmas via S3 para uso pelo dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.core.config import AppConfig
from src.core.s3_client import S3Client, get_s3_client
from src.face.recognizer import FaceRecognizer


@dataclass(frozen=True)
class S3EnrollmentStudent:
    student_id: str
    student_name: str
    s3_key: str
    success: bool
    samples_captured: int
    message: str


@dataclass(frozen=True)
class S3EnrollmentSummary:
    turma: str
    total: int
    ok: int
    failed: int
    pkl_path: Path | None
    students: list[S3EnrollmentStudent]


class S3EnrollmentError(RuntimeError):
    """Falha operacional no enrollment via S3."""


class S3EnrollmentService:
    def __init__(
        self,
        app_config: AppConfig,
        *,
        s3_factory: Callable[[], S3Client] | None = None,
        recognizer_factory: Callable[[], FaceRecognizer] | None = None,
    ):
        self._app_config = app_config
        self._s3_factory = s3_factory or (lambda: get_s3_client(config=app_config.s3))
        self._recognizer_factory = recognizer_factory or (
            lambda: FaceRecognizer(app_config.face.model_copy(update={"num_jitters": 3}))
        )

    def list_turmas(self) -> list[str]:
        return self._s3_factory().list_photo_turmas()

    def enroll_turma(self, turma: str, *, force: bool = False) -> S3EnrollmentSummary:
        turma = turma.strip()
        if not turma:
            raise S3EnrollmentError("Turma é obrigatória.")

        recognizer = self._recognizer_factory()
        pkl_path = Path(recognizer.config.encodings_dir) / f"{turma}.pkl"
        if pkl_path.exists() and not force:
            raise S3EnrollmentError(
                f"Já existe enrollment local para '{turma}'. Use reprocessar para sobrescrever."
            )

        s3 = self._s3_factory()
        try:
            photos = s3.download_all_photos(turma)
        except ValueError as exc:
            raise S3EnrollmentError(str(exc)) from exc

        recognizer.create_turma(turma)
        students: list[S3EnrollmentStudent] = []
        ok = 0

        for photo in photos:
            frames = [photo.image] * max(3, int(recognizer.config.num_jitters))
            result = recognizer.enroll_from_frames(
                student_id=photo.student_name,
                student_name=photo.student_name,
                frames=frames,
            )
            if result.success:
                ok += 1
            students.append(
                S3EnrollmentStudent(
                    student_id=result.student_id,
                    student_name=result.student_name,
                    s3_key=photo.s3_key,
                    success=result.success,
                    samples_captured=result.samples_captured,
                    message=result.message,
                )
            )

        saved_path = recognizer.save_turma() if ok else None
        if saved_path is None:
            raise S3EnrollmentError(
                f"Nenhum aluno da turma '{turma}' gerou samples válidos; .pkl não foi criado."
            )

        return S3EnrollmentSummary(
            turma=turma,
            total=len(photos),
            ok=ok,
            failed=len(photos) - ok,
            pkl_path=saved_path,
            students=students,
        )
