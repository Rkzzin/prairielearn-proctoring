"""Modelos de dados do sistema."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class IdentifyStatus(str, Enum):
    """Resultado de uma tentativa de identificação."""

    MATCH = "MATCH"
    NO_MATCH = "NO_MATCH"
    NO_FACE = "NO_FACE"
    MULTIPLE_FACES = "MULTIPLE_FACES"


@dataclass
class IdentifyResult:
    """Resultado completo de uma identificação facial."""

    status: IdentifyStatus
    student_id: str | None = None
    student_name: str | None = None
    confidence: float | None = None  # 1 - distance (maior = melhor)
    face_location: tuple[int, int, int, int] | None = None  # top, right, bottom, left
    face_count: int = 0

    @property
    def is_match(self) -> bool:
        return self.status == IdentifyStatus.MATCH


@dataclass
class StudentEncoding:
    """Encoding facial de um aluno."""

    student_id: str  # RA
    student_name: str
    encodings: list[np.ndarray] = field(default_factory=list)  # múltiplos ângulos
    mean_encoding: np.ndarray | None = None  # média dos encodings

    def compute_mean(self) -> None:
        """Calcula o encoding médio a partir dos samples."""
        if self.encodings:
            self.mean_encoding = np.mean(self.encodings, axis=0)

    def add_encoding(self, encoding: np.ndarray) -> None:
        self.encodings.append(encoding)
        self.compute_mean()


@dataclass
class EnrollmentResult:
    """Resultado de um enrollment."""

    student_id: str
    student_name: str
    success: bool
    samples_captured: int
    message: str


@dataclass
class TurmaEncodings:
    """Encodings de toda uma turma."""

    turma_id: str
    students: dict[str, StudentEncoding] = field(default_factory=dict)

    def add_student(self, student: StudentEncoding) -> None:
        self.students[student.student_id] = student

    def remove_student(self, student_id: str) -> bool:
        if student_id in self.students:
            del self.students[student_id]
            return True
        return False

    @property
    def student_count(self) -> int:
        return len(self.students)

    def get_all_mean_encodings(self) -> tuple[list[str], list[np.ndarray]]:
        """Retorna IDs e mean_encodings alinhados para comparação vetorizada."""
        ids = []
        encs = []
        for sid, student in self.students.items():
            if student.mean_encoding is not None:
                ids.append(sid)
                encs.append(student.mean_encoding)
        return ids, encs
