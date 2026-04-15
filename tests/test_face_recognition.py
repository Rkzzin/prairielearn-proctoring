"""Testes do módulo de reconhecimento facial.

Roda sem câmera — usa imagens sintéticas para testar a lógica.
Para testes com câmera real, use scripts/test_camera.py.
"""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.core.config import FaceConfig
from src.core.models import (
    IdentifyStatus,
    StudentEncoding,
    TurmaEncodings,
)
from src.face.recognizer import FaceRecognizer


# ── Fixtures ──


@pytest.fixture
def tmp_encodings_dir(tmp_path: Path) -> Path:
    """Diretório temporário para encodings."""
    d = tmp_path / "encodings"
    d.mkdir()
    return d


@pytest.fixture
def config(tmp_encodings_dir: Path) -> FaceConfig:
    return FaceConfig(
        encodings_dir=tmp_encodings_dir,
        match_threshold=0.45,
        detection_model="hog",
        detection_scale=1.0,
    )


@pytest.fixture
def recognizer(config: FaceConfig) -> FaceRecognizer:
    return FaceRecognizer(config)


@pytest.fixture
def sample_turma() -> TurmaEncodings:
    """Turma com 3 alunos e encodings sintéticos."""
    turma = TurmaEncodings(turma_id="TEST-001")

    rng = np.random.default_rng(42)
    for i, (ra, nome) in enumerate([
        ("11111", "Alice Silva"),
        ("22222", "Bob Santos"),
        ("33333", "Carol Oliveira"),
    ]):
        # Gerar encoding "base" e adicionar ruído para simular variações
        base = rng.standard_normal(128).astype(np.float64)
        base = base / np.linalg.norm(base)  # normalizar

        student = StudentEncoding(student_id=ra, student_name=nome)
        for _ in range(3):
            noisy = base + rng.normal(0, 0.02, 128)
            student.add_encoding(noisy)

        turma.add_student(student)

    return turma


# ── Testes de TurmaEncodings ──


class TestTurmaEncodings:
    def test_add_and_count(self, sample_turma: TurmaEncodings):
        assert sample_turma.student_count == 3

    def test_remove_student(self, sample_turma: TurmaEncodings):
        assert sample_turma.remove_student("11111") is True
        assert sample_turma.student_count == 2
        assert sample_turma.remove_student("99999") is False

    def test_get_all_mean_encodings(self, sample_turma: TurmaEncodings):
        ids, encs = sample_turma.get_all_mean_encodings()
        assert len(ids) == 3
        assert len(encs) == 3
        assert all(e.shape == (128,) for e in encs)


class TestStudentEncoding:
    def test_compute_mean(self):
        student = StudentEncoding(student_id="123", student_name="Test")
        enc1 = np.ones(128)
        enc2 = np.ones(128) * 3
        student.add_encoding(enc1)
        student.add_encoding(enc2)
        assert student.mean_encoding is not None
        np.testing.assert_array_almost_equal(student.mean_encoding, np.ones(128) * 2)


# ── Testes de persistência ──


class TestPersistence:
    def test_save_and_load_turma(
        self, recognizer: FaceRecognizer, sample_turma: TurmaEncodings
    ):
        recognizer.turma = sample_turma
        path = recognizer.save_turma()
        assert path.exists()
        assert path.suffix == ".pkl"

        # Criar novo recognizer e carregar
        r2 = FaceRecognizer(recognizer.config)
        loaded = r2.load_turma("TEST-001")
        assert loaded.student_count == 3
        assert "11111" in loaded.students

    def test_load_nonexistent_turma(self, recognizer: FaceRecognizer):
        with pytest.raises(FileNotFoundError):
            recognizer.load_turma("NONEXISTENT")

    def test_list_turmas(
        self, recognizer: FaceRecognizer, sample_turma: TurmaEncodings
    ):
        recognizer.turma = sample_turma
        recognizer.save_turma()

        turmas = recognizer.list_turmas()
        assert "TEST-001" in turmas

    def test_create_turma(self, recognizer: FaceRecognizer):
        turma = recognizer.create_turma("NEW-001")
        assert turma.turma_id == "NEW-001"
        assert turma.student_count == 0
        assert recognizer.turma is turma


# ── Testes de identificação (com mocks) ──


class TestIdentify:
    @patch("src.face.recognizer.face_recognition")
    def test_identify_match(
        self, mock_fr: MagicMock, recognizer: FaceRecognizer, sample_turma: TurmaEncodings
    ):
        recognizer.turma = sample_turma

        # Simular detecção de 1 rosto
        mock_fr.face_locations.return_value = [(50, 200, 200, 50)]

        # Retornar encoding próximo ao do aluno "Alice"
        alice_enc = sample_turma.students["11111"].mean_encoding
        mock_fr.face_encodings.return_value = [alice_enc + np.random.normal(0, 0.01, 128)]

        # face_distance real (não mockar para testar a lógica)
        mock_fr.face_distance.side_effect = lambda known, unknown: np.array([
            np.linalg.norm(k - unknown) for k in known
        ])

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = recognizer.identify(frame)

        assert result.is_match
        assert result.student_id == "11111"
        assert result.student_name == "Alice Silva"
        assert result.confidence is not None and result.confidence > 0.5

    @patch("src.face.recognizer.face_recognition")
    def test_identify_no_face(
        self, mock_fr: MagicMock, recognizer: FaceRecognizer, sample_turma: TurmaEncodings
    ):
        recognizer.turma = sample_turma
        mock_fr.face_locations.return_value = []

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = recognizer.identify(frame)

        assert result.status == IdentifyStatus.NO_FACE
        assert result.face_count == 0

    @patch("src.face.recognizer.face_recognition")
    def test_identify_multiple_faces(
        self, mock_fr: MagicMock, recognizer: FaceRecognizer, sample_turma: TurmaEncodings
    ):
        recognizer.turma = sample_turma
        mock_fr.face_locations.return_value = [
            (50, 200, 200, 50),
            (50, 500, 200, 350),
        ]

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = recognizer.identify(frame)

        assert result.status == IdentifyStatus.MULTIPLE_FACES
        assert result.face_count == 2

    @patch("src.face.recognizer.face_recognition")
    def test_identify_no_match(
        self, mock_fr: MagicMock, recognizer: FaceRecognizer, sample_turma: TurmaEncodings
    ):
        recognizer.turma = sample_turma
        mock_fr.face_locations.return_value = [(50, 200, 200, 50)]

        # Encoding aleatório, distante de todos os alunos
        random_enc = np.random.default_rng(99).standard_normal(128)
        mock_fr.face_encodings.return_value = [random_enc]
        mock_fr.face_distance.side_effect = lambda known, unknown: np.array([
            np.linalg.norm(k - unknown) for k in known
        ])

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = recognizer.identify(frame)

        assert result.status == IdentifyStatus.NO_MATCH

    def test_identify_without_turma(self, recognizer: FaceRecognizer):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="Nenhuma turma carregada"):
            recognizer.identify(frame)


# ── Testes de enrollment (com mocks) ──


class TestEnrollment:
    @patch("src.face.recognizer.face_recognition")
    def test_enroll_from_frames_success(
        self, mock_fr: MagicMock, recognizer: FaceRecognizer
    ):
        recognizer.create_turma("TEST")

        # 5 frames, cada um com 1 rosto
        mock_fr.face_locations.return_value = [(50, 200, 200, 50)]
        mock_fr.face_encodings.return_value = [np.random.standard_normal(128)]

        frames = [np.zeros((720, 1280, 3), dtype=np.uint8)] * 5
        result = recognizer.enroll_from_frames("12345", "Test Student", frames)

        assert result.success is True
        assert result.samples_captured == 5
        assert "12345" in recognizer.turma.students

    @patch("src.face.recognizer.face_recognition")
    def test_enroll_from_frames_not_enough_faces(
        self, mock_fr: MagicMock, recognizer: FaceRecognizer
    ):
        recognizer.create_turma("TEST")

        # Nenhum rosto detectado
        mock_fr.face_locations.return_value = []

        frames = [np.zeros((720, 1280, 3), dtype=np.uint8)] * 5
        result = recognizer.enroll_from_frames("12345", "Test Student", frames)

        assert result.success is False
        assert result.samples_captured == 0

    @patch("src.face.recognizer.face_recognition")
    def test_enroll_discards_multi_face_frames(
        self, mock_fr: MagicMock, recognizer: FaceRecognizer
    ):
        recognizer.create_turma("TEST")

        # 3 frames com 1 rosto, 2 frames com 2 rostos
        call_count = 0

        def mock_locations(img, model="hog"):
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                return [(50, 200, 200, 50)]
            return [(50, 200, 200, 50), (50, 500, 200, 350)]

        mock_fr.face_locations.side_effect = mock_locations
        mock_fr.face_encodings.return_value = [np.random.standard_normal(128)]

        frames = [np.zeros((720, 1280, 3), dtype=np.uint8)] * 5
        result = recognizer.enroll_from_frames("12345", "Test Student", frames)

        assert result.success is True
        assert result.samples_captured == 3
