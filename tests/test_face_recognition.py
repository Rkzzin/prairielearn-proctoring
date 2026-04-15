"""Testes do módulo de reconhecimento facial.

Roda sem câmera — usa imagens sintéticas para testar a lógica.
Para testes com câmera real, use scripts/test_camera.py.

Estratégia de mock: patchar dlib no nível do módulo recognizer
ANTES de instanciar FaceRecognizer, para que o __init__ não
tente carregar arquivos .dat reais.
"""

from __future__ import annotations

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


# ── Helpers ──


def _make_dlib_rect(top: int, right: int, bottom: int, left: int) -> MagicMock:
    """Cria um mock de dlib.rectangle."""
    rect = MagicMock()
    rect.top.return_value = top
    rect.right.return_value = right
    rect.bottom.return_value = bottom
    rect.left.return_value = left
    return rect


def _make_config(tmp_encodings_dir: Path) -> FaceConfig:
    """Cria config apontando para diretório temporário com modelos 'fake'."""
    # Criar arquivos .dat fake para que validate_models() passe
    models_dir = tmp_encodings_dir.parent / "models"
    models_dir.mkdir(exist_ok=True)
    for name in [
        "shape_predictor_68_face_landmarks.dat",
        "dlib_face_recognition_resnet_model_v1.dat",
    ]:
        (models_dir / name).touch()

    return FaceConfig(
        encodings_dir=tmp_encodings_dir,
        models_dir=models_dir,
        match_threshold=0.45,
        use_cnn_detector=False,
        detection_scale=1.0,
    )


def _make_recognizer(config: FaceConfig):
    """Cria FaceRecognizer com dlib mockado no nível do módulo."""
    with patch("src.face.recognizer.dlib") as mock_dlib:
        mock_dlib.get_frontal_face_detector.return_value = MagicMock(return_value=[])
        mock_dlib.shape_predictor.return_value = MagicMock()
        mock_dlib.face_recognition_model_v1.return_value = MagicMock()

        from src.face.recognizer import FaceRecognizer

        rec = FaceRecognizer(config)
        rec._mock_dlib = mock_dlib
    return rec


# ── Fixtures ──


@pytest.fixture
def tmp_encodings_dir(tmp_path: Path) -> Path:
    d = tmp_path / "encodings"
    d.mkdir()
    return d


@pytest.fixture
def config(tmp_encodings_dir: Path) -> FaceConfig:
    return _make_config(tmp_encodings_dir)


@pytest.fixture
def sample_turma() -> TurmaEncodings:
    """Turma com 3 alunos e encodings sintéticos."""
    turma = TurmaEncodings(turma_id="TEST-001")

    rng = np.random.default_rng(42)
    for ra, nome in [
        ("11111", "Alice Silva"),
        ("22222", "Bob Santos"),
        ("33333", "Carol Oliveira"),
    ]:
        base = rng.standard_normal(128).astype(np.float64)
        base = base / np.linalg.norm(base)

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
        self, config: FaceConfig, sample_turma: TurmaEncodings
    ):
        rec = _make_recognizer(config)
        rec.turma = sample_turma
        path = rec.save_turma()
        assert path.exists()
        assert path.suffix == ".pkl"

        rec2 = _make_recognizer(config)
        loaded = rec2.load_turma("TEST-001")
        assert loaded.student_count == 3
        assert "11111" in loaded.students

    def test_load_nonexistent_turma(self, config: FaceConfig):
        rec = _make_recognizer(config)
        with pytest.raises(FileNotFoundError):
            rec.load_turma("NONEXISTENT")

    def test_list_turmas(
        self, config: FaceConfig, sample_turma: TurmaEncodings
    ):
        rec = _make_recognizer(config)
        rec.turma = sample_turma
        rec.save_turma()

        turmas = rec.list_turmas()
        assert "TEST-001" in turmas

    def test_create_turma(self, config: FaceConfig):
        rec = _make_recognizer(config)
        turma = rec.create_turma("NEW-001")
        assert turma.turma_id == "NEW-001"
        assert turma.student_count == 0
        assert rec.turma is turma


# ── Testes de identificação ──


class TestIdentify:
    def test_identify_match(
        self, config: FaceConfig, sample_turma: TurmaEncodings
    ):
        alice_enc = sample_turma.students["11111"].mean_encoding
        probe = alice_enc + np.random.default_rng(7).normal(0, 0.01, 128)

        face_rect = _make_dlib_rect(50, 200, 200, 50)

        rec = _make_recognizer(config)
        rec._detector = MagicMock(return_value=[face_rect])
        rec._face_encoder.compute_face_descriptor.return_value = probe
        rec.turma = sample_turma

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = rec.identify(frame)

        assert result.is_match
        assert result.student_id == "11111"
        assert result.student_name == "Alice Silva"
        assert result.confidence is not None and result.confidence > 0.5

    def test_identify_no_face(
        self, config: FaceConfig, sample_turma: TurmaEncodings
    ):
        rec = _make_recognizer(config)
        rec._detector = MagicMock(return_value=[])
        rec.turma = sample_turma

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = rec.identify(frame)

        assert result.status == IdentifyStatus.NO_FACE
        assert result.face_count == 0

    def test_identify_multiple_faces(
        self, config: FaceConfig, sample_turma: TurmaEncodings
    ):
        face1 = _make_dlib_rect(50, 200, 200, 50)
        face2 = _make_dlib_rect(50, 500, 200, 350)

        rec = _make_recognizer(config)
        rec._detector = MagicMock(return_value=[face1, face2])
        rec.turma = sample_turma

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = rec.identify(frame)

        assert result.status == IdentifyStatus.MULTIPLE_FACES
        assert result.face_count == 2

    def test_identify_no_match(
        self, config: FaceConfig, sample_turma: TurmaEncodings
    ):
        random_enc = np.random.default_rng(99).standard_normal(128)
        face_rect = _make_dlib_rect(50, 200, 200, 50)

        rec = _make_recognizer(config)
        rec._detector = MagicMock(return_value=[face_rect])
        rec._face_encoder.compute_face_descriptor.return_value = random_enc
        rec.turma = sample_turma

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = rec.identify(frame)

        assert result.status == IdentifyStatus.NO_MATCH

    def test_identify_without_turma(self, config: FaceConfig):
        rec = _make_recognizer(config)

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="Nenhuma turma carregada"):
            rec.identify(frame)


# ── Testes de enrollment ──


class TestEnrollment:
    def test_enroll_from_frames_success(self, config: FaceConfig):
        face_rect = _make_dlib_rect(50, 200, 200, 50)

        rec = _make_recognizer(config)
        rec.create_turma("TEST")
        rec._detector = MagicMock(return_value=[face_rect])
        rec._face_encoder.compute_face_descriptor.return_value = (
            np.random.standard_normal(128)
        )

        frames = [np.zeros((720, 1280, 3), dtype=np.uint8)] * 5
        result = rec.enroll_from_frames("12345", "Test Student", frames)

        assert result.success is True
        assert result.samples_captured == 5
        assert "12345" in rec.turma.students

    def test_enroll_from_frames_not_enough_faces(self, config: FaceConfig):
        rec = _make_recognizer(config)
        rec.create_turma("TEST")
        rec._detector = MagicMock(return_value=[])

        frames = [np.zeros((720, 1280, 3), dtype=np.uint8)] * 5
        result = rec.enroll_from_frames("12345", "Test Student", frames)

        assert result.success is False
        assert result.samples_captured == 0

    def test_enroll_discards_multi_face_frames(self, config: FaceConfig):
        face1 = _make_dlib_rect(50, 200, 200, 50)
        face2 = _make_dlib_rect(50, 500, 200, 350)

        rec = _make_recognizer(config)
        rec.create_turma("TEST")

        call_count = 0

        def mock_detect(rgb, upsample):
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                return [face1]
            return [face1, face2]

        rec._detector = MagicMock(side_effect=mock_detect)
        rec._face_encoder.compute_face_descriptor.return_value = (
            np.random.standard_normal(128)
        )

        frames = [np.zeros((720, 1280, 3), dtype=np.uint8)] * 5
        result = rec.enroll_from_frames("12345", "Test Student", frames)

        assert result.success is True
        assert result.samples_captured == 3