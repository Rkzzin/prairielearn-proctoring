"""Módulo de reconhecimento facial — enrollment e identificação.

Usa dlib diretamente para detecção (HOG/CNN), landmark prediction,
e encoding (ResNet 128-d). Sem dependência de face_recognition.

Uso típico:
    recognizer = FaceRecognizer(config)
    recognizer.load_turma("ES2025-T1")
    result = recognizer.identify(frame)
    if result.is_match:
        print(f"Aluno identificado: {result.student_id}")
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import cv2
import dlib
import numpy as np

from src.core.config import FaceConfig
from src.core.models import (
    EnrollmentResult,
    IdentifyResult,
    IdentifyStatus,
    StudentEncoding,
    TurmaEncodings,
)

logger = logging.getLogger(__name__)


def _dlib_rect_to_tlbr(rect: dlib.rectangle) -> tuple[int, int, int, int]:
    """Converte dlib.rectangle para (top, right, bottom, left)."""
    return (rect.top(), rect.right(), rect.bottom(), rect.left())


def _face_distance(known_encodings: list[np.ndarray], face_encoding: np.ndarray) -> np.ndarray:
    """Calcula distância euclidiana entre um encoding e uma lista de conhecidos."""
    if not known_encodings:
        return np.array([])
    known = np.array(known_encodings)
    return np.linalg.norm(known - face_encoding, axis=1)


class FaceRecognizer:
    """Gerencia enrollment e identificação facial de alunos."""

    def __init__(self, config: FaceConfig | None = None):
        self.config = config or FaceConfig()
        self.turma: TurmaEncodings | None = None
        self._encodings_dir = Path(self.config.encodings_dir)
        self._encodings_dir.mkdir(parents=True, exist_ok=True)

        # Carregar modelos dlib
        missing = self.config.validate_models()
        if missing:
            raise FileNotFoundError(
                f"Modelos dlib não encontrados: {missing}. "
                f"Rode: ./scripts/download_models.sh {self.config.models_dir}"
            )

        self._detector = dlib.get_frontal_face_detector()
        self._shape_predictor = dlib.shape_predictor(
            str(self.config.shape_predictor_path)
        )
        self._face_encoder = dlib.face_recognition_model_v1(
            str(self.config.recognition_model_path)
        )

        if self.config.use_cnn_detector:
            self._cnn_detector = dlib.cnn_face_detection_model_v1(
                str(self.config.cnn_detector_path)
            )

    # ──────────────────────────────────────────────
    #  Detecção interna
    # ──────────────────────────────────────────────

    def _detect_faces(self, rgb: np.ndarray) -> list[dlib.rectangle]:
        """Detecta rostos e retorna lista de dlib.rectangle."""
        if self.config.use_cnn_detector:
            detections = self._cnn_detector(rgb, 1)
            return [d.rect for d in detections]
        return self._detector(rgb, 1)

    def _compute_encoding(
        self, rgb: np.ndarray, face_rect: dlib.rectangle
    ) -> np.ndarray:
        """Computa o encoding 128-d de um rosto detectado."""
        shape = self._shape_predictor(rgb, face_rect)
        encoding = self._face_encoder.compute_face_descriptor(
            rgb, shape, self.config.num_jitters
        )
        return np.array(encoding)

    # ──────────────────────────────────────────────
    #  Enrollment
    # ──────────────────────────────────────────────

    def enroll_from_frames(
        self,
        student_id: str,
        student_name: str,
        frames: list[np.ndarray],
    ) -> EnrollmentResult:
        """Cadastra um aluno a partir de uma lista de frames BGR (OpenCV).

        Cada frame deve conter exatamente 1 rosto. Frames com 0 ou >1
        rostos são descartados silenciosamente.
        """
        student = StudentEncoding(student_id=student_id, student_name=student_name)
        discarded = 0

        for i, frame in enumerate(frames):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = self._detect_faces(rgb)

            if len(faces) != 1:
                discarded += 1
                logger.debug(
                    "Frame %d descartado: %d rostos detectados", i, len(faces)
                )
                continue

            encoding = self._compute_encoding(rgb, faces[0])
            student.add_encoding(encoding)

        if len(student.encodings) < 2:
            return EnrollmentResult(
                student_id=student_id,
                student_name=student_name,
                success=False,
                samples_captured=len(student.encodings),
                message=f"Apenas {len(student.encodings)} samples válidos (mínimo 2). "
                f"{discarded} frames descartados.",
            )

        if self.turma is not None:
            self.turma.add_student(student)

        logger.info(
            "Enrollment OK: %s (%s) — %d samples, %d descartados",
            student_name,
            student_id,
            len(student.encodings),
            discarded,
        )

        return EnrollmentResult(
            student_id=student_id,
            student_name=student_name,
            success=True,
            samples_captured=len(student.encodings),
            message=f"{len(student.encodings)} samples capturados com sucesso.",
        )

    def enroll_from_camera(
        self,
        student_id: str,
        student_name: str,
        camera_index: int | None = None,
        num_samples: int | None = None,
        delay_between_samples_ms: int = 500,
    ) -> EnrollmentResult:
        """Captura frames da webcam e faz enrollment."""
        cam_idx = camera_index if camera_index is not None else self.config.camera_index
        n_samples = num_samples or self.config.samples_per_student

        cap = cv2.VideoCapture(cam_idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)

        if not cap.isOpened():
            return EnrollmentResult(
                student_id=student_id,
                student_name=student_name,
                success=False,
                samples_captured=0,
                message=f"Não foi possível abrir a câmera {cam_idx}.",
            )

        frames = []
        try:
            for i in range(n_samples):
                for _ in range(5):
                    cap.grab()

                ret, frame = cap.read()
                if not ret:
                    logger.warning("Falha ao capturar frame %d", i)
                    continue

                frames.append(frame)
                logger.info(
                    "Captura %d/%d — mova levemente a cabeça", i + 1, n_samples
                )

                if i < n_samples - 1:
                    cv2.waitKey(delay_between_samples_ms)
        finally:
            cap.release()

        return self.enroll_from_frames(student_id, student_name, frames)

    # ──────────────────────────────────────────────
    #  Identificação
    # ──────────────────────────────────────────────

    def identify(self, frame: np.ndarray) -> IdentifyResult:
        """Identifica o aluno presente no frame.

        Args:
            frame: Imagem BGR (OpenCV) capturada da webcam.

        Returns:
            IdentifyResult com status, ID do aluno e confiança.
        """
        if self.turma is None or self.turma.student_count == 0:
            raise RuntimeError("Nenhuma turma carregada. Use load_turma() primeiro.")

        # Redimensionar para performance
        scale = self.config.detection_scale
        small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        # Detectar rostos
        faces = self._detect_faces(rgb_small)

        if len(faces) == 0:
            return IdentifyResult(status=IdentifyStatus.NO_FACE, face_count=0)

        if len(faces) > 1:
            return IdentifyResult(
                status=IdentifyStatus.MULTIPLE_FACES,
                face_count=len(faces),
            )

        # Encoding do rosto detectado
        encoding = self._compute_encoding(rgb_small, faces[0])

        # Comparar com todos os alunos da turma
        student_ids, known_encodings = self.turma.get_all_mean_encodings()

        if not known_encodings:
            return IdentifyResult(status=IdentifyStatus.NO_MATCH, face_count=1)

        distances = _face_distance(known_encodings, encoding)
        best_idx = int(np.argmin(distances))
        best_distance = float(distances[best_idx])

        # Escalar location de volta para resolução original
        rect = faces[0]
        inv_scale = 1.0 / scale
        face_loc = (
            int(rect.top() * inv_scale),
            int(rect.right() * inv_scale),
            int(rect.bottom() * inv_scale),
            int(rect.left() * inv_scale),
        )

        if best_distance < self.config.match_threshold:
            matched_id = student_ids[best_idx]
            student = self.turma.students[matched_id]
            return IdentifyResult(
                status=IdentifyStatus.MATCH,
                student_id=matched_id,
                student_name=student.student_name,
                confidence=round(1.0 - best_distance, 4),
                face_location=face_loc,
                face_count=1,
            )

        return IdentifyResult(
            status=IdentifyStatus.NO_MATCH,
            confidence=round(1.0 - best_distance, 4),
            face_location=face_loc,
            face_count=1,
        )

    def identify_best_of_n(
        self, frames: list[np.ndarray], n: int = 3
    ) -> IdentifyResult:
        """Roda identificação em N frames e retorna o melhor match."""
        best: IdentifyResult | None = None

        for frame in frames[:n]:
            result = self.identify(frame)
            if result.is_match:
                if best is None or (result.confidence or 0) > (best.confidence or 0):
                    best = result

        if best is not None:
            return best

        return result  # type: ignore[return-value]

    # ──────────────────────────────────────────────
    #  Persistência
    # ──────────────────────────────────────────────

    def save_turma(self, turma: TurmaEncodings | None = None) -> Path:
        """Salva os encodings da turma em arquivo .pkl."""
        turma = turma or self.turma
        if turma is None:
            raise RuntimeError("Nenhuma turma para salvar.")

        filepath = self._encodings_dir / f"{turma.turma_id}.pkl"
        with open(filepath, "wb") as f:
            pickle.dump(turma, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(
            "Turma '%s' salva em %s (%d alunos)",
            turma.turma_id,
            filepath,
            turma.student_count,
        )
        return filepath

    def load_turma(self, turma_id: str) -> TurmaEncodings:
        """Carrega encodings de uma turma a partir do arquivo .pkl."""
        filepath = self._encodings_dir / f"{turma_id}.pkl"
        if not filepath.exists():
            raise FileNotFoundError(
                f"Encodings não encontrados para turma '{turma_id}' em {filepath}"
            )

        with open(filepath, "rb") as f:
            turma = pickle.load(f)  # noqa: S301 — conteúdo confiável

        if not isinstance(turma, TurmaEncodings):
            raise TypeError(f"Arquivo {filepath} não contém TurmaEncodings válido.")

        self.turma = turma
        logger.info(
            "Turma '%s' carregada: %d alunos", turma.turma_id, turma.student_count
        )
        return turma

    def create_turma(self, turma_id: str) -> TurmaEncodings:
        """Cria uma nova turma vazia e a define como ativa."""
        self.turma = TurmaEncodings(turma_id=turma_id)
        logger.info("Turma '%s' criada (vazia)", turma_id)
        return self.turma

    def list_turmas(self) -> list[str]:
        """Lista todas as turmas com encodings salvos."""
        return [p.stem for p in self._encodings_dir.glob("*.pkl")]