"""Módulo de reconhecimento facial — enrollment e identificação.

Usa dlib diretamente para detecção (HOG/CNN), landmark prediction,
e encoding (ResNet 128-d). Sem dependência de face_recognition.

Fluxo de enrollment:
    1. Operador roda: python scripts/enroll.py --turma ES2025-T1
    2. enroll.py baixa fotos do S3  (s3://bucket/fotos/ES2025-T1/*.png)
    3. enroll.py chama recognizer.enroll_from_frames() para cada aluno
    4. recognizer.save_turma() persiste o .pkl local na NUC

Fluxo de identificação:
    recognizer = FaceRecognizer(config)
    recognizer.load_turma("ES2025-T1")
    result = recognizer.identify(frame)
    if result.is_match:
        print(f"Aluno: {result.student_id} — {result.student_name}")
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


def _face_distance(known_encodings: list[np.ndarray], face_encoding: np.ndarray) -> np.ndarray:
    """Distância euclidiana entre um encoding e uma lista de conhecidos."""
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

    def _detect_faces(self, rgb: np.ndarray) -> list:
        """Detecta rostos e retorna lista de dlib.rectangle."""
        if self.config.use_cnn_detector:
            detections = self._cnn_detector(rgb, 1)
            return [d.rect for d in detections]
        return self._detector(rgb, 1)

    def _compute_encoding(self, rgb: np.ndarray, face_rect) -> np.ndarray:
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
        """Cadastra um aluno a partir de uma lista de frames BGR.

        Frames com 0 ou >1 rostos são descartados silenciosamente.
        Requer mínimo de 2 frames válidos para aceitar o enrollment.

        No fluxo S3, cada aluno tipicamente tem 1 foto, então enroll.py
        passa [frame] * num_jitters para gerar variações via dlib jitter.
        """
        student = StudentEncoding(student_id=student_id, student_name=student_name)
        discarded = 0

        for i, frame in enumerate(frames):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = self._detect_faces(rgb)

            if len(faces) != 1:
                discarded += 1
                logger.debug("Frame %d descartado: %d rostos detectados", i, len(faces))
                continue

            encoding = self._compute_encoding(rgb, faces[0])
            student.add_encoding(encoding)

        if len(student.encodings) < 2:
            return EnrollmentResult(
                student_id=student_id,
                student_name=student_name,
                success=False,
                samples_captured=len(student.encodings),
                message=(
                    f"Apenas {len(student.encodings)} sample(s) válido(s) "
                    f"(mínimo 2). {discarded} frame(s) descartado(s) — "
                    f"verifique se a foto contém exatamente 1 rosto."
                ),
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

    # ──────────────────────────────────────────────
    #  Identificação
    # ──────────────────────────────────────────────

    def identify(self, frame: np.ndarray) -> IdentifyResult:
        """Identifica o aluno presente no frame.

        Args:
            frame: Imagem BGR (OpenCV) capturada da webcam.

        Returns:
            IdentifyResult com status, ID do aluno e confiança.

        Raises:
            RuntimeError: Se nenhuma turma estiver carregada.
        """
        if self.turma is None or self.turma.student_count == 0:
            raise RuntimeError("Nenhuma turma carregada. Use load_turma() primeiro.")

        scale = self.config.detection_scale
        small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        faces = self._detect_faces(rgb_small)

        if len(faces) == 0:
            return IdentifyResult(status=IdentifyStatus.NO_FACE, face_count=0)

        if len(faces) > 1:
            return IdentifyResult(status=IdentifyStatus.MULTIPLE_FACES, face_count=len(faces))

        encoding = self._compute_encoding(rgb_small, faces[0])

        student_ids, known_encodings = self.turma.get_all_mean_encodings()
        if not known_encodings:
            return IdentifyResult(status=IdentifyStatus.NO_MATCH, face_count=1)

        distances = _face_distance(known_encodings, encoding)
        best_idx = int(np.argmin(distances))
        best_distance = float(distances[best_idx])

        inv_scale = 1.0 / scale
        rect = faces[0]
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

    def identify_best_of_n(self, frames: list[np.ndarray], n: int = 3) -> IdentifyResult:
        """Roda identificação em N frames e retorna o melhor match."""
        best: IdentifyResult | None = None
        last: IdentifyResult | None = None

        for frame in frames[:n]:
            result = self.identify(frame)
            last = result
            if result.is_match:
                if best is None or (result.confidence or 0) > (best.confidence or 0):
                    best = result

        return best if best is not None else last  # type: ignore[return-value]

    # ──────────────────────────────────────────────
    #  Persistência (local na NUC)
    # ──────────────────────────────────────────────

    def save_turma(self, turma: TurmaEncodings | None = None) -> Path:
        """Persiste os encodings da turma em arquivo .pkl local."""
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
        """Carrega encodings de uma turma a partir do .pkl local."""
        filepath = self._encodings_dir / f"{turma_id}.pkl"
        if not filepath.exists():
            raise FileNotFoundError(
                f"Encodings não encontrados para turma '{turma_id}' em {filepath}. "
                f"Rode: python scripts/enroll.py --turma {turma_id}"
            )

        with open(filepath, "rb") as f:
            turma = pickle.load(f)  # noqa: S301 — conteúdo local confiável

        if not isinstance(turma, TurmaEncodings):
            raise TypeError(f"Arquivo {filepath} não contém TurmaEncodings válido.")

        self.turma = turma
        logger.info("Turma '%s' carregada: %d alunos", turma.turma_id, turma.student_count)
        return turma

    def create_turma(self, turma_id: str) -> TurmaEncodings:
        """Cria uma nova turma vazia e a define como ativa."""
        self.turma = TurmaEncodings(turma_id=turma_id)
        logger.info("Turma '%s' criada (vazia)", turma_id)
        return self.turma

    def list_turmas(self) -> list[str]:
        """Lista todas as turmas com encodings salvos localmente."""
        return [p.stem for p in self._encodings_dir.glob("*.pkl")]