"""Detector de rostos para monitoramento contínuo.

Módulo leve que roda a cada frame durante a prova.
Diferente do recognizer (que identifica *quem*), o detector
apenas conta quantos rostos estão visíveis e onde estão.

Usa face_recognition.face_locations() com modelo HOG para
performance em CPU.
"""

from __future__ import annotations

import logging

import cv2
import face_recognition
import numpy as np

from src.core.config import FaceConfig

logger = logging.getLogger(__name__)


class FaceDetector:
    """Detecção rápida de rostos para proctoring contínuo."""

    def __init__(self, config: FaceConfig | None = None):
        self.config = config or FaceConfig()

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Detecta rostos no frame e retorna suas localizações.

        Args:
            frame: Imagem BGR (OpenCV).

        Returns:
            Lista de (top, right, bottom, left) para cada rosto.
            Coordenadas na escala original do frame.
        """
        scale = self.config.detection_scale
        small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        locations = face_recognition.face_locations(
            rgb_small, model=self.config.detection_model
        )

        # Escalar de volta para resolução original
        inv = 1.0 / scale
        return [
            (int(t * inv), int(r * inv), int(b * inv), int(l * inv))
            for t, r, b, l in locations
        ]

    def count_faces(self, frame: np.ndarray) -> int:
        """Retorna a quantidade de rostos detectados no frame."""
        return len(self.detect(frame))

    def has_single_face(self, frame: np.ndarray) -> bool:
        """Verifica se há exatamente 1 rosto no frame."""
        return self.count_faces(frame) == 1

    def annotate_frame(
        self,
        frame: np.ndarray,
        locations: list[tuple[int, int, int, int]] | None = None,
        color: tuple[int, int, int] = (0, 255, 0),
        label: str | None = None,
    ) -> np.ndarray:
        """Desenha retângulos nos rostos detectados (para debug/preview).

        Args:
            frame: Imagem BGR.
            locations: Localizações dos rostos (se None, detecta automaticamente).
            color: Cor BGR do retângulo.
            label: Texto a exibir acima do rosto.

        Returns:
            Cópia do frame com anotações.
        """
        annotated = frame.copy()
        if locations is None:
            locations = self.detect(frame)

        for top, right, bottom, left in locations:
            cv2.rectangle(annotated, (left, top), (right, bottom), color, 2)
            if label:
                cv2.putText(
                    annotated,
                    label,
                    (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )

        return annotated
