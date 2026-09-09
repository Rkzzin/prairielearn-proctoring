"""Detecção passiva de prova de vida sobre um rosto já localizado.

O modelo é uma conversão ONNX do MiniFASNetV2, derivado do projeto
Silent-Face-Anti-Spoofing (Apache 2.0). A inferência usa o OpenCV já presente
na estação, sem incluir PyTorch ou TensorFlow no runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


class PassiveLivenessDetector:
    """Adaptador do MiniFASNet fornecido pelo DeepFace.

    O reconhecimento de identidade continua no dlib. Este componente recebe o
    mesmo frame e retângulo facial e devolve somente a confiança de rosto real.
    """

    def __init__(self, model_path: str | Path | None = None, *, net: Any | None = None):
        if net is None:
            if model_path is None or not Path(model_path).is_file():
                raise FileNotFoundError(
                    f"Modelo de prova de vida não encontrado: {model_path}. "
                    "Rode ./scripts/download_models.sh models"
                )
            net = cv2.dnn.readNetFromONNX(str(model_path))
        self._net = net

    def score(
        self,
        frame: np.ndarray,
        face_location: tuple[int, int, int, int],
    ) -> float:
        """Retorna confiança de rosto real entre 0 e 1."""
        crop = _expanded_face_crop(frame, face_location, scale=2.7)
        resized = cv2.resize(crop, (80, 80))
        blob = np.transpose(resized.astype(np.float32), (2, 0, 1))[None, ...]
        self._net.setInput(blob)
        logits = np.asarray(self._net.forward(), dtype=np.float32).reshape(-1)
        probabilities = np.exp(logits - np.max(logits))
        probabilities /= probabilities.sum()
        # A classe 1 é "real" nos pesos originais do MiniFASNet.
        return float(probabilities[1])


def _expanded_face_crop(
    frame: np.ndarray,
    face_location: tuple[int, int, int, int],
    *,
    scale: float,
) -> np.ndarray:
    top, right, bottom, left = face_location
    frame_height, frame_width = frame.shape[:2]
    width = max(1, right - left)
    height = max(1, bottom - top)
    center_x = left + width / 2
    center_y = top + height / 2
    expanded_width = min(width * scale, frame_width)
    expanded_height = min(height * scale, frame_height)
    crop_left = max(0, int(center_x - expanded_width / 2))
    crop_top = max(0, int(center_y - expanded_height / 2))
    crop_right = min(frame_width, int(center_x + expanded_width / 2))
    crop_bottom = min(frame_height, int(center_y + expanded_height / 2))
    if crop_right <= crop_left or crop_bottom <= crop_top:
        raise ValueError("Retângulo facial inválido para prova de vida")
    return frame[crop_top:crop_bottom, crop_left:crop_right]
