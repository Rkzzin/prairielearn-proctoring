"""Estimação de pose da cabeça e direção do olhar.

Usa dlib HOG para detecção e shape_predictor_68 para landmarks.
A pose (yaw, pitch, roll) é calculada via solvePnP com 6 pontos
canônicos do rosto. O ratio ocular é opcional e serve como sinal
secundário de desvio.

Integração com o projeto:
- Recebe FaceConfig para localizar os modelos dlib (mesmos .dat
  já baixados por download_models.sh, sem duplicar caminhos).
- Todas as operações internas usam RGB, consistente com recognizer.py.
- Coordenadas dos landmarks são reescaladas para o frame original
  antes de calcular a pose.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import dlib
import numpy as np

from src.core.config import FaceConfig

logger = logging.getLogger(__name__)


@dataclass
class GazeData:
    """Resultado do processamento de um frame."""

    yaw: float          # graus, positivo = direita
    pitch: float        # graus, positivo = baixo
    roll: float         # graus
    eye_ratio: float | None  # None se enable_eye_gaze=False
    face_count: int     # quantos rostos foram detectados


# Pontos 3D canônicos do rosto em mm (sistema de coordenadas centrado no nariz)
_MODEL_POINTS_3D = np.array([
    (0.0,    0.0,    0.0),    # ponta do nariz (landmark 30)
    (0.0,  -330.0,  -65.0),   # queixo (8)
    (-225.0, 170.0, -135.0),  # canto ext. olho esq. (36)
    ( 225.0, 170.0, -135.0),  # canto ext. olho dir. (45)
    (-150.0,-150.0, -125.0),  # canto esq. boca (48)
    ( 150.0,-150.0, -125.0),  # canto dir. boca (54)
], dtype=np.float64)

# Índices dos landmarks usados para pose
_POSE_LANDMARK_IDX = [30, 8, 36, 45, 48, 54]


class GazeEstimator:
    """Estima pose da cabeça (yaw/pitch/roll) e desvio ocular por frame.

    Args:
        face_config: Configuração do projeto. Se None, usa defaults.
        enable_eye_gaze: Se True, calcula também o ratio pupila/olho.
            Útil para detectar desvios leves que o yaw não captura.
    """

    def __init__(
        self,
        face_config: FaceConfig | None = None,
        enable_eye_gaze: bool = False,
    ):
        cfg = face_config or FaceConfig()
        self.enable_eye_gaze = enable_eye_gaze
        self._scale = cfg.detection_scale

        missing = cfg.validate_models()
        if missing:
            raise FileNotFoundError(
                f"Modelos dlib não encontrados: {missing}. "
                f"Rode: ./scripts/download_models.sh {cfg.models_dir}"
            )

        self._detector = dlib.get_frontal_face_detector()
        self._predictor = dlib.shape_predictor(str(cfg.shape_predictor_path))
        logger.info("GazeEstimator inicializado (eye_gaze=%s)", enable_eye_gaze)

    # ──────────────────────────────────────────────
    #  API pública
    # ──────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> GazeData | None:
        """Processa um frame BGR e retorna GazeData, ou None se sem rosto.

        Args:
            frame: Imagem BGR capturada da webcam (OpenCV).

        Returns:
            GazeData com yaw/pitch/roll e eye_ratio, ou None se nenhum
            rosto for detectado.
        """
        # Redimensionar para detecção (performance)
        scale = self._scale
        small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        faces = self._detector(rgb_small, 1)

        if len(faces) == 0:
            return None

        # Usar o rosto mais central se houver múltiplos
        face_rect = _pick_central_face(faces, rgb_small.shape)

        shape = self._predictor(rgb_small, face_rect)

        # Reescalar landmarks para resolução original antes da pose
        inv = 1.0 / scale
        landmarks_full = _scale_landmarks(shape, inv)

        yaw, pitch, roll = self._get_head_pose(landmarks_full, frame.shape)

        eye_ratio: float | None = None
        if self.enable_eye_gaze:
            gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            eye_ratio = self._get_eye_ratio(shape, gray_small)

        return GazeData(
            yaw=float(yaw),
            pitch=float(pitch),
            roll=float(roll),
            eye_ratio=eye_ratio,
            face_count=len(faces),
        )

    # ──────────────────────────────────────────────
    #  Pose estimation
    # ──────────────────────────────────────────────

    def _get_head_pose(
        self,
        landmarks: list[tuple[float, float]],
        frame_shape: tuple,
    ) -> tuple[float, float, float]:
        """Retorna (yaw, pitch, roll) em graus via solvePnP.

        Args:
            landmarks: Lista de (x, y) para TODOS os 68 pontos,
                       na escala do frame original.
            frame_shape: (height, width, channels) do frame original.
        """
        h, w = frame_shape[:2]
        focal_length = w
        cx, cy = w / 2.0, h / 2.0

        camera_matrix = np.array([
            [focal_length, 0,            cx],
            [0,            focal_length, cy],
            [0,            0,            1 ],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        image_points = np.array(
            [landmarks[i] for i in _POSE_LANDMARK_IDX], dtype=np.float64
        )

        _, rvec, tvec = cv2.solvePnP(
            _MODEL_POINTS_3D, image_points,
            camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        rmat, _ = cv2.Rodrigues(rvec)
        proj_matrix = np.hstack((rmat, tvec))
        _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj_matrix)

        pitch, yaw, roll = euler.flatten()
        return yaw, pitch, roll

    # ──────────────────────────────────────────────
    #  Eye ratio (sinal secundário)
    # ──────────────────────────────────────────────

    def _get_eye_ratio(self, shape, gray_small: np.ndarray) -> float:
        """Calcula ratio pupila esq/dir como indicador de desvio ocular.

        Retorna valores próximos de 1.0 quando o olhar está centralizado.
        Valores < 0.5 ou > 2.0 indicam desvio lateral significativo.
        """
        def _eye_region_ratio(eye_indices: range) -> float:
            pts = np.array(
                [(shape.part(p).x, shape.part(p).y) for p in eye_indices]
            )
            min_x, max_x = int(pts[:, 0].min()), int(pts[:, 0].max())
            min_y, max_y = int(pts[:, 1].min()), int(pts[:, 1].max())

            eye_img = gray_small[min_y:max_y, min_x:max_x]
            if eye_img.size == 0:
                return 1.0

            blurred = cv2.GaussianBlur(eye_img, (7, 7), 0)
            _, thresh = cv2.threshold(
                blurred, 70, 255, cv2.THRESH_BINARY_INV
            )

            w = max_x - min_x
            mid = w // 2
            left_px = cv2.countNonZero(thresh[:, :mid])
            right_px = cv2.countNonZero(thresh[:, mid:])

            return left_px / right_px if right_px > 0 else 1.0

        ratio_left = _eye_region_ratio(range(36, 42))
        ratio_right = _eye_region_ratio(range(42, 48))
        return (ratio_left + ratio_right) / 2.0


# ──────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────

def _pick_central_face(faces, frame_shape: tuple):
    """Retorna o rosto mais próximo do centro do frame."""
    if len(faces) == 1:
        return faces[0]
    h, w = frame_shape[:2]
    cx, cy = w / 2, h / 2
    return min(
        faces,
        key=lambda r: (
            ((r.left() + r.right()) / 2 - cx) ** 2
            + ((r.top() + r.bottom()) / 2 - cy) ** 2
        ),
    )


def _scale_landmarks(
    shape, scale: float
) -> list[tuple[float, float]]:
    """Converte landmarks do shape_predictor para escala desejada."""
    return [
        (shape.part(i).x * scale, shape.part(i).y * scale)
        for i in range(68)
    ]