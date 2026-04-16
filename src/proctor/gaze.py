import cv2
import dlib
import numpy as np
from pathlib import Path

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "shape_predictor_68_face_landmarks.dat"
)

class GazeEstimator:
    def __init__(self, model_path=None, enable_eye_gaze=True):
        model_path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
        self.detector = dlib.get_frontal_face_detector()
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo dlib nao encontrado: {model_path}")
        self.predictor = dlib.shape_predictor(str(model_path))
        
        # Toggle para ativar/desativar verificacao ocular
        self.enable_eye_gaze = enable_eye_gaze

        # Pontos 3D de referencia para Pose Estimation
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Ponta do Nariz (30)
            (0.0, -330.0, -65.0),        # Queixo (8)
            (-225.0, 170.0, -135.0),     # Canto esquerdo olho esquerdo (36)
            (225.0, 170.0, -135.0),      # Canto direito olho direito (45)
            (-150.0, -150.0, -125.0),    # Canto esquerdo boca (48)
            (150.0, -150.0, -125.0)      # Canto direito boca (54)
        ])

    def _get_head_pose(self, shape, frame_size):
        """Calcula orientacao da cabeca (Pitch, Yaw, Roll)."""
        image_points = np.array([
            (shape.part(30).x, shape.part(30).y),
            (shape.part(8).x, shape.part(8).y),
            (shape.part(36).x, shape.part(36).y),
            (shape.part(45).x, shape.part(45).y),
            (shape.part(48).x, shape.part(48).y),
            (shape.part(54).x, shape.part(54).y)
        ], dtype="double")

        focal_length = frame_size[1]
        center = (frame_size[1] / 2, frame_size[0] / 2)
        camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double")
        
        _, rotation_vector, translation_vector = cv2.solvePnP(self.model_points, image_points, camera_matrix, np.zeros((4,1)))
        rmat, _ = cv2.Rodrigues(rotation_vector)
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(np.hstack((rmat, translation_vector)))
        
        return euler_angles.flatten()

    def _get_eye_ratio(self, shape, gray):
        """Isola a pupila e calcula o desvio ocular."""
        def get_ratio(eye_points):
            region = np.array([(shape.part(p).x, shape.part(p).y) for p in eye_points])
            min_x, max_x = np.min(region[:, 0]), np.max(region[:, 0])
            min_y, max_y = np.min(region[:, 1]), np.max(region[:, 1])
            eye_img = gray[min_y:max_y, min_x:max_x]
            if eye_img.size == 0: return 1.0
            
            # Threshold para encontrar pupila
            _, thresh = cv2.threshold(cv2.GaussianBlur(eye_img, (7,7), 0), 70, 255, cv2.THRESH_BINARY_INV)
            w = max_x - min_x
            left_area = cv2.countNonZero(thresh[:, 0:int(w/2)])
            right_area = cv2.countNonZero(thresh[:, int(w/2):])
            return left_area / right_area if right_area > 0 else 1.0

        return (get_ratio(range(36, 42)) + get_ratio(range(42, 48))) / 2

    def process_frame(self, frame):
        """Retorna Pose e Ratio Ocular (se ativo)."""
        small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        
        if not faces: return None
        
        shape = self.predictor(gray, faces[0])
        p, y, r = self._get_head_pose(shape, gray.shape)
        
        data = {"pitch": p, "yaw": y, "roll": r, "eye_ratio": None}
        if self.enable_eye_gaze:
            data["eye_ratio"] = self._get_eye_ratio(shape, gray)
        
        return data
