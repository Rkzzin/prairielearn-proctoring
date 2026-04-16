import cv2
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.proctor.gaze import GazeEstimator

def run_calibration():
    # Garante que olhos estao ligados para o teste visual
    estimator = GazeEstimator(model_path="models/shape_predictor_68_face_landmarks.dat", enable_eye_gaze=True)
    cap = cv2.VideoCapture(0)
    
    print("=== CALIBRACAO: CABECA + OLHOS ===")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        res = estimator.process_frame(frame)
        if res:
            # Info da Cabeca
            cv2.putText(frame, f"Yaw (Cabeca): {res['yaw']:.1f}", (20, 50), 1, 1.5, (255,255,255), 2)
            # Info dos Olhos
            cv2.putText(frame, f"Eye Ratio: {res['eye_ratio']:.2f}", (20, 90), 1, 1.5, (0,255,255), 2)
            
            # Alerta visual
            color = (0,255,0) if abs(res['yaw']) < 22 else (0,0,255)
            cv2.circle(frame, (frame.shape[1]-40, 40), 15, color, -1)

        cv2.imshow('Calibrador', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_calibration()