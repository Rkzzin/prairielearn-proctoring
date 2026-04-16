import time
from enum import Enum
from .events import EventLogger
from .gaze import GazeEstimator

class ProctorState(Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    BLOCKED = "BLOCKED"

class ProctorEngine:
    def __init__(self, config=None):
        self.config = config or {
            "YAW_THRESHOLD": 30.0,
            "EYE_RATIO_LOW": 0.4,
            "EYE_RATIO_HIGH": 0.8,
            "SMOOTHING_WINDOW": 10,
            "WARNING_DURATION_SEC": 5.0
        }
        # Inicia com olhos ativos por padrao
        self.gaze_estimator = GazeEstimator(enable_eye_gaze=False)
        self.logger = EventLogger()
        self.state = ProctorState.NORMAL
        self.yaw_history, self.eye_history = [], []
        self.warning_start_time, self.frame_count = 0, 0

    def update(self, frame):
        """Process a camera frame and return the current proctoring state."""
        data = self.gaze_estimator.process_frame(frame)
        return self._update_state(data)

    def _update_state(self, data):
        self.frame_count += 1
        if data is None: return "ABSENCE"

        # Suavizacao
        self.yaw_history.append(data['yaw'])
        if len(self.yaw_history) > self.config["SMOOTHING_WINDOW"]: self.yaw_history.pop(0)
        smooth_yaw = sum(self.yaw_history) / len(self.yaw_history)

        # Logica combinada
        is_deviated = abs(smooth_yaw) > self.config["YAW_THRESHOLD"]
        
        # Se olhos estiverem ativos, eles tambem podem gerar alerta
        if data["eye_ratio"] is not None:
            is_deviated = is_deviated or (data["eye_ratio"] < self.config["EYE_RATIO_LOW"] or 
                                          data["eye_ratio"] > self.config["EYE_RATIO_HIGH"])

        if self.state == ProctorState.NORMAL and is_deviated:
            self.state = ProctorState.WARNING
            self.warning_start_time = time.time()
            self.logger.create_event(self.frame_count, "PROCTOR_WARNING", "WARNING", data)
        
        elif self.state == ProctorState.WARNING:
            if not is_deviated:
                self.state = ProctorState.NORMAL
            elif (time.time() - self.warning_start_time) > self.config["WARNING_DURATION_SEC"]:
                self.state = ProctorState.BLOCKED
                self.logger.create_event(self.frame_count, "SESSION_BLOCKED", "CRITICAL")
        
        return self.state.value
