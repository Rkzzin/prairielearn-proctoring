import json
import time
from dataclasses import dataclass, asdict
from typing import Dict, Any

@dataclass
class ProctorEvent:
    timestamp: float
    frame_number: int
    event_type: str       # 'HEAD_POSE_WARNING', 'EYE_GAZE_WARNING', 'BLOCKED', 'NORMAL'
    severity: str         # 'INFO', 'WARNING', 'CRITICAL'
    details: Dict[str, Any]

class EventLogger:
    def __init__(self, log_path: str = "events.jsonl"):
        self.log_path = log_path

    def log(self, event: ProctorEvent):
        """Append de evento no arquivo de log."""
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(asdict(event)) + '\n')
            
    def create_event(self, frame_num: int, ev_type: str, severity: str, details: dict = None):
        event = ProctorEvent(
            timestamp=time.time(),
            frame_number=frame_num,
            event_type=ev_type,
            severity=severity,
            details=details or {}
        )
        self.log(event)
        return event