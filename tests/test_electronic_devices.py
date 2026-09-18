import time

import numpy as np

from src.proctor.electronic_devices import (
    ElectronicDeviceDetection,
    ElectronicDeviceMonitor,
    YoloXElectronicDeviceDetector,
    YoloXInferenceResult,
)


class FakeNet:
    def __init__(self, output):
        self.output = output
        self.input = None
        self.forward_calls = 0

    def setInput(self, value):
        self.input = value

    def forward(self):
        self.forward_calls += 1
        return self.output


def test_yolox_detector_returns_person_presence_separately_from_electronics():
    output = np.zeros((1, 8400, 85), dtype=np.float32)
    output[0, 0, :4] = [40, 40, np.log(10), np.log(8)]
    output[0, 0, 4] = 0.9
    output[0, 0, 5 + 67] = 0.9
    output[0, 1, :4] = [20, 20, np.log(20), np.log(30)]
    output[0, 1, 4] = 0.8
    output[0, 1, 5] = 0.9
    output[0, 2, :4] = [60, 20, np.log(20), np.log(30)]
    output[0, 2, 4] = 0.8
    output[0, 2, 5] = 0.875
    net = FakeNet(output)
    detector = YoloXElectronicDeviceDetector(
        "unused.onnx",
        confidence_threshold=0.45,
        net=net,
    )

    result = detector.infer(np.zeros((360, 640, 3), dtype=np.uint8))

    assert len(result.electronic_devices) == 1
    assert result.electronic_devices[0].label == "celular"
    assert result.electronic_devices[0].confidence == 0.81
    assert result.person_present is True
    assert result.person_confidence == 0.72
    assert result.person_count == 2
    assert {person.confidence for person in result.people} == {0.72, 0.7}
    assert len({person.box for person in result.people}) == 2
    assert net.forward_calls == 1
    assert net.input.shape == (1, 3, 640, 640)


def test_monitor_requires_two_of_three_detections_and_emits_clear():
    detection = ElectronicDeviceDetection("notebook", 0.9, (1, 2, 3, 4))
    responses = iter([[detection], [detection], [], []])

    class Detector:
        def detect(self, _frame):
            return next(responses, [])

    monitor = ElectronicDeviceMonitor(
        detector=Detector(),
        primary_enabled=True,
        secondary_enabled=False,
        secondary_preview_url=None,
        interval_sec=0.01,
        confirmation_sec=0.0,
    )
    monitor.submit_primary(np.zeros((10, 10, 3), dtype=np.uint8))
    monitor.start()
    deadline = time.monotonic() + 1
    transitions = []
    while time.monotonic() < deadline and len(transitions) < 2:
        transitions.extend(monitor.drain_transitions())
        time.sleep(0.01)
    monitor.stop()

    assert [(item.camera, item.active) for item in transitions] == [
        ("principal", True),
        ("principal", False),
    ]


def test_monitor_ignores_calibrated_device_region():
    detection = ElectronicDeviceDetection("notebook", 0.9, (100, 50, 200, 100))

    class Detector:
        def infer(self, _frame):
            return YoloXInferenceResult((detection,))

    monitor = ElectronicDeviceMonitor(
        detector=Detector(),
        primary_enabled=True,
        secondary_enabled=False,
        secondary_preview_url=None,
        interval_sec=0.01,
        confirmation_sec=0.0,
        ignored_regions={
            "principal": [
                {"label": "notebook", "box": [0.15, 0.12, 0.33, 0.30]}
            ]
        },
    )
    monitor.submit_primary(np.zeros((360, 640, 3), dtype=np.uint8))
    monitor.start()
    time.sleep(0.06)
    transitions = monitor.drain_transitions()
    monitor.stop()

    assert transitions == []


def test_monitor_exposes_fresh_primary_person_without_electronic_transition():
    now = [10.0]

    class Detector:
        def infer(self, _frame):
            return YoloXInferenceResult((), person_confidence=0.82)

    monitor = ElectronicDeviceMonitor(
        detector=Detector(),
        primary_enabled=True,
        secondary_enabled=False,
        secondary_preview_url=None,
        interval_sec=0.01,
        person_presence_ttl_sec=5.0,
        clock_fn=lambda: now[0],
    )
    monitor.submit_primary(np.zeros((10, 10, 3), dtype=np.uint8))
    monitor.start()
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline and monitor.latest_primary_person_present() is None:
        time.sleep(0.01)

    assert monitor.latest_primary_person_present() is True
    assert monitor.drain_transitions() == []

    now[0] = 15.1
    assert monitor.latest_primary_person_present() is None
    monitor.stop()


def test_monitor_waits_shared_confirmation_time_before_emitting_detection():
    detection = ElectronicDeviceDetection("celular", 0.9, (1, 2, 3, 4))
    now = [0.0]
    monitor = ElectronicDeviceMonitor(
        detector=None,
        primary_enabled=True,
        secondary_enabled=False,
        secondary_preview_url=None,
        confirmation_sec=10.0,
        clock_fn=lambda: now[0],
    )

    monitor._histories["principal"].append(True)
    monitor._update_detection_state("principal", [detection])
    now[0] = 1.0
    monitor._histories["principal"].append(True)
    monitor._update_detection_state("principal", [detection])
    now[0] = 10.9
    monitor._histories["principal"].append(True)
    monitor._update_detection_state("principal", [detection])
    assert monitor.drain_transitions() == []

    now[0] = 11.0
    monitor._histories["principal"].append(True)
    monitor._update_detection_state("principal", [detection])

    transitions = monitor.drain_transitions()
    assert [(item.camera, item.active) for item in transitions] == [("principal", True)]


def test_monitor_cancels_confirmation_when_detection_does_not_persist():
    detection = ElectronicDeviceDetection("celular", 0.9, (1, 2, 3, 4))
    now = [0.0]
    monitor = ElectronicDeviceMonitor(
        detector=None,
        primary_enabled=True,
        secondary_enabled=False,
        secondary_preview_url=None,
        confirmation_sec=10.0,
        clock_fn=lambda: now[0],
    )

    for detected in (True, True, False, False):
        monitor._histories["principal"].append(detected)
        monitor._update_detection_state(
            "principal", [detection] if detected else []
        )
        now[0] += 1.0
    now[0] = 20.0
    monitor._histories["principal"].append(True)
    monitor._update_detection_state("principal", [detection])

    assert monitor.drain_transitions() == []
