import numpy as np
import pytest

from src.face.liveness import PassiveLivenessDetector


def test_liveness_converts_face_location_and_returns_real_confidence():
    class FakeNet:
        def setInput(self, value):
            self.input = value

        def forward(self):
            return np.array([[0.0, 2.0, 0.0]], dtype=np.float32)

    net = FakeNet()
    detector = PassiveLivenessDetector(net=net)
    frame = np.zeros((200, 300, 3), dtype=np.uint8)

    score = detector.score(frame, (20, 140, 120, 40))

    assert score == pytest.approx(0.786986)
    assert net.input.shape == (1, 3, 80, 80)


def test_liveness_returns_low_probability_for_spoof_class():
    class FakeNet:
        def setInput(self, _value):
            pass

        def forward(self):
            return np.array([[0.0, 0.0, 5.0]], dtype=np.float32)

    score = PassiveLivenessDetector(net=FakeNet()).score(
        np.zeros((100, 100, 3), dtype=np.uint8),
        (10, 90, 90, 10),
    )

    assert score < 0.01
