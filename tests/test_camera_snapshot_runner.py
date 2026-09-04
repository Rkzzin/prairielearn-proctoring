import base64
import json
import threading
import time

import httpx

from src.core.camera_snapshot_runner import CameraSnapshotRunner
from src.core.config import DashboardConfig


class SnapshotManager:
    def capture_camera_snapshots(self):
        return (
            [
                {
                    "index": 2,
                    "name": "C922 Pro Stream Webcam",
                    "device": "/dev/video2",
                    "jpeg": b"\xff\xd8photo",
                }
            ],
            [],
        )


def test_camera_snapshot_runner_uploads_each_camera_with_station_auth():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(201, json={})

    config = DashboardConfig(
        base_url="https://dashboard.test",
        station_id="nuc-01",
        station_token="secret",
    )
    runner = CameraSnapshotRunner(
        config=config,
        session_manager=SnapshotManager(),
        client_factory=lambda: httpx.Client(
            base_url=config.base_url,
            headers={"X-Station-Id": "nuc-01", "X-Station-Token": "secret"},
            transport=httpx.MockTransport(handler),
        ),
    )

    runner.start("batch-1")
    deadline = time.monotonic() + 2
    while runner.status_dict()["camera_capture_status"] == "running" and time.monotonic() < deadline:
        time.sleep(0.01)

    assert runner.status_dict()["camera_capture_status"] == "done"
    assert requests[0].url.path == "/api/camera-snapshots"
    assert requests[0].headers["x-station-id"] == "nuc-01"
    payload = json.loads(requests[0].read())
    assert base64.b64decode(payload["image_base64"]) == b"\xff\xd8photo"


def test_camera_snapshot_runner_runs_replacement_batch_after_current_batch():
    class BlockingManager:
        def __init__(self):
            self.calls = []
            self.first_started = threading.Event()
            self.release_first = threading.Event()

        def capture_camera_snapshots(self):
            self.calls.append(len(self.calls) + 1)
            if len(self.calls) == 1:
                self.first_started.set()
                self.release_first.wait(timeout=2)
            return [], ["Nenhuma câmera detectada"]

    manager = BlockingManager()
    runner = CameraSnapshotRunner(
        config=DashboardConfig(),
        session_manager=manager,
        client_factory=lambda: httpx.Client(transport=httpx.MockTransport(lambda _request: httpx.Response(201))),
    )

    runner.start("batch-1")
    assert manager.first_started.wait(timeout=1)
    runner.start("batch-2")
    manager.release_first.set()
    deadline = time.monotonic() + 2
    while len(manager.calls) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)

    assert manager.calls == [1, 2]
    assert runner.status_dict()["camera_capture_batch_id"] == "batch-2"
