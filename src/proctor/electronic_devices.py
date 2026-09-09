"""Detecção periódica e não bloqueante de celular e notebook."""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.core.camera import open_video_capture

logger = logging.getLogger(__name__)

TARGET_CLASSES = {63: "notebook", 67: "celular"}


@dataclass(frozen=True)
class ElectronicDeviceDetection:
    label: str
    confidence: float
    box: tuple[int, int, int, int]


@dataclass(frozen=True)
class ElectronicDeviceTransition:
    camera: str
    active: bool
    detections: tuple[ElectronicDeviceDetection, ...]

    def details(self) -> dict[str, Any]:
        return {
            "camera": self.camera,
            "detections": [asdict(detection) for detection in self.detections],
        }


class YoloXElectronicDeviceDetector:
    """YOLOX-S COCO do OpenCV Zoo executado pelo OpenCV DNN."""

    INPUT_SIZE = 640
    STRIDES = (8, 16, 32)

    def __init__(
        self,
        model_path: str | Path,
        *,
        confidence_threshold: float = 0.45,
        nms_threshold: float = 0.45,
        net: Any | None = None,
    ):
        if net is None:
            if not Path(model_path).is_file():
                raise FileNotFoundError(
                    f"Modelo de detecção de eletrônicos não encontrado: {model_path}. "
                    "Rode ./scripts/download_models.sh models"
                )
            net = cv2.dnn.readNetFromONNX(str(model_path))
        self._net = net
        self._confidence_threshold = confidence_threshold
        self._nms_threshold = nms_threshold
        self._grid, self._expanded_strides = self._generate_anchors()

    def detect(self, frame: np.ndarray) -> list[ElectronicDeviceDetection]:
        height, width = frame.shape[:2]
        scale = min(self.INPUT_SIZE / width, self.INPUT_SIZE / height)
        resized = cv2.resize(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            (int(width * scale), int(height * scale)),
        ).astype(np.float32)
        padded = np.full(
            (self.INPUT_SIZE, self.INPUT_SIZE, 3),
            114.0,
            dtype=np.float32,
        )
        padded[: resized.shape[0], : resized.shape[1]] = resized
        blob = np.transpose(padded, (2, 0, 1))[None, ...]
        self._net.setInput(blob)
        output = self._net.forward()
        if isinstance(output, (list, tuple)):
            output = output[0]
        predictions = np.asarray(output, dtype=np.float32).reshape(-1, 85).copy()
        predictions[:, :2] = (
            predictions[:, :2] + self._grid
        ) * self._expanded_strides
        predictions[:, 2:4] = np.exp(predictions[:, 2:4]) * self._expanded_strides

        class_scores = predictions[:, 4:5] * predictions[:, 5:]
        class_ids = np.argmax(class_scores, axis=1)
        scores = class_scores[np.arange(len(class_ids)), class_ids]
        selected = [
            index
            for index, (class_id, score) in enumerate(zip(class_ids, scores, strict=True))
            if int(class_id) in TARGET_CLASSES and float(score) >= self._confidence_threshold
        ]
        if not selected:
            return []

        boxes: list[list[int]] = []
        selected_scores: list[float] = []
        selected_classes: list[int] = []
        for index in selected:
            center_x, center_y, box_width, box_height = predictions[index, :4]
            boxes.append([
                int((center_x - box_width / 2) / scale),
                int((center_y - box_height / 2) / scale),
                int(box_width / scale),
                int(box_height / scale),
            ])
            selected_scores.append(float(scores[index]))
            selected_classes.append(int(class_ids[index]))

        keep: list[int] = []
        for class_id in TARGET_CLASSES:
            class_indexes = [
                index for index, value in enumerate(selected_classes) if value == class_id
            ]
            if not class_indexes:
                continue
            class_keep = cv2.dnn.NMSBoxes(
                [boxes[index] for index in class_indexes],
                [selected_scores[index] for index in class_indexes],
                self._confidence_threshold,
                self._nms_threshold,
            )
            keep.extend(class_indexes[int(index)] for index in np.asarray(class_keep).reshape(-1))

        return [
            ElectronicDeviceDetection(
                label=TARGET_CLASSES[selected_classes[index]],
                confidence=round(selected_scores[index], 4),
                box=tuple(boxes[index]),
            )
            for index in keep
        ]

    @classmethod
    def _generate_anchors(cls) -> tuple[np.ndarray, np.ndarray]:
        grids = []
        expanded_strides = []
        for stride in cls.STRIDES:
            size = cls.INPUT_SIZE // stride
            x_values, y_values = np.meshgrid(np.arange(size), np.arange(size))
            grid = np.stack((x_values, y_values), axis=2).reshape(-1, 2)
            grids.append(grid)
            expanded_strides.append(np.full((len(grid), 1), stride))
        return np.concatenate(grids), np.concatenate(expanded_strides)


class _LatestPreviewReader:
    def __init__(
        self,
        source: str,
        capture_factory: Callable[[int | str], Any] = open_video_capture,
    ):
        self._source = source
        self._capture_factory = capture_factory
        self._capture: Any | None = None
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="environment-preview", daemon=True)
        self._thread.start()

    def latest(self) -> np.ndarray | None:
        with self._lock:
            return self._frame

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
            if self._thread.is_alive():
                logger.warning("Preview da câmera ambiente não encerrou em 3s")

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._capture = self._capture_factory(self._source)
                if hasattr(self._capture, "isOpened") and not self._capture.isOpened():
                    raise RuntimeError("preview ambiente não abriu")
                while not self._stop.is_set():
                    ok, frame = self._capture.read()
                    if not ok or frame is None:
                        break
                    with self._lock:
                        self._frame = frame
            except (OSError, RuntimeError, cv2.error) as exc:  # pragma: no cover
                logger.warning("Falha no preview da câmera ambiente: %s", exc)
            finally:
                if self._capture is not None and hasattr(self._capture, "release"):
                    self._capture.release()
                self._capture = None
            self._stop.wait(0.5)


class ElectronicDeviceMonitor:
    """Agenda inferências sem atrasar o loop principal do proctoring."""

    def __init__(
        self,
        *,
        detector: YoloXElectronicDeviceDetector,
        primary_enabled: bool,
        secondary_enabled: bool,
        secondary_preview_url: str | None,
        interval_sec: float = 1.0,
        preview_capture_factory: Callable[[int | str], Any] = open_video_capture,
    ):
        self._detector = detector
        self._primary_enabled = primary_enabled
        self._interval_sec = interval_sec
        self._primary_frame: np.ndarray | None = None
        self._frame_lock = threading.Lock()
        self._reader = (
            _LatestPreviewReader(secondary_preview_url, preview_capture_factory)
            if secondary_enabled and secondary_preview_url
            else None
        )
        self._histories = {
            "principal": deque(maxlen=3),
            "ambiente": deque(maxlen=3),
        }
        self._active = {"principal": False, "ambiente": False}
        self._results: queue.SimpleQueue[ElectronicDeviceTransition] = queue.SimpleQueue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._reader is not None:
            self._reader.start()
        self._thread = threading.Thread(target=self._run, name="electronic-device-monitor", daemon=True)
        self._thread.start()

    def submit_primary(self, frame: np.ndarray) -> None:
        if not self._primary_enabled:
            return
        with self._frame_lock:
            self._primary_frame = frame

    def drain_transitions(self) -> list[ElectronicDeviceTransition]:
        transitions = []
        while True:
            try:
                transitions.append(self._results.get_nowait())
            except queue.Empty:
                return transitions

    def stop(self) -> None:
        self._stop.set()
        if self._reader is not None:
            self._reader.stop()
        if self._thread is not None:
            self._thread.join(timeout=3)

    def _run(self) -> None:
        while not self._stop.wait(self._interval_sec):
            frames: dict[str, np.ndarray | None] = {}
            if self._primary_enabled:
                with self._frame_lock:
                    frames["principal"] = self._primary_frame
            if self._reader is not None:
                frames["ambiente"] = self._reader.latest()
            for camera, frame in frames.items():
                if frame is None:
                    continue
                started_at = time.monotonic()
                try:
                    detections = self._detector.detect(frame)
                except (RuntimeError, ValueError, cv2.error) as exc:  # pragma: no cover
                    logger.warning("Falha ao detectar eletrônicos na câmera %s: %s", camera, exc)
                    continue
                logger.debug(
                    "Detecção de eletrônicos em %s concluída em %.3fs",
                    camera,
                    time.monotonic() - started_at,
                )
                history = self._histories[camera]
                history.append(bool(detections))
                active = sum(history) >= 2
                if active != self._active[camera]:
                    self._active[camera] = active
                    self._results.put(
                        ElectronicDeviceTransition(camera, active, tuple(detections))
                    )
