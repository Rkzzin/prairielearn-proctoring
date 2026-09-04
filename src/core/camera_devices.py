"""Descoberta leve das câmeras V4L2 disponíveis na estação."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def discover_video_devices(sys_class_path: Path = Path("/sys/class/video4linux")) -> list[dict[str, Any]]:
    """Lista dispositivos ``/dev/videoN`` com o nome informado pelo kernel."""
    devices: list[dict[str, Any]] = []
    if not sys_class_path.exists():
        return devices

    for entry in sys_class_path.glob("video*"):
        suffix = entry.name.removeprefix("video")
        if not suffix.isdigit():
            continue
        try:
            interface_index = (entry / "index").read_text(encoding="utf-8").strip()
        except OSError:
            interface_index = "0"
        if interface_index != "0":
            continue
        try:
            name = (entry / "name").read_text(encoding="utf-8").strip()
        except OSError:
            name = "Câmera sem nome"
        index = int(suffix)
        devices.append(
            {
                "index": index,
                "name": name or "Câmera sem nome",
                "device": f"/dev/video{index}",
            }
        )
    return sorted(devices, key=lambda item: item["index"])
