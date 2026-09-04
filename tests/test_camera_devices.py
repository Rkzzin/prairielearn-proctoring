from pathlib import Path

from src.core.camera_devices import discover_video_devices


def test_discover_video_devices_reports_kernel_names_and_skips_metadata_nodes(tmp_path: Path):
    primary = tmp_path / "video2"
    primary.mkdir()
    (primary / "name").write_text("Logitech BRIO", encoding="utf-8")
    (primary / "index").write_text("0", encoding="utf-8")
    metadata = tmp_path / "video3"
    metadata.mkdir()
    (metadata / "name").write_text("Logitech BRIO", encoding="utf-8")
    (metadata / "index").write_text("1", encoding="utf-8")

    assert discover_video_devices(tmp_path) == [
        {"index": 2, "name": "Logitech BRIO", "device": "/dev/video2"}
    ]
