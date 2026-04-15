#!/usr/bin/env python3
"""Patch face_recognition_models to remove pkg_resources dependency.

face_recognition_models uses pkg_resources.resource_filename() to locate
its model files. setuptools >= 82 removed pkg_resources, breaking the
import on Python 3.12+.

This script replaces the __init__.py with a pathlib-based version that
works on any Python 3.6+ without setuptools.

Run AFTER pip install:
    pip install -e ".[dev]"
    python scripts/patch_face_models.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PATCHED_INIT = '''\
# -*- coding: utf-8 -*-
# Patched by proctor-station to remove pkg_resources dependency.
# Original used pkg_resources.resource_filename which requires setuptools < 82.
__author__ = """Adam Geitgey"""
__email__ = "ageitgey@gmail.com"
__version__ = "0.3.0"

from pathlib import Path as _Path

_models_dir = _Path(__file__).parent / "models"

def pose_predictor_model_location():
    return str(_models_dir / "shape_predictor_68_face_landmarks.dat")

def pose_predictor_five_point_model_location():
    return str(_models_dir / "shape_predictor_5_face_landmarks.dat")

def face_recognition_model_location():
    return str(_models_dir / "dlib_face_recognition_resnet_model_v1.dat")

def cnn_face_detector_model_location():
    return str(_models_dir / "mmod_human_face_detector.dat")
'''


def find_package_init() -> Path | None:
    """Locate face_recognition_models/__init__.py in the current environment."""
    spec = importlib.util.find_spec("face_recognition_models")
    if spec is None or spec.origin is None:
        return None
    return Path(spec.origin)


def needs_patch(init_path: Path) -> bool:
    """Check if the file still uses pkg_resources."""
    content = init_path.read_text()
    return "pkg_resources" in content


def patch(init_path: Path) -> None:
    """Replace __init__.py with pathlib-based version."""
    # Validate that model files actually exist
    models_dir = init_path.parent / "models"
    expected_files = [
        "shape_predictor_68_face_landmarks.dat",
        "shape_predictor_5_face_landmarks.dat",
        "dlib_face_recognition_resnet_model_v1.dat",
        "mmod_human_face_detector.dat",
    ]

    missing = [f for f in expected_files if not (models_dir / f).exists()]
    if missing:
        print(f"WARNING: Model files missing in {models_dir}:")
        for f in missing:
            print(f"  - {f}")
        print("The patch will proceed but face_recognition may not work.")
        print("Try: pip install face-recognition-models --force-reinstall")

    # Write patched file
    init_path.write_text(PATCHED_INIT)


def verify() -> bool:
    """Verify the patch works by importing the module."""
    # Force reimport
    for mod_name in list(sys.modules.keys()):
        if "face_recognition" in mod_name:
            del sys.modules[mod_name]

    try:
        import face_recognition_models

        # Verify all functions return valid paths
        funcs = [
            face_recognition_models.pose_predictor_model_location,
            face_recognition_models.pose_predictor_five_point_model_location,
            face_recognition_models.face_recognition_model_location,
            face_recognition_models.cnn_face_detector_model_location,
        ]
        for fn in funcs:
            path = Path(fn())
            status = "OK" if path.exists() else "MISSING"
            print(f"  {status}: {path.name}")

        import face_recognition  # noqa: F401

        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def main() -> None:
    print("Patching face_recognition_models...")

    init_path = find_package_init()
    if init_path is None:
        print("ERROR: face_recognition_models not installed.")
        print("Run: pip install face-recognition-models")
        sys.exit(1)

    print(f"  Found: {init_path}")

    if not needs_patch(init_path):
        print("  Already patched — skipping.")
    else:
        patch(init_path)
        print("  Patched successfully.")

    print("\nVerifying...")
    if verify():
        print("\nface_recognition is ready.")
    else:
        print("\nPatch applied but verification failed — check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
