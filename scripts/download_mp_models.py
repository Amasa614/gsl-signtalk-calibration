"""Download MediaPipe Tasks landmarker models into mp_models/."""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import PATHS

MODELS = {
    "holistic_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"
    ),
    "pose_landmarker_full.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
    ),
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    ),
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
    ),
}


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {dest.name} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"  -> {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    PATHS.mp_models.mkdir(parents=True, exist_ok=True)
    for name, url in MODELS.items():
        dest = PATHS.mp_models / name
        if dest.exists() and dest.stat().st_size > 1000:
            print(f"skip (exists): {dest.name}")
            continue
        _download(url, dest)
    print("done.")


if __name__ == "__main__":
    main()
