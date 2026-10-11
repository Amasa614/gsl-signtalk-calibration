"""Convert Fragkiadakis OpenPose JSON dumps into MediaPipe-layout landmark npz.

The lexicon ships as BODY_25 + 21-point hands (no face). We map overlapping
body joints into the 33-pose / 21×2-hand / 25-NMM layout used by SignTalk so
SSL can pool both sources. Unmapped joints (including all face NMMs) stay zero.
After torso normalization the two sources share a shoulder-width coordinate
frame; they are never mixed into the labeled SignTalk train/val/test split.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.config import FEATURE_DIM, N_HAND, N_POSE, PATHS
from src.landmark_extraction import landmark_out_path, save_landmarks
from src.normalize import empty_frame, flatten_frame, normalize_sequence

JSON_NAME = re.compile(r"^(?P<stem>.+)_(?P<frame>\d+)_keypoints\.json$", re.IGNORECASE)
CONF_MIN = 0.05

# OpenPose BODY_25 index -> MediaPipe Pose index (33). Neck / mid-hip have no MP slot.
BODY25_TO_MP: dict[int, int] = {
    0: 0,  # nose
    16: 2,  # left eye
    15: 5,  # right eye
    18: 7,  # left ear
    17: 8,  # right ear
    5: 11,  # left shoulder
    2: 12,  # right shoulder
    6: 13,  # left elbow
    3: 14,  # right elbow
    7: 15,  # left wrist
    4: 16,  # right wrist
    12: 23,  # left hip
    9: 24,  # right hip
    13: 25,  # left knee
    10: 26,  # right knee
    14: 27,  # left ankle
    11: 28,  # right ankle
    21: 29,  # left heel
    24: 30,  # right heel
    19: 31,  # left foot index
    22: 32,  # right foot index
}


def fragkiadakis_root(root: Path | None = None) -> Path:
    if root is not None:
        return root
    for candidate in (
        PATHS.repo_root / "GSL_openpose_data",
        PATHS.external / "fragkiadakis" / "GSL_openpose_data",
        PATHS.fragkiadakis,
    ):
        if candidate.exists():
            return candidate
    return PATHS.fragkiadakis


def _xyz_from_openpose(flat: list | tuple | None, n_expected: int) -> np.ndarray:
    """Return (n_expected, 3) as x, y, z=0. Confidence below CONF_MIN -> zeros."""
    out = np.zeros((n_expected, 3), dtype=np.float32)
    if not flat:
        return out
    n = len(flat) // 3
    for i in range(min(n, n_expected)):
        x, y, c = float(flat[3 * i]), float(flat[3 * i + 1]), float(flat[3 * i + 2])
        if c < CONF_MIN:
            continue
        out[i, 0] = x
        out[i, 1] = y
    return out


def person_to_frame(person: dict) -> np.ndarray:
    """Pack one OpenPose `people[]` entry into (N_JOINTS, 3) MediaPipe layout."""
    frame = empty_frame()
    pose_op = _xyz_from_openpose(person.get("pose_keypoints_2d"), 25)
    for op_i, mp_i in BODY25_TO_MP.items():
        if mp_i < N_POSE:
            frame[mp_i] = pose_op[op_i]

    left = _xyz_from_openpose(person.get("hand_left_keypoints_2d"), N_HAND)
    right = _xyz_from_openpose(person.get("hand_right_keypoints_2d"), N_HAND)
    frame[N_POSE : N_POSE + N_HAND] = left
    frame[N_POSE + N_HAND : N_POSE + 2 * N_HAND] = right
    return frame


def load_frame_json(path: Path) -> np.ndarray:
    data = json.loads(path.read_text(encoding="utf-8"))
    people = data.get("people") or []
    if not people:
        return empty_frame()
    return person_to_frame(people[0])


def list_openpose_clips(root: Path | None = None) -> list[dict]:
    """One clip per gloss folder (JSON frames sorted by OpenPose frame index)."""
    root = fragkiadakis_root(root)
    if not root.exists():
        return []
    clips: list[dict] = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        grouped: dict[str, list[tuple[int, Path]]] = {}
        for path in folder.glob("*_keypoints.json"):
            m = JSON_NAME.match(path.name)
            if not m:
                continue
            grouped.setdefault(m.group("stem"), []).append((int(m.group("frame")), path))
        for stem, frames in grouped.items():
            frames.sort(key=lambda item: item[0])
            clips.append(
                {
                    "stem": stem,
                    "folder": folder,
                    "json_paths": [p for _, p in frames],
                    "n_frames": len(frames),
                }
            )
    return clips


def load_openpose_index(root: Path | None = None) -> "pd.DataFrame":
    import pandas as pd

    from src.corpora import _video_id

    root = fragkiadakis_root(root)
    rows = []
    for clip in list_openpose_clips(root):
        gloss = clip["folder"].name
        rows.append(
            {
                "video_id": _video_id("fragkiadakis", clip["folder"], root),
                "path": str(clip["folder"].resolve()),
                "signer_id": "nyarko",
                "sentence_id": gloss,
                "sentence_text": gloss.replace("_", " "),
                "category": "lexicon",
                "source": "fragkiadakis",
                "split": "ssl",
                "pose_source": "openpose",
                "n_json": clip["n_frames"],
            }
        )
    return pd.DataFrame(rows)


def sequence_from_jsons(json_paths: list[Path]) -> np.ndarray:
    frames = [flatten_frame(load_frame_json(p)) for p in json_paths]
    if not frames:
        return np.zeros((0, FEATURE_DIM), dtype=np.float32)
    seq = np.stack(frames, axis=0)
    return normalize_sequence(seq)


def convert_openpose_index(
    index,
    *,
    skip_existing: bool = True,
) -> list[Path]:
    """Write one normalized .npz per OpenPose clip. Returns paths written or skipped."""
    by_folder: dict[Path, list[dict]] = {}
    for clip in list_openpose_clips():
        by_folder.setdefault(clip["folder"].resolve(), []).append(clip)

    written: list[Path] = []
    jobs = []
    for _, row in index.iterrows():
        video_id = str(row["video_id"])
        out = landmark_out_path(video_id)
        if skip_existing and out.exists():
            written.append(out)
            continue
        jobs.append(row)
    print(f"[openpose] already done={len(index) - len(jobs)}  remaining={len(jobs)}")

    for row in tqdm(jobs, desc="openpose -> npz"):
        folder = Path(str(row["path"])).resolve()
        video_id = str(row["video_id"])
        candidates = by_folder.get(folder, [])
        clip = candidates[0] if candidates else None
        if clip is None:
            print(f"[openpose] skip {video_id}: no JSON clip in {folder}")
            continue
        seq = sequence_from_jsons(clip["json_paths"])
        written.append(
            save_landmarks(
                video_id,
                seq,
                extra={
                    "signer_id": np.array(str(row["signer_id"])),
                    "sentence_id": np.array(str(row["sentence_id"])),
                    "pose_source": np.array("openpose"),
                },
            )
        )
    meta = {
        "feature_dim": FEATURE_DIM,
        "n_videos": len(written),
        "pose_source": "openpose",
        "body": "BODY_25_mapped_to_mediapipe_pose33",
        "face": "absent",
    }
    (PATHS.landmarks / "openpose_conversion_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return written
