"""Per-frame landmark packing, torso centering, and shoulder-width scaling."""

from __future__ import annotations

import numpy as np

from src.config import (
    FACE_NMM_INDICES,
    FEATURE_DIM,
    LEFT_SHOULDER,
    N_COORDS,
    N_FACE,
    N_HAND,
    N_JOINTS,
    N_POSE,
    RIGHT_SHOULDER,
)


def empty_frame() -> np.ndarray:
    return np.zeros((N_JOINTS, N_COORDS), dtype=np.float32)


def _xyz(landmark) -> tuple[float, float, float]:
    return float(landmark.x), float(landmark.y), float(getattr(landmark, "z", 0.0) or 0.0)


def pack_landmarks(
    pose=None,
    left_hand=None,
    right_hand=None,
    face=None,
) -> np.ndarray:
    """Pack MediaPipe results into a (N_JOINTS, 3) array in a fixed joint order.

    Layout is [pose, left hand, right hand, face-NMM]. Methodology §3.2 writes
    the concatenation as [h^L, h^R, p, f]; this is the same keypoints, just a
    fixed permutation. Do not reorder without re-extracting landmarks.
    Missing detections stay at zero so the feature dimension is constant.
    """
    frame = empty_frame()
    offset = 0

    if pose:
        n = min(len(pose), N_POSE)
        for i in range(n):
            frame[offset + i] = _xyz(pose[i])
    offset += N_POSE

    if left_hand:
        n = min(len(left_hand), N_HAND)
        for i in range(n):
            frame[offset + i] = _xyz(left_hand[i])
    offset += N_HAND

    if right_hand:
        n = min(len(right_hand), N_HAND)
        for i in range(n):
            frame[offset + i] = _xyz(right_hand[i])
    offset += N_HAND

    if face:
        if len(face) >= max(FACE_NMM_INDICES) + 1:
            for j, idx in enumerate(FACE_NMM_INDICES):
                frame[offset + j] = _xyz(face[idx])
        else:
            n = min(len(face), N_FACE)
            for j in range(n):
                frame[offset + j] = _xyz(face[j])

    return frame


def flatten_frame(frame: np.ndarray) -> np.ndarray:
    return frame.reshape(-1).astype(np.float32)


def unflatten_frame(vec: np.ndarray) -> np.ndarray:
    return vec.reshape(N_JOINTS, N_COORDS)


def normalize_sequence(seq: np.ndarray) -> np.ndarray:
    """Center each frame on the shoulder midpoint and scale by shoulder width.

    `seq` is (T, N_JOINTS, 3) or (T, FEATURE_DIM). The model should see signing
    motion, not signer size or camera distance.
    """
    frames = seq.reshape(seq.shape[0], N_JOINTS, N_COORDS).astype(np.float32).copy()
    left = frames[:, LEFT_SHOULDER]
    right = frames[:, RIGHT_SHOULDER]
    origin = 0.5 * (left + right)
    width = np.linalg.norm(left - right, axis=-1, keepdims=True)
    width = np.clip(width, 1e-6, None)
    frames = (frames - origin[:, None, :]) / width[:, None, :]
    # Joints that were never detected (all zeros before normalize) stay zero.
    detected = np.any(np.abs(seq.reshape(seq.shape[0], N_JOINTS, N_COORDS)) > 0, axis=-1)
    frames[~detected] = 0.0
    return frames.reshape(seq.shape[0], FEATURE_DIM).astype(np.float32)


def spatial_jitter(seq: np.ndarray, std: float = 0.01, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    noise = rng.normal(0.0, std, size=seq.shape).astype(np.float32)
    return (seq + noise).astype(np.float32)


def frame_dropout(seq: np.ndarray, drop_p: float = 0.15, rng: np.random.Generator | None = None) -> np.ndarray:
    """Zero out random frames (temporal dropout). Do not mirror left/right."""
    rng = rng or np.random.default_rng()
    out = seq.copy()
    t = out.shape[0]
    mask = rng.random(t) < drop_p
    out[mask] = 0.0
    return out


def temporal_crop(seq: np.ndarray, min_keep: float = 0.7, rng: np.random.Generator | None = None) -> np.ndarray:
    """Crop a contiguous temporal span and resize back to the original length."""
    rng = rng or np.random.default_rng()
    t = seq.shape[0]
    keep = int(t * rng.uniform(min_keep, 1.0))
    keep = max(keep, 1)
    start = int(rng.integers(0, t - keep + 1))
    crop = seq[start : start + keep]
    idx = np.linspace(0, keep - 1, t)
    lo = np.floor(idx).astype(int)
    hi = np.clip(lo + 1, 0, keep - 1)
    w = (idx - lo).astype(np.float32)[:, None]
    return ((1.0 - w) * crop[lo] + w * crop[hi]).astype(np.float32)
