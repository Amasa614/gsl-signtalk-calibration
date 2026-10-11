"""Landmark-space synthetic signer diversity (RQ5).

Cheap approximation of avatar-driven synthesis: perturb already-normalized
pose windows so each real signer spawns a few internally consistent
"pseudo-signers." No L/R mirroring (phonological).

Perturbation axes (applied as a fixed profile per pseudo-signer):
  - proportion: rescale limb-segment lengths
  - speed: temporal warp via interpolation
  - jitter: small per-joint Gaussian noise (calibrated, consistent seed)
  - combined: all of the above
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import FEATURE_DIM, LEFT_SHOULDER, N_COORDS, N_HAND, N_JOINTS, N_POSE, RIGHT_SHOULDER

# MediaPipe Pose edges used for body-proportion scaling (parent -> child).
# Indices are within the packed [pose | L-hand | R-hand | face] layout.
_POSE_BONES: tuple[tuple[int, int], ...] = (
    (11, 13),  # L shoulder -> elbow
    (13, 15),  # L elbow -> wrist
    (12, 14),  # R shoulder -> elbow
    (14, 16),  # R elbow -> wrist
    (11, 23),  # L shoulder -> hip
    (12, 24),  # R shoulder -> hip
    (23, 25),  # L hip -> knee
    (25, 27),  # L knee -> ankle
    (24, 26),  # R hip -> knee
    (26, 28),  # R knee -> ankle
)

# Hand root (wrist) within each hand block; scale tips relative to wrist.
_HAND_ROOT = 0  # wrist in MediaPipe Hands
_HAND_TIPS = (4, 8, 12, 16, 20)


@dataclass(frozen=True)
class PerturbProfile:
    """One consistent 'body' / signing style for a pseudo-signer."""

    name: str
    limb_scale: float = 1.0
    hand_span_scale: float = 1.0
    speed: float = 1.0  # >1 = faster (compress time), <1 = slower
    jitter_std: float = 0.0
    seed: int = 0


def default_profiles(n: int = 4, axis: str = "combined", base_seed: int = 0) -> list[PerturbProfile]:
    """Build ``n`` distinct profiles for a given ablation axis."""
    limb_opts = [0.92, 0.96, 1.04, 1.08, 1.12]
    hand_opts = [0.90, 0.95, 1.05, 1.10, 1.15]
    speed_opts = [0.85, 0.92, 1.08, 1.15, 1.25]
    jitter_opts = [0.004, 0.006, 0.008, 0.010, 0.012]
    profiles = []
    for i in range(n):
        limb = limb_opts[i % len(limb_opts)]
        hand = hand_opts[i % len(hand_opts)]
        speed = speed_opts[i % len(speed_opts)]
        jitter = jitter_opts[i % len(jitter_opts)]
        if axis == "proportion":
            profiles.append(
                PerturbProfile(
                    name=f"prop{i}",
                    limb_scale=limb,
                    hand_span_scale=hand,
                    seed=base_seed + 17 * i,
                )
            )
        elif axis == "speed":
            profiles.append(
                PerturbProfile(name=f"spd{i}", speed=speed, seed=base_seed + 17 * i)
            )
        elif axis == "jitter":
            profiles.append(
                PerturbProfile(name=f"jit{i}", jitter_std=jitter, seed=base_seed + 17 * i)
            )
        elif axis == "combined":
            profiles.append(
                PerturbProfile(
                    name=f"syn{i}",
                    limb_scale=limb,
                    hand_span_scale=hand,
                    speed=speed,
                    jitter_std=jitter,
                    seed=base_seed + 17 * i,
                )
            )
        else:
            raise ValueError(f"unknown axis {axis}")
    return profiles


def _as_joints(seq: np.ndarray) -> np.ndarray:
    return seq.reshape(seq.shape[0], N_JOINTS, N_COORDS).astype(np.float32)


def _flatten(joints: np.ndarray) -> np.ndarray:
    return joints.reshape(joints.shape[0], FEATURE_DIM).astype(np.float32)


def scale_limb_proportions(
    seq: np.ndarray,
    limb_scale: float,
    hand_span_scale: float,
) -> np.ndarray:
    """Rescale selected bone lengths about the parent joint; shoulders fixed."""
    if abs(limb_scale - 1.0) < 1e-6 and abs(hand_span_scale - 1.0) < 1e-6:
        return seq.astype(np.float32)
    joints = _as_joints(seq).copy()
    for parent, child in _POSE_BONES:
        vec = joints[:, child] - joints[:, parent]
        joints[:, child] = joints[:, parent] + limb_scale * vec
    # Hands: scale tip offsets from wrist within each hand block.
    for hand_i, span in ((0, hand_span_scale), (1, hand_span_scale)):
        base = N_POSE + hand_i * N_HAND
        root = base + _HAND_ROOT
        for tip in _HAND_TIPS:
            j = base + tip
            vec = joints[:, j] - joints[:, root]
            joints[:, j] = joints[:, root] + span * vec
    # Re-anchor to shoulders so torso frame stays comparable.
    left = joints[:, LEFT_SHOULDER]
    right = joints[:, RIGHT_SHOULDER]
    origin = 0.5 * (left + right)
    width = np.linalg.norm(left - right, axis=-1, keepdims=True)
    width = np.clip(width, 1e-6, None)
    joints = (joints - origin[:, None, :]) / width[:, None, :]
    return _flatten(joints)


def temporal_warp(seq: np.ndarray, speed: float) -> np.ndarray:
    """Resample time so ``speed`` > 1 is faster signing and ``speed`` < 1 is slower.

    Output length always matches the input window length.
    """
    if abs(speed - 1.0) < 1e-6:
        return seq.astype(np.float32)
    t = seq.shape[0]
    speed = float(np.clip(speed, 0.5, 2.0))
    # Source span used before resampling back to t.
    src_len = int(round(t / speed))
    src_len = int(np.clip(src_len, 2, max(t * 2, 2)))
    if src_len <= t:
        start = (t - src_len) // 2
        crop = seq[start : start + src_len]
    else:
        # Slower: stretch the full clip then we resample down to t via linspace.
        idx_up = np.linspace(0, t - 1, src_len)
        lo = np.floor(idx_up).astype(int)
        hi = np.clip(lo + 1, 0, t - 1)
        w = (idx_up - lo).astype(np.float32)[:, None]
        crop = ((1.0 - w) * seq[lo] + w * seq[hi]).astype(np.float32)
    idx = np.linspace(0, crop.shape[0] - 1, t)
    lo = np.floor(idx).astype(int)
    hi = np.clip(lo + 1, 0, crop.shape[0] - 1)
    w = (idx - lo).astype(np.float32)[:, None]
    return ((1.0 - w) * crop[lo] + w * crop[hi]).astype(np.float32)


def consistent_jitter(seq: np.ndarray, std: float, seed: int) -> np.ndarray:
    """Per-joint Gaussian noise with a fixed RNG seed (same 'tremor' style)."""
    if std <= 0:
        return seq.astype(np.float32)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, std, size=seq.shape).astype(np.float32)
    return (seq + noise).astype(np.float32)


def apply_profile(seq: np.ndarray, profile: PerturbProfile) -> np.ndarray:
    out = seq.astype(np.float32)
    out = scale_limb_proportions(out, profile.limb_scale, profile.hand_span_scale)
    out = temporal_warp(out, profile.speed)
    out = consistent_jitter(out, profile.jitter_std, profile.seed)
    return out
