"""Visualize SSL frame/joint masks and reconstruction on a real SignTalk window."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.config import FEATURE_DIM, N_COORDS, N_FACE, N_HAND, N_JOINTS, N_POSE, PATHS
from src.model import MaskedPoseTransformer, make_frame_mask, make_joint_mask
from src.splits import pad_window
from src.train_utils import get_device, load_checkpoint

OUT = PATHS.results / "mask_viz"
OUT.mkdir(parents=True, exist_ok=True)

POSE_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (27, 29), (27, 31),
    (24, 26), (26, 28), (28, 30), (28, 32),
]
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def _xy(frame: np.ndarray, start: int, n: int) -> np.ndarray:
    return frame[start : start + n, :2].copy()


def _draw_skeleton(ax, frame: np.ndarray, color: str, alpha: float = 1.0, lw: float = 1.4):
    pose = _xy(frame, 0, N_POSE)
    lh = _xy(frame, N_POSE, N_HAND)
    rh = _xy(frame, N_POSE + N_HAND, N_HAND)
    face = _xy(frame, N_POSE + 2 * N_HAND, N_FACE)
    pose[:, 1] *= -1
    lh[:, 1] *= -1
    rh[:, 1] *= -1
    face[:, 1] *= -1
    for a, b in POSE_EDGES:
        ax.plot([pose[a, 0], pose[b, 0]], [pose[a, 1], pose[b, 1]], color=color, lw=lw, alpha=alpha)
    for a, b in HAND_EDGES:
        ax.plot([lh[a, 0], lh[b, 0]], [lh[a, 1], lh[b, 1]], color=color, lw=lw, alpha=alpha)
        ax.plot([rh[a, 0], rh[b, 0]], [rh[a, 1], rh[b, 1]], color=color, lw=lw, alpha=alpha)
    ax.scatter(pose[:, 0], pose[:, 1], s=8, c=color, alpha=alpha, zorder=3)
    ax.scatter(lh[:, 0], lh[:, 1], s=6, c=color, alpha=alpha, zorder=3)
    ax.scatter(rh[:, 0], rh[:, 1], s=6, c=color, alpha=alpha, zorder=3)
    ax.scatter(face[:, 0], face[:, 1], s=5, c=color, alpha=0.7 * alpha, zorder=3)
    ax.set_aspect("equal")
    ax.axis("off")


def main() -> None:
    npz_path = PATHS.landmarks / "10c.npz"
    if not npz_path.exists():
        npz_path = PATHS.landmarks / "1000a.npz"
    seq = np.load(npz_path)["landmarks"].astype(np.float32)
    window = pad_window(seq, 64)
    x = torch.from_numpy(window).unsqueeze(0)

    g = torch.Generator().manual_seed(0)
    frame_mask = make_frame_mask(x, 0.3, generator=g)
    joint_mask = make_joint_mask(x, 0.3, generator=torch.Generator().manual_seed(1))

    device = get_device()
    model = MaskedPoseTransformer(feature_dim=FEATURE_DIM).to(device)
    ckpt = load_checkpoint(PATHS.ssl_encoder, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    with torch.no_grad():
        recon = model.reconstruct(x.to(device), frame_mask=frame_mask.to(device)).cpu().numpy()[0]

    fm = frame_mask[0].numpy()
    jm = joint_mask[0].numpy().reshape(64, N_JOINTS, N_COORDS)[:, :, 0]

    # 1) frame-mask timeline
    fig, ax = plt.subplots(figsize=(10, 1.6))
    ax.imshow(fm[None, :], aspect="auto", cmap="coolwarm", vmin=0, vmax=1, interpolation="nearest")
    ax.set_yticks([])
    ax.set_xlabel("frame index in 64-frame window")
    ax.set_title(f"Frame mask (30%) on {npz_path.stem}  — red = masked / reconstructed")
    for i, flag in enumerate(fm):
        if flag:
            ax.text(i, 0, str(i), ha="center", va="center", fontsize=6, color="white")
    fig.tight_layout()
    fig.savefig(OUT / "frame_mask_timeline.png", dpi=160)
    plt.close()

    # 2) joint-mask heatmap
    fig, ax = plt.subplots(figsize=(10, 4.2))
    im = ax.imshow(jm.T, aspect="auto", cmap="coolwarm", vmin=0, vmax=1, interpolation="nearest")
    ax.axhline(N_POSE - 0.5, color="white", lw=0.6)
    ax.axhline(N_POSE + N_HAND - 0.5, color="white", lw=0.6)
    ax.axhline(N_POSE + 2 * N_HAND - 0.5, color="white", lw=0.6)
    ax.set_xlabel("frame")
    ax.set_ylabel("joint index")
    ax.set_title("Joint mask (30% of joints per frame)  — pose | left hand | right hand | face")
    ax.text(65.5, N_POSE / 2, "pose", va="center", fontsize=8)
    ax.text(65.5, N_POSE + N_HAND / 2, "L hand", va="center", fontsize=8)
    ax.text(65.5, N_POSE + N_HAND + N_HAND / 2, "R hand", va="center", fontsize=8)
    ax.text(65.5, N_POSE + 2 * N_HAND + N_FACE / 2, "face", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.08, label="masked")
    fig.tight_layout()
    fig.savefig(OUT / "joint_mask_heatmap.png", dpi=160)
    plt.close()

    # 3) skeletons: visible / masked input / reconstruction
    masked_idx = int(np.where(fm)[0][len(np.where(fm)[0]) // 2])
    visible_candidates = np.where(~fm)[0]
    vis_idx = int(visible_candidates[len(visible_candidates) // 2])
    frames_gt = window.reshape(64, N_JOINTS, N_COORDS)
    frames_re = recon.reshape(64, N_JOINTS, N_COORDS)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    _draw_skeleton(axes[0], frames_gt[vis_idx], "#1f4e79")
    axes[0].set_title(f"Unmasked frame {vis_idx}\n(model sees this)")
    _draw_skeleton(axes[1], frames_gt[masked_idx], "#9aa0a6", alpha=0.25, lw=1.0)
    axes[1].set_title(f"Masked frame {masked_idx}\n(replaced by mask token)")
    _draw_skeleton(axes[2], frames_re[masked_idx], "#b3261e")
    axes[2].set_title(f"Reconstruction of frame {masked_idx}\n(SSL encoder output)")
    fig.suptitle(f"Masked pose prediction  —  {npz_path.stem}.npz  window 0:64", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "skeleton_mask_vs_recon.png", dpi=160)
    plt.close()

    # 4) overlay GT vs recon on the masked frame
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    _draw_skeleton(ax, frames_gt[masked_idx], "#1f4e79", alpha=0.45, lw=1.2)
    _draw_skeleton(ax, frames_re[masked_idx], "#b3261e", alpha=0.95, lw=1.6)
    ax.set_title(f"Frame {masked_idx}: ground truth (blue) vs reconstruction (red)")
    fig.tight_layout()
    fig.savefig(OUT / "overlay_gt_vs_recon.png", dpi=160)
    plt.close()

    print(f"clip={npz_path.stem} T={seq.shape[0]} masked_frames={int(fm.sum())}/64")
    print(f"example masked frame indices: {np.where(fm)[0].tolist()}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
