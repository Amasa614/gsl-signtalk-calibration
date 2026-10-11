"""Fine-tune-time augmentations for RQ6 (not a separate SSL stage).

- Frame replacement: splice frames from another rendition of the same
  category (different clip), a lightweight non-generative pose stitch.
- Mask substitution: zero / replace joint channels on a random subset of
  frames during supervised training (regularizer, not pretext).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import CFG, FEATURE_DIM, N_COORDS, N_JOINTS
from src.dataset import _read_window
from src.model import make_joint_mask
from src.splits import load_label_map


class AugmentedLabeledWindowDataset(Dataset):
    """Labeled windows with optional frame replacement at load time.

    Mask substitution is applied in the training loop (needs a torch Generator
    and the model's input mask token), not here.
    """

    def __init__(
        self,
        table: pd.DataFrame | str,
        label_map: dict[str, int] | None = None,
        *,
        label_col: str = "category",
        window_size: int | None = None,
        frame_replace_prob: float = 0.0,
        frame_replace_frac: float = 0.25,
        seed: int = 42,
        drop_unseen: bool = True,
    ):
        df = pd.read_csv(table) if not isinstance(table, pd.DataFrame) else table.reset_index(drop=True)
        self.label_col = label_col
        self.label_map = label_map or load_label_map()
        df = df.copy()
        df["y"] = df[label_col].astype(str).map(self.label_map)
        if drop_unseen:
            df = df.dropna(subset=["y"])
        self.table = df.reset_index(drop=True)
        self.window_size = window_size or CFG.window.window_size
        self.frame_replace_prob = float(frame_replace_prob)
        self.frame_replace_frac = float(frame_replace_frac)
        self.rng = np.random.default_rng(seed)

        self._by_label: dict[int, list[int]] = defaultdict(list)
        for i, y in enumerate(self.table["y"].astype(int).tolist()):
            self._by_label[y].append(i)

    def __len__(self) -> int:
        return len(self.table)

    def _maybe_replace_frames(self, seq: np.ndarray, idx: int, y: int) -> np.ndarray:
        if self.frame_replace_prob <= 0 or self.rng.random() >= self.frame_replace_prob:
            return seq
        donors = [j for j in self._by_label.get(y, []) if j != idx]
        if not donors:
            return seq
        donor_idx = int(self.rng.choice(donors))
        donor = _read_window(self.table.iloc[donor_idx], self.window_size)
        t = seq.shape[0]
        span = max(1, int(round(t * self.frame_replace_frac)))
        start = int(self.rng.integers(0, max(t - span + 1, 1)))
        out = seq.copy()
        out[start : start + span] = donor[start : start + span]
        return out

    def __getitem__(self, idx: int):
        row = self.table.iloc[idx]
        seq = _read_window(row, self.window_size)
        y = int(row["y"])
        if self.frame_replace_prob > 0:
            seq = self._maybe_replace_frames(seq, idx, y)
        return {
            "x": torch.from_numpy(seq),
            "y": torch.tensor(y, dtype=torch.long),
            "video_id": str(row["video_id"]),
            "signer_id": str(row["signer_id"]),
            "sentence_id": str(row["sentence_id"]),
            "category": str(row.get("category", "")),
        }


def apply_mask_substitution(
    x: torch.Tensor,
    *,
    prob: float = 0.5,
    mask_ratio_range: tuple[float, float] = (0.10, 0.30),
    mode: str = "zero",
    mask_token: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Randomly mask joints on a subset of the batch (training regularizer).

    mode='zero' zeros masked coords; mode='token' replaces with ``mask_token``.
    """
    if prob <= 0:
        return x
    b = x.size(0)
    device = x.device
    if generator is None:
        apply = torch.rand(b, device=device) < prob
    else:
        apply = torch.rand(b, device=device, generator=generator) < prob
    if not apply.any():
        return x
    mask = make_joint_mask(x, mask_ratio=None, ratio_range=mask_ratio_range, generator=generator)
    mask = mask & apply.view(b, 1, 1)
    if mode == "token" and mask_token is not None:
        token = mask_token.to(device=device, dtype=x.dtype)
        if token.dim() == 3 and token.size(0) == 1:
            token = token.expand_as(x)
        return torch.where(mask, token.expand_as(x), x)
    return torch.where(mask, torch.zeros_like(x), x)


class SyntheticProfileDataset(Dataset):
    """Apply a fixed ``PerturbProfile`` to every window (RQ5 pseudo-signer)."""

    def __init__(
        self,
        table: pd.DataFrame,
        label_map: dict[str, int],
        profile,
        *,
        label_col: str = "category",
        window_size: int | None = None,
        signer_id_override: str | None = None,
    ):
        from src.synthetic_signers import apply_profile

        self._apply = apply_profile
        self.profile = profile
        self.label_col = label_col
        self.label_map = label_map
        df = table.copy()
        df["y"] = df[label_col].astype(str).map(self.label_map)
        df = df.dropna(subset=["y"]).reset_index(drop=True)
        self.table = df
        self.window_size = window_size or CFG.window.window_size
        self.signer_id_override = signer_id_override

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, idx: int):
        row = self.table.iloc[idx]
        seq = _read_window(row, self.window_size)
        seq = self._apply(seq, self.profile)
        signer = self.signer_id_override or f"{row['signer_id']}__{self.profile.name}"
        return {
            "x": torch.from_numpy(seq),
            "y": torch.tensor(int(row["y"]), dtype=torch.long),
            "video_id": f"{row['video_id']}__{self.profile.name}",
            "signer_id": str(signer),
            "sentence_id": str(row["sentence_id"]),
            "category": str(row.get("category", "")),
        }
