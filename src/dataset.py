"""Datasets for SSL pretraining (labels ignored) and labeled fine-tuning."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import CFG, FEATURE_DIM
from src.normalize import frame_dropout, spatial_jitter, temporal_crop
from src.splits import load_label_map, pad_window


def _read_window(row, window_size: int) -> np.ndarray:
    with np.load(row["npz_path"], allow_pickle=False) as data:
        seq = data["landmarks"].astype(np.float32)
    start = int(row["start_frame"])
    end = int(row["end_frame"])
    chunk = seq[start:end]
    return pad_window(chunk, window_size)


class SSLWindowDataset(Dataset):
    """Every extracted landmark window, ignoring the sentence label.

    This is the unlabeled pretraining corpus for RQ1.
    """

    def __init__(
        self,
        table: pd.DataFrame | Path,
        window_size: int | None = None,
        augment: bool = True,
        seed: int = 42,
    ):
        self.table = pd.read_csv(table) if isinstance(table, Path) else table.reset_index(drop=True)
        self.window_size = window_size or CFG.window.window_size
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.table)

    def _augment(self, seq: np.ndarray) -> np.ndarray:
        seq = temporal_crop(seq, rng=self.rng)
        seq = frame_dropout(seq, rng=self.rng)
        seq = spatial_jitter(seq, rng=self.rng)
        return seq

    def __getitem__(self, idx: int):
        row = self.table.iloc[idx]
        seq = _read_window(row, self.window_size)
        if seq.shape[-1] != FEATURE_DIM:
            raise ValueError(f"expected feature_dim={FEATURE_DIM}, got {seq.shape[-1]}")
        view_a = self._augment(seq) if self.augment else seq
        view_b = self._augment(seq) if self.augment else seq
        return {
            "x": torch.from_numpy(seq),
            "view_a": torch.from_numpy(view_a.copy()),
            "view_b": torch.from_numpy(view_b.copy()),
            "video_id": str(row["video_id"]),
        }


class LabeledWindowDataset(Dataset):
    """Windows with class labels for fine-tuning / evaluation.

    `label_col` is `sentence_id` (RQ1 default, ~4,025-way) or `category`
    (coarse diagnostic, 26-way).
    """

    def __init__(
        self,
        table: pd.DataFrame | Path,
        label_map: dict[str, int] | None = None,
        window_size: int | None = None,
        drop_unseen: bool = True,
        label_col: str = "sentence_id",
    ):
        df = pd.read_csv(table) if isinstance(table, Path) else table.reset_index(drop=True)
        self.label_col = label_col
        self.label_map = label_map or load_label_map()
        df = df.copy()
        df["y"] = df[label_col].astype(str).map(self.label_map)
        if drop_unseen:
            n_drop = int(df["y"].isna().sum())
            if n_drop:
                print(
                    f"[dataset] dropping {n_drop} windows whose {label_col} is not in the train label map"
                )
            df = df.dropna(subset=["y"])
        self.table = df.reset_index(drop=True)
        self.window_size = window_size or CFG.window.window_size

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, idx: int):
        row = self.table.iloc[idx]
        seq = _read_window(row, self.window_size)
        return {
            "x": torch.from_numpy(seq),
            "y": torch.tensor(int(row["y"]), dtype=torch.long),
            "video_id": str(row["video_id"]),
            "signer_id": str(row["signer_id"]),
            "sentence_id": str(row["sentence_id"]),
            "category": str(row.get("category", "")),
        }
