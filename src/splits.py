"""Signer-independent splits and overlapping landmark windows."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CFG, PATHS
from src.landmark_extraction import landmark_out_path, load_landmarks


def assign_signer_splits(
    index: pd.DataFrame,
    test_signers: tuple[str, ...] | None = None,
    val_signers: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Assign train/val/test by signer ID. Never shuffle clips across signers."""
    out = index.copy()
    extra_mask = pd.Series(False, index=out.index)
    if "source" in out.columns:
        extra_mask = out["source"].astype(str).str.lower().ne("signtalk")
    labeled = out.loc[~extra_mask].copy()
    extra = out.loc[extra_mask].copy()

    if extra.empty:
        work = labeled if not labeled.empty else out
    else:
        work = labeled

    signers = sorted(work["signer_id"].astype(str).unique().tolist()) if not work.empty else []
    if len(signers) < 2:
        print(
            "[split] WARNING: fewer than 2 signers. This is NOT a signer-independent "
            "split. Use it only to debug the pipeline on a subset."
        )

    test = list(test_signers) if test_signers else list(CFG.split.test_signers)
    val = list(val_signers) if val_signers else list(CFG.split.val_signers)

    if not test:
        test = [signers[-1]] if signers else []
    if not val:
        remaining = [s for s in signers if s not in test]
        val = [remaining[-1]] if remaining else []

    test_set = {str(s) for s in test}
    val_set = {str(s) for s in val}
    overlap = test_set & val_set
    if overlap:
        raise ValueError(f"test and val signers overlap: {overlap}")

    def _split(signer: str) -> str:
        signer = str(signer)
        if signer in test_set:
            return "test"
        if signer in val_set:
            return "val"
        return "train"

    work["split"] = work["signer_id"].map(_split)
    if not extra.empty:
        extra["split"] = "ssl"
        out = pd.concat([work, extra], ignore_index=True)
    else:
        out = work
    print(
        "[split] train signers="
        f"{sorted(out.loc[out.split=='train','signer_id'].astype(str).unique().tolist())}  "
        f"val={sorted(val_set)}  test={sorted(test_set)}"
        + (f"  ssl={int((out.split=='ssl').sum())} clips" if "ssl" in set(out["split"]) else "")
    )
    return out


def holdout_clips_per_signer(
    table: pd.DataFrame,
    val_frac: float = 0.20,
    rng: np.random.Generator | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out ``val_frac`` of each training signer's clips for validation.

    Split is by ``video_id``, not window, so overlapping windows of the same
    clip never appear in both train and val. A whole extra signer is not used,
    so k=1 still has a non-empty val set and the test-signer boundary stays clean.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    empty = table.iloc[0:0].copy()
    if table.empty or val_frac <= 0:
        return table.copy(), empty
    val_ids: set[str] = set()
    train_ids: set[str] = set()
    for signer in sorted(table["signer_id"].astype(str).unique().tolist()):
        clips = table.loc[table["signer_id"].astype(str).eq(signer), "video_id"].astype(str).unique()
        clips = clips[rng.permutation(len(clips))]
        n = len(clips)
        if n < 2:
            train_ids.update(clips.tolist())
            continue
        n_val = int(round(n * val_frac))
        n_val = min(max(n_val, 1), n - 1)
        val_ids.update(clips[:n_val].tolist())
        train_ids.update(clips[n_val:].tolist())
    vid = table["video_id"].astype(str)
    train = table.loc[vid.isin(train_ids)].reset_index(drop=True)
    val = table.loc[vid.isin(val_ids)].reset_index(drop=True)
    return train, val


def _windows_for_length(n_frames: int, window_size: int, stride: int, min_frames: int) -> list[tuple[int, int]]:
    if n_frames < min_frames:
        return []
    if n_frames <= window_size:
        return [(0, n_frames)]
    starts = list(range(0, n_frames - window_size + 1, stride))
    last = n_frames - window_size
    if starts[-1] != last:
        starts.append(last)
    return [(s, s + window_size) for s in starts]


def build_windows(
    index: pd.DataFrame,
    window_size: int | None = None,
    stride: int | None = None,
) -> pd.DataFrame:
    window_size = window_size or CFG.window.window_size
    stride = stride or CFG.window.stride
    min_frames = CFG.window.min_frames
    rows = []
    skipped = 0
    for _, clip in index.iterrows():
        npz = landmark_out_path(str(clip["video_id"]))
        if not npz.exists():
            skipped += 1
            continue
        seq = load_landmarks(npz)
        n_frames = int(seq.shape[0])
        for w_idx, (start, end) in enumerate(
            _windows_for_length(n_frames, window_size, stride, min_frames)
        ):
            rows.append(
                {
                    "video_id": clip["video_id"],
                    "path": clip.get("path", ""),
                    "npz_path": str(npz),
                    "signer_id": clip["signer_id"],
                    "sentence_id": clip["sentence_id"],
                    "sentence_text": clip.get("sentence_text", ""),
                    "category": clip.get("category", ""),
                    "source": clip.get("source", "signtalk"),
                    "split": clip.get("split", ""),
                    "n_frames": n_frames,
                    "window_idx": w_idx,
                    "start_frame": start,
                    "end_frame": end,
                    "window_size": window_size,
                }
            )
    if skipped:
        print(f"[windows] skipped {skipped} clips with no landmark file")
    windows = pd.DataFrame(rows)
    PATHS.splits.mkdir(parents=True, exist_ok=True)
    windows.to_csv(PATHS.splits / "windows.csv", index=False)
    for split in ("train", "val", "test", "ssl"):
        part = windows[windows["split"] == split] if "split" in windows.columns else windows
        part.to_csv(PATHS.splits / f"{split}.csv", index=False)
    return windows


def build_label_map(
    windows: pd.DataFrame,
    column: str = "sentence_id",
    dest: Path | None = None,
) -> dict[str, int]:
    """Map label strings -> class index using TRAIN SignTalk rows only."""
    train = windows[windows["split"] == "train"] if "split" in windows else windows
    if "source" in train.columns:
        train = train[train["source"].astype(str).str.lower().eq("signtalk")]
    labels = sorted(train[column].astype(str).unique().tolist())
    mapping = {lab: i for i, lab in enumerate(labels)}
    PATHS.splits.mkdir(parents=True, exist_ok=True)
    path = dest or (PATHS.category_map if column == "category" else PATHS.label_map)
    path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    return mapping


def load_label_map(path: Path | None = None) -> dict[str, int]:
    path = path or PATHS.label_map
    return json.loads(path.read_text(encoding="utf-8"))


def pad_window(seq: np.ndarray, window_size: int) -> np.ndarray:
    t, d = seq.shape
    if t == window_size:
        return seq.astype(np.float32)
    if t > window_size:
        return seq[:window_size].astype(np.float32)
    out = np.zeros((window_size, d), dtype=np.float32)
    out[:t] = seq
    return out
