"""Frozen Paper-1 Access revision protocol: IDs, seeds, hashes, candidate rules.

Reconstructs the historical RQ4 calibration selections from the same
``windows.csv`` order and seed formulas used by ``09_fewshot_calibration.py``
(five-draw) and ``14_rescore_rq4_metrics.py`` (single-draw / v2).
Historical JSONs store counts, not video IDs; reconstructed IDs are accepted
only when those counts match.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.adaptation import sample_calib_clips, split_calib_test
from src.config import CFG, FEATURE_DIM, PATHS
from src.train_utils import load_checkpoint

REVISION_ROOT = PATHS.results / "paper1_access_revision"
PROTOCOL_NAME = "paper1_access_revision_v1"
MAIN_BUDGETS = (1, 5, 20)
N_DRAWS = 5
ADAPT_SEEDS = (42, 43, 44)
SIGNERS = ("1", "2", "3", "4", "5")
CALIB_FRAC = 0.80
BASE_SEED = 42
ADAPT_EPOCHS = 8
UNFREEZE_LAYERS = 2
BATCH_SIZE = 32
LR = 1e-4
WEIGHT_DECAY = 1e-2
GRAD_CLIP = 1.0


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj) -> str:
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def load_signtalk_windows() -> pd.DataFrame:
    windows = pd.read_csv(PATHS.splits / "windows.csv")
    if "source" in windows.columns:
        windows = windows[
            windows["source"].astype(str).str.lower().eq("signtalk")
        ].reset_index(drop=True)
    windows["signer_id"] = windows["signer_id"].astype(str)
    windows["video_id"] = windows["video_id"].astype(str)
    windows["category"] = windows["category"].astype(str)
    return windows


def category_map(windows: pd.DataFrame | None = None) -> dict[str, int]:
    if windows is None:
        windows = load_signtalk_windows()
    labels = sorted(windows["category"].astype(str).unique().tolist())
    return {lab: i for i, lab in enumerate(labels)}


def id_to_category(label_map: dict[str, int]) -> dict[int, str]:
    return {int(v): str(k) for k, v in label_map.items()}


def split_seed(test_signer: str, seed: int = BASE_SEED) -> int:
    return int(seed) + 100 * int(test_signer)


def sample_seed_draws(
    test_signer: str, n: int, draw: int, seed: int = BASE_SEED
) -> int:
    """Seed used by ``09_fewshot_calibration.py`` for the five-draw grid."""
    return int(seed) + 1000 * int(test_signer) + 17 * int(n) + 997 * int(draw)


def sample_seed_v2(test_signer: str, n: int, seed: int = BASE_SEED) -> int:
    """Seed used by ``14_rescore_rq4_metrics.py`` (single-draw / Table B3)."""
    return int(seed) + 1000 * int(test_signer) + int(n)


def reconstruct_selection(
    windows: pd.DataFrame,
    *,
    test_signer: str,
    n_per_category: int,
    draw_id: int,
    family: str = "draws",
    seed: int = BASE_SEED,
    calib_frac: float = CALIB_FRAC,
) -> dict:
    """Replay split + sampling. ``family`` is ``draws`` or ``v2``."""
    test_all = windows[windows["signer_id"].eq(str(test_signer))].reset_index(drop=True)
    pool_seed = split_seed(test_signer, seed)
    calib_pool, final_test = split_calib_test(
        test_all, calib_frac=calib_frac, seed=pool_seed
    )
    if family == "v2":
        samp = sample_seed_v2(test_signer, n_per_category, seed)
    elif family == "draws":
        samp = sample_seed_draws(test_signer, n_per_category, draw_id, seed)
    else:
        raise ValueError(f"unknown family {family!r}")
    calib = sample_calib_clips(
        calib_pool, n_per_category=n_per_category, seed=samp
    )
    calib_ids = sorted(calib["video_id"].astype(str).unique().tolist())
    test_ids = sorted(final_test["video_id"].astype(str).unique().tolist())
    pool_ids = sorted(calib_pool["video_id"].astype(str).unique().tolist())
    cats = sorted(calib["category"].astype(str).unique().tolist()) if len(calib) else []
    clip_cats = (
        calib.drop_duplicates("video_id")[["video_id", "category"]]
        .sort_values("video_id")
        .values.tolist()
        if len(calib)
        else []
    )
    achieved = {}
    if len(calib):
        for cat, g in calib.drop_duplicates("video_id").groupby("category"):
            achieved[str(cat)] = int(g["video_id"].nunique())
    return {
        "target_signer": str(test_signer),
        "budget_per_category": int(n_per_category),
        "draw_id": int(draw_id),
        "family": family,
        "pool_seed": int(pool_seed),
        "sampling_seed": int(samp),
        "calibration_video_ids": calib_ids,
        "final_test_video_ids": test_ids,
        "calib_pool_video_ids": pool_ids,
        "clip_categories": [{"video_id": v, "category": c} for v, c in clip_cats],
        "calibration_candidate_categories": cats,
        "n_calib_clips": len(calib_ids),
        "n_calib_windows": int(len(calib)),
        "n_final_test_clips": len(test_ids),
        "n_final_test_windows": int(len(final_test)),
        "n_allowed_categories": len(cats),
        "achieved_clips_per_category": achieved,
        "requested_minus_achieved": {
            c: int(n_per_category) - n for c, n in achieved.items() if n < int(n_per_category)
        },
    }


def inspect_base_checkpoint(path: Path, test_signer: str) -> dict:
    ckpt = load_checkpoint(path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if not isinstance(state, dict):
        raise TypeError(f"unexpected checkpoint payload: {type(state)}")
    n_params = sum(int(np.prod(tuple(v.shape))) for v in state.values() if hasattr(v, "shape"))
    head = state.get("cls_head.weight")
    n_classes = int(head.shape[0]) if head is not None else None
    extra = {}
    if isinstance(ckpt, dict):
        for key in (
            "feature_dim",
            "d_model",
            "n_classes",
            "arch",
            "best_val_acc",
            "epoch",
            "tag",
            "label_col",
        ):
            if key in ckpt and not torch.is_tensor(ckpt[key]):
                extra[key] = ckpt[key]
        extra["payload_keys"] = sorted(str(k) for k in ckpt.keys())
    source = [s for s in SIGNERS if s != str(test_signer)]
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
        "target_signer": str(test_signer),
        "source_signers": source,
        "n_state_tensors": len(state),
        "n_parameters": n_params,
        "n_classes_from_head": n_classes,
        "feature_dim": extra.get("feature_dim", FEATURE_DIM),
        "model_config": {
            "arch": "transformer",
            "d_model": CFG.model.d_model,
            "nhead": CFG.model.nhead,
            "num_layers": CFG.model.num_layers,
            "dim_feedforward": CFG.model.dim_feedforward,
            "dropout": CFG.model.dropout,
        },
        "source_training_metadata": extra,
    }


def lookup_historical_counts(runs: list[dict], *, signer: str, n: int, draw: int, method: str) -> dict | None:
    for r in runs:
        if (
            str(r.get("test_signer")) == str(signer)
            and int(r.get("n_per_category", -1)) == int(n)
            and int(r.get("calib_draw", 0)) == int(draw)
            and str(r.get("method")) == method
            and str(r.get("arm", "scratch")) == "scratch"
        ):
            return r
    return None
