"""RQ1 validity: sentence-label overlap and restricted evaluation.

Addresses the review concern that near-zero SI sentence accuracy may conflate
signer transfer with evaluating on sentence classes never seen in training.

Reports:
  - train / val / test sentence-ID counts and intersections
  - % of test clips whose sentence appears in train
  - training support (clips + windows) per test sentence
  - RQ1 accuracy on ALL test vs ONLY sentences with train support
  - optional min-support thresholds (e.g. >=2 train clips)
  - window-level and clip-level metrics

    python -u scripts/13_rq1_sentence_overlap.py
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import FEATURE_DIM, PATHS
from src.dataset import LabeledWindowDataset
from src.metrics import pool_clips, predict_loader, topk_accuracy
from src.model import MaskedPoseTransformer
from src.splits import load_label_map
from src.train_utils import get_device, load_checkpoint, write_json


def _clip_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per video_id (first window metadata)."""
    return (
        df.sort_values(["video_id", "window_idx"] if "window_idx" in df.columns else ["video_id"])
        .drop_duplicates("video_id")
        .reset_index(drop=True)
    )


def _set_stats(name: str, sents: set[str], clips: pd.DataFrame) -> dict:
    return {
        "split": name,
        "n_windows": int(len(clips)) if "window_idx" in clips.columns else None,
        "n_clips": int(clips["video_id"].nunique()) if "video_id" in clips.columns else int(len(clips)),
        "n_sentence_ids": len(sents),
        "n_signers": int(clips["signer_id"].nunique()) if "signer_id" in clips.columns else None,
    }


def analyze_overlap(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> dict:
    train_clips = _clip_table(train)
    val_clips = _clip_table(val)
    test_clips = _clip_table(test)
    for df in (train_clips, val_clips, test_clips, train, val, test):
        df["sentence_id"] = df["sentence_id"].astype(str)
        if "video_id" in df.columns:
            df["video_id"] = df["video_id"].astype(str)

    train_s = set(train_clips["sentence_id"])
    val_s = set(val_clips["sentence_id"])
    test_s = set(test_clips["sentence_id"])

    inter_tv = train_s & val_s
    inter_tt = train_s & test_s
    inter_vt = val_s & test_s
    inter_all = train_s & val_s & test_s

    # Support: train clips / windows per sentence
    train_clip_support = (
        train_clips.groupby("sentence_id")["video_id"].nunique().astype(int).to_dict()
    )
    train_win_support = train.groupby("sentence_id").size().astype(int).to_dict()

    test_clip_sents = test_clips["sentence_id"].astype(str)
    in_train = test_clip_sents.isin(train_s)
    support_counts = test_clip_sents.map(lambda s: int(train_clip_support.get(s, 0)))

    support_hist = Counter(support_counts.tolist())
    support_hist_sorted = {str(k): int(v) for k, v in sorted(support_hist.items())}

    # Per-test-sentence table (clip-level)
    rows = []
    for sid, g in test_clips.groupby(test_clip_sents):
        rows.append(
            {
                "sentence_id": sid,
                "n_test_clips": int(len(g)),
                "in_train": bool(sid in train_s),
                "n_train_clips": int(train_clip_support.get(sid, 0)),
                "n_train_windows": int(train_win_support.get(sid, 0)),
            }
        )
    support_df = pd.DataFrame(rows).sort_values(
        ["in_train", "n_train_clips", "sentence_id"], ascending=[False, False, True]
    )

    payload = {
        "splits": {
            "train": _set_stats("train", train_s, train),
            "val": _set_stats("val", val_s, val),
            "test": _set_stats("test", test_s, test),
        },
        "intersections": {
            "train_cap_val": len(inter_tv),
            "train_cap_test": len(inter_tt),
            "val_cap_test": len(inter_vt),
            "train_cap_val_cap_test": len(inter_all),
            "test_only": len(test_s - train_s - val_s),
            "test_minus_train": len(test_s - train_s),
            "frac_test_sentence_ids_in_train": float(len(inter_tt) / max(len(test_s), 1)),
        },
        "test_clip_coverage": {
            "n_test_clips": int(len(test_clips)),
            "n_test_clips_sentence_in_train": int(in_train.sum()),
            "frac_test_clips_sentence_in_train": float(in_train.mean()),
            "n_test_clips_sentence_not_in_train": int((~in_train).sum()),
            "mean_train_clips_per_test_clip_sentence": float(support_counts.mean()),
            "median_train_clips_per_test_clip_sentence": float(support_counts.median()),
            "train_clip_support_histogram_over_test_clips": support_hist_sorted,
        },
    }
    return payload, support_df, train_s, inter_tt


def _load_model(path: Path, n_classes: int, device):
    model = MaskedPoseTransformer(feature_dim=FEATURE_DIM, n_classes=n_classes).to(device)
    ckpt = load_checkpoint(path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    return model


def _score_subset(raw: dict, mask: np.ndarray) -> dict:
    """Score window-level predictions on a boolean mask; also clip-pool that subset."""
    if mask.sum() == 0:
        return {
            "n_windows": 0,
            "n_clips": 0,
            "window_top1": 0.0,
            "window_top5": 0.0,
            "clip_top1": 0.0,
            "clip_top5": 0.0,
        }
    idx = np.where(mask)[0]
    logits = raw["logits"][idx]
    y = raw["y"][idx]
    sub = {
        "logits": logits,
        "y": y,
        "signer_id": [raw["signer_id"][i] for i in idx],
        "sentence_id": [raw["sentence_id"][i] for i in idx],
        "category": [raw["category"][i] for i in idx],
        "video_id": [raw["video_id"][i] for i in idx],
    }
    pooled = pool_clips(sub)
    return {
        "n_windows": int(mask.sum()),
        "n_clips": int(pooled["logits"].size(0)),
        "window_top1": topk_accuracy(logits, y, k=1),
        "window_top5": topk_accuracy(logits, y, k=5),
        "clip_top1": topk_accuracy(pooled["logits"], pooled["y"], k=1),
        "clip_top5": topk_accuracy(pooled["logits"], pooled["y"], k=5),
    }


def evaluate_restricted(
    train: pd.DataFrame,
    test: pd.DataFrame,
    label_map: dict[str, int],
    train_sentences: set[str],
    device,
    *,
    scratch_ckpt: Path,
    pretrained_ckpt: Path,
    batch_size: int,
    min_train_clips: list[int],
) -> dict:
    train_clip_support = (
        _clip_table(train).groupby("sentence_id")["video_id"].nunique().astype(int).to_dict()
    )
    ds = LabeledWindowDataset(test, label_map=label_map, label_col="sentence_id", drop_unseen=True)
    # After drop_unseen, only sentences in label_map remain (train-built map).
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    n_classes = len(label_map)
    print(f"[eval] test windows kept after drop_unseen={len(ds)}  classes={n_classes}")

    out = {"n_test_windows_in_label_map": len(ds), "arms": {}}
    for arm, ckpt in (("scratch", scratch_ckpt), ("pretrained", pretrained_ckpt)):
        if not ckpt.exists():
            print(f"[eval] missing {ckpt}")
            continue
        model = _load_model(ckpt, n_classes, device)
        raw = predict_loader(model, loader, device)
        sents = np.array([str(s) for s in raw["sentence_id"]])
        support = np.array([int(train_clip_support.get(s, 0)) for s in sents])

        # All kept windows (already in label_map = train sentences by construction of map,
        # but val-only sentences could theoretically be in test if map included them —
        # our map is train-only, so "all" ≈ in_train).
        masks = {
            "all_in_label_map": np.ones(len(sents), dtype=bool),
            "sentence_in_train": np.array([s in train_sentences for s in sents]),
        }
        for m in min_train_clips:
            masks[f"train_clips_ge_{m}"] = support >= m

        arm_scores = {}
        for name, mask in masks.items():
            arm_scores[name] = _score_subset(raw, mask)
            sc = arm_scores[name]
            print(
                f"  [{arm}] {name}: windows={sc['n_windows']} clips={sc['n_clips']}  "
                f"win top1={100*sc['window_top1']:.2f}% top5={100*sc['window_top5']:.2f}%  "
                f"clip top1={100*sc['clip_top1']:.2f}% top5={100*sc['clip_top5']:.2f}%"
            )
        out["arms"][arm] = arm_scores
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scratch-ckpt", type=str, default=str(PATHS.finetune_scratch))
    p.add_argument("--pretrained-ckpt", type=str, default=str(PATHS.finetune_pretrained))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--min-train-clips",
        type=str,
        default="1,2,3",
        help="comma list of min train-clip support thresholds for restricted eval",
    )
    p.add_argument(
        "--out",
        type=str,
        default=str(PATHS.results / "rq1_sentence_overlap.json"),
    )
    args = p.parse_args()
    mins = [int(x) for x in args.min_train_clips.split(",") if x.strip()]

    train = pd.read_csv(PATHS.splits / "train.csv")
    val = pd.read_csv(PATHS.splits / "val.csv")
    test = pd.read_csv(PATHS.splits / "test.csv")
    # SignTalk only
    for name, df in (("train", train), ("val", val), ("test", test)):
        if "source" in df.columns:
            df = df[df["source"].astype(str).str.lower().eq("signtalk")].reset_index(drop=True)
            if name == "train":
                train = df
            elif name == "val":
                val = df
            else:
                test = df

    overlap, support_df, train_s, inter_tt = analyze_overlap(train, val, test)
    support_csv = PATHS.results / "rq1_sentence_support.csv"
    support_df.to_csv(support_csv, index=False)

    print("=== RQ1 sentence-ID overlap ===")
    for split, slot in overlap["splits"].items():
        print(
            f"  {split}: clips={slot['n_clips']} windows={slot['n_windows']} "
            f"sentences={slot['n_sentence_ids']} signers={slot['n_signers']}"
        )
    ix = overlap["intersections"]
    print(
        f"  train AND test sentences={ix['train_cap_test']}  "
        f"(|test|={overlap['splits']['test']['n_sentence_ids']})  "
        f"frac sentence IDs in train={100*ix['frac_test_sentence_ids_in_train']:.1f}%"
    )
    print(f"  test sentences NOT in train={ix['test_minus_train']}")
    tc = overlap["test_clip_coverage"]
    print(
        f"  test clips with sentence in train: "
        f"{tc['n_test_clips_sentence_in_train']}/{tc['n_test_clips']} "
        f"({100*tc['frac_test_clips_sentence_in_train']:.1f}%)"
    )
    print(
        f"  mean/median train clips supporting each test clip's sentence: "
        f"{tc['mean_train_clips_per_test_clip_sentence']:.2f} / "
        f"{tc['median_train_clips_per_test_clip_sentence']:.2f}"
    )
    print(f"wrote {support_csv}")

    label_map = load_label_map()
    device = get_device()
    print(f"\n=== Restricted RQ1 re-eval (device={device}) ===")
    eval_payload = evaluate_restricted(
        train,
        test,
        label_map,
        train_s,
        device,
        scratch_ckpt=Path(args.scratch_ckpt),
        pretrained_ckpt=Path(args.pretrained_ckpt),
        batch_size=args.batch_size,
        min_train_clips=mins,
    )

    # Label-map vs dataset sanity
    map_sents = set(label_map.keys())
    payload = {
        "protocol": "rq1_sentence_overlap",
        "label_map_n_classes": len(label_map),
        "label_map_equals_train_sentences": map_sents == train_s,
        "n_train_sentences_not_in_label_map": len(train_s - map_sents),
        "n_label_map_not_in_train": len(map_sents - train_s),
        "overlap": overlap,
        "restricted_eval": eval_payload,
        "support_csv": str(support_csv),
    }
    out = Path(args.out)
    write_json(out, payload)
    print(f"\nwrote {out}")

    # One-line takeaway for the paper
    frac = tc["frac_test_clips_sentence_in_train"]
    print("\n=== Takeaway for the paper ===")
    if frac >= 0.99:
        print(
            "Almost all signer-5 test clips use sentence IDs that appear in train. "
            "RQ1 near-floor accuracy is NOT explained by missing sentence classes."
        )
    elif frac >= 0.5:
        print(
            f"Only {100*frac:.1f}% of test clips have sentence IDs in train. "
            "Report both full and restricted metrics; do not attribute all error to signer transfer."
        )
    else:
        print(
            f"Only {100*frac:.1f}% of test clips have sentence IDs in train. "
            "RQ1 absolute accuracy is heavily confounded by unseen sentence classes."
        )


if __name__ == "__main__":
    main()
