"""Score RQ1 scratch vs SSL checkpoints across training seeds.

    python -u scripts/16_eval_rq1_multiseed.py --seeds 42,43,44,45,46
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from torch.utils.data import DataLoader

from src.config import FEATURE_DIM, PATHS
from src.dataset import LabeledWindowDataset
from src.metrics import pool_clips, predict_loader, slim_stats, summarize_predictions
from src.model import build_pose_model
from src.splits import load_label_map
from src.train_utils import get_device, load_checkpoint, write_json


def _ids(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _eval_ckpt(path: Path, n_classes: int, loader, device) -> dict | None:
    if not path.exists():
        print(f"missing {path}")
        return None
    ckpt = load_checkpoint(path, map_location=device)
    arch = str(ckpt.get("arch", "transformer")) if isinstance(ckpt, dict) else "transformer"
    model = build_pose_model(arch, feature_dim=FEATURE_DIM, n_classes=n_classes).to(device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    raw = predict_loader(model, loader, device)
    win = summarize_predictions(raw)
    clip = summarize_predictions(pool_clips(raw))
    return {
        "checkpoint": str(path),
        "window": slim_stats(win),
        "clip": slim_stats(clip),
        "window_top1": float(win.get("top1", 0.0)),
        "window_top5": float(win.get("top5", 0.0)),
        "clip_top1": float(clip.get("top1", 0.0)),
        "clip_top5": float(clip.get("top5", 0.0)),
    }


def _slot(vals: list[float]) -> dict:
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        "n": len(vals),
        "values": [float(v) for v in vals],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=str, default="42,43,44,45,46")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out", type=str, default=str(PATHS.results / "rq1_multiseed.json"))
    args = p.parse_args()

    seeds = _ids(args.seeds)
    device = get_device()
    sent_map = load_label_map()
    n_sent = len(sent_map)
    test_ds = LabeledWindowDataset(PATHS.splits / "test.csv", label_map=sent_map)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    print(f"[rq1-ms] seeds={seeds}  test_windows={len(test_ds)}  classes={n_sent}  device={device}")

    runs = []
    for seed in seeds:
        scr = PATHS.checkpoints / f"finetune_scratch_s{seed}.pt"
        pre = PATHS.checkpoints / f"finetune_pretrained_s{seed}.pt"
        print(f"\n=== seed {seed} ===")
        s = _eval_ckpt(scr, n_sent, loader, device)
        p_ = _eval_ckpt(pre, n_sent, loader, device)
        if s is None or p_ is None:
            continue
        print(
            f"  scratch     win top-1={100 * s['window_top1']:.3f}%  "
            f"clip={100 * s['clip_top1']:.3f}%"
        )
        print(
            f"  pretrained  win top-1={100 * p_['window_top1']:.3f}%  "
            f"clip={100 * p_['clip_top1']:.3f}%  "
            f"d_win={100 * (p_['window_top1'] - s['window_top1']):+.3f} pp"
        )
        runs.append({"seed": seed, "scratch": s, "pretrained": p_})

    summary = {}
    for arm in ("scratch", "pretrained"):
        summary[arm] = {
            "window_top1": _slot([r[arm]["window_top1"] for r in runs]),
            "window_top5": _slot([r[arm]["window_top5"] for r in runs]),
            "clip_top1": _slot([r[arm]["clip_top1"] for r in runs]),
            "clip_top5": _slot([r[arm]["clip_top5"] for r in runs]),
        }
    deltas = [r["pretrained"]["window_top1"] - r["scratch"]["window_top1"] for r in runs]
    clip_deltas = [r["pretrained"]["clip_top1"] - r["scratch"]["clip_top1"] for r in runs]
    summary["delta_pretrained_minus_scratch"] = {
        "window_top1": _slot(deltas),
        "clip_top1": _slot(clip_deltas),
    }

    payload = {"seeds": seeds, "n_test_windows": len(test_ds), "runs": runs, "summary": summary}
    write_json(Path(args.out), payload)

    print("\nRQ1 multi-seed summary (window top-1 %)")
    for arm in ("scratch", "pretrained"):
        d = summary[arm]["window_top1"]
        print(f"  {arm:<12} {100 * d['mean']:.3f} +/- {100 * d['std']:.3f}%  n={d['n']}")
    d = summary["delta_pretrained_minus_scratch"]["window_top1"]
    print(f"  delta SSL-scratch  {100 * d['mean']:+.3f} +/- {100 * d['std']:.3f} pp")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
