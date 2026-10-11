"""RQ1 evaluation: window vs clip scoring, plus category diagnostics.

Default: score the 4,025-way sentence models at window level (legacy) and
clip level (mean-pooled logits over windows of the same video). Also maps
those sentence predictions onto thematic categories (post-hoc).

If category-trained checkpoints exist, they are scored the same way:

    python scripts/05_evaluate.py
    python scripts/04_finetune.py --task category --pretrained checkpoints/ssl_encoder.pt
    python scripts/04_finetune.py --task category --pretrained none
    python scripts/05_evaluate.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from torch.utils.data import DataLoader

from src.config import FEATURE_DIM, PATHS
from src.dataset import LabeledWindowDataset
from src.metrics import (
    format_comparison,
    mapped_category_summary,
    majority_accuracy,
    pool_clips,
    predict_loader,
    slim_stats,
    summarize_predictions,
)
from src.model import build_pose_model
from src.splits import load_label_map
from src.train_utils import get_device, load_checkpoint, write_json


def _eval_ckpt(path: Path, n_classes: int, loader, device, arch: str = "transformer"):
    if not path.exists():
        print(f"missing checkpoint: {path}")
        return None, None
    ckpt = load_checkpoint(path, map_location=device)
    ckpt_arch = arch
    if isinstance(ckpt, dict) and ckpt.get("arch"):
        ckpt_arch = str(ckpt["arch"])
    model = build_pose_model(ckpt_arch, feature_dim=FEATURE_DIM, n_classes=n_classes).to(device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    raw = predict_loader(model, loader, device)
    return raw, str(path)


def _write_group_table(stats: dict, key: str, dest: Path) -> None:
    rows = [{"name": k, "top1": v, "error": 1.0 - v} for k, v in sorted(stats.get(key, {}).items())]
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["name", "top1", "error"])
        w.writeheader()
        w.writerows(rows)


def _sentence_to_category() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for name in ("train.csv", "val.csv", "test.csv"):
        path = PATHS.splits / name
        if not path.exists():
            continue
        df = pd.read_csv(path, usecols=["sentence_id", "category"])
        for sid, cat in zip(df["sentence_id"].astype(str), df["category"].astype(str)):
            mapping.setdefault(sid, cat)
    return mapping


def _score_pair(raw_s, raw_p, ckpt_s, ckpt_p) -> tuple[dict | None, dict | None]:
    def _one(raw, ckpt):
        if raw is None:
            return None
        stats = summarize_predictions(raw)
        stats["checkpoint"] = ckpt
        return stats

    return _one(raw_s, ckpt_s), _one(raw_p, ckpt_p)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained-ckpt", type=str, default=str(PATHS.finetune_pretrained))
    p.add_argument("--scratch-ckpt", type=str, default=str(PATHS.finetune_scratch))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--arch",
        choices=("transformer", "bigru"),
        default="transformer",
        help="fallback if checkpoint has no arch field",
    )
    args = p.parse_args()

    device = get_device()
    sent_map = load_label_map()
    id_to_sent = {int(v): str(k) for k, v in sent_map.items()}
    sent_to_cat = _sentence_to_category()
    test_ds = LabeledWindowDataset(PATHS.splits / "test.csv", label_map=sent_map)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    n_sent = len(sent_map)
    print(f"test windows: {len(test_ds)}  sentence classes: {n_sent}  device: {device}")

    raw_s, ckpt_s = _eval_ckpt(Path(args.scratch_ckpt), n_sent, loader, device, arch=args.arch)
    raw_p, ckpt_p = _eval_ckpt(Path(args.pretrained_ckpt), n_sent, loader, device, arch=args.arch)

    scratch, pretrained = _score_pair(raw_s, raw_p, ckpt_s, ckpt_p)
    clip_s = summarize_predictions(pool_clips(raw_s)) if raw_s is not None else None
    clip_p = summarize_predictions(pool_clips(raw_p)) if raw_p is not None else None
    if clip_s:
        clip_s["checkpoint"] = ckpt_s
    if clip_p:
        clip_p["checkpoint"] = ckpt_p

    print()
    print(format_comparison(scratch or {}, pretrained or {}, "RQ1  sentence ID  (window-level)"))
    print()
    print(format_comparison(clip_s or {}, clip_p or {}, "RQ1  sentence ID  (clip-level, mean-pooled logits)"))

    posthoc = {}
    for unit, rs, rp in (("window", raw_s, raw_p), ("clip", None, None)):
        if unit == "clip":
            rs = pool_clips(raw_s) if raw_s is not None else None
            rp = pool_clips(raw_p) if raw_p is not None else None
        if rs is None or rp is None:
            continue
        ms = mapped_category_summary(rs, id_to_sent, sent_to_cat)
        mp = mapped_category_summary(rp, id_to_sent, sent_to_cat)
        posthoc[unit] = {"scratch": ms, "pretrained": mp}
        print()
        print(
            format_comparison(
                ms,
                mp,
                f"RQ1  category via sentence argmax  ({unit}-level, {ms['n']} items, "
                f"majority {100 * ms['majority']:.1f}%)",
            )
        )

    for arm, stats in (("scratch", scratch), ("pretrained", pretrained)):
        if not stats:
            continue
        _write_group_table(stats, "per_category_top1", PATHS.results / f"rq1_per_category_{arm}.csv")
        _write_group_table(stats, "per_sentence_top1", PATHS.results / f"rq1_per_sentence_{arm}.csv")

    payload = {
        "scratch": slim_stats(scratch),
        "pretrained": slim_stats(pretrained),
        "clip": {"scratch": slim_stats(clip_s), "pretrained": slim_stats(clip_p)},
    }
    write_json(PATHS.results / "rq1_comparison.json", payload)

    cat_trained = None
    cat_path = PATHS.category_map
    pre_cat = PATHS.finetune_pretrained_category
    scr_cat = PATHS.finetune_scratch_category
    if cat_path.exists() and (pre_cat.exists() or scr_cat.exists()):
        cat_map = load_label_map(cat_path)
        cat_ds = LabeledWindowDataset(
            PATHS.splits / "test.csv", label_map=cat_map, label_col="category"
        )
        cat_loader = DataLoader(cat_ds, batch_size=args.batch_size, shuffle=False)
        raw_cs, ckpt_cs = _eval_ckpt(scr_cat, len(cat_map), cat_loader, device, arch=args.arch)
        raw_cp, ckpt_cp = _eval_ckpt(pre_cat, len(cat_map), cat_loader, device, arch=args.arch)
        cw_s, cw_p = _score_pair(raw_cs, raw_cp, ckpt_cs, ckpt_cp)
        cc_s = summarize_predictions(pool_clips(raw_cs)) if raw_cs is not None else None
        cc_p = summarize_predictions(pool_clips(raw_cp)) if raw_cp is not None else None
        maj_w = majority_accuracy(list(cat_ds.table["category"].astype(str)))
        maj_c = majority_accuracy(
            cat_ds.table.drop_duplicates("video_id")["category"].astype(str).tolist()
        )
        print()
        print(
            format_comparison(
                cw_s or {},
                cw_p or {},
                f"RQ1  category-trained  (window-level, {len(cat_map)}-way, majority {100 * maj_w:.1f}%)",
            )
        )
        print()
        print(
            format_comparison(
                cc_s or {},
                cc_p or {},
                f"RQ1  category-trained  (clip-level, {len(cat_map)}-way, majority {100 * maj_c:.1f}%)",
            )
        )
        cat_trained = {
            "n_classes": len(cat_map),
            "majority_window": maj_w,
            "majority_clip": maj_c,
            "window": {"scratch": slim_stats(cw_s), "pretrained": slim_stats(cw_p)},
            "clip": {"scratch": slim_stats(cc_s), "pretrained": slim_stats(cc_p)},
        }

    diagnostics = {
        "sentence": {
            "n_classes": n_sent,
            "chance_top1": 1.0 / max(n_sent, 1),
            "window": {"scratch": slim_stats(scratch), "pretrained": slim_stats(pretrained)},
            "clip": {"scratch": slim_stats(clip_s), "pretrained": slim_stats(clip_p)},
        },
        "category_posthoc": posthoc,
        "category_trained": cat_trained,
    }
    diag_path = PATHS.results / "rq1_diagnostics.json"
    write_json(diag_path, diagnostics)
    print(f"\nwrote {PATHS.results / 'rq1_comparison.json'}")
    print(f"wrote {diag_path}")


if __name__ == "__main__":
    main()
