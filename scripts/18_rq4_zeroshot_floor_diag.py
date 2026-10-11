"""Zero-shot category floor diagnostic for reviewer comment.

Answers: difficult transfer vs prediction collapse vs both.

Computes:
  - clip counts by (signer, category)
  - train-roster coverage of each held-out signer's categories
  - test-majority baseline (always predict test gold majority)
  - source-majority baseline (always predict majority category of the
    *training* roster; evaluated on the held-out signer)
  - for each RQ4 k=4 scratch base: predicted-category histogram on the
    held-out signer (full clips), entropy, top predicted mass, familiar-
    signer val accuracy if recoverable from logs

    python -u scripts/18_rq4_zeroshot_floor_diag.py
"""

from __future__ import annotations

import argparse
import json
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

from src.adaptation import build_classifier, eval_category_clips
from src.config import CFG, PATHS
from src.dataset import LabeledWindowDataset
from src.metrics import predict_loader, pool_clips
from src.train_utils import get_device, set_seed, write_json

PROTOCOL = "rq4_zeroshot_floor_diag"


def _ids(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _category_map(windows: pd.DataFrame) -> dict[str, int]:
    labels = sorted(windows["category"].astype(str).unique().tolist())
    return {lab: i for i, lab in enumerate(labels)}


def _majority_label(labels: list[str]) -> str:
    return Counter(labels).most_common(1)[0][0]


def _acc(pred: list[str], gold: list[str]) -> float:
    return float(np.mean([p == g for p, g in zip(pred, gold)])) if gold else 0.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--signers", type=str, default="1,2,3,4,5")
    p.add_argument("--arms", type=str, default="scratch")
    p.add_argument("--batch-size", type=int, default=CFG.finetune.batch_size)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ckpt-dir", type=str, default=str(PATHS.checkpoints / "rq4"))
    p.add_argument(
        "--out",
        type=str,
        default=str(PATHS.results / "rq4_zeroshot_floor_diag.json"),
    )
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device()
    signers = _ids(args.signers)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    ckpt_dir = Path(args.ckpt_dir)

    windows = pd.read_csv(PATHS.splits / "windows.csv")
    if "source" in windows.columns:
        windows = windows[
            windows["source"].astype(str).str.lower().eq("signtalk")
        ].reset_index(drop=True)
    clips = windows.drop_duplicates("video_id").copy()
    clips["signer_id"] = clips["signer_id"].astype(str)
    clips["category"] = clips["category"].astype(str)

    label_map = _category_map(windows)
    id_to_cat = {int(v): str(k) for k, v in label_map.items()}
    n_classes = len(label_map)

    # --- table: clip counts by signer x category ---
    count_tbl = (
        clips.groupby(["signer_id", "category"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=signers)
        .reindex(columns=sorted(label_map.keys()), fill_value=0)
    )
    counts_records = {
        s: {c: int(count_tbl.loc[s, c]) for c in count_tbl.columns}
        for s in signers
    }

    fold_rows: list[dict] = []
    for test_s in signers:
        train_signers = [s for s in signers if s != test_s]
        test_clips = clips[clips["signer_id"].eq(test_s)]
        train_clips = clips[clips["signer_id"].isin(train_signers)]
        gold = test_clips["category"].tolist()
        train_labels = train_clips["category"].tolist()

        test_maj = _majority_label(gold)
        source_maj = _majority_label(train_labels)
        test_maj_acc = _acc([test_maj] * len(gold), gold)
        source_maj_acc = _acc([source_maj] * len(gold), gold)

        tgt_cats = set(gold)
        src_cats = set(train_labels)
        covered = sorted(tgt_cats & src_cats)
        missing = sorted(tgt_cats - src_cats)

        # mass of source majority among train clips
        src_maj_mass = train_labels.count(source_maj) / max(len(train_labels), 1)
        # how often test gold equals source majority category
        test_has_src_maj = gold.count(source_maj) / max(len(gold), 1)

        row_base = {
            "test_signer": test_s,
            "train_signers": train_signers,
            "n_test_clips": len(test_clips),
            "n_train_clips": len(train_clips),
            "n_target_categories": len(tgt_cats),
            "n_source_categories": len(src_cats),
            "n_target_covered_by_source": len(covered),
            "target_categories_missing_from_source": missing,
            "test_majority_category": test_maj,
            "source_majority_category": source_maj,
            "test_majority_baseline": test_maj_acc,
            "source_majority_baseline": source_maj_acc,
            "source_majority_train_mass": src_maj_mass,
            "test_mass_on_source_majority_category": test_has_src_maj,
            "test_category_histogram": dict(Counter(gold)),
            "train_category_histogram": dict(Counter(train_labels)),
        }

        for arm in arms:
            if arm == "pretrained" and test_s != "5":
                continue
            base_path = ckpt_dir / f"base_t{test_s}_k4_{arm}.pt"
            if not base_path.exists():
                row = {**row_base, "arm": arm, "status": "missing_base"}
                fold_rows.append(row)
                continue

            model = build_classifier(n_classes, device, full_checkpoint=base_path)
            # Evaluate on ALL held-out signer clips (RQ3-style), not RQ4 20% final test.
            test_windows = windows[windows["signer_id"].astype(str).eq(test_s)].reset_index(
                drop=True
            )
            ds = LabeledWindowDataset(
                test_windows, label_map=label_map, label_col="category"
            )
            loader = DataLoader(
                ds, batch_size=args.batch_size, shuffle=False, num_workers=0
            )
            raw = predict_loader(model, loader, device)
            pooled = pool_clips(raw)
            logits = pooled["logits"].float()
            pred_idx = logits.argmax(dim=-1).tolist()
            pred = [id_to_cat.get(int(i), "") for i in pred_idx]
            gold_c = [str(c) for c in pooled["category"]]
            probs = torch.softmax(logits, dim=-1)

            pred_counts = Counter(pred)
            n = len(pred)
            top_pred, top_n = pred_counts.most_common(1)[0]
            # mean predictive entropy (nats)
            ent = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=-1)
            mean_ent = float(ent.mean())
            uniform_ent = float(np.log(n_classes))

            scores = eval_category_clips(
                model,
                test_windows,
                label_map,
                id_to_cat,
                device,
                batch_size=args.batch_size,
                allowed_categories=None,
            )

            row = {
                **row_base,
                "arm": arm,
                "status": "ok",
                "base_checkpoint": str(base_path),
                "model_clip_top1": float(scores["clip_top1"]),
                "model_clip_macro_f1": float(scores["clip_macro_f1"]),
                "model_clip_majority_ref": float(scores["clip_majority"]),
                "n_unique_predicted_categories": len(pred_counts),
                "top_predicted_category": top_pred,
                "top_predicted_mass": top_n / max(n, 1),
                "predicted_category_histogram": dict(pred_counts),
                "mean_predictive_entropy_nats": mean_ent,
                "uniform_entropy_nats": uniform_ent,
                "frac_preds_in_source_majority": pred.count(source_maj) / max(n, 1),
                "frac_preds_in_test_majority": pred.count(test_maj) / max(n, 1),
            }
            fold_rows.append(row)
            print(
                f"test={test_s} arm={arm}  "
                f"model={100 * row['model_clip_top1']:.2f}%  "
                f"test-maj={100 * test_maj_acc:.2f}%  "
                f"src-maj={100 * source_maj_acc:.2f}%  "
                f"preds={len(pred_counts)} cats  "
                f"top={top_pred!r} ({100 * top_n / max(n, 1):.1f}%)  "
                f"H={mean_ent:.2f}/{uniform_ent:.2f}"
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # summary
    ok = [r for r in fold_rows if r.get("status") == "ok"]
    summary = {
        "n_folds": len(ok),
        "model_clip_top1": {
            "mean": float(np.mean([r["model_clip_top1"] for r in ok])) if ok else 0.0,
            "std": float(np.std([r["model_clip_top1"] for r in ok], ddof=1))
            if len(ok) > 1
            else 0.0,
        },
        "test_majority_baseline": {
            "mean": float(np.mean([r["test_majority_baseline"] for r in ok])) if ok else 0.0,
        },
        "source_majority_baseline": {
            "mean": float(np.mean([r["source_majority_baseline"] for r in ok])) if ok else 0.0,
        },
        "mean_unique_predicted_categories": float(
            np.mean([r["n_unique_predicted_categories"] for r in ok])
        )
        if ok
        else 0.0,
        "mean_top_predicted_mass": float(
            np.mean([r["top_predicted_mass"] for r in ok])
        )
        if ok
        else 0.0,
        "all_target_cats_covered_by_source": all(
            r["n_target_covered_by_source"] == r["n_target_categories"] for r in ok
        ),
    }

    payload = {
        "protocol": PROTOCOL,
        "note": (
            "Zero-shot floor diagnostic on RQ4 k=4 bases evaluated on the full "
            "held-out signer (not the RQ4 20% final-test split). "
            "source_majority_baseline always predicts the training-roster "
            "majority category; test_majority_baseline always predicts the "
            "held-out signer's empirical majority."
        ),
        "clip_counts_by_signer_category": counts_records,
        "folds": fold_rows,
        "summary": summary,
    }
    write_json(Path(args.out), payload)

    print("\n=== Summary ===")
    print(
        f"model top-1 mean {100 * summary['model_clip_top1']['mean']:.2f}%  "
        f"test-majority {100 * summary['test_majority_baseline']['mean']:.2f}%  "
        f"source-majority {100 * summary['source_majority_baseline']['mean']:.2f}%"
    )
    print(
        f"mean unique predicted cats {summary['mean_unique_predicted_categories']:.1f}/26  "
        f"mean top-pred mass {100 * summary['mean_top_predicted_mass']:.1f}%  "
        f"all tgt cats covered by source? {summary['all_target_cats_covered_by_source']}"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
