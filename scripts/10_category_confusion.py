"""Confusion matrix and predicted-label distribution for 26-way category models.

Resolves whether test top-1 below the test-majority baseline is a real
signer-transfer failure or an artifact of the majority class being computed
on a different label distribution than the model emits.

    python -u scripts/10_category_confusion.py
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import FEATURE_DIM, PATHS
from src.dataset import LabeledWindowDataset
from src.metrics import pool_clips
from src.model import MaskedPoseTransformer
from src.paper_figures import GRAY, NAVY, RUST, _save, _style
from src.splits import load_label_map
from src.train_utils import get_device, load_checkpoint, write_json


def _eval_ckpt(path: Path, n_classes: int, loader, device):
    if not path.exists():
        raise FileNotFoundError(path)
    model = MaskedPoseTransformer(feature_dim=FEATURE_DIM, n_classes=n_classes).to(device)
    ckpt = load_checkpoint(path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()
    logits_all, y_all, cat_all, vid_all = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            logits_all.append(model.classify(x).cpu())
            y_all.append(batch["y"])
            cat_all.extend([str(s) for s in batch["category"]])
            vid_all.extend([str(s) for s in batch["video_id"]])
    return {
        "logits": torch.cat(logits_all, dim=0),
        "y": torch.cat(y_all, dim=0),
        "category": cat_all,
        "video_id": vid_all,
        "sentence_id": cat_all,
        "signer_id": ["5"] * len(cat_all),
    }


def _labels_from_raw(raw: dict, id_to_cat: dict[int, str], clip: bool) -> tuple[list[str], list[str]]:
    data = pool_clips(raw) if clip else raw
    pred_idx = data["logits"].argmax(dim=-1).tolist()
    pred = [id_to_cat[int(i)] for i in pred_idx]
    gold = [str(c) for c in data["category"]]
    return gold, pred


def _clip_labels(df: pd.DataFrame) -> list[str]:
    return df.drop_duplicates("video_id")["category"].astype(str).tolist()


def analyze(
    gold: list[str],
    pred: list[str],
    class_names: list[str],
    train_gold: list[str],
) -> dict:
    n = len(gold)
    gold_c = Counter(gold)
    pred_c = Counter(pred)
    train_c = Counter(train_gold)
    acc = float(np.mean([p == g for p, g in zip(pred, gold)])) if n else 0.0

    test_mode, test_mode_n = gold_c.most_common(1)[0]
    train_mode, _ = train_c.most_common(1)[0]
    pred_mode, pred_mode_n = pred_c.most_common(1)[0]

    test_classes = set(gold_c)
    mass_on_absent = float(np.mean([p not in test_classes for p in pred])) if n else 0.0
    idx = {c: i for i, c in enumerate(class_names)}
    cm = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    for g, p in zip(gold, pred):
        if g in idx and p in idx:
            cm[idx[g], idx[p]] += 1

    gold_frac = {c: gold_c.get(c, 0) / n for c in class_names}
    pred_frac = {c: pred_c.get(c, 0) / n for c in class_names}
    train_n = max(len(train_gold), 1)
    train_frac = {c: train_c.get(c, 0) / train_n for c in class_names}

    def _tv(a: dict[str, float], b: dict[str, float]) -> float:
        return 0.5 * sum(abs(a[c] - b[c]) for c in class_names)

    return {
        "n": n,
        "top1": acc,
        "n_gold_classes": len(test_classes),
        "n_pred_classes": len(pred_c),
        "test_majority_class": test_mode,
        "test_majority_acc": test_mode_n / n,
        "train_majority_class": train_mode,
        "train_majority_on_test": float(np.mean([g == train_mode for g in gold])),
        "pred_majority_class": pred_mode,
        "pred_majority_mass": pred_mode_n / n,
        "constant_pred_mode_acc": float(np.mean([g == pred_mode for g in gold])),
        "uniform_chance": 1.0 / max(len(class_names), 1),
        "mass_on_classes_absent_from_test": mass_on_absent,
        "tv_pred_vs_gold": _tv(pred_frac, gold_frac),
        "tv_pred_vs_train": _tv(pred_frac, train_frac),
        "gold_frac": gold_frac,
        "pred_frac": pred_frac,
        "train_frac": train_frac,
        "confusion": cm.tolist(),
    }


def _short(name: str, n: int = 22) -> str:
    return name if len(name) <= n else name[: n - 1] + "…"


def _print_block(title: str, stats: dict) -> None:
    print(title)
    print(
        f"  n={stats['n']}  top-1={100 * stats['top1']:.2f}%  "
        f"pred classes used={stats['n_pred_classes']}/{stats['n_gold_classes']} gold classes on this split"
    )
    print(
        f"  test majority     always '{stats['test_majority_class']}' "
        f"-> {100 * stats['test_majority_acc']:.2f}%"
    )
    print(
        f"  train majority    always '{stats['train_majority_class']}' on test "
        f"-> {100 * stats['train_majority_on_test']:.2f}%"
    )
    print(
        f"  model pred mode   '{stats['pred_majority_class']}' "
        f"(mass {100 * stats['pred_majority_mass']:.1f}%)  "
        f"if constant, test acc would be {100 * stats['constant_pred_mode_acc']:.2f}%"
    )
    print(
        f"  uniform 26-way chance {100 * stats['uniform_chance']:.2f}%  "
        f"mass on classes with 0 test support {100 * stats['mass_on_classes_absent_from_test']:.1f}%"
    )
    print(
        f"  TV(pred, test gold)={stats['tv_pred_vs_gold']:.3f}  "
        f"TV(pred, train)={stats['tv_pred_vs_train']:.3f}  (0=identical, 1=disjoint)"
    )
    vs_test_maj = 100 * (stats["top1"] - stats["test_majority_acc"])
    vs_train_maj = 100 * (stats["top1"] - stats["train_majority_on_test"])
    print(f"  top-1 minus test-majority {vs_test_maj:+.2f} pp  minus train-majority-on-test {vs_train_maj:+.2f} pp")
    print()


def fig_distributions(payload: dict, class_names: list[str], unit: str) -> list[Path]:
    block = payload[unit]
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.4), sharey=True)
    x = np.arange(len(class_names))
    shorts = [_short(c) for c in class_names]
    gold = np.array([block["scratch"]["gold_frac"][c] for c in class_names])
    for ax, arm, color in zip(axes, ("scratch", "pretrained"), (RUST, NAVY)):
        pred = np.array([block[arm]["pred_frac"][c] for c in class_names])
        ax.bar(x - 0.2, 100 * gold, 0.4, label="Test gold", color=GRAY, alpha=0.85)
        ax.bar(x + 0.2, 100 * pred, 0.4, label="Model pred", color=color)
        ax.set_xticks(x, shorts, rotation=90, fontsize=6)
        ax.set_title(f"{unit} / {arm}")
        ax.set_ylabel("Percent of items")
        ax.legend(frameon=False, fontsize=8)
        ax.axhline(100 / len(class_names), color="#888", lw=0.8, ls="--")
    fig.suptitle("Category-trained models: predicted vs gold label distribution (held-out signer)")
    fig.tight_layout()
    return _save(fig, f"fig_category_pred_dist_{unit}")


def fig_confusion(payload: dict, class_names: list[str], unit: str, arm: str) -> list[Path]:
    cm = np.array(payload[unit][arm]["confusion"], dtype=float)
    row_sum = cm.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    norm = cm / row_sum
    _style()
    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)), [_short(c, 16) for c in class_names], rotation=90, fontsize=6)
    ax.set_yticks(range(len(class_names)), [_short(c, 16) for c in class_names], fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Gold")
    ax.set_title(f"Confusion (row-normalized)  {unit} / {arm}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return _save(fig, f"fig_category_confusion_{unit}_{arm}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    cat_map = load_label_map(PATHS.category_map)
    id_to_cat = {int(v): str(k) for k, v in cat_map.items()}
    class_names = [id_to_cat[i] for i in range(len(cat_map))]
    device = get_device()
    train = pd.read_csv(PATHS.splits / "train.csv")
    test_ds = LabeledWindowDataset(
        PATHS.splits / "test.csv", label_map=cat_map, label_col="category"
    )
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    print(f"category classes={len(cat_map)}  test windows={len(test_ds)}  device={device}")

    raw = {}
    for arm, path in (
        ("scratch", PATHS.finetune_scratch_category),
        ("pretrained", PATHS.finetune_pretrained_category),
    ):
        raw[arm] = _eval_ckpt(path, len(cat_map), loader, device)

    payload = {"class_names": class_names}
    train_window = train["category"].astype(str).tolist()
    train_clip = _clip_labels(train)
    for unit, clip, train_gold in (("window", False, train_window), ("clip", True, train_clip)):
        payload[unit] = {}
        print("=" * 72)
        for arm in ("scratch", "pretrained"):
            gold, pred = _labels_from_raw(raw[arm], id_to_cat, clip=clip)
            stats = analyze(gold, pred, class_names, train_gold)
            payload[unit][arm] = stats
            _print_block(f"category-trained  {unit}  {arm}", stats)

    out = PATHS.results / "rq1_category_confusion.json"
    write_json(out, payload)
    print(f"wrote {out}")

    written = []
    for unit in ("window", "clip"):
        written += fig_distributions(payload, class_names, unit)
        for arm in ("scratch", "pretrained"):
            written += fig_confusion(payload, class_names, unit, arm)
    rows = []
    for unit in ("window", "clip"):
        for arm in ("scratch", "pretrained"):
            s = payload[unit][arm]
            rows.append(
                {
                    "Unit": unit,
                    "Arm": arm,
                    "n": s["n"],
                    "Top-1 (%)": f"{100 * s['top1']:.2f}",
                    "Test majority (%)": f"{100 * s['test_majority_acc']:.2f}",
                    "Train-maj on test (%)": f"{100 * s['train_majority_on_test']:.2f}",
                    "Pred mode mass (%)": f"{100 * s['pred_majority_mass']:.1f}",
                    "Mass on absent classes (%)": f"{100 * s['mass_on_classes_absent_from_test']:.1f}",
                    "TV pred vs test": f"{s['tv_pred_vs_gold']:.3f}",
                }
            )
    df = pd.DataFrame(rows)
    PATHS.tables.mkdir(parents=True, exist_ok=True)
    csv_path = PATHS.tables / "table_category_confusion.csv"
    df.to_csv(csv_path, index=False)
    written.append(csv_path)
    print("wrote figures/tables:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
