"""Evaluation metrics for signer-independent sign-to-text (sentence classification)."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def predict_loader(model, loader, device) -> dict:
    model.eval()
    logits_all = []
    y_all = []
    signer_all = []
    sent_all = []
    cat_all = []
    vid_all = []
    for batch in loader:
        x = batch["x"].to(device)
        logits = model.classify(x)
        logits_all.append(logits.cpu())
        y_all.append(batch["y"])
        signer_all.extend([str(s) for s in batch["signer_id"]])
        sent_all.extend([str(s) for s in batch["sentence_id"]])
        cat_all.extend([str(s) for s in batch["category"]])
        vid_all.extend([str(s) for s in batch["video_id"]])
    logits = torch.cat(logits_all, dim=0)
    y = torch.cat(y_all, dim=0)
    return {
        "logits": logits,
        "y": y,
        "signer_id": signer_all,
        "sentence_id": sent_all,
        "category": cat_all,
        "video_id": vid_all,
    }


def topk_accuracy(logits: torch.Tensor, y: torch.Tensor, k: int = 1) -> float:
    k = min(k, logits.size(-1))
    pred = logits.topk(k, dim=-1).indices
    return float((pred == y.unsqueeze(-1)).any(dim=-1).float().mean().item())


def per_group_accuracy(pred: np.ndarray, y: np.ndarray, groups: list[str]) -> dict[str, float]:
    buckets: dict[str, list[bool]] = defaultdict(list)
    for p, t, g in zip(pred, y, groups):
        buckets[str(g)].append(bool(p == t))
    return {g: float(np.mean(v)) for g, v in sorted(buckets.items()) if g and g != "nan"}


def summarize_predictions(out: dict) -> dict:
    logits = out["logits"]
    y = out["y"]
    pred = logits.argmax(dim=-1).numpy()
    y_np = y.numpy()
    probs = F.softmax(logits.float(), dim=-1)
    entropy = -(probs * (probs.clamp_min(1e-12).log())).sum(dim=-1).numpy()
    summary = {
        "n": int(len(y_np)),
        "top1": topk_accuracy(logits, y, k=1),
        "top5": topk_accuracy(logits, y, k=5),
        "mean_entropy": float(entropy.mean()) if len(entropy) else 0.0,
        "per_signer_top1": per_group_accuracy(pred, y_np, out["signer_id"]),
        "per_category_top1": per_group_accuracy(pred, y_np, out["category"]),
        "per_sentence_top1": per_group_accuracy(pred, y_np, out["sentence_id"]),
    }
    summary["per_category_error"] = {k: 1.0 - v for k, v in summary["per_category_top1"].items()}
    summary["per_sentence_error"] = {k: 1.0 - v for k, v in summary["per_sentence_top1"].items()}
    return summary


def pool_clips(out: dict) -> dict:
    """Mean-pool window logits that share a video_id (one score per clip)."""
    logits = out["logits"]
    groups: dict[str, list[int]] = defaultdict(list)
    for i, vid in enumerate(out["video_id"]):
        groups[str(vid)].append(i)
    pooled = []
    y = []
    signer_all = []
    sent_all = []
    cat_all = []
    vid_all = []
    n_windows = []
    for vid, idxs in groups.items():
        idx = torch.tensor(idxs, dtype=torch.long)
        pooled.append(logits.index_select(0, idx).mean(dim=0))
        i0 = idxs[0]
        y.append(out["y"][i0])
        signer_all.append(out["signer_id"][i0])
        sent_all.append(out["sentence_id"][i0])
        cat_all.append(out["category"][i0])
        vid_all.append(vid)
        n_windows.append(len(idxs))
    return {
        "logits": torch.stack(pooled, dim=0),
        "y": torch.stack(y, dim=0),
        "signer_id": signer_all,
        "sentence_id": sent_all,
        "category": cat_all,
        "video_id": vid_all,
        "n_windows": n_windows,
    }


def macro_f1_from_preds(pred: list[str], gold: list[str]) -> float:
    """Unweighted mean of per-class F1 over classes present in gold."""
    if not gold:
        return 0.0
    labels = sorted(set(gold))
    f1s = []
    for lab in labels:
        tp = sum(1 for p, g in zip(pred, gold) if p == lab and g == lab)
        fp = sum(1 for p, g in zip(pred, gold) if p == lab and g != lab)
        fn = sum(1 for p, g in zip(pred, gold) if p != lab and g == lab)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(f1s)) if f1s else 0.0


def score_clip_category(
    out: dict,
    id_to_cat: dict[int, str],
    *,
    allowed_categories: set[str] | list[str] | None = None,
    oracle_test_coverage: bool = False,
) -> dict:
    """Clip-level 26-way category metrics.

    Primary metrics (always):
      - clip_top1: unrestricted argmax over all classes
      - clip_macro_f1: macro-F1 on unrestricted predictions
      - clip_majority: majority-class baseline on gold

    Restricted / coverage-aware (optional, never uses test gold by default):
      - If ``allowed_categories`` is set (e.g. categories seen in calibration
        or in the training roster), also report ``clip_restricted_top1``.
      - ``oracle_test_coverage=True`` restores the old behaviour of restricting
        to categories that appear in the *test* gold labels. Prefer not to use
        this as a primary deployment metric; it is kept for ablation only and
        stored as ``clip_oracle_coverage_aware_top1``.
    """
    pooled = pool_clips(out)
    logits = pooled["logits"].float()
    gold = [str(c) for c in pooled["category"]]
    n = len(gold)
    empty = {
        "n_clips": 0,
        "n_gold_categories": 0,
        "n_allowed_categories": 0,
        "clip_top1": 0.0,
        "clip_macro_f1": 0.0,
        "clip_majority": 0.0,
        "clip_restricted_top1": 0.0,
        "clip_coverage_aware_top1": 0.0,  # alias of restricted when allowed set given
        "clip_oracle_coverage_aware_top1": 0.0,
        "mass_on_covered": 0.0,
        "covered_categories": [],
        "allowed_categories": [],
    }
    if n == 0:
        return empty

    pred_idx = logits.argmax(dim=-1).tolist()
    pred = [id_to_cat.get(int(i), "") for i in pred_idx]
    gold_set = set(gold)
    n_classes = logits.size(-1)

    def _restrict(allow_names: set[str]) -> tuple[list[str], float]:
        allow = torch.zeros(n_classes, dtype=torch.bool)
        for i in range(n_classes):
            if id_to_cat.get(i, "") in allow_names:
                allow[i] = True
        masked = logits.clone()
        if allow.any():
            masked[:, ~allow] = -1e9
        pred_r = [id_to_cat.get(int(i), "") for i in masked.argmax(dim=-1).tolist()]
        acc = float(np.mean([p == g for p, g in zip(pred_r, gold)]))
        return pred_r, acc

    restricted_acc = 0.0
    allowed_list: list[str] = []
    if allowed_categories is not None:
        allow_set = {str(c) for c in allowed_categories}
        allowed_list = sorted(allow_set)
        _, restricted_acc = _restrict(allow_set)

    oracle_acc = 0.0
    if oracle_test_coverage:
        _, oracle_acc = _restrict(gold_set)

    # Backward-compatible field: prefer explicit allow-list; else equal unrestricted
    # (do NOT silently use test-gold oracle).
    cov_alias = (
        restricted_acc
        if allowed_categories is not None
        else float(np.mean([p == g for p, g in zip(pred, gold)]))
    )

    return {
        "n_clips": n,
        "n_gold_categories": len(gold_set),
        "n_allowed_categories": len(allowed_list),
        "clip_top1": float(np.mean([p == g for p, g in zip(pred, gold)])),
        "clip_macro_f1": macro_f1_from_preds(pred, gold),
        "clip_majority": majority_accuracy(gold),
        "clip_restricted_top1": restricted_acc,
        "clip_coverage_aware_top1": cov_alias,
        "clip_oracle_coverage_aware_top1": oracle_acc,
        "mass_on_covered": float(np.mean([p in gold_set for p in pred])),
        "covered_categories": sorted(gold_set),
        "allowed_categories": allowed_list,
    }


def majority_accuracy(labels: list[str]) -> float:
    if not labels:
        return 0.0
    counts: dict[str, int] = defaultdict(int)
    for lab in labels:
        counts[str(lab)] += 1
    return max(counts.values()) / len(labels)


def mapped_category_summary(
    out: dict,
    id_to_label: dict[int, str],
    label_to_category: dict[str, str],
) -> dict:
    """Treat a sentence-ID model's predictions as category predictions."""
    logits = out["logits"]
    gold = [str(c) for c in out["category"]]
    pred_idx = logits.argmax(dim=-1).tolist()
    pred_cat = []
    for i in pred_idx:
        sent = id_to_label.get(int(i), "")
        pred_cat.append(str(label_to_category.get(sent, "")))
    top1 = float(np.mean([p == g and bool(p) for p, g in zip(pred_cat, gold)])) if gold else 0.0
    k = min(5, logits.size(-1))
    top5_idx = logits.topk(k, dim=-1).indices.tolist()
    top5_hits = []
    for row, gold_c in zip(top5_idx, gold):
        cats = {
            str(label_to_category.get(id_to_label.get(int(j), ""), ""))
            for j in row
        }
        top5_hits.append(gold_c in cats)
    pred_arr = np.array(pred_cat)
    gold_arr = np.array(gold)
    return {
        "n": int(len(gold)),
        "top1": top1,
        "top5": float(np.mean(top5_hits)) if top5_hits else 0.0,
        "majority": majority_accuracy(gold),
        "per_category_top1": per_group_accuracy(pred_arr, gold_arr, gold),
    }


def slim_stats(stats: dict | None) -> dict | None:
    if not stats:
        return stats
    skip = {"per_sentence_top1", "per_sentence_error"}
    return {k: v for k, v in stats.items() if k not in skip}


def format_comparison(scratch: dict, pretrained: dict, title: str | None = None) -> str:
    lines = [
        title or "RQ1  signer-independent sign-to-text  (sentence classification)",
        f"{'arm':<22} {'n':>6} {'top-1':>8} {'top-5':>8}",
        "-" * 48,
    ]
    for name, s in (("from-scratch", scratch), ("ssl-pretrained", pretrained)):
        if not s:
            lines.append(f"{name:<22} {'—':>6} {'—':>8} {'—':>8}")
            continue
        lines.append(f"{name:<22} {s['n']:>6d} {100*s['top1']:7.2f}% {100*s['top5']:7.2f}%")
    if scratch and pretrained:
        delta = 100 * (pretrained["top1"] - scratch["top1"])
        lines.append("-" * 48)
        lines.append(f"{'delta (pre - scratch)':<22} {'':>6} {delta:7.2f} pp")
    return "\n".join(lines)
