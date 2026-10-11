"""RQ2: recording prioritization vs matched-size random selection.

Active learning here is not "which unlabeled clip to send to an annotator".
Labels are known because recording is directed. The decision is which
sentences/categories the next signer should prioritize.

Revised RQ2 (session protocol): simulate one donor's recording session with a
fixed unit budget, ranked by seed-signer error (preferred) or model
uncertainty, vs a matched random fill of the same budget.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def prediction_entropy(probs: torch.Tensor) -> torch.Tensor:
    return -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)


def prediction_margin(probs: torch.Tensor) -> torch.Tensor:
    """Smaller margin => more uncertain. Returned value is uncertainty = 1 - (p1 - p2)."""
    top2 = probs.topk(k=min(2, probs.size(-1)), dim=-1).values
    if top2.size(-1) == 1:
        margin = top2[:, 0]
    else:
        margin = top2[:, 0] - top2[:, 1]
    return 1.0 - margin


@torch.no_grad()
def mc_dropout_variance(model, x: torch.Tensor, n_passes: int = 8) -> torch.Tensor:
    """Mean variance of class probabilities across stochastic forward passes."""
    was_training = model.training
    model.train()
    samples = []
    for _ in range(n_passes):
        samples.append(F.softmax(model.classify(x), dim=-1))
    stacked = torch.stack(samples, dim=0)
    var = stacked.var(dim=0).mean(dim=-1)
    if not was_training:
        model.eval()
    return var


@torch.no_grad()
def score_loader(model, loader, device, method: str = "entropy", mc_passes: int = 8) -> pd.DataFrame:
    model.eval()
    rows = []
    for batch in loader:
        x = batch["x"].to(device)
        logits = model.classify(x)
        probs = F.softmax(logits.float(), dim=-1)
        if method == "entropy":
            score = prediction_entropy(probs)
        elif method == "margin":
            score = prediction_margin(probs)
        elif method == "mc_dropout":
            score = mc_dropout_variance(model, x, n_passes=mc_passes)
        else:
            raise ValueError(f"unknown uncertainty method: {method}")
        pred = logits.argmax(dim=-1)
        for i in range(x.size(0)):
            rows.append(
                {
                    "video_id": str(batch["video_id"][i]),
                    "signer_id": str(batch["signer_id"][i]),
                    "sentence_id": str(batch["sentence_id"][i]),
                    "category": str(batch["category"][i]),
                    "y": int(batch["y"][i]),
                    "pred": int(pred[i].cpu()),
                    "uncertainty": float(score[i].cpu()),
                }
            )
    return pd.DataFrame(rows)


def aggregate_priority(
    scores: pd.DataFrame,
    by: str = "category",
    rank_by: str = "uncertainty",
) -> pd.DataFrame:
    """Rank categories or sentences for the next recording session.

    rank_by:
      - uncertainty: high mean uncertainty first (entropy / margin / MC)
      - error: high error rate first (seed-signer diagnostic; preferred for revised RQ2)
    """
    if by not in scores.columns:
        raise ValueError(f"cannot aggregate by {by}")
    if rank_by not in {"uncertainty", "error"}:
        raise ValueError(f"rank_by must be uncertainty|error, got {rank_by}")
    tmp = scores.copy()
    tmp["correct"] = tmp["pred"] == tmp["y"]
    grouped = (
        tmp.groupby(by, dropna=False)
        .agg(
            n=("uncertainty", "size"),
            mean_uncertainty=("uncertainty", "mean"),
            accuracy=("correct", "mean"),
        )
        .reset_index()
    )
    grouped["error"] = 1.0 - grouped["accuracy"]
    if rank_by == "error":
        grouped = grouped.sort_values(["error", "n", "mean_uncertainty"], ascending=[False, False, False])
    else:
        grouped = grouped.sort_values(["mean_uncertainty", "error", "n"], ascending=[False, False, False])
    grouped = grouped.reset_index(drop=True)
    grouped["rank"] = np.arange(1, len(grouped) + 1)
    return grouped


def order_units(
    available: list[str],
    ranked: pd.DataFrame,
    unit_col: str,
) -> list[str]:
    """Intersect a ranked priority list with units that exist in the donor pool."""
    avail = set(str(u) for u in available)
    ordered = [str(u) for u in ranked[unit_col].astype(str).tolist() if str(u) in avail]
    leftovers = [u for u in available if u not in ordered]
    return ordered + leftovers


def simulate_addition_curve(
    scores: pd.DataFrame,
    order: list[str],
    group_col: str = "category",
    steps: int | None = None,
) -> list[dict]:
    """DEPRECATED for RQ2: this does not retrain.

    Kept only as a diagnostic of whether high-uncertainty groups also have
    high error on a frozen model. The paper protocol is
    scripts/06_simulate_active_learning.py (retrain vs matched random).
    """
    steps = steps or len(order)
    included = set()
    curve = []
    for i, g in enumerate(order[:steps], start=1):
        included.add(g)
        subset = scores[scores[group_col].isin(included)]
        acc = float((subset["pred"] == subset["y"]).mean()) if len(subset) else 0.0
        curve.append(
            {
                "step": i,
                "added": g,
                "n_clips": int(len(subset)),
                "accuracy": acc,
            }
        )
    return curve
