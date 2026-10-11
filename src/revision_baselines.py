"""Revision calibration methods: prototypes (unchanged), cosine k-NN, head-only FT."""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.adaptation import (
    _pool_clip_embeddings,
    extract_embeddings,
    set_partial_trainable,
)
from src.config import CFG
from src.dataset import LabeledWindowDataset
from src.train_utils import set_seed


def trainable_names(model) -> list[str]:
    return [n for n, p in model.named_parameters() if p.requires_grad]


def snapshot_state(model) -> dict[str, torch.Tensor]:
    snap = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
    snap.update({n: b.detach().cpu().clone() for n, b in model.named_buffers()})
    return snap


def changed_names(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor], atol: float = 0.0) -> list[str]:
    keys = sorted(set(before) | set(after))
    out = []
    for k in keys:
        if k not in before or k not in after:
            out.append(k)
            continue
        if before[k].shape != after[k].shape or not torch.allclose(before[k], after[k], atol=atol):
            out.append(k)
    return out


def set_head_only_trainable(model) -> list:
    """Freeze encoder; update the existing source-trained classification head."""
    return set_partial_trainable(model, n_unfreeze_layers=0)


def adapt_finetune(
    model,
    train_table,
    label_map: dict[str, int],
    device,
    *,
    mode: str,
    epochs: int = 8,
    batch_size: int | None = None,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    adapt_seed: int = 42,
    tag: str = "adapt",
) -> dict:
    """Reset is the caller's job: load a fresh base before calling.

    ``mode`` is ``head_only`` or ``partial_ft``. Training uses ``model.train()``
    (dropout on) even when encoder weights are frozen.
    """
    if train_table is None or len(train_table) == 0:
        return {"trainable": [], "n_windows": 0, "skipped": True}
    set_seed(int(adapt_seed))
    batch_size = batch_size or CFG.finetune.batch_size
    ds = LabeledWindowDataset(train_table, label_map=label_map, label_col="category")
    if len(ds) == 0:
        return {"trainable": [], "n_windows": 0, "skipped": True}
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    n_unfreeze = 0 if mode == "head_only" else 2
    params = set_partial_trainable(model, n_unfreeze_layers=n_unfreeze)
    names = trainable_names(model)
    before = snapshot_state(model)
    if not params:
        return {"trainable": names, "n_windows": len(ds), "skipped": True}
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    model.train()
    for epoch in range(1, epochs + 1):
        running = 0.0
        n = 0
        correct = 0
        for batch in tqdm(loader, desc=f"{tag} {epoch}/{epochs}", leave=False):
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            logits = model.classify(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            running += float(loss.item()) * x.size(0)
            n += x.size(0)
            correct += int((logits.argmax(-1) == y).sum().item())
        print(
            f"  [{tag}] epoch {epoch:02d}  loss={running / max(n, 1):.4f}  "
            f"train_acc={100 * correct / max(n, 1):.1f}%"
        )
    after = snapshot_state(model)
    changed = changed_names(before, after)
    unexpected = [n for n in changed if n not in set(names)]
    return {
        "trainable": names,
        "changed": changed,
        "unexpected_changes": unexpected,
        "n_windows": len(ds),
        "skipped": False,
        "encoder_frozen_ok": mode != "head_only" or not unexpected,
    }


def clip_support_bank(calib_emb: dict) -> dict:
    """One L2-normalized embedding per calibration clip, sorted for ties."""
    pooled = _pool_clip_embeddings(calib_emb)
    if pooled["emb"].numel() == 0:
        return {
            "emb": torch.zeros(0, 1),
            "y": torch.zeros(0, dtype=torch.long),
            "video_id": [],
            "category": [],
        }
    order = sorted(
        range(len(pooled["video_id"])),
        key=lambda i: (int(pooled["y"][i]), str(pooled["video_id"][i])),
    )
    idx = torch.tensor(order, dtype=torch.long)
    emb = F.normalize(pooled["emb"].index_select(0, idx).float(), dim=-1)
    return {
        "emb": emb,
        "y": pooled["y"].index_select(0, idx),
        "video_id": [pooled["video_id"][i] for i in order],
        "category": [pooled["category"][i] for i in order],
    }


def _window_cosine(model, loader, support_emb: torch.Tensor, device) -> tuple[dict, torch.Tensor]:
    model.eval()
    emb_out = extract_embeddings(model, loader, device)
    query = F.normalize(emb_out["emb"].float(), dim=-1)
    if support_emb.numel() == 0:
        sim = query.new_zeros(query.size(0), 0)
    else:
        sim = query @ support_emb.to(query.device).float().T
    return emb_out, sim


def knn_window_logits(
    sim: torch.Tensor,
    support_y: torch.Tensor,
    n_classes: int,
    *,
    k: int,
) -> torch.Tensor:
    """Per-window class scores from cosine k-NN.

    k=1 uses max cosine per class (identical to a one-example prototype).
    k=3 uses uniform majority among the top-3 clips; ties break by summed
    cosine among the tied labels, then by category index.
    """
    w = sim.size(0)
    logits = sim.new_full((w, n_classes), -1e9)
    if sim.size(1) == 0:
        return logits
    y = support_y.to(sim.device).long()
    if k <= 1:
        for c in range(n_classes):
            mask = y == c
            if bool(mask.any()):
                logits[:, c] = sim[:, mask].max(dim=-1).values
        return logits

    kk = min(int(k), sim.size(1))
    topv, topi = sim.topk(kk, dim=-1)
    topy = y[topi]
    for i in range(w):
        labels = topy[i].tolist()
        values = topv[i].tolist()
        counts: dict[int, int] = defaultdict(int)
        sums: dict[int, float] = defaultdict(float)
        for lab, val in zip(labels, values):
            counts[int(lab)] += 1
            sums[int(lab)] += float(val)
        best_count = max(counts.values())
        tied = [c for c, n in counts.items() if n == best_count]
        if len(tied) == 1:
            winner = tied[0]
        else:
            best_sum = max(sums[c] for c in tied)
            tied = [c for c in tied if sums[c] == best_sum]
            winner = min(tied)
        for c, n in counts.items():
            logits[i, c] = float(n) + 1e-3 * sums[c]
        # ensure argmax is the documented winner if float noise collides
        logits[i, winner] = logits[i, winner] + 1e-6
    return logits


@torch.no_grad()
def predict_knn(
    model,
    loader,
    support: dict,
    device,
    *,
    n_classes: int,
    k: int = 1,
) -> dict:
    emb_out, sim = _window_cosine(model, loader, support["emb"], device)
    logits = knn_window_logits(sim, support["y"], n_classes, k=k)
    return {
        "logits": logits.cpu(),
        "y": emb_out["y"],
        "signer_id": emb_out["signer_id"],
        "sentence_id": emb_out["sentence_id"],
        "category": emb_out["category"],
        "video_id": emb_out["video_id"],
    }


def support_categories(support: dict, id_to_cat: dict[int, str]) -> list[str]:
    labs = sorted({int(y) for y in support["y"].tolist()})
    return [id_to_cat[c] for c in labs if c in id_to_cat]
