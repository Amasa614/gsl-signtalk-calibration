"""Signer-adaptive few-shot calibration helpers (RQ4).

Supports (a) nearest-centroid / prototypical classification in embedding space
and (b) short partial fine-tuning of the classification head plus the last
encoder layers. Both operate on a calibration pool carved from the held-out
signer; the final test pool is never used for adaptation.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import CFG, FEATURE_DIM
from src.dataset import LabeledWindowDataset
from src.finetune_loop import load_encoder_weights
from src.metrics import predict_loader, score_clip_category
from src.model import build_pose_model
from src.train_utils import load_checkpoint


def build_classifier(
    n_classes: int,
    device,
    *,
    pretrained: str | Path | None = None,
    full_checkpoint: str | Path | None = None,
    arch: str = "transformer",
):
    """Load either an SSL encoder init or a full fine-tuned checkpoint."""
    model = build_pose_model(arch, feature_dim=FEATURE_DIM, n_classes=n_classes).to(device)
    if full_checkpoint is not None:
        ckpt = load_checkpoint(Path(full_checkpoint), map_location=device)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state, strict=False)
    elif pretrained is not None:
        if str(arch).lower().strip() != "transformer":
            print(f"[adapt] SSL init skipped for arch={arch}")
        else:
            load_encoder_weights(model, Path(pretrained), device)
    return model


def split_calib_test(
    table: pd.DataFrame,
    *,
    calib_frac: float = 0.80,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split one signer's windows into calibration vs final test by video_id."""
    rng = np.random.default_rng(seed)
    clips = table["video_id"].astype(str).unique()
    clips = clips[rng.permutation(len(clips))]
    n_calib = int(round(len(clips) * calib_frac))
    n_calib = min(max(n_calib, 1), len(clips) - 1) if len(clips) > 1 else len(clips)
    calib_ids = set(clips[:n_calib].tolist())
    test_ids = set(clips[n_calib:].tolist()) if len(clips) > 1 else set()
    if not test_ids:
        # Degenerate: keep one clip for test if possible
        if len(clips) >= 2:
            test_ids = {clips[-1]}
            calib_ids = set(clips[:-1].tolist())
        else:
            test_ids = set(clips.tolist())
            calib_ids = set()
    calib = table[table["video_id"].astype(str).isin(calib_ids)].reset_index(drop=True)
    test = table[table["video_id"].astype(str).isin(test_ids)].reset_index(drop=True)
    return calib, test


def sample_calib_clips(
    calib_pool: pd.DataFrame,
    *,
    n_per_category: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Sample up to ``n_per_category`` distinct video_ids per category.

    ``n_per_category == 0`` returns an empty table (zero-shot).
    """
    if n_per_category <= 0 or calib_pool.empty:
        return calib_pool.iloc[0:0].copy()
    rng = np.random.default_rng(seed)
    keep: set[str] = set()
    for cat in sorted(calib_pool["category"].astype(str).unique().tolist()):
        sub = calib_pool[calib_pool["category"].astype(str).eq(cat)]
        vids = sub["video_id"].astype(str).unique()
        vids = vids[rng.permutation(len(vids))]
        keep.update(vids[: min(n_per_category, len(vids))].tolist())
    return calib_pool[calib_pool["video_id"].astype(str).isin(keep)].reset_index(drop=True)


@torch.no_grad()
def extract_embeddings(model, loader, device) -> dict:
    model.eval()
    emb_all = []
    y_all = []
    signer_all = []
    sent_all = []
    cat_all = []
    vid_all = []
    for batch in loader:
        x = batch["x"].to(device)
        emb_all.append(model.embed(x).cpu())
        y_all.append(batch["y"])
        signer_all.extend([str(s) for s in batch["signer_id"]])
        sent_all.extend([str(s) for s in batch["sentence_id"]])
        cat_all.extend([str(s) for s in batch["category"]])
        vid_all.extend([str(s) for s in batch["video_id"]])
    return {
        "emb": torch.cat(emb_all, dim=0) if emb_all else torch.zeros(0, model.d_model),
        "y": torch.cat(y_all, dim=0) if y_all else torch.zeros(0, dtype=torch.long),
        "signer_id": signer_all,
        "sentence_id": sent_all,
        "category": cat_all,
        "video_id": vid_all,
    }


def _pool_clip_embeddings(out: dict) -> dict:
    """Mean-pool window embeddings that share a video_id."""
    emb = out["emb"]
    groups: dict[str, list[int]] = defaultdict(list)
    for i, vid in enumerate(out["video_id"]):
        groups[str(vid)].append(i)
    pooled = []
    y = []
    signer_all = []
    sent_all = []
    cat_all = []
    vid_all = []
    for vid, idxs in groups.items():
        idx = torch.tensor(idxs, dtype=torch.long)
        pooled.append(emb.index_select(0, idx).mean(dim=0))
        i0 = idxs[0]
        y.append(out["y"][i0])
        signer_all.append(out["signer_id"][i0])
        sent_all.append(out["sentence_id"][i0])
        cat_all.append(out["category"][i0])
        vid_all.append(vid)
    return {
        "emb": torch.stack(pooled, dim=0) if pooled else emb,
        "y": torch.stack(y, dim=0) if y else out["y"],
        "signer_id": signer_all,
        "sentence_id": sent_all,
        "category": cat_all,
        "video_id": vid_all,
    }


def build_prototypes(calib_emb: dict, n_classes: int) -> torch.Tensor:
    """One L2-normalized prototype per class (clip-pooled), zeros if unseen."""
    pooled = _pool_clip_embeddings(calib_emb)
    protos = torch.zeros(n_classes, pooled["emb"].size(-1), dtype=torch.float32)
    counts = torch.zeros(n_classes, dtype=torch.long)
    for e, y in zip(pooled["emb"], pooled["y"].tolist()):
        yi = int(y)
        if 0 <= yi < n_classes:
            protos[yi] += e
            counts[yi] += 1
    for c in range(n_classes):
        if counts[c] > 0:
            protos[c] = protos[c] / float(counts[c])
    return F.normalize(protos, dim=-1)


@torch.no_grad()
def predict_prototypical(
    model,
    loader,
    prototypes: torch.Tensor,
    device,
) -> dict:
    """Nearest-prototype classification; logits = cosine similarity to prototypes."""
    model.eval()
    emb_out = extract_embeddings(model, loader, device)
    emb = F.normalize(emb_out["emb"].float(), dim=-1)
    protos = prototypes.to(emb.device).float()
    # Unseen classes (zero proto) stay at -inf after normalize of zeros -> nan; fix.
    proto_norm = protos.norm(dim=-1)
    logits = emb @ protos.T
    logits[:, proto_norm < 1e-6] = -1e9
    return {
        "logits": logits.cpu(),
        "y": emb_out["y"],
        "signer_id": emb_out["signer_id"],
        "sentence_id": emb_out["sentence_id"],
        "category": emb_out["category"],
        "video_id": emb_out["video_id"],
    }


def set_partial_trainable(model, n_unfreeze_layers: int = 2) -> list:
    """Freeze everything except cls_head and the last ``n_unfreeze_layers`` encoder layers."""
    for p in model.parameters():
        p.requires_grad = False
    if model.cls_head is not None:
        for p in model.cls_head.parameters():
            p.requires_grad = True
    encoder = getattr(model, "encoder", None)
    layers = list(getattr(encoder, "layers", []) or [])
    n = max(0, min(n_unfreeze_layers, len(layers)))
    for layer in layers[-n:] if n else []:
        for p in layer.parameters():
            p.requires_grad = True
    return [p for p in model.parameters() if p.requires_grad]


def partial_finetune(
    model,
    train_table: pd.DataFrame,
    label_map: dict[str, int],
    device,
    *,
    epochs: int = 8,
    batch_size: int | None = None,
    lr: float = 1e-4,
    n_unfreeze_layers: int = 2,
    label_col: str = "category",
    tag: str = "adapt",
):
    """Short adaptation pass on the calibration pool only."""
    if train_table.empty:
        return model
    batch_size = batch_size or CFG.finetune.batch_size
    ds = LabeledWindowDataset(train_table, label_map=label_map, label_col=label_col)
    if len(ds) == 0:
        return model
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    params = set_partial_trainable(model, n_unfreeze_layers=n_unfreeze_layers)
    if not params:
        return model
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=CFG.finetune.weight_decay)
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
    return model


def eval_category_clips(
    model,
    test_table: pd.DataFrame,
    label_map: dict[str, int],
    id_to_cat: dict[int, str],
    device,
    *,
    batch_size: int | None = None,
    predicter=None,
    allowed_categories: set[str] | list[str] | None = None,
    oracle_test_coverage: bool = False,
) -> dict:
    """Clip metrics on a held-out table. Primary: unrestricted top-1 + macro-F1."""
    batch_size = batch_size or CFG.finetune.batch_size
    ds = LabeledWindowDataset(test_table, label_map=label_map, label_col="category")
    if len(ds) == 0:
        return {
            "n_clips": 0,
            "clip_top1": 0.0,
            "clip_macro_f1": 0.0,
            "clip_restricted_top1": 0.0,
            "clip_coverage_aware_top1": 0.0,
            "clip_majority": 0.0,
            "n_allowed_categories": 0,
        }
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    if predicter is None:
        raw = predict_loader(model, loader, device)
    else:
        raw = predicter(model, loader, device)
    scores = score_clip_category(
        raw,
        id_to_cat,
        allowed_categories=allowed_categories,
        oracle_test_coverage=oracle_test_coverage,
    )
    skip = {"covered_categories", "allowed_categories"}
    return {k: v for k, v in scores.items() if k not in skip}
