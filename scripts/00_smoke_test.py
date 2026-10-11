"""End-to-end smoke test with synthetic landmarks (no videos / MediaPipe required).

Creates 3 fake signers × a handful of sentences, then runs windowing, 1-epoch
SSL, both fine-tune arms, evaluation, and the RQ2 simulation.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import FEATURE_DIM, PATHS
from src.dataset import LabeledWindowDataset, SSLWindowDataset
from src.landmark_extraction import save_landmarks
from src.metrics import format_comparison, predict_loader, summarize_predictions
from src.model import (
    MaskedPoseTransformer,
    make_frame_mask,
    nt_xent,
    reconstruction_loss,
)
from src.splits import assign_signer_splits, build_label_map, build_windows
from src.train_utils import get_device, save_checkpoint, set_seed, write_json
from src.active_learning import aggregate_priority, score_loader, simulate_addition_curve


def _make_synthetic_corpus(n_signers: int = 3, n_sentences: int = 6, n_frames: int = 80) -> pd.DataFrame:
    PATHS.landmarks.mkdir(parents=True, exist_ok=True)
    PATHS.splits.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    categories = ["dermatology", "consultation"]
    for signer in range(1, n_signers + 1):
        for sent in range(1, n_sentences + 1):
            video_id = f"s{signer}_sent{sent}"
            # Signer-specific bias + sentence-specific motion so there is a learnable signal.
            seq = rng.normal(0, 0.05, size=(n_frames, FEATURE_DIM)).astype(np.float32)
            seq += 0.15 * signer
            seq[:, :10] += 0.2 * sent
            save_landmarks(video_id, seq)
            rows.append(
                {
                    "video_id": video_id,
                    "path": f"synthetic/{video_id}.mp4",
                    "signer_id": str(signer),
                    "sentence_id": str(sent),
                    "sentence_text": f"sentence {sent}",
                    "category": categories[(sent - 1) % len(categories)],
                }
            )
    index = pd.DataFrame(rows)
    index.to_csv(PATHS.splits / "video_index.csv", index=False)
    return index


def _one_epoch_ssl(table: pd.DataFrame, device) -> MaskedPoseTransformer:
    ds = SSLWindowDataset(table, augment=True, seed=0)
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    model = MaskedPoseTransformer(feature_dim=FEATURE_DIM).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    for batch in loader:
        x = batch["x"].to(device)
        mask = make_frame_mask(x, 0.3)
        pred = model.reconstruct(x, frame_mask=mask)
        loss = reconstruction_loss(pred, x, mask)
        za = model.project(batch["view_a"].to(device))
        zb = model.project(batch["view_b"].to(device))
        loss = loss + 0.5 * nt_xent(za, zb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        break
    return model


def _one_epoch_ft(encoder: MaskedPoseTransformer | None, n_classes: int, device) -> MaskedPoseTransformer:
    label_map = {str(k): v for k, v in json.loads(PATHS.label_map.read_text()).items()}
    ds = LabeledWindowDataset(PATHS.splits / "train.csv", label_map=label_map)
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    model = MaskedPoseTransformer(feature_dim=FEATURE_DIM, n_classes=n_classes).to(device)
    if encoder is not None:
        own = model.state_dict()
        filtered = {k: v for k, v in encoder.state_dict().items() if k in own and own[k].shape == v.shape}
        model.load_state_dict(filtered, strict=False)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        loss = torch.nn.functional.cross_entropy(model.classify(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        break
    return model


def main() -> None:
    set_seed(0)
    device = get_device()
    print(f"smoke test on {device}")
    index = _make_synthetic_corpus()
    split_index = assign_signer_splits(index)
    windows = build_windows(split_index, window_size=64, stride=32)
    label_map = build_label_map(windows)
    print(f"windows={len(windows)} classes={len(label_map)}")
    assert len(windows) > 0
    assert set(split_index["split"]) >= {"train", "test"}

    ssl_table = windows[windows["split"] != "test"]
    encoder = _one_epoch_ssl(ssl_table, device)
    save_checkpoint(PATHS.ssl_encoder, {"model": encoder.state_dict(), "feature_dim": FEATURE_DIM})

    n_classes = len(label_map)
    pretrained = _one_epoch_ft(encoder, n_classes, device)
    scratch = _one_epoch_ft(None, n_classes, device)
    save_checkpoint(PATHS.finetune_pretrained, {"model": pretrained.state_dict(), "n_classes": n_classes})
    save_checkpoint(PATHS.finetune_scratch, {"model": scratch.state_dict(), "n_classes": n_classes})

    test_ds = LabeledWindowDataset(PATHS.splits / "test.csv", label_map=label_map)
    loader = DataLoader(test_ds, batch_size=4)
    s_stats = summarize_predictions(predict_loader(scratch, loader, device))
    p_stats = summarize_predictions(predict_loader(pretrained, loader, device))
    print(format_comparison(s_stats, p_stats))

    scores = score_loader(pretrained, loader, device, method="entropy")
    pri = aggregate_priority(scores, by="category", rank_by="uncertainty")
    pri_err = aggregate_priority(scores, by="category", rank_by="error")
    print("RQ2 priority (uncertainty):\n", pri.to_string(index=False))
    print("RQ2 priority (error):\n", pri_err.to_string(index=False))
    simulate_addition_curve(scores, pri["category"].astype(str).tolist())
    write_json(PATHS.results / "smoke.json", {"ok": True, "windows": int(len(windows))})
    print("smoke test passed")


if __name__ == "__main__":
    main()
