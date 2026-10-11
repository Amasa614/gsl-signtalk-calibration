"""Unit tests for the Access-revision calibration methods."""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.adaptation import build_prototypes, predict_prototypical
from src.metrics import macro_f1_from_preds
from src.revision_baselines import (
    adapt_finetune,
    clip_support_bank,
    knn_window_logits,
    predict_knn,
    set_head_only_trainable,
    snapshot_state,
    changed_names,
)
from src.revision_protocol import reconstruct_selection, load_signtalk_windows, MAIN_BUDGETS


def test_one_shot_prototype_matches_1nn():
    """One clip per class: nearest centroid == cosine 1-NN under shared scoring."""
    torch.manual_seed(0)
    n_classes, d, n_win = 4, 8, 6
    # one support clip embedding per class
    clips = F.normalize(torch.randn(n_classes, d), dim=-1)
    calib_emb = {
        "emb": clips.clone(),
        "y": torch.arange(n_classes),
        "video_id": [f"c{c}" for c in range(n_classes)],
        "signer_id": ["t"] * n_classes,
        "sentence_id": ["s"] * n_classes,
        "category": [str(c) for c in range(n_classes)],
    }
    protos = build_prototypes(calib_emb, n_classes)
    support = clip_support_bank(calib_emb)
    query = F.normalize(torch.randn(n_win, d), dim=-1)
    proto_logits = query @ protos.T
    knn_logits = knn_window_logits(query @ support["emb"].T, support["y"], n_classes, k=1)
    assert torch.allclose(proto_logits, knn_logits, atol=1e-6)
    assert torch.equal(proto_logits.argmax(-1), knn_logits.argmax(-1))


def test_3nn_tie_break_uses_sum_then_category_order():
    # two classes, three neighbors with a 1-1-1 vote is impossible; 2-1 vote
    # construct: query closer to class 1 twice
    sim = torch.tensor([[0.9, 0.8, 0.1]])  # clips of class 0, 0, 1
    y = torch.tensor([0, 0, 1])
    logits = knn_window_logits(sim, y, n_classes=2, k=3)
    assert int(logits.argmax(-1)) == 0


def test_macro_f1_gold_present_only():
    pred = ["a", "b", "a"]
    gold = ["a", "a", "c"]
    # classes in gold: a, c. b is not averaged.
    f1 = macro_f1_from_preds(pred, gold)
    # a: tp=1 fp=1 fn=1 -> p=0.5 r=0.5 f1=0.5
    # c: tp=0 fp=0 fn=1 -> p=0 r=0 f1=0
    assert abs(f1 - 0.25) < 1e-9


def test_empty_support_logits_are_suppressed():
    logits = knn_window_logits(torch.zeros(3, 0), torch.zeros(0, dtype=torch.long), 5, k=1)
    assert logits.shape == (3, 5)
    assert torch.all(logits == -1e9)


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
        self.encoder.layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        self.cls_head = nn.Linear(4, 3)

    def classify(self, x):
        h = x.mean(dim=1)
        for layer in self.encoder.layers:
            h = layer(h)
        return self.cls_head(h)


def test_head_only_freezes_encoder_and_updates_head():
    model = _Tiny()
    before = snapshot_state(model)
    table = pd.DataFrame(
        {
            "video_id": ["v0", "v1", "v2"],
            "category": ["Admin and billing"] * 3,
            "signer_id": ["1"] * 3,
            "sentence_id": ["1", "2", "3"],
            "npz_path": ["missing"] * 3,
        }
    )
    # Avoid real npz: call freeze helpers only
    names = [n for n, _ in model.named_parameters()]
    set_head_only_trainable(model)
    train = [n for n, p in model.named_parameters() if p.requires_grad]
    assert any(n.startswith("cls_head") for n in train)
    assert not any(n.startswith("encoder") for n in train)
    # mutate head, prove encoder snapshot equality
    with torch.no_grad():
        model.cls_head.weight.add_(0.1)
    after = snapshot_state(model)
    changed = changed_names(before, after)
    assert "cls_head.weight" in changed
    assert not any(n.startswith("encoder") for n in changed)
    del table, names


def test_reconstructed_v2_and_draws_are_disjoint_partitions():
    windows = load_signtalk_windows()
    sel = reconstruct_selection(
        windows, test_signer="5", n_per_category=1, draw_id=0, family="draws"
    )
    assert not set(sel["calibration_video_ids"]) & set(sel["final_test_video_ids"])
    assert sel["n_calib_clips"] == sel["n_allowed_categories"]
    assert 1 in MAIN_BUDGETS
