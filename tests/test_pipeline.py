from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import FEATURE_DIM, N_COORDS, N_JOINTS
from src.metadata import SIGNTALK_LETTER_TO_SIGNER, infer_signer_from_path
from src.model import MaskedPoseTransformer, make_frame_mask, reconstruction_loss
from src.normalize import flatten_frame, normalize_sequence, pack_landmarks
from src.splits import _windows_for_length, assign_signer_splits, holdout_clips_per_signer


def test_signtalk_letter_maps_to_paper_signers():
    assert infer_signer_from_path(Path("1A.mp4")) == "1"
    assert infer_signer_from_path(Path("1001B.mp4")) == "2"
    assert infer_signer_from_path(Path("9C.mp4")) == "3"
    assert infer_signer_from_path(Path("1000E.mp4")) == "4"
    assert infer_signer_from_path(Path("998D.mp4")) == "5"
    assert infer_signer_from_path(Path("1Z.mp4")) is None
    assert set(SIGNTALK_LETTER_TO_SIGNER) == set("ABCDE")


class _L:
    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z


def test_pack_and_normalize_centers_shoulders():
    pose = [_L(0, 0, 0) for _ in range(33)]
    pose[11] = _L(0.4, 0.5, 0.0)  # left shoulder
    pose[12] = _L(0.6, 0.5, 0.0)  # right shoulder
    frame = pack_landmarks(pose=pose)
    seq = flatten_frame(frame)[None, :]
    normed = normalize_sequence(seq).reshape(1, N_JOINTS, N_COORDS)
    mid = 0.5 * (normed[0, 11] + normed[0, 12])
    assert np.allclose(mid, 0.0, atol=1e-5)
    width = np.linalg.norm(normed[0, 11] - normed[0, 12])
    assert np.isclose(width, 1.0, atol=1e-5)


def test_windows_cover_sequence():
    wins = _windows_for_length(100, window_size=64, stride=32, min_frames=16)
    assert wins[0] == (0, 64)
    assert wins[-1][1] == 100


def test_pool_clips_mean_logits_one_score_per_video():
    from src.metrics import pool_clips, summarize_predictions

    out = {
        "logits": torch.tensor([[1.0, 0.0], [3.0, 0.0], [0.0, 4.0]]),
        "y": torch.tensor([0, 0, 1]),
        "signer_id": ["5", "5", "5"],
        "sentence_id": ["a", "a", "b"],
        "category": ["c1", "c1", "c2"],
        "video_id": ["v1", "v1", "v2"],
    }
    pooled = pool_clips(out)
    assert pooled["logits"].shape[0] == 2
    v1 = pooled["video_id"].index("v1")
    assert torch.allclose(pooled["logits"][v1], torch.tensor([2.0, 0.0]))
    stats = summarize_predictions(pooled)
    assert stats["n"] == 2
    assert stats["top1"] == 1.0


def test_coverage_aware_masks_unseen_categories():
    from src.metrics import score_clip_category

    # Gold is only c0. Unrestricted argmax picks c1; restricted to {c0} must pick c0.
    out = {
        "logits": torch.tensor([[0.1, 5.0], [0.2, 4.0]]),
        "y": torch.tensor([0, 0]),
        "signer_id": ["5", "5"],
        "sentence_id": ["a", "a"],
        "category": ["c0", "c0"],
        "video_id": ["v1", "v1"],
    }
    # Default: no oracle — unrestricted top-1 fails, restricted needs allow-list
    stats = score_clip_category(out, {0: "c0", 1: "c1"})
    assert stats["n_clips"] == 1
    assert stats["clip_top1"] == 0.0
    assert stats["clip_restricted_top1"] == 0.0

    stats_allow = score_clip_category(
        out, {0: "c0", 1: "c1"}, allowed_categories={"c0"}
    )
    assert stats_allow["clip_restricted_top1"] == 1.0
    assert stats_allow["clip_coverage_aware_top1"] == 1.0

    stats_oracle = score_clip_category(
        out, {0: "c0", 1: "c1"}, oracle_test_coverage=True
    )
    assert stats_oracle["clip_oracle_coverage_aware_top1"] == 1.0
    assert stats_oracle["mass_on_covered"] == 0.0


def test_holdout_clips_per_signer_is_nonempty_at_k1_and_does_not_leak_videos():
    rows = []
    for signer in ("1",):
        for i in range(10):
            # two windows per clip, so a window-wise split would leak
            for w in range(2):
                rows.append(
                    {
                        "video_id": f"{signer}_{i}",
                        "signer_id": signer,
                        "window": w,
                    }
                )
    df = pd.DataFrame(rows)
    train, val = holdout_clips_per_signer(df, val_frac=0.20, seed=0)
    assert len(val) > 0
    assert len(train) > 0
    assert set(train["video_id"]) & set(val["video_id"]) == set()
    assert set(train["signer_id"]) == {"1"}
    assert set(val["signer_id"]) == {"1"}
    n_clips = df["video_id"].nunique()
    n_val_clips = val["video_id"].nunique()
    assert n_val_clips == 2  # 20% of 10 clips
    assert train["video_id"].nunique() == n_clips - n_val_clips
    # every window of a val clip is in val
    for vid in val["video_id"].unique():
        assert set(df.loc[df.video_id == vid, "window"]) == set(val.loc[val.video_id == vid, "window"])

    two = pd.concat(
        [
            df,
            df.assign(signer_id="2", video_id=lambda d: "2_" + d["video_id"].str.split("_").str[1]),
        ],
        ignore_index=True,
    )
    train2, val2 = holdout_clips_per_signer(two, val_frac=0.20, seed=0)
    assert set(train2["signer_id"]) == {"1", "2"}
    assert set(val2["signer_id"]) == {"1", "2"}
    assert val2.groupby("signer_id")["video_id"].nunique().min() == 2


def test_signer_split_never_mixes_signers():
    df = pd.DataFrame(
        {
            "video_id": ["a", "b", "c", "d"],
            "signer_id": ["1", "1", "2", "3"],
            "sentence_id": ["s", "s", "s", "s"],
        }
    )
    out = assign_signer_splits(df, test_signers=("3",), val_signers=("2",))
    assert set(out.loc[out.split == "test", "signer_id"]) == {"3"}
    assert set(out.loc[out.split == "train", "signer_id"]) == {"1"}


def test_fragkiadakis_is_ssl_only_never_in_label_map():
    df = pd.DataFrame(
        {
            "video_id": ["1A", "2B", "3C", "fragkiadakis__hello"],
            "signer_id": ["1", "2", "3", "nyarko"],
            "sentence_id": ["10", "10", "10", "HELLO"],
            "source": ["signtalk", "signtalk", "signtalk", "fragkiadakis"],
        }
    )
    out = assign_signer_splits(df, test_signers=("3",), val_signers=("2",))
    extra = out[out["source"] == "fragkiadakis"]
    assert len(extra) == 1
    assert (extra["split"] == "ssl").all()
    labeled = out[out["split"].isin(["train", "val", "test"])]
    assert set(labeled["source"]) == {"signtalk"}
    assert "nyarko" not in set(labeled["signer_id"].astype(str))
    train = out[out["split"] == "train"]
    train = train[train["source"].astype(str).str.lower().eq("signtalk")]
    assert "HELLO" not in set(train["sentence_id"].astype(str))
    assert "10" in set(train["sentence_id"].astype(str))


def test_masked_reconstruction_runs():
    model = MaskedPoseTransformer(feature_dim=FEATURE_DIM, d_model=32, nhead=4, num_layers=1, dim_feedforward=64)
    x = torch.randn(2, 16, FEATURE_DIM)
    mask = make_frame_mask(x, 0.3)
    pred = model.reconstruct(x, frame_mask=mask)
    loss = reconstruction_loss(pred, x, mask)
    assert pred.shape == x.shape
    assert torch.isfinite(loss)


def test_pose_bigru_classify():
    from src.model import PoseBiGRU, build_pose_model

    model = PoseBiGRU(feature_dim=FEATURE_DIM, hidden_size=16, num_layers=1, n_classes=5)
    x = torch.randn(2, 12, FEATURE_DIM)
    logits = model.classify(x)
    assert logits.shape == (2, 5)
    assert model.embed(x).shape == (2, 32)
    # encoder view exposes layers for partial-FT freeze helpers
    assert len(model.encoder.layers) == 1
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == sum(p.numel() for p in build_pose_model("bigru", n_classes=5, hidden_size=16, num_layers=1).parameters())
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1]))
    loss.backward()
    assert torch.isfinite(loss)


def test_mask_ratio_samples_methodology_range():
    x = torch.rand(32, 64, FEATURE_DIM)
    mask = make_frame_mask(x, mask_ratio=None, ratio_range=(0.15, 0.40))
    frac = mask.float().mean(dim=1)
    assert mask.any(dim=1).all()
    assert float(frac.min()) >= 0.05
    assert float(frac.max()) <= 0.55
    assert 0.15 <= float(frac.mean()) <= 0.40


def test_openpose_body25_maps_shoulders_then_normalizes():
    from src.openpose import person_to_frame

    pose = [0.0] * (25 * 3)
    # BODY_25 left shoulder=5, right shoulder=2
    pose[5 * 3 : 5 * 3 + 3] = [100.0, 200.0, 0.9]
    pose[2 * 3 : 2 * 3 + 3] = [300.0, 200.0, 0.9]
    pose[0:3] = [200.0, 80.0, 0.9]  # nose
    person = {
        "pose_keypoints_2d": pose,
        "hand_left_keypoints_2d": [1.0, 2.0, 0.8] + [0.0] * (20 * 3),
        "hand_right_keypoints_2d": [],
        "face_keypoints_2d": [],
    }
    frame = person_to_frame(person)
    assert frame[11, 0] == 100.0 and frame[11, 1] == 200.0
    assert frame[12, 0] == 300.0 and frame[12, 1] == 200.0
    seq = flatten_frame(frame)[None, :]
    normed = normalize_sequence(seq).reshape(1, N_JOINTS, N_COORDS)
    mid = 0.5 * (normed[0, 11] + normed[0, 12])
    assert np.allclose(mid, 0.0, atol=1e-5)
    width = np.linalg.norm(normed[0, 11] - normed[0, 12])
    assert np.isclose(width, 1.0, atol=1e-5)


def test_info_nce_and_projection_head():
    from src.model import info_nce

    model = MaskedPoseTransformer(feature_dim=FEATURE_DIM, d_model=32, nhead=4, num_layers=1, dim_feedforward=64)
    x = torch.randn(4, 16, FEATURE_DIM)
    z = model.project(x)
    assert z.shape[-1] == 64
    loss = info_nce(z, z)
    assert torch.isfinite(loss)


def test_rq4_calib_split_and_sample_no_leak():
    from src.adaptation import sample_calib_clips, split_calib_test

    rows = []
    for i in range(20):
        cat = f"c{i % 5}"
        rows.append(
            {
                "video_id": f"v{i}",
                "signer_id": "5",
                "category": cat,
                "sentence_id": f"s{i}",
            }
        )
    df = pd.DataFrame(rows)
    calib, test = split_calib_test(df, calib_frac=0.8, seed=0)
    assert len(calib) + len(test) == len(df)
    assert set(calib["video_id"]) & set(test["video_id"]) == set()
    sampled = sample_calib_clips(calib, n_per_category=2, seed=1)
    assert sampled["video_id"].nunique() <= 5 * 2
    empty = sample_calib_clips(calib, n_per_category=0, seed=1)
    assert len(empty) == 0


def test_rq5_perturbation_preserves_shape_no_mirror():
    from src.synthetic_signers import PerturbProfile, apply_profile, temporal_warp

    seq = np.random.randn(64, FEATURE_DIM).astype(np.float32)
    # Plant shoulders so proportion scaling has a torso frame.
    joints = seq.reshape(64, N_JOINTS, N_COORDS)
    joints[:, 11] = [-0.5, 0.0, 0.0]
    joints[:, 12] = [0.5, 0.0, 0.0]
    seq = joints.reshape(64, FEATURE_DIM)
    out = apply_profile(
        seq,
        PerturbProfile(name="t", limb_scale=1.1, hand_span_scale=0.9, speed=1.2, jitter_std=0.01, seed=3),
    )
    assert out.shape == seq.shape
    assert np.isfinite(out).all()
    warped = temporal_warp(seq, 0.8)
    assert warped.shape == seq.shape


def test_rq6_mask_substitution_zeros_some_coords():
    from src.augment_finetune import apply_mask_substitution

    x = torch.ones(4, 16, FEATURE_DIM)
    y = apply_mask_substitution(x, prob=1.0, mask_ratio_range=(0.2, 0.2), mode="zero")
    assert y.shape == x.shape
    assert (y == 0).any()
    assert (y == 1).any()
