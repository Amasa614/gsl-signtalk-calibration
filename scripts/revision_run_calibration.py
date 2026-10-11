"""Stage D: run the revision calibration grid on frozen selections.

    python -u scripts/revision_run_calibration.py --pilot
    python -u scripts/revision_run_calibration.py
    python -u scripts/revision_run_calibration.py --skip-adapt  # support methods only
    python -u scripts/revision_run_calibration.py --resume      # skip finished cells (default)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from src.adaptation import (
    build_classifier,
    build_prototypes,
    eval_category_clips,
    extract_embeddings,
    predict_prototypical,
)
from src.config import CFG
from src.dataset import LabeledWindowDataset
from src.metrics import predict_loader, score_clip_category
from src.revision_baselines import (
    adapt_finetune,
    clip_support_bank,
    predict_knn,
    support_categories,
)
from src.revision_protocol import (
    ADAPT_EPOCHS,
    ADAPT_SEEDS,
    BATCH_SIZE,
    GRAD_CLIP,
    LR,
    REVISION_ROOT,
    WEIGHT_DECAY,
    category_map,
    id_to_category,
    load_signtalk_windows,
    sha256_json,
)
from src.train_utils import get_device, set_seed, write_json


def _table(windows, signer: str, video_ids: list[str]):
    return windows[
        windows["signer_id"].eq(str(signer)) & windows["video_id"].isin(video_ids)
    ].reset_index(drop=True)


def _save_preds(path: Path, pooled, pred_n, pred_c, candidates: list[str]) -> None:
    rows = []
    for i, vid in enumerate(pooled["video_id"]):
        rows.append(
            {
                "video_id": vid,
                "true_category": pooled["category"][i],
                "prediction_normal_rule": pred_n[i],
                "prediction_matched_candidates": pred_c[i],
                "candidate_categories": candidates,
            }
        )
    write_json(path, rows)


def _dual_scores(raw, id_to_cat, allowed):
    full = score_clip_category(raw, id_to_cat, allowed_categories=None)
    restr = score_clip_category(raw, id_to_cat, allowed_categories=allowed)
    return full, restr


def _selection_complete(path: Path, seeds: list[int], skip_adapt: bool) -> dict | None:
    """Return the saved record if this cell is finished for the requested seeds."""
    if not path.exists():
        return None
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if "scores" not in rec:
        return None
    if skip_adapt:
        return rec
    need = {(mode, int(seed)) for mode in ("head_only", "partial_ft") for seed in seeds}
    have = {
        (str(a.get("mode")), int(a.get("adapt_seed")))
        for a in rec.get("adapt") or []
    }
    if need <= have:
        return rec
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pilot", action="store_true", help="signer 5, n=1, draw 0 only")
    p.add_argument("--skip-adapt", action="store_true")
    p.add_argument("--signers", type=str, default="")
    p.add_argument("--seeds", type=str, default="")
    p.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="skip selections that already have a complete run JSON (default)",
    )
    p.add_argument("--no-resume", action="store_true", help="recompute every selection")
    args = p.parse_args()
    if args.no_resume:
        args.resume = False

    set_seed(42)
    device = get_device()
    manifest = json.loads((REVISION_ROOT / "revision_manifest.json").read_text(encoding="utf-8"))
    windows = load_signtalk_windows()
    label_map = category_map(windows)
    id_to_cat = id_to_category(label_map)
    n_classes = len(label_map)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()] or list(ADAPT_SEEDS)
    want_signers = [x.strip() for x in args.signers.split(",") if x.strip()]

    selections = list(manifest["selections"])
    if args.pilot:
        selections = [
            s
            for s in selections
            if s["target_signer"] == "5"
            and int(s["budget_per_category"]) == 1
            and int(s["draw_id"]) == 0
        ]
        seeds = [42]
    elif want_signers:
        selections = [s for s in selections if s["target_signer"] in want_signers]

    pred_dir = REVISION_ROOT / "predictions"
    run_dir = REVISION_ROOT / "runs"
    pred_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Cache unadapted base logits once per signer (final-test IDs are split-constant).
    base_cache: dict[str, dict] = {}
    records = []

    for i, sel in enumerate(selections, start=1):
        s = sel["target_signer"]
        n = int(sel["budget_per_category"])
        d = int(sel["draw_id"])
        key = f"s{s}_n{n}_d{d}"
        rec_path = run_dir / f"{key}.json"
        if args.resume:
            prev = _selection_complete(rec_path, seeds, args.skip_adapt)
            if prev is not None:
                records.append(prev)
                print(f"[{i}/{len(selections)}] {key}  skip (already complete)")
                continue
        print(f"[{i}/{len(selections)}] {key}")
        final_test = _table(windows, s, sel["final_test_video_ids"])
        calib = _table(windows, s, sel["calibration_video_ids"])
        allowed = list(sel["calibration_candidate_categories"])
        allowed_set = set(allowed)
        test_ds = LabeledWindowDataset(final_test, label_map=label_map, label_col="category")
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        if s not in base_cache:
            t0 = time.perf_counter()
            model = build_classifier(n_classes, device, full_checkpoint=sel["base_checkpoint"])
            raw = predict_loader(model, test_loader, device)
            base_cache[s] = raw
            torch.save(
                {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in raw.items()},
                REVISION_ROOT / f"base_logits_signer{s}.pt",
            )
            print(f"  cached base logits signer {s} in {time.perf_counter()-t0:.1f}s")
            del model
        raw_base = base_cache[s]
        base_full, base_restr = _dual_scores(raw_base, id_to_cat, allowed_set)

        model = build_classifier(n_classes, device, full_checkpoint=sel["base_checkpoint"])
        calib_loader = DataLoader(
            LabeledWindowDataset(calib, label_map=label_map, label_col="category"),
            batch_size=BATCH_SIZE,
            shuffle=False,
        )
        t0 = time.perf_counter()
        calib_emb = extract_embeddings(model, calib_loader, device)
        protos = build_prototypes(calib_emb, n_classes)
        support = clip_support_bank(calib_emb)
        proto_valid = [
            id_to_cat[c]
            for c in range(n_classes)
            if float(protos[c].norm()) >= 1e-6
        ]
        support_cats = support_categories(support, id_to_cat)
        candidate_flag = sorted(proto_valid) != sorted(allowed)
        proto_raw = predict_prototypical(model, test_loader, protos, device)
        knn1_raw = predict_knn(model, test_loader, support, device, n_classes=n_classes, k=1)
        proto_full, proto_restr = _dual_scores(proto_raw, id_to_cat, allowed_set)
        knn1_full, knn1_restr = _dual_scores(knn1_raw, id_to_cat, allowed_set)
        knn3 = None
        if n in (5, 20):
            knn3_raw = predict_knn(model, test_loader, support, device, n_classes=n_classes, k=3)
            knn3_full, knn3_restr = _dual_scores(knn3_raw, id_to_cat, allowed_set)
            knn3 = {"full": knn3_full, "restr": knn3_restr}
        feat_s = time.perf_counter() - t0

        from src.metrics import pool_clips

        def _dump(raw, method: str):
            pooled = pool_clips(raw)
            pred_n = [id_to_cat[int(i)] for i in pooled["logits"].argmax(-1).tolist()]
            masked = pooled["logits"].clone()
            allow = torch.zeros(n_classes, dtype=torch.bool)
            for idx, name in id_to_cat.items():
                if name in allowed_set:
                    allow[idx] = True
            if allow.any():
                masked[:, ~allow] = -1e9
            pred_c = [id_to_cat[int(i)] for i in masked.argmax(-1).tolist()]
            _save_preds(pred_dir / f"{key}_{method}.json", pooled, pred_n, pred_c, allowed)

        _dump(raw_base, "base")
        _dump(proto_raw, "prototype")
        _dump(knn1_raw, "knn1")
        if knn3:
            _dump(knn3_raw, "knn3")

        rec = {
            "selection_key": key,
            "test_signer": s,
            "n_per_category": n,
            "draw_id": d,
            "base_checkpoint_sha256": sel["base_checkpoint_sha256"],
            "calib_id_hash": sha256_json(sel["calibration_video_ids"]),
            "test_id_hash": sha256_json(sel["final_test_video_ids"]),
            "n_calib_clips": sel["n_calib_clips"],
            "n_calib_windows": sel["n_calib_windows"],
            "n_final_test_clips": sel["n_final_test_clips"],
            "n_candidates": len(allowed),
            "prototype_valid_categories": proto_valid,
            "knn_support_categories": support_cats,
            "candidate_set_mismatch": candidate_flag,
            "feature_seconds": feat_s,
            "scores": {
                "base26": {"clip_top1": base_full["clip_top1"], "clip_macro_f1": base_full["clip_macro_f1"]},
                "baseC": {
                    "clip_top1": base_restr["clip_restricted_top1"],
                    "clip_macro_f1": base_restr["clip_macro_f1"],
                },
                "prototypeC": {
                    "clip_top1": proto_restr["clip_restricted_top1"],
                    "clip_macro_f1": proto_restr["clip_macro_f1"],
                },
                "knn1C": {
                    "clip_top1": knn1_restr["clip_restricted_top1"],
                    "clip_macro_f1": knn1_restr["clip_macro_f1"],
                },
            },
        }
        if knn3:
            rec["scores"]["knn3C"] = {
                "clip_top1": knn3["restr"]["clip_top1"],
                "clip_macro_f1": knn3["restr"]["clip_macro_f1"],
            }

        if not args.skip_adapt:
            for seed in seeds:
                for mode in ("head_only", "partial_ft"):
                    tag = f"{key}_{mode}_s{seed}"
                    out_ckpt = run_dir / f"{tag}.pt"
                    t1 = time.perf_counter()
                    adapted = build_classifier(
                        n_classes, device, full_checkpoint=sel["base_checkpoint"]
                    )
                    info = adapt_finetune(
                        adapted,
                        calib,
                        label_map,
                        device,
                        mode=mode,
                        epochs=ADAPT_EPOCHS,
                        batch_size=BATCH_SIZE,
                        lr=LR,
                        weight_decay=WEIGHT_DECAY,
                        adapt_seed=seed,
                        tag=tag,
                    )
                    raw_ft = predict_loader(adapted, test_loader, device)
                    full, restr = _dual_scores(raw_ft, id_to_cat, allowed_set)
                    _dump(raw_ft, f"{mode}_s{seed}")
                    torch.save(
                        {"model": adapted.state_dict(), "adapt": info, "seed": seed},
                        out_ckpt,
                    )
                    rec.setdefault("adapt", []).append(
                        {
                            "mode": mode,
                            "adapt_seed": seed,
                            "seconds": time.perf_counter() - t1,
                            "trainable": info.get("trainable"),
                            "unexpected_changes": info.get("unexpected_changes"),
                            "scores26": {
                                "clip_top1": full["clip_top1"],
                                "clip_macro_f1": full["clip_macro_f1"],
                            },
                            "scoresC": {
                                "clip_top1": restr["clip_restricted_top1"],
                                "clip_macro_f1": restr["clip_macro_f1"],
                            },
                        }
                    )
                    print(
                        f"  {mode} seed={seed}  26={100*full['clip_top1']:.2f}%  "
                        f"C={100*restr['clip_restricted_top1']:.2f}%"
                    )
                    del adapted
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        records.append(rec)
        write_json(run_dir / f"{key}.json", rec)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_json(REVISION_ROOT / "run_records.json", {"runs": records})
    print(f"wrote {len(records)} selection records")


if __name__ == "__main__":
    main()
