"""Stage B: replay historical base / prototype (and optional partial-FT) scores.

Deterministic methods must match the recorded v2 equal-candidate table and the
five-draw prototype summaries to displayed precision. Partial-FT is optional
because shuffle RNG depends on job order.

    python -u scripts/revision_historical_replay.py
    python -u scripts/revision_historical_replay.py --partial-ft-v2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.adaptation import (
    build_classifier,
    build_prototypes,
    eval_category_clips,
    extract_embeddings,
    partial_finetune,
    predict_prototypical,
)
from src.config import CFG
from src.dataset import LabeledWindowDataset
from src.revision_protocol import (
    REVISION_ROOT,
    category_map,
    id_to_category,
    load_signtalk_windows,
)
from src.train_utils import get_device, set_seed, write_json


def _table(windows, signer: str, video_ids: list[str]):
    return windows[
        windows["signer_id"].eq(str(signer))
        & windows["video_id"].isin(video_ids)
    ].reset_index(drop=True)


def _mean_pct(vals: list[float]) -> float:
    return float(100.0 * np.mean(vals)) if vals else float("nan")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--partial-ft-v2", action="store_true")
    p.add_argument("--draws-prototype", action="store_true", default=True)
    args = p.parse_args()

    set_seed(42)
    device = get_device()
    manifest = json.loads((REVISION_ROOT / "revision_manifest.json").read_text(encoding="utf-8"))
    windows = load_signtalk_windows()
    label_map = category_map(windows)
    id_to_cat = id_to_category(label_map)
    n_classes = len(label_map)

    eq = json.loads((ROOT / "results" / "rq4_equal_candidate_v2.json").read_text(encoding="utf-8"))
    eq_by = {
        (str(r["test_signer"]), int(r["n_per_category"])): r for r in eq["runs"]
    }

    v2_rows = []
    for sel in manifest["v2_single_draw_selections"]:
        s = sel["target_signer"]
        n = int(sel["budget_per_category"])
        model = build_classifier(
            n_classes, device, full_checkpoint=sel["base_checkpoint"]
        )
        final_test = _table(windows, s, sel["final_test_video_ids"])
        calib = _table(windows, s, sel["calibration_video_ids"])
        allowed = set(sel["calibration_candidate_categories"])

        base_full = eval_category_clips(
            model, final_test, label_map, id_to_cat, device, allowed_categories=None
        )
        base_restr = eval_category_clips(
            model, final_test, label_map, id_to_cat, device, allowed_categories=allowed
        )
        calib_loader = DataLoader(
            LabeledWindowDataset(calib, label_map=label_map, label_col="category"),
            batch_size=CFG.finetune.batch_size,
            shuffle=False,
        )
        calib_emb = extract_embeddings(model, calib_loader, device)
        protos = build_prototypes(calib_emb, n_classes)
        proto_norm = protos.norm(dim=-1)
        n_zero = int((proto_norm < 1e-6).sum().item())
        proto_scores = eval_category_clips(
            model,
            final_test,
            label_map,
            id_to_cat,
            device,
            predicter=lambda m, loader, dev, _p=protos: predict_prototypical(m, loader, _p, dev),
            allowed_categories=allowed,
        )
        ref = eq_by[(s, n)]
        row = {
            "test_signer": s,
            "n_per_category": n,
            "replay_base26": base_full["clip_top1"],
            "replay_baseC": base_restr["clip_restricted_top1"],
            "replay_proto": proto_scores["clip_top1"],
            "hist_base26": ref["base_unrestricted_top1"],
            "hist_baseC": ref["base_calib_restricted_top1"],
            "hist_proto": ref["prototype_top1"],
            "n_zero_prototypes": n_zero,
            "n_calib_cats": len(allowed),
            "candidates_matched": n_zero == 0 or n_zero == (n_classes - len(allowed)),
        }
        row["abs_diff"] = {
            "base26": abs(row["replay_base26"] - row["hist_base26"]),
            "baseC": abs(row["replay_baseC"] - row["hist_baseC"]),
            "proto": abs(row["replay_proto"] - row["hist_proto"]),
        }
        if args.partial_ft_v2:
            model_ft = build_classifier(
                n_classes, device, full_checkpoint=sel["base_checkpoint"]
            )
            partial_finetune(
                model_ft,
                calib,
                label_map,
                device,
                epochs=8,
                tag=f"replay-v2-t{s}-n{n}",
            )
            ft = eval_category_clips(
                model_ft,
                final_test,
                label_map,
                id_to_cat,
                device,
                allowed_categories=allowed,
            )
            row["replay_ft26"] = ft["clip_top1"]
            row["replay_ftC"] = ft["clip_restricted_top1"]
            row["hist_ft26"] = ref["partial_ft_unrestricted_top1"]
            row["hist_ftC"] = ref["partial_ft_calib_restricted_top1"]
            del model_ft
        v2_rows.append(row)
        print(
            f"v2 t={s} n={n}  base26={100*row['replay_base26']:.2f} "
            f"(hist {100*row['hist_base26']:.2f})  "
            f"baseC={100*row['replay_baseC']:.2f}  "
            f"proto={100*row['replay_proto']:.2f} "
            f"(hist {100*row['hist_proto']:.2f})"
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _slot(key_replay, key_hist):
        return {
            "replay_mean": _mean_pct([r[key_replay] for r in v2_rows]),
            "hist_mean": _mean_pct([r[key_hist] for r in v2_rows]),
            "max_abs_diff": float(max(r["abs_diff"][key_replay.replace("replay_", "")] for r in v2_rows)),
        }

    # grouped by n
    v2_by_n = {}
    for n in (1, 5, 20):
        pts = [r for r in v2_rows if r["n_per_category"] == n]
        v2_by_n[str(n)] = {
            "base26": {
                "replay": _mean_pct([r["replay_base26"] for r in pts]),
                "hist": 4.04,
            },
            "baseC": {
                "replay": _mean_pct([r["replay_baseC"] for r in pts]),
                "hist": 11.80,
            },
            "proto": {
                "replay": _mean_pct([r["replay_proto"] for r in pts]),
                "hist": {1: 39.45, 5: 49.95, 20: 55.33}[n],
            },
        }

    draw_rows = []
    if args.draws_prototype:
        for sel in manifest["selections"]:
            if sel.get("arm") and sel["arm"] != "scratch":
                continue
            s = sel["target_signer"]
            n = int(sel["budget_per_category"])
            d = int(sel["draw_id"])
            model = build_classifier(
                n_classes, device, full_checkpoint=sel["base_checkpoint"]
            )
            final_test = _table(windows, s, sel["final_test_video_ids"])
            calib = _table(windows, s, sel["calibration_video_ids"])
            allowed = set(sel["calibration_candidate_categories"])
            calib_loader = DataLoader(
                LabeledWindowDataset(calib, label_map=label_map, label_col="category"),
                batch_size=CFG.finetune.batch_size,
                shuffle=False,
            )
            calib_emb = extract_embeddings(model, calib_loader, device)
            protos = build_prototypes(calib_emb, n_classes)
            proto_scores = eval_category_clips(
                model,
                final_test,
                label_map,
                id_to_cat,
                device,
                predicter=lambda m, loader, dev, _p=protos: predict_prototypical(
                    m, loader, _p, dev
                ),
                allowed_categories=allowed,
            )
            hist = sel.get("historical_prototype_clip_top1")
            draw_rows.append(
                {
                    "test_signer": s,
                    "n_per_category": n,
                    "draw_id": d,
                    "replay_proto": proto_scores["clip_top1"],
                    "hist_proto": hist,
                    "abs_diff": None if hist is None else abs(proto_scores["clip_top1"] - hist),
                }
            )
            print(
                f"draw t={s} n={n} d={d}  proto={100*proto_scores['clip_top1']:.2f}"
                + (f" hist={100*hist:.2f}" if hist is not None else "")
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    draw_summary = {}
    for n in (1, 5, 20):
        pts = [r for r in draw_rows if r["n_per_category"] == n]
        diffs = [r["abs_diff"] for r in pts if r["abs_diff"] is not None]
        draw_summary[str(n)] = {
            "replay_mean": _mean_pct([r["replay_proto"] for r in pts]),
            "hist_mean": _mean_pct([r["hist_proto"] for r in pts if r["hist_proto"] is not None]),
            "max_abs_diff": float(max(diffs)) if diffs else None,
        }

    displayed = {
        "1": {"base26": 4.04, "baseC": 11.80, "proto": 39.45},
        "5": {"base26": 4.04, "baseC": 11.80, "proto": 49.95},
        "20": {"base26": 4.04, "baseC": 11.80, "proto": 55.33},
    }
    v2_ok = True
    explanations = []
    for n, slot in v2_by_n.items():
        for k, target in displayed[n].items():
            got = slot[k]["replay"]
            if abs(got - target) > 0.015:  # displayed hundredths
                v2_ok = False
                explanations.append(f"v2 n={n} {k}: replay {got:.2f} vs displayed {target:.2f}")

    draws_ok = True
    for n, slot in draw_summary.items():
        if slot["max_abs_diff"] is not None and slot["max_abs_diff"] > 1e-8:
            draws_ok = False
            explanations.append(
                f"five-draw proto n={n}: max |replay-hist|={slot['max_abs_diff']:.3g}"
            )

    payload = {
        "v2_rows": v2_rows,
        "v2_by_n": v2_by_n,
        "draw_rows": draw_rows,
        "draw_summary": draw_summary,
        "v2_matches_displayed": v2_ok,
        "five_draw_prototype_matches": draws_ok,
        "explanations": explanations,
        "verdict": "match" if v2_ok and draws_ok else "inspect",
    }
    write_json(REVISION_ROOT / "historical_replay.json", payload)
    print("\n=== v2 means (replay %) ===")
    for n, slot in v2_by_n.items():
        print(
            f"n={n}  base26={slot['base26']['replay']:.2f}  "
            f"baseC={slot['baseC']['replay']:.2f}  proto={slot['proto']['replay']:.2f}"
        )
    print("verdict:", payload["verdict"])
    for e in explanations:
        print(" ", e)
    print("wrote", REVISION_ROOT / "historical_replay.json")


if __name__ == "__main__":
    main()
