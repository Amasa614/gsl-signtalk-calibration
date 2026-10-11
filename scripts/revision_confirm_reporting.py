"""Confirm grid completion, ± convention, paired signer gaps, and macro-F1."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.metrics import macro_f1_from_preds
from src.revision_protocol import REVISION_ROOT, SIGNERS
from src.train_utils import write_json

METHODS = {
    "base26": ("base", "prediction_normal_rule"),
    "baseC": ("base", "prediction_matched_candidates"),
    "prototypeC": ("prototype", "prediction_matched_candidates"),
    "knn1C": ("knn1", "prediction_matched_candidates"),
    "head26": ("head_only_s{seed}", "prediction_normal_rule"),
    "headC": ("head_only_s{seed}", "prediction_matched_candidates"),
    "partial26": ("partial_ft_s{seed}", "prediction_normal_rule"),
    "partialC": ("partial_ft_s{seed}", "prediction_matched_candidates"),
}
ADAPT_SEEDS = (42, 43, 44)
BUDGETS = (1, 5, 20)


def _f1_top1(path: Path, pred_key: str) -> tuple[float, float] | None:
    if not path.exists():
        return None
    rows = json.loads(path.read_text(encoding="utf-8"))
    gold = [r["true_category"] for r in rows]
    pred = [r[pred_key] for r in rows]
    top1 = float(np.mean([p == g for p, g in zip(pred, gold)])) if gold else 0.0
    return top1, float(macro_f1_from_preds(pred, gold))


def main() -> None:
    runs_path = REVISION_ROOT / "run_records.json"
    runs = json.loads(runs_path.read_text(encoding="utf-8"))["runs"]
    pred_dir = REVISION_ROOT / "predictions"

    planned = {(s, n, d) for s in SIGNERS for n in BUDGETS for d in range(5)}
    have = {(str(r["test_signer"]), int(r["n_per_category"]), int(r["draw_id"])) for r in runs}
    missing_sel = sorted(planned - have)

    adapt_gaps = []
    for r in runs:
        modes = defaultdict(set)
        for a in r.get("adapt") or []:
            modes[a["mode"]].add(int(a["adapt_seed"]))
        for mode in ("head_only", "partial_ft"):
            missing_seeds = [s for s in ADAPT_SEEDS if s not in modes[mode]]
            if missing_seeds:
                adapt_gaps.append(
                    {
                        "key": r.get("selection_key"),
                        "mode": mode,
                        "missing_seeds": missing_seeds,
                    }
                )

    # Per-selection metrics from prediction files (exact C-rule F1).
    cells = []
    pred_missing = []
    for r in runs:
        s, n, d = str(r["test_signer"]), int(r["n_per_category"]), int(r["draw_id"])
        key = r.get("selection_key") or f"s{s}_n{n}_d{d}"
        cell = {"signer": s, "n": n, "draw": d, "key": key, "metrics": {}}
        for mname, (stem, pkey) in METHODS.items():
            if "{seed}" in stem:
                vals_t, vals_f = [], []
                for seed in ADAPT_SEEDS:
                    path = pred_dir / f"{key}_{stem.format(seed=seed)}.json"
                    got = _f1_top1(path, pkey)
                    if got is None:
                        pred_missing.append(str(path.name))
                        continue
                    vals_t.append(got[0])
                    vals_f.append(got[1])
                if vals_t:
                    cell["metrics"][mname] = {
                        "top1": float(np.mean(vals_t)),
                        "macro_f1": float(np.mean(vals_f)),
                        "n_seeds": len(vals_t),
                    }
            else:
                path = pred_dir / f"{key}_{stem}.json"
                got = _f1_top1(path, pkey)
                if got is None:
                    pred_missing.append(str(path.name))
                    continue
                cell["metrics"][mname] = {
                    "top1": got[0],
                    "macro_f1": got[1],
                    "n_seeds": 1,
                }
        cells.append(cell)

    def aggregate(n: int, method: str, field: str) -> dict:
        signer_means = []
        pooled = []
        per_signer = {}
        n_draws = {}
        for s in SIGNERS:
            pts = [
                c["metrics"][method][field]
                for c in cells
                if c["signer"] == s and c["n"] == n and method in c["metrics"]
            ]
            n_draws[s] = len(pts)
            if not pts:
                continue
            sm = float(np.mean(pts))
            per_signer[s] = sm
            signer_means.append(sm)
            pooled.extend(pts)
        return {
            "n_signers": len(signer_means),
            "draws_per_signer": n_draws,
            "mean_of_signer_means": float(np.mean(signer_means)) if signer_means else None,
            "std_across_signer_means_ddof1": (
                float(np.std(signer_means, ddof=1)) if len(signer_means) > 1 else 0.0
            ),
            "pooled_n": len(pooled),
            "pooled_mean": float(np.mean(pooled)) if pooled else None,
            "pooled_std_ddof1": float(np.std(pooled, ddof=1)) if len(pooled) > 1 else 0.0,
            "per_signer": per_signer,
        }

    by_budget = {}
    paired = {}
    for n in BUDGETS:
        by_budget[str(n)] = {
            m: {
                "top1": aggregate(n, m, "top1"),
                "macro_f1": aggregate(n, m, "macro_f1"),
            }
            for m in METHODS
        }
        # paired signer-level differences after averaging draws (and seeds)
        pairs = {}
        for a, b, name in (
            ("partialC", "knn1C", "partialC_minus_knn1C"),
            ("knn1C", "prototypeC", "knn1C_minus_prototypeC"),
            ("partialC", "prototypeC", "partialC_minus_prototypeC"),
            ("partialC", "headC", "partialC_minus_headC"),
            ("prototypeC", "baseC", "prototypeC_minus_baseC"),
        ):
            diffs = {}
            for s in SIGNERS:
                va = by_budget[str(n)][a]["top1"]["per_signer"].get(s)
                vb = by_budget[str(n)][b]["top1"]["per_signer"].get(s)
                if va is None or vb is None:
                    continue
                diffs[s] = float(va - vb)
            vals = list(diffs.values())
            pairs[name] = {
                "mean_of_signer_diffs": float(np.mean(vals)) if vals else None,
                "n_signers_positive": int(sum(1 for v in vals if v > 0)),
                "n_signers_negative": int(sum(1 for v in vals if v < 0)),
                "n_signers_zero": int(sum(1 for v in vals if v == 0)),
                "min": float(min(vals)) if vals else None,
                "max": float(max(vals)) if vals else None,
                "per_signer": diffs,
            }
        paired[str(n)] = pairs

    # Compare old pooled-25 convention on deterministic prototype n=1 and
    # seed-42-only partial FT n=20 (closest to historical Table 4).
    old_style = {}
    proto1 = [
        c["metrics"]["prototypeC"]["top1"]
        for c in cells
        if c["n"] == 1 and "prototypeC" in c["metrics"]
    ]
    old_style["prototype_n1_pooled_25"] = {
        "n": len(proto1),
        "mean": float(np.mean(proto1)),
        "std_ddof1": float(np.std(proto1, ddof=1)),
    }
    # seed-42 only partial 26 at n=20 from run JSON
    ft20_s42 = []
    ft20_allseeds_signer = defaultdict(list)
    for r in runs:
        if int(r["n_per_category"]) != 20:
            continue
        s = str(r["test_signer"])
        seed_vals = [
            a["scores26"]["clip_top1"]
            for a in r.get("adapt") or []
            if a["mode"] == "partial_ft"
        ]
        s42 = [
            a["scores26"]["clip_top1"]
            for a in r.get("adapt") or []
            if a["mode"] == "partial_ft" and int(a["adapt_seed"]) == 42
        ]
        if s42:
            ft20_s42.append(s42[0])
        if seed_vals:
            ft20_allseeds_signer[s].append(float(np.mean(seed_vals)))
    old_style["partial26_n20_seed42_pooled"] = {
        "n": len(ft20_s42),
        "mean": float(np.mean(ft20_s42)) if ft20_s42 else None,
        "std_ddof1": float(np.std(ft20_s42, ddof=1)) if len(ft20_s42) > 1 else 0.0,
        "note": "same pooling as old Table 4 (25 draw-level values, seed 42 only)",
    }
    signer_means_ft = [float(np.mean(v)) for v in ft20_allseeds_signer.values()]
    old_style["partial26_n20_new_convention"] = {
        "n_signers": len(signer_means_ft),
        "mean": float(np.mean(signer_means_ft)) if signer_means_ft else None,
        "std_across_signer_means_ddof1": (
            float(np.std(signer_means_ft, ddof=1)) if len(signer_means_ft) > 1 else 0.0
        ),
        "note": "seeds averaged within draw, draws within signer, then 5 signer means",
    }

    payload = {
        "grid": {
            "planned_selections": 75,
            "completed_selections": len(have),
            "missing_selections": missing_sel,
            "planned_adapt_fits": 75 * 2 * 3,
            "adapt_gaps": adapt_gaps,
            "prediction_files_missing": pred_missing,
            "complete": not missing_sel and not adapt_gaps and not pred_missing,
        },
        "plus_minus_definition": {
            "new_table": (
                "Sample std (ddof=1) across the five signer means. "
                "Each signer mean averages five draws; gradient methods first "
                "average three adaptation seeds inside each draw. "
                "This is NOT the old 25-run pooled std."
            ),
            "old_table_4": (
                "Sample std (ddof=1) over 25 pooled (signer x draw) values; "
                "partial FT used one optimization seed."
            ),
        },
        "old_vs_new_std": old_style,
        "by_budget": by_budget,
        "paired_signer_top1": paired,
    }
    out = REVISION_ROOT / "tables" / "reporting_confirmation.json"
    write_json(out, payload)

    print("GRID", "complete" if payload["grid"]["complete"] else "INCOMPLETE")
    print("selections", len(have), "/ 75  adapt_gaps", len(adapt_gaps), "pred_missing", len(pred_missing))
    print("\n± check  prototype n=1")
    print("  new (5 signer means)", f"{100*by_budget['1']['prototypeC']['top1']['mean_of_signer_means']:.2f} ± {100*by_budget['1']['prototypeC']['top1']['std_across_signer_means_ddof1']:.2f}")
    print("  old-style pooled 25", f"{100*old_style['prototype_n1_pooled_25']['mean']:.2f} ± {100*old_style['prototype_n1_pooled_25']['std_ddof1']:.2f}")
    print("\n± check  partial-FT 26-way n=20")
    print("  new (5 signer means, 3 seeds)", f"{100*old_style['partial26_n20_new_convention']['mean']:.2f} ± {100*old_style['partial26_n20_new_convention']['std_across_signer_means_ddof1']:.2f}")
    print("  seed-42 pooled 25", f"{100*old_style['partial26_n20_seed42_pooled']['mean']:.2f} ± {100*old_style['partial26_n20_seed42_pooled']['std_ddof1']:.2f}")

    print("\n=== top-1 / macro-F1 (signer-mean convention) ===")
    for n in BUDGETS:
        print(f"n={n}")
        for m in METHODS:
            t = by_budget[str(n)][m]["top1"]
            f = by_budget[str(n)][m]["macro_f1"]
            print(
                f"  {m:12s}  top1 {100*t['mean_of_signer_means']:5.2f} ± {100*t['std_across_signer_means_ddof1']:4.2f}"
                f"   F1 {100*f['mean_of_signer_means']:5.2f} ± {100*f['std_across_signer_means_ddof1']:4.2f}"
            )

    print("\n=== paired signer diffs (pp) ===")
    for n, block in paired.items():
        print(f"n={n}")
        for name, d in block.items():
            ps = "  ".join(f"s{s}:{100*v:+.1f}" for s, v in d["per_signer"].items())
            print(
                f"  {name:28s} mean {100*d['mean_of_signer_diffs']:+.2f}  "
                f"+{d['n_signers_positive']}/5  {ps}"
            )
    print("wrote", out)


if __name__ == "__main__":
    main()
