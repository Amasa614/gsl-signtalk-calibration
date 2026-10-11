"""Stage E: signer-then-draw-then-seed summaries and paired contrasts.

    python -u scripts/revision_summarize.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.revision_protocol import REVISION_ROOT, SIGNERS
from src.train_utils import write_json


def _seed_mean(rec: dict, mode: str, field: str) -> float | None:
    rows = [a for a in rec.get("adapt") or [] if a["mode"] == mode]
    if not rows:
        return None
    return float(np.mean([a[field]["clip_top1"] for a in rows]))


def _slot_stats(vals: list[float]) -> dict:
    arr = np.asarray(vals, dtype=float)
    return {
        "mean": float(arr.mean()) if len(arr) else float("nan"),
        "std_across_signers": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "n": len(arr),
        "per_signer": None,
    }


def main() -> None:
    path = REVISION_ROOT / "run_records.json"
    if not path.exists():
        # fall back to concatenating per-selection files
        runs = []
        for f in sorted((REVISION_ROOT / "runs").glob("s*_n*_d*.json")):
            if f.name.endswith(".pt"):
                continue
            runs.append(json.loads(f.read_text(encoding="utf-8")))
    else:
        runs = json.loads(path.read_text(encoding="utf-8")).get("runs") or []
    if not runs:
        print("no run records yet")
        return

    tables = {}
    contrasts = {}
    for n in (1, 5, 20):
        signer_means = {m: [] for m in (
            "base26", "baseC", "prototypeC", "knn1C", "head26", "headC", "partial26", "partialC"
        )}
        signer_deltas = {m: [] for m in (
            "coverage", "prototype", "head", "encoder_update", "proto_minus_knn1", "proto_minus_partialC"
        )}
        per_signer = {}
        for s in SIGNERS:
            pts = [r for r in runs if str(r["test_signer"]) == s and int(r["n_per_category"]) == n]
            if not pts:
                continue

            def dmean(key):
                return float(np.mean([r["scores"][key]["clip_top1"] for r in pts]))

            b26 = dmean("base26")
            bC = dmean("baseC")
            pC = dmean("prototypeC")
            kC = dmean("knn1C")
            h26 = float(np.mean([_seed_mean(r, "head_only", "scores26") for r in pts]))
            hC = float(np.mean([_seed_mean(r, "head_only", "scoresC") for r in pts]))
            f26 = float(np.mean([_seed_mean(r, "partial_ft", "scores26") for r in pts]))
            fC = float(np.mean([_seed_mean(r, "partial_ft", "scoresC") for r in pts]))
            per_signer[s] = {
                "base26": b26, "baseC": bC, "prototypeC": pC, "knn1C": kC,
                "head26": h26, "headC": hC, "partial26": f26, "partialC": fC,
                "delta_coverage": bC - b26,
                "delta_prototype": pC - bC,
                "delta_head": hC - bC,
                "delta_encoder_update": fC - hC,
                "delta_proto_minus_knn1": pC - kC,
                "delta_proto_minus_partialC": pC - fC,
            }
            for k, v in [
                ("base26", b26), ("baseC", bC), ("prototypeC", pC), ("knn1C", kC),
                ("head26", h26), ("headC", hC), ("partial26", f26), ("partialC", fC),
            ]:
                signer_means[k].append(v)
            signer_deltas["coverage"].append(bC - b26)
            signer_deltas["prototype"].append(pC - bC)
            signer_deltas["head"].append(hC - bC)
            signer_deltas["encoder_update"].append(fC - hC)
            signer_deltas["proto_minus_knn1"].append(pC - kC)
            signer_deltas["proto_minus_partialC"].append(pC - fC)

        tables[str(n)] = {
            k: {**_slot_stats(v), "per_signer": {s: per_signer[s][k] for s in per_signer}}
            for k, v in signer_means.items()
            if v
        }
        contrasts[str(n)] = {
            k: {**_slot_stats(v), "per_signer": {s: per_signer[s][
                {
                    "coverage": "delta_coverage",
                    "prototype": "delta_prototype",
                    "head": "delta_head",
                    "encoder_update": "delta_encoder_update",
                    "proto_minus_knn1": "delta_proto_minus_knn1",
                    "proto_minus_partialC": "delta_proto_minus_partialC",
                }[k]
            ] for s in per_signer}}
            for k, v in signer_deltas.items()
            if v
        }

    out = {"n_records": len(runs), "by_budget": tables, "paired_contrasts": contrasts}
    write_json(REVISION_ROOT / "tables" / "summary.json", out)
    print("=== clip top-1 means across 5 signers ===")
    for n, slot in tables.items():
        print(f"n={n}")
        for k, st in slot.items():
            print(f"  {k:12s} {100*st['mean']:6.2f} ± {100*st['std_across_signers']:5.2f}")
    print("wrote", REVISION_ROOT / "tables" / "summary.json")


if __name__ == "__main__":
    main()
