"""Selective multi-seed + optional BiGRU baseline (revision support).

Runs a small, reviewer-facing grid instead of re-seeding every RQ.

Profiles
--------
  rq1     RQ1 sentence FT: scratch + SSL, one seed per outer loop
  rq3     RQ3 category LOSO: scratch, k in {1,4}, sample draws=2, all test folds
  rq6     RQ6: scratch, none+mask_sub, k=4, sample draws=1
  bigru   Cheap alt arch: BiGRU scratch, k=4, sample draws=1, test signer 5
  all     rq1 → rq3 → bigru (skip rq6 unless listed)

Examples
--------
  python -u scripts/15_selective_multiseed.py --dry-run
  python -u scripts/15_selective_multiseed.py --profile rq1 --seeds 42,43,44,45,46
  python -u scripts/15_selective_multiseed.py --profile rq3 --seeds 42,43,44
  python -u scripts/15_selective_multiseed.py --profile bigru
  python -u scripts/15_selective_multiseed.py --profile rq1,rq3,bigru --seeds 42,43,44

After RQ1 finishes, score each seed with::

  python -u scripts/05_evaluate.py --scratch-ckpt checkpoints/finetune_scratch_s42.pt \\
      --pretrained-ckpt checkpoints/finetune_pretrained_s42.pt
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import PATHS
from src.train_utils import write_json

PROFILES = ("rq1", "rq3", "rq6", "bigru")


def _ids(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _seeds(s: str) -> list[int]:
    return [int(x) for x in _ids(s)]


def _run(cmd: list[str], *, dry_run: bool) -> int:
    print("\n>>>", " ".join(cmd), flush=True)
    if dry_run:
        return 0
    proc = subprocess.run(cmd, cwd=str(ROOT))
    return int(proc.returncode)


def _py() -> str:
    return sys.executable


def _rq1_cmds(seed: int, epochs: int | None) -> list[list[str]]:
    out_s = PATHS.checkpoints / f"finetune_scratch_s{seed}.pt"
    out_p = PATHS.checkpoints / f"finetune_pretrained_s{seed}.pt"
    base = [_py(), "-u", "scripts/04_finetune.py", "--task", "sentence", "--seed", str(seed)]
    if epochs is not None:
        base += ["--epochs", str(epochs)]
    return [
        base + ["--pretrained", "none", "--out", str(out_s)],
        base + ["--pretrained", str(PATHS.ssl_encoder), "--out", str(out_p)],
    ]


def _rq3_cmd(seed: int, epochs: int | None) -> list[str]:
    out = PATHS.results / f"rq3_seeds_s{seed}.json"
    cmd = [
        _py(),
        "-u",
        "scripts/07_signer_scaling.py",
        "--arms",
        "scratch",
        "--ks",
        "1,4",
        "--mode",
        "sample",
        "--draws",
        "2",
        "--ssl-clean-test",
        "5",
        "--seed",
        str(seed),
        "--out",
        str(out),
    ]
    if epochs is not None:
        cmd += ["--epochs", str(epochs)]
    return cmd


def _rq6_cmd(seed: int, epochs: int | None) -> list[str]:
    out = PATHS.results / f"rq6_seeds_s{seed}.json"
    cmd = [
        _py(),
        "-u",
        "scripts/11_finetune_augment.py",
        "--arms",
        "scratch",
        "--augs",
        "none,mask_sub",
        "--ks",
        "4",
        "--mode",
        "sample",
        "--draws",
        "1",
        "--seed",
        str(seed),
        "--out",
        str(out),
    ]
    if epochs is not None:
        cmd += ["--epochs", str(epochs)]
    return cmd


def _bigru_cmd(seed: int, epochs: int | None) -> list[str]:
    """Lean BiGRU SI check: k=4, one sampled train set, test signer 5."""
    out = PATHS.results / f"rq3_bigru_s{seed}.json"
    cmd = [
        _py(),
        "-u",
        "scripts/07_signer_scaling.py",
        "--arch",
        "bigru",
        "--arms",
        "scratch",
        "--ks",
        "4",
        "--mode",
        "sample",
        "--draws",
        "1",
        "--test-signers",
        "5",
        "--seed",
        str(seed),
        "--out",
        str(out),
    ]
    if epochs is not None:
        cmd += ["--epochs", str(epochs)]
    return cmd


def _summarize_seed_jsons(paths: list[Path], metric: str = "clip_top1") -> dict:
    """Mean±std of arm×k (or aug) from per-seed RQ3/RQ6 JSON files."""
    by_key: dict[tuple, list[float]] = {}
    for path in paths:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        seed = int(data.get("seed", -1))
        for r in data.get("runs") or []:
            key = (
                str(r.get("arch", data.get("arch", "transformer"))),
                str(r.get("arm")),
                int(r["k"]),
                str(r.get("aug", "")),
            )
            by_key.setdefault(key, []).append(float(r[metric]))
            _ = seed
    out = {}
    for (arch, arm, k, aug), vals in sorted(by_key.items()):
        import numpy as np

        slot = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "n": len(vals),
        }
        label = f"{arch}/{arm}/k{k}" + (f"/{aug}" if aug else "")
        out[label] = slot
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--profile",
        type=str,
        default="rq1,rq3,bigru",
        help="comma list: rq1,rq3,rq6,bigru,all",
    )
    p.add_argument("--seeds", type=str, default="42,43,44", help="outer RNG seeds")
    p.add_argument(
        "--rq1-seeds",
        type=str,
        default="",
        help="RQ1 seeds (default: 42,43,44,45,46)",
    )
    p.add_argument("--epochs", type=int, default=0, help="override FT epochs (0 = script default)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--aggregate",
        action="store_true",
        help="only fold existing per-seed JSONs into results/selective_multiseed_summary.json",
    )
    args = p.parse_args()

    raw = _ids(args.profile.lower().replace(" ", ""))
    if "all" in raw:
        profiles = ["rq1", "rq3", "bigru"]
    else:
        profiles = [x for x in raw if x in PROFILES]
    if not profiles:
        raise SystemExit(f"no valid profiles in {args.profile!r}; choose from {PROFILES}")

    seeds = _seeds(args.seeds)
    rq1_seeds = _seeds(args.rq1_seeds) if args.rq1_seeds else list(range(42, 47))
    epochs = args.epochs if args.epochs > 0 else None

    if args.aggregate:
        rq3_paths = [PATHS.results / f"rq3_seeds_s{s}.json" for s in seeds]
        rq6_paths = [PATHS.results / f"rq6_seeds_s{s}.json" for s in seeds]
        bigru_paths = [PATHS.results / f"rq3_bigru_s{s}.json" for s in seeds]
        summary = {
            "rq3": _summarize_seed_jsons(rq3_paths),
            "rq6": _summarize_seed_jsons(rq6_paths),
            "bigru": _summarize_seed_jsons(bigru_paths),
            "rq1_checkpoints": {
                "scratch": [str(PATHS.checkpoints / f"finetune_scratch_s{s}.pt") for s in rq1_seeds],
                "pretrained": [
                    str(PATHS.checkpoints / f"finetune_pretrained_s{s}.pt") for s in rq1_seeds
                ],
            },
        }
        dest = PATHS.results / "selective_multiseed_summary.json"
        write_json(dest, summary)
        print(json.dumps(summary, indent=2))
        print(f"wrote {dest}")
        return

    print(f"[15] profiles={profiles}  seeds={seeds}  rq1_seeds={rq1_seeds}  dry_run={args.dry_run}")
    failures: list[str] = []

    for profile in profiles:
        if profile == "rq1":
            for seed in rq1_seeds:
                for cmd in _rq1_cmds(seed, epochs):
                    rc = _run(cmd, dry_run=args.dry_run)
                    if rc != 0:
                        failures.append(" ".join(cmd))
        elif profile == "rq3":
            for seed in seeds:
                rc = _run(_rq3_cmd(seed, epochs), dry_run=args.dry_run)
                if rc != 0:
                    failures.append(f"rq3 seed={seed}")
        elif profile == "rq6":
            for seed in seeds:
                rc = _run(_rq6_cmd(seed, epochs), dry_run=args.dry_run)
                if rc != 0:
                    failures.append(f"rq6 seed={seed}")
        elif profile == "bigru":
            # One seed is enough for the cheap alt-arch control.
            bigru_seeds = [seeds[0]]
            for seed in bigru_seeds:
                rc = _run(_bigru_cmd(seed, epochs), dry_run=args.dry_run)
                if rc != 0:
                    failures.append(f"bigru seed={seed}")

    if failures:
        print(f"[15] {len(failures)} command(s) failed:")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print("[15] done" + (" (dry-run)" if args.dry_run else ""))
    if not args.dry_run:
        print(
            "[15] tip: python -u scripts/15_selective_multiseed.py --aggregate "
            f"--seeds {','.join(str(s) for s in seeds)}"
        )


if __name__ == "__main__":
    main()
