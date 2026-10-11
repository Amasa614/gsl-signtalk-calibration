"""Generate methodology diagrams and results figures/tables for the paper.

    python scripts/08_make_figures.py
    python scripts/08_make_figures.py --methods-only
    python scripts/08_make_figures.py --results-only

Re-run after 03/04/05/06/07 so result curves pick up the latest JSON.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import PATHS
from src.paper_figures import generate_all


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--methods-only", action="store_true")
    p.add_argument("--results-only", action="store_true")
    args = p.parse_args()
    methods = not args.results_only
    results = not args.methods_only
    written = generate_all(methods=methods, results=results)
    print(f"wrote {len(written)} files under {PATHS.figures} and {PATHS.tables}")
    for path in written:
        print(f"  {path.relative_to(PATHS.repo_root)}")


if __name__ == "__main__":
    main()
