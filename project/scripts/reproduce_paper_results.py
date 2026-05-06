from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from atq_repro.paper import reproduce_paper_results
from atq_repro.paths import PAPER_REPRO_ROOT

from scripts._manifest import append_command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce AlphaTensor-Quantum paper results from official decompositions."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PAPER_REPRO_ROOT,
        help="Directory for CSV/JSON/figure reproduction artifacts.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = reproduce_paper_results(args.output_dir)
    append_command(
        {
            "tool": "reproduce_paper_results.py",
            "command": f"{sys.executable} scripts/reproduce_paper_results.py --output-dir {args.output_dir}",
            "cwd": str(PROJECT_ROOT),
            "exit_code": 0,
            "artifacts": {name: str(path) for name, path in paths.items()},
        }
    )
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
