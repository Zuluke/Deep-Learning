from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._fig4_reproduction import DEFAULT_FIG4_OUTPUT_DIR
from scripts._fig4_reproduction import generate_fig4_artifacts
from scripts._fig4_reproduction import timestamp_utc
from scripts._manifest import append_command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce Fig. 4 (finite-field multiplication + binary addition)."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_FIG4_OUTPUT_DIR,
        help="Directory where CSV/JSON/README/PNG/PDF outputs will be written.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifact_paths = generate_fig4_artifacts(args.output_dir)
    append_command(
        {
            "tool": "reproduce_fig4.py",
            "command": f"{sys.executable} scripts/reproduce_fig4.py --output-dir {args.output_dir}",
            "cwd": str(PROJECT_ROOT),
            "exit_code": 0,
            "output_dir": str(args.output_dir),
            "artifacts": {name: str(path) for name, path in artifact_paths.items()},
        }
    )
    summary = {
        "figure": "Fig. 4",
        "output_dir": str(args.output_dir),
        "artifacts": {name: str(path) for name, path in artifact_paths.items()},
        "generated_at_utc": timestamp_utc(),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
