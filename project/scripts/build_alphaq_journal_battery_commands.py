from __future__ import annotations

import argparse
import csv
import re
import shlex
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_BATTERY_CSV = PROJECT_ROOT / "results" / "csv" / "alphaq_journal_next_battery.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "reports" / "alphaq_journal_battery_commands.md"

STAGE_CONFIG = {
    "full-action-repair": {
        "time_limit_sec": 3600,
        "beam_widths": "4,16,32",
        "job_name": "alphaq_journal_repair",
        "suffix": "journal_repair",
    },
    "full-action-expansion": {
        "time_limit_sec": 3600,
        "beam_widths": "4,16,32",
        "job_name": "alphaq_journal_full",
        "suffix": "journal_full",
    },
    "tensor-v3-screen": {
        "time_limit_sec": 0,
        "beam_widths": "",
        "job_name": "alphaq_journal_tensor_v3_screen",
        "suffix": "journal_tensor_v3_screen",
        "submit_supported": False,
        "blocked_reason": (
            "Run a tensor-v3/profile screening step first; do not repeat full-action "
            "objective-grid jobs for this target without a new screening signal."
        ),
    },
    "restricted-action-pilot": {
        "time_limit_sec": 3600,
        "beam_widths": "4,16,32",
        "job_name": "alphaq_journal_restricted",
        "suffix": "journal_restricted",
        "submit_supported": False,
        "blocked_reason": "No restricted-action submit path is implemented yet.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build reproducible submit commands for the AlphaQ journal evidence battery."
    )
    parser.add_argument("--battery-csv", type=Path, default=DEFAULT_BATTERY_CSV)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-targets-per-job", type=int, default=1)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def command_blocks(rows: list[dict[str, str]], max_targets_per_job: int) -> list[dict[str, Any]]:
    by_stage: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        stage = row.get("recommended_stage", "")
        if stage in STAGE_CONFIG:
            by_stage[stage].append(row)
    blocks = []
    for stage in STAGE_CONFIG:
        targets = [row["target"] for row in by_stage.get(stage, [])]
        if not targets:
            continue
        if not STAGE_CONFIG[stage].get("submit_supported", True):
            blocks.append(
                {
                    "stage": stage,
                    "targets": targets,
                    "command": "",
                    "blocked_reason": STAGE_CONFIG[stage].get("blocked_reason", "Submit path is not implemented."),
                }
            )
            continue
        for index, chunk in enumerate(chunks(targets, max_targets_per_job), start=1):
            config = STAGE_CONFIG[stage]
            if len(targets) <= max_targets_per_job:
                suffix = config["suffix"]
                job_name = config["job_name"]
            elif len(chunk) == 1:
                suffix = f"{config['suffix']}_{slug(chunk[0])}"
                job_name = f"{config['job_name']}_{slug(chunk[0])}"
            else:
                suffix = f"{config['suffix']}_{index}"
                job_name = f"{config['job_name']}_{index}"
            command = [
                "./submit_external_validation_apuana.sh",
                "--targets",
                ",".join(chunk),
                "--time-limit-sec",
                str(config["time_limit_sec"]),
                "--beam-widths",
                config["beam_widths"],
                "--output-suffix",
                suffix,
                "--job-name",
                job_name,
            ]
            blocks.append(
                {
                    "stage": stage,
                    "targets": chunk,
                    "command": " ".join(shlex.quote(item) for item in command),
                }
            )
    return blocks


def chunks(items: list[str], size: int) -> list[list[str]]:
    if size <= 0:
        raise ValueError("--max-targets-per-job must be positive.")
    return [items[index : index + size] for index in range(0, len(items), size)]


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")


def write_report(path: Path, blocks: list[dict[str, Any]], battery_csv: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# AlphaQ Journal Battery Commands",
        "",
        f"Battery CSV: `{battery_csv}`.",
        "",
        "These commands are generated from the journal evidence audit. Run them from `/Users/caio/Deep-Learning` when the Apuana environment is reachable. They intentionally keep outputs separated by suffix so the current baseline artifacts are not overwritten.",
        "",
    ]
    for block in blocks:
        lines.extend(
            [
                f"## {block['stage']}",
                "",
                f"Targets: `{','.join(block['targets'])}`.",
                "",
            ]
        )
        if block.get("command"):
            lines.extend(["```bash", block["command"], "```", ""])
        else:
            lines.extend([f"Blocked: {block['blocked_reason']}", ""])
    if not blocks:
        lines.append("No runnable command blocks were produced.")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    blocks = command_blocks(read_csv(args.battery_csv), args.max_targets_per_job)
    write_report(args.output_path, blocks, args.battery_csv)
    print(f"Wrote {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
