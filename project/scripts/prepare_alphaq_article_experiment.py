from __future__ import annotations

import argparse
import csv
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_READINESS_CSV = PROJECT_ROOT / "results" / "csv" / "alphaq_external_validation_readiness.csv"
DEFAULT_PROTOCOL_CSV = PROJECT_ROOT / "results" / "csv" / "alphaq_article_experiment_protocol.csv"
DEFAULT_REPORT = PROJECT_ROOT / "results" / "reports" / "alphaq_article_experiment_protocol.md"
DEFAULT_COMMANDS = PROJECT_ROOT / "results" / "reports" / "alphaq_article_apuana_commands.md"

ARTICLE_OBJECTIVES = (
    "factor_count",
    "factor_count_pair_cap",
    "mixed_pair",
    "frontier_pair",
    "depth_guarded_mixed_pair",
    "t_preserving_frontier_pair",
)

CORE_TARGETS = (
    "nc_tof_4",
    "barenco_tof_4",
    "vbe_adder_3",
    "gf_2pow2_mult",
    "mod_5_4",
    "hamming_weight_n4",
    "hamming_weight_n5",
)

EXTENDED_TARGETS = (
    "mod_mult_55",
    "cuccaro_adder_n4",
    "gf_2pow4_mult",
    "hamming_weight_n6",
    "hamming_weight_n7",
    "nc_tof_5",
    "gf_2pow5_mult",
    "cuccaro_adder_n5",
)


@dataclass(frozen=True)
class ExperimentBatch:
    name: str
    targets: tuple[str, ...]
    time_limit_sec: int
    beam_widths: str


DEFAULT_BATCHES = (
    ExperimentBatch("article_core", CORE_TARGETS, 900, "4,16,32"),
    ExperimentBatch("article_extended", EXTENDED_TARGETS, 1800, "4,16,32"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the fixed AlphaQ article experiment protocol without "
            "submitting or running the heavy Apuana jobs."
        )
    )
    parser.add_argument("--readiness-csv", type=Path, default=DEFAULT_READINESS_CSV)
    parser.add_argument("--protocol-csv", type=Path, default=DEFAULT_PROTOCOL_CSV)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--commands-path", type=Path, default=DEFAULT_COMMANDS)
    parser.add_argument(
        "--target-preset",
        choices=("core", "extended", "all"),
        default="all",
        help="Which article target batch descriptions to emit.",
    )
    return parser.parse_args()


def selected_batches(preset: str) -> tuple[ExperimentBatch, ...]:
    if preset == "core":
        return (DEFAULT_BATCHES[0],)
    if preset == "extended":
        return (DEFAULT_BATCHES[1],)
    return DEFAULT_BATCHES


def load_readiness(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            row["target"]: row
            for row in csv.DictReader(handle)
            if row.get("target")
        }


def protocol_rows(
    batches: tuple[ExperimentBatch, ...],
    readiness: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for batch in batches:
        for target in batch.targets:
            meta = readiness.get(target, {})
            rows.append(
                {
                    "batch": batch.name,
                    "target": target,
                    "target_status": target_status(meta),
                    "family": meta.get("family", infer_family(target)),
                    "tensor_size": meta.get("tensor_size", ""),
                    "readiness_status": meta.get("readiness_status", "not-in-readiness-csv"),
                    "objectives": ",".join(ARTICLE_OBJECTIVES),
                    "time_limit_sec": batch.time_limit_sec,
                    "beam_widths": batch.beam_widths,
                    "primary_external_metric": "paper_primary_nc_depth_ratio",
                    "secondary_metrics": "tcount,qasm_depth_ratio,paper_zx_best_nonclifford_depth",
                }
            )
    return rows


def target_status(meta: dict[str, str]) -> str:
    if not meta:
        return "needs-readiness-entry"
    readiness = meta.get("readiness_status", "")
    if readiness == "ready-full-action":
        return "ready-new-run"
    if readiness == "current-grid-control":
        return "ready-control-refresh"
    if readiness == "needs-tensor-v3-screen":
        return "pre-screen-required"
    if readiness == "ready-restricted-action":
        return "restricted-action-required"
    if readiness == "evaluation-only-large-action":
        return "evaluation-only"
    return "review"


def infer_family(target: str) -> str:
    if "hamming_weight" in target:
        return "hamming_weight"
    if "gf_2pow" in target:
        return "gf_multiplier"
    if "adder" in target:
        return "adder"
    if "tof" in target:
        return "toffoli"
    if "mod" in target:
        return "modular"
    return "unknown"


def pipeline_command(batch: ExperimentBatch) -> list[str]:
    suffix = batch.name
    csv_root = "results/csv"
    report_root = "results/reports"
    figure_root = "results/figures"
    return [
        "PYTHONPATH=.:external",
        ".venv/bin/python",
        "scripts/run_alphaq_external_validation_pipeline.py",
        "--targets",
        ",".join(batch.targets),
        "--objective-variants",
        ",".join(ARTICLE_OBJECTIVES),
        "--time-limit-sec",
        str(batch.time_limit_sec),
        "--beam-widths",
        batch.beam_widths,
        "--output-root",
        f"results/alphaq_decomposition_objective_external_validation_{suffix}",
        "--output-csv",
        f"{csv_root}/alphaq_decomposition_objective_external_validation_{suffix}.csv",
        "--report-path",
        f"{report_root}/alphaq_decomposition_objective_external_validation_{suffix}.md",
        "--figure-path",
        f"{figure_root}/alphaq_decomposition_objective_external_validation_{suffix}.png",
        "--beam-output-root",
        f"results/alphaq_objective_beam_policy_external_validation_{suffix}",
        "--beam-grid-csv",
        f"{csv_root}/alphaq_objective_beam_policy_external_validation_{suffix}_grid.csv",
        "--beam-policy-csv",
        f"{csv_root}/alphaq_objective_beam_policy_external_validation_{suffix}_summary.csv",
        "--beam-report-path",
        f"{report_root}/alphaq_objective_beam_policy_external_validation_{suffix}.md",
        "--beam-figure-path",
        f"{figure_root}/alphaq_objective_beam_policy_external_validation_{suffix}.png",
        "--combined-grid-csv",
        f"{csv_root}/alphaq_objective_beam_policy_grid_plus_external_validation_{suffix}.csv",
        "--transfer-summary-csv",
        f"{csv_root}/alphaq_external_selector_transfer_{suffix}_summary.csv",
        "--transfer-detail-csv",
        f"{csv_root}/alphaq_external_selector_transfer_{suffix}_details.csv",
        "--transfer-report-path",
        f"{report_root}/alphaq_external_selector_transfer_{suffix}.md",
        "--status-csv",
        f"{csv_root}/alphaq_external_validation_status_{suffix}.csv",
        "--status-report-path",
        f"{report_root}/alphaq_external_validation_status_{suffix}.md",
        "--paper-zx-csv",
        f"{csv_root}/alphaq_external_validation_paper_zx_audit_{suffix}.csv",
        "--paper-zx-report-path",
        f"{report_root}/alphaq_external_validation_paper_zx_audit_{suffix}.md",
    ]


def shell_line(command: list[str]) -> str:
    rendered = []
    for index, part in enumerate(command):
        if index == 0 and "=" in part and not part.startswith("-"):
            rendered.append(part)
        else:
            rendered.append(shlex.quote(part))
    return " ".join(rendered)


def write_protocol_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "batch",
        "target",
        "target_status",
        "family",
        "tensor_size",
        "readiness_status",
        "objectives",
        "time_limit_sec",
        "beam_widths",
        "primary_external_metric",
        "secondary_metrics",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict[str, Any]], protocol_csv: Path, commands_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ready = sum(1 for row in rows if str(row["target_status"]).startswith("ready-"))
    pre_screen = sum(1 for row in rows if row["target_status"] == "pre-screen-required")
    review = len(rows) - ready - pre_screen
    lines = [
        "# AlphaQ Article Experiment Protocol",
        "",
        f"Protocol CSV: `{display_path(protocol_csv)}`.",
        f"Apuana commands: `{display_path(commands_path)}`.",
        "",
        "This protocol freezes the article-facing comparison before the heavy jobs are submitted. The method sees only AlphaQ objectives; the paper-style ZX detector is used after materialization as an external audit.",
        "",
        "## Fixed Objectives",
        "",
        "- `factor_count`: original AlphaQuantum tensor objective.",
        "- `factor_count_pair_cap`: original objective with pair-overlap cap.",
        "- `mixed_pair`: previous split-aware AlphaQ-only objective.",
        "- `frontier_pair`: article-inspired AlphaQ-only objective aligned with paper-style frontier auditing.",
        "",
        "## Metrics",
        "",
        "- Primary external metric: `paper_primary_nc_depth_ratio`.",
        "- Secondary metrics: `tcount`, `qasm_depth_ratio`, `paper_zx_best_nonclifford_depth`.",
        "- No ZX, PyZX, or feynver metric is used inside the AlphaQ objective.",
        "",
        "## Target Coverage",
        "",
        f"Targets listed: {len(rows)}.",
        f"Ready for direct run/control refresh: {ready}.",
        f"Require tensor/profile pre-screen first: {pre_screen}.",
        f"Need review/readiness update: {review}.",
        "",
        "| batch | target | status | family | tensor size |",
        "|---|---|---|---|---:|",
    ]
    for row in rows:
        lines.append(
            "| {batch} | {target} | {target_status} | {family} | {tensor_size} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_commands(path: Path, batches: tuple[ExperimentBatch, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# AlphaQ Article Apuana Commands",
        "",
        "Run these from `/home/CIN/cacl2/Deep-Learning/project` on Apuana after syncing the current code.",
        "Each command writes suffixed outputs and includes the paper-style ZX audit.",
        "",
    ]
    for batch in batches:
        lines.extend(
            [
                f"## {batch.name}",
                "",
                f"Targets: `{','.join(batch.targets)}`.",
                "",
                "```bash",
                shell_line(pipeline_command(batch)),
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def main() -> int:
    args = parse_args()
    batches = selected_batches(args.target_preset)
    rows = protocol_rows(batches, load_readiness(args.readiness_csv))
    write_protocol_csv(args.protocol_csv, rows)
    write_commands(args.commands_path, batches)
    write_report(args.report_path, rows, args.protocol_csv, args.commands_path)
    print(f"Wrote {args.protocol_csv}")
    print(f"Wrote {args.commands_path}")
    print(f"Wrote {args.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
