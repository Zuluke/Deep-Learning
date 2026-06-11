"""Report objective-specific win sizes and factor-structure shifts."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_alphaq_portfolio_budget import portfolio_rows
from scripts.analyze_alphaq_split_select import (
    BASELINE_OBJECTIVE,
    grouped,
    oracle_row,
    read_csv,
    train_ready_rows,
    write_csv,
)
from scripts.structural_target import coerce_float

DEFAULT_DATASET = PROJECT_ROOT / "results" / "csv" / "alphaq_objective_selection_dataset.csv"
DEFAULT_CSV = PROJECT_ROOT / "results" / "csv" / "alphaq_objective_mechanism_diagnostic.csv"
DEFAULT_REPORT = PROJECT_ROOT / "results" / "reports" / "alphaq_objective_mechanism_diagnostic.md"

STRUCTURE_FIELDS = (
    "factor_count",
    "factor_pairwise_support_overlap_mean",
    "factor_pairwise_jaccard_mean",
    "factor_qubit_concentration_index",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize factor-structure shifts for objective wins."
    )
    parser.add_argument("--dataset-csv", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def fmt(value: Any) -> str:
    numeric = coerce_float(value)
    return "" if numeric is None or math.isnan(numeric) else f"{numeric:.6g}"


def delta(row: dict[str, Any], baseline: dict[str, Any], field: str) -> float | None:
    left = coerce_float(row.get(field))
    right = coerce_float(baseline.get(field))
    if left is None or right is None:
        return None
    return left - right


def win_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out = []
    for (source_split, target), items in sorted(grouped(rows).items()):
        baseline = next(
            (row for row in items if row["objective_variant"] == BASELINE_OBJECTIVE),
            None,
        )
        if baseline is None:
            continue
        oracle = oracle_row(items)
        if oracle["objective_variant"] == BASELINE_OBJECTIVE:
            continue
        baseline_t = coerce_float(baseline.get("best_beam_tcount"))
        oracle_t = coerce_float(oracle.get("best_beam_tcount"))
        if baseline_t is None or oracle_t is None or oracle_t >= baseline_t:
            continue
        row = {
            "row_type": "win",
            "objective_variant": oracle["objective_variant"],
            "source_split": source_split,
            "target": target,
            "wins": "",
            "win_t_gates": fmt(baseline_t - oracle_t),
            "median_win_t_gates": "",
            "max_win_t_gates": "",
        }
        for field in STRUCTURE_FIELDS:
            row[f"{field}_delta_vs_baseline"] = fmt(delta(oracle, baseline, field))
            row[f"median_{field}_delta_vs_baseline"] = ""
        out.append(row)
    return out


def summary_rows(wins: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for objective in sorted({row["objective_variant"] for row in wins}):
        rows = [row for row in wins if row["objective_variant"] == objective]
        win_sizes = [
            value
            for row in rows
            if (value := coerce_float(row.get("win_t_gates"))) is not None
        ]
        summary = {
            "row_type": "objective_summary",
            "objective_variant": objective,
            "source_split": "",
            "target": "",
            "wins": len(rows),
            "win_t_gates": "",
            "median_win_t_gates": fmt(median(win_sizes)),
            "max_win_t_gates": fmt(max(win_sizes) if win_sizes else None),
        }
        for field in STRUCTURE_FIELDS:
            values = [
                value
                for row in rows
                if (value := coerce_float(row.get(f"{field}_delta_vs_baseline"))) is not None
            ]
            summary[f"{field}_delta_vs_baseline"] = ""
            summary[f"median_{field}_delta_vs_baseline"] = fmt(median(values))
        out.append(summary)
    return out


def write_report(path: Path, rows: list[dict[str, Any]], *, dataset_csv: Path, output_csv: Path) -> None:
    summaries = [row for row in rows if row["row_type"] == "objective_summary"]
    wins = [row for row in rows if row["row_type"] == "win"]
    lines = [
        "# AlphaQ Objective Mechanism Diagnostic",
        "",
        f"Dataset: `{dataset_csv}`.",
        f"CSV: `{output_csv}`.",
        "",
        "Rows where a non-baseline objective attains a lower T-count than `factor_count` are summarized below. "
        "Structure deltas are objective minus baseline within the same (source split, target) group.",
        "",
        "| objective | wins | median win (T) | max win (T) | median factor-count delta | median overlap delta | median Jaccard delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {objective_variant} | {wins} | {median_win_t_gates} | {max_win_t_gates} | {factor} | {overlap} | {jaccard} |".format(
                objective_variant=row["objective_variant"],
                wins=row["wins"],
                median_win_t_gates=row["median_win_t_gates"],
                max_win_t_gates=row["max_win_t_gates"],
                factor=row["median_factor_count_delta_vs_baseline"],
                overlap=row["median_factor_pairwise_support_overlap_mean_delta_vs_baseline"],
                jaccard=row["median_factor_pairwise_jaccard_mean_delta_vs_baseline"],
            )
        )
    lines.extend(["", "## Winning Groups", ""])
    for row in wins:
        lines.append(
            "- `{target}` ({source_split}): `{objective_variant}` improves by {win_t_gates} T gates; "
            "overlap delta {overlap}, Jaccard delta {jaccard}.".format(
                target=row["target"],
                source_split=row["source_split"],
                objective_variant=row["objective_variant"],
                win_t_gates=row["win_t_gates"],
                overlap=row["factor_pairwise_support_overlap_mean_delta_vs_baseline"],
                jaccard=row["factor_pairwise_jaccard_mean_delta_vs_baseline"],
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


CSV_FIELDS = [
    "row_type",
    "objective_variant",
    "source_split",
    "target",
    "wins",
    "win_t_gates",
    "median_win_t_gates",
    "max_win_t_gates",
    "factor_count_delta_vs_baseline",
    "median_factor_count_delta_vs_baseline",
    "factor_pairwise_support_overlap_mean_delta_vs_baseline",
    "median_factor_pairwise_support_overlap_mean_delta_vs_baseline",
    "factor_pairwise_jaccard_mean_delta_vs_baseline",
    "median_factor_pairwise_jaccard_mean_delta_vs_baseline",
    "factor_qubit_concentration_index_delta_vs_baseline",
    "median_factor_qubit_concentration_index_delta_vs_baseline",
]


def main() -> int:
    args = parse_args()
    rows = portfolio_rows(train_ready_rows(read_csv(args.dataset_csv)))
    wins = win_rows(rows)
    out = [*summary_rows(wins), *wins]
    write_csv(args.output_csv, out, CSV_FIELDS)
    write_report(args.report_path, out, dataset_csv=args.dataset_csv, output_csv=args.output_csv)
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
