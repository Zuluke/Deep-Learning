"""Summarize MILP certificate status for AlphaQ linear-span runs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_alphaq_split_select import write_csv
from scripts.structural_target import coerce_float

DEFAULT_SUMMARY = PROJECT_ROOT / "results" / "csv" / "alphaq_certificates.csv"
DEFAULT_DETAILS = PROJECT_ROOT / "results" / "csv" / "alphaq_certificate_details.csv"
DEFAULT_HEADLINES = PROJECT_ROOT / "results" / "csv" / "alphaq_certificate_headline_wins.csv"
DEFAULT_REPORT = PROJECT_ROOT / "results" / "reports" / "alphaq_certificates.md"

HEADLINE_WINS = (
    ("barenco_tof_4", "mixed_pair"),
    ("vbe_adder_3", "frontier_pair"),
    ("hamming_weight_n7", "frontier_pair"),
    ("mod_mult_55", "mixed_pair"),
    ("nc_tof_4", "frontier_pair"),
    ("gf_2pow3_mult", "factor_count_pair_cap"),
    ("gf_2pow4_mult", "factor_count_pair_cap"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build certificate-status tables from linear-span summary JSON files."
    )
    parser.add_argument("--results-root", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--detail-csv", type=Path, default=DEFAULT_DETAILS)
    parser.add_argument("--headline-csv", type=Path, default=DEFAULT_HEADLINES)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def objective_from_path(path: Path) -> str:
    parts = path.parts
    index = parts.index("linear_span")
    return parts[index + 1]


def source_run_from_path(path: Path) -> str:
    parts = path.parts
    results_index = parts.index("results")
    linear_index = parts.index("linear_span")
    return "/".join(parts[results_index + 1 : linear_index])


def is_current_run(row: dict[str, Any]) -> bool:
    source = str(row["source_run"])
    return source.startswith("alphaq_decomposition_objective")


def detail_rows(results_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(results_root.glob("**/linear_span/**/summary.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        objective = objective_from_path(path)
        rows.append(
            {
                "target": data.get("target", ""),
                "objective_variant": objective,
                "source_run": source_run_from_path(path),
                "certificate_status": "certified-optimal"
                if data.get("is_optimal")
                else "time-limit-incumbent",
                "is_optimal": bool(data.get("is_optimal")),
                "solver_status": data.get("solver_status", ""),
                "solver_message": data.get("solver_message", ""),
                "status": data.get("status", ""),
                "num_factors": data.get("num_factors", ""),
                "span_objective_value": data.get("span_objective_value", ""),
                "elapsed_sec": data.get("elapsed_sec", ""),
                "max_action_weight": data.get("max_action_weight", ""),
                "num_actions": data.get("num_actions", ""),
                "summary_path": str(path),
                "current_run": is_current_source(source_run_from_path(path)),
            }
        )
    return rows


def is_current_source(source_run: str) -> bool:
    return source_run.startswith("alphaq_decomposition_objective")


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not row["target"] or not row["objective_variant"]:
            continue
        grouped[(row["target"], row["objective_variant"])].append(row)
    out = []
    for (target, objective), items in sorted(grouped.items()):
        current_items = [row for row in items if row["current_run"]]
        status_items = current_items or items
        optimal_items = [row for row in status_items if row["is_optimal"]]
        factors = [
            value
            for row in status_items
            if (value := coerce_float(row.get("num_factors"))) is not None
        ]
        status = "certified-optimal" if optimal_items else "time-limit-incumbent"
        out.append(
            {
                "target": target,
                "objective_variant": objective,
                "certificate_status": status,
                "current_runs": len(current_items),
                "all_runs": len(items),
                "certified_runs": sum(bool(row["is_optimal"]) for row in status_items),
                "best_num_factors": "" if not factors else min(factors),
                "source_runs": ";".join(sorted({str(row["source_run"]) for row in status_items})),
            }
        )
    return out


def headline_rows(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (row["target"], row["objective_variant"]): row
        for row in summary
    }
    rows = []
    for target, objective in HEADLINE_WINS:
        row = by_key.get((target, objective))
        if row is None:
            rows.append(
                {
                    "target": target,
                    "objective_variant": objective,
                    "certificate_status": "no-linear-span-summary",
                    "current_runs": 0,
                    "all_runs": 0,
                    "certified_runs": 0,
                    "best_num_factors": "",
                    "source_runs": "",
                }
            )
        else:
            rows.append(dict(row))
    return rows


def write_report(
    path: Path,
    summary: list[dict[str, Any]],
    headlines: list[dict[str, Any]],
    details: list[dict[str, Any]],
    *,
    results_root: Path,
) -> None:
    current = [row for row in details if row["current_run"]]
    current_certified = sum(bool(row["is_optimal"]) for row in current)
    current_total = len(current)
    all_certified = sum(bool(row["is_optimal"]) for row in details)
    lines = [
        "# AlphaQ Certificate Audit",
        "",
        f"Results root: `{results_root}`.",
        "",
        f"Current linear-span summaries: {current_certified}/{current_total} certified optimal.",
        f"All discovered linear-span summaries: {all_certified}/{len(details)} certified optimal.",
        "",
        "## Headline Wins",
        "",
        "| target | objective | certificate status | current runs | certified runs | best factors |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in headlines:
        lines.append(
            "| {target} | {objective_variant} | {certificate_status} | {current_runs} | {certified_runs} | {best_num_factors} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Per Target/Objective Summary",
            "",
            "| target | objective | status | current runs | all runs | certified runs | best factors |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary:
        lines.append(
            "| {target} | {objective_variant} | {certificate_status} | {current_runs} | {all_runs} | {certified_runs} | {best_num_factors} |".format(
                **row
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


DETAIL_FIELDS = [
    "target",
    "objective_variant",
    "source_run",
    "certificate_status",
    "is_optimal",
    "solver_status",
    "solver_message",
    "status",
    "num_factors",
    "span_objective_value",
    "elapsed_sec",
    "max_action_weight",
    "num_actions",
    "summary_path",
    "current_run",
]

SUMMARY_FIELDS = [
    "target",
    "objective_variant",
    "certificate_status",
    "current_runs",
    "all_runs",
    "certified_runs",
    "best_num_factors",
    "source_runs",
]


def main() -> int:
    args = parse_args()
    details = detail_rows(args.results_root)
    summary = aggregate_rows(details)
    headlines = headline_rows(summary)
    write_csv(args.detail_csv, details, DETAIL_FIELDS)
    write_csv(args.summary_csv, summary, SUMMARY_FIELDS)
    write_csv(args.headline_csv, headlines, SUMMARY_FIELDS)
    write_report(
        args.report_path,
        summary,
        headlines,
        details,
        results_root=args.results_root,
    )
    print(f"Wrote {args.detail_csv}")
    print(f"Wrote {args.summary_csv}")
    print(f"Wrote {args.headline_csv}")
    print(f"Wrote {args.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
