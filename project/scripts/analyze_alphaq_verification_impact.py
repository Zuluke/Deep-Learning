"""Cross-reference portfolio-selection wins with formal verification status.

For each target-level guarded portfolio decision, report the verification
status of the selected candidate (and of the baseline candidate it is
compared against), so T-count win claims can be qualified by proof status.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_alphaq_journal_evidence import (
    PROVEN_VERIFICATION_STATUSES,
    merge_verification_rows,
)
from scripts.structural_target import coerce_float

DEFAULT_PORTFOLIO_DETAILS = PROJECT_ROOT / "results" / "csv" / "alphaq_portfolio_budget_details.csv"
DEFAULT_VERIFICATION_CSVS = (
    PROJECT_ROOT / "results" / "verification" / "alphaq_external_article_core" / "verification_summary.csv",
    PROJECT_ROOT / "results" / "verification" / "alphaq_external_article_extended" / "verification_summary.csv",
    PROJECT_ROOT / "results" / "verification" / "alphaq_external_night_long" / "verification_summary.csv",
    PROJECT_ROOT / "results" / "verification" / "alphaq_external_article_repair2_barenco" / "verification_summary.csv",
    PROJECT_ROOT / "results" / "verification" / "alphaq_external_article_repair2_vbe" / "verification_summary.csv",
    PROJECT_ROOT / "results" / "verification" / "alphaq_external_journal_full_nc_tof_5_long" / "verification_summary.csv",
    PROJECT_ROOT / "results" / "verification" / "alphaq_external_numeric" / "verification_numeric.csv",
    PROJECT_ROOT / "results" / "verification" / "alphaq_external_numeric_new" / "verification_numeric.csv",
)
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "results" / "csv" / "alphaq_verification_impact.csv"
DEFAULT_REPORT = PROJECT_ROOT / "results" / "reports" / "alphaq_verification_impact.md"

POLICY = "guarded_top2"
SCOPE = "targets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join guarded portfolio decisions with verification statuses."
    )
    parser.add_argument("--portfolio-details-csv", type=Path, default=DEFAULT_PORTFOLIO_DETAILS)
    parser.add_argument(
        "--verification-csvs",
        default=",".join(str(path) for path in DEFAULT_VERIFICATION_CSVS),
    )
    parser.add_argument("--policy", default=POLICY)
    parser.add_argument("--scope", default=SCOPE)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def read_csv_path(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def verification_status_by_key(paths: list[Path]) -> dict[tuple[str, str], str]:
    rows = merge_verification_rows(
        [row for path in paths for row in read_csv_path(path)]
    )
    statuses: dict[tuple[str, str], str] = {}
    for row in rows:
        key = (row.get("target", ""), row.get("objective_variant", ""))
        status = row.get("verification_status", "")
        current = statuses.get(key)
        # Keep the strongest claim per (target, objective).
        if current is None or (
            current not in PROVEN_VERIFICATION_STATUSES
            and status in PROVEN_VERIFICATION_STATUSES
        ):
            statuses[key] = status
    return statuses


def impact_rows(
    details: list[dict[str, str]],
    statuses: dict[tuple[str, str], str],
    *,
    policy: str,
    scope: str,
) -> list[dict[str, Any]]:
    rows = []
    for row in details:
        if row.get("policy") != policy or row.get("scope") != scope:
            continue
        target = row.get("target", "")
        final_objective = row.get("final_objective", "")
        ratio = coerce_float(row.get("tcount_ratio_vs_baseline"))
        outcome = (
            "win" if ratio is not None and ratio < 1.0
            else "loss" if ratio is not None and ratio > 1.0
            else "tie"
        )
        selected_status = statuses.get((target, final_objective), "unverified")
        baseline_status = statuses.get((target, "factor_count"), "unverified")
        rows.append(
            {
                "target": target,
                "source_split": row.get("source_split", ""),
                "final_objective": final_objective,
                "tcount_ratio_vs_baseline": row.get("tcount_ratio_vs_baseline", ""),
                "outcome_vs_baseline": outcome,
                "selected_verification_status": selected_status,
                "baseline_verification_status": baseline_status,
                "selected_proven": selected_status in PROVEN_VERIFICATION_STATUSES,
                "win_on_proven_candidate": outcome == "win"
                and selected_status in PROVEN_VERIFICATION_STATUSES,
            }
        )
    return sorted(rows, key=lambda item: (item["outcome_vs_baseline"], item["target"]))


def write_outputs(
    rows: list[dict[str, Any]],
    *,
    output_csv: Path,
    report_path: Path,
    policy: str,
    scope: str,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["target"]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    wins = [row for row in rows if row["outcome_vs_baseline"] == "win"]
    proven_wins = [row for row in wins if row["win_on_proven_candidate"]]
    lines = [
        "# Verification Status Of Portfolio Selection Wins",
        "",
        f"Policy: `{policy}` (scope `{scope}`). CSV: `{output_csv}`.",
        "",
        f"T-count wins vs baseline: {len(wins)}; wins whose selected candidate "
        f"is fully proven: {len(proven_wins)}.",
        "",
        "| target | split | selected objective | T ratio | outcome | selected proof | baseline proof |",
        "|---|---|---|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {target} | {split} | {objective} | {ratio} | {outcome} | {sel} | {base} |".format(
                target=row["target"],
                split=row["source_split"],
                objective=row["final_objective"],
                ratio=row["tcount_ratio_vs_baseline"],
                outcome=row["outcome_vs_baseline"],
                sel=row["selected_verification_status"],
                base=row["baseline_verification_status"],
            )
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    details = read_csv_path(args.portfolio_details_csv)
    statuses = verification_status_by_key(
        [Path(item.strip()) for item in args.verification_csvs.split(",") if item.strip()]
    )
    rows = impact_rows(details, statuses, policy=args.policy, scope=args.scope)
    write_outputs(
        rows,
        output_csv=args.output_csv,
        report_path=args.report_path,
        policy=args.policy,
        scope=args.scope,
    )
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
