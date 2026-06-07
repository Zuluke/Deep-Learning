from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_alphaq_guarded_pairwise_override import baseline_objective_by_target
from scripts.analyze_alphaq_guarded_pairwise_override import best_grid_row_for_objective
from scripts.analyze_alphaq_guarded_pairwise_override import numeric
from scripts.analyze_alphaq_learned_objective_selector import normalize_rows
from scripts.analyze_alphaq_oracle_equivalence import oracle_info
from scripts.analyze_alphaq_pairwise_selector_separability import clear_pair_rows
from scripts.analyze_alphaq_pairwise_selector_separability import selector_profiles
from scripts.analyze_alphaq_pairwise_selector_separability import train_weights
from scripts.analyze_alphaq_pairwise_tournament_selector import select_tournament_objective
from scripts.run_best_objective_beam_ablation import parse_paths
from scripts.run_best_objective_beam_ablation import read_csv


DEFAULT_TRAIN_DECOMP_CSVS = (
    PROJECT_ROOT / "results" / "csv" / "alphaq_decomposition_objective_ablation.csv",
    PROJECT_ROOT / "results" / "csv" / "alphaq_decomposition_objective_holdout_ablation.csv",
)
DEFAULT_EXTERNAL_DECOMP_CSV = PROJECT_ROOT / "results" / "csv" / "alphaq_decomposition_objective_external_validation.csv"
DEFAULT_GRID_CSV = PROJECT_ROOT / "results" / "csv" / "alphaq_objective_beam_policy_grid_plus_external_validation.csv"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "results" / "csv" / "alphaq_external_selector_transfer_summary.csv"
DEFAULT_DETAIL_CSV = PROJECT_ROOT / "results" / "csv" / "alphaq_external_selector_transfer_details.csv"
DEFAULT_REPORT = PROJECT_ROOT / "results" / "reports" / "alphaq_external_selector_transfer.md"
PAIRWISE_SELECTOR = "loto_pairwise_linear_alphaq"
REQUIRED_OBJECTIVES = frozenset({"factor_count", "factor_count_pair_cap", "mixed_pair"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transfer the guarded pairwise selector from current targets to external targets.")
    parser.add_argument("--train-decomposition-csvs", default=",".join(str(path) for path in DEFAULT_TRAIN_DECOMP_CSVS))
    parser.add_argument("--external-decomposition-csv", type=Path, default=DEFAULT_EXTERNAL_DECOMP_CSV)
    parser.add_argument("--grid-csv", type=Path, default=DEFAULT_GRID_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--detail-csv", type=Path, default=DEFAULT_DETAIL_CSV)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def detail_rows(
    *,
    train_decomp_rows: list[dict[str, str]],
    external_decomp_rows: list[dict[str, str]],
    grid_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    train_decomp_rows = ok_rows(train_decomp_rows)
    external_decomp_rows = complete_target_rows(ok_rows(external_decomp_rows))
    profile = next(profile for profile in selector_profiles() if profile["selector"] == PAIRWISE_SELECTOR)
    features = tuple(profile["features"])
    combined = [*train_decomp_rows, *external_decomp_rows]
    normalized = normalize_rows(combined, features)
    pairs = clear_pair_rows(normalized_rows=normalized, grid_rows=grid_rows, features=features)
    train_targets = sorted({row["target"] for row in train_decomp_rows})
    external_targets = sorted({row["target"] for row in external_decomp_rows})
    weights = train_weights(pairs=pairs, train_targets=train_targets, profile=profile)
    baseline = baseline_objective_by_target(combined)
    oracle = oracle_info(grid_rows)
    rows = []
    for target in external_targets:
        target_rows = [row for row in normalized if row["target"] == target]
        tournament_objective, wins, margins = select_tournament_objective(
            target_rows=target_rows,
            features=features,
            weights=weights,
        )
        baseline_objective = baseline[target]
        baseline_beam = best_grid_row_for_objective(grid_rows, target, baseline_objective)
        tournament_beam = best_grid_row_for_objective(grid_rows, target, tournament_objective)
        if baseline_beam is None or tournament_beam is None:
            continue
        baseline_qasm = numeric(baseline_beam.get("qasm_depth"))
        tournament_qasm = numeric(tournament_beam.get("qasm_depth"))
        accepted = tournament_objective != baseline_objective and tournament_qasm <= baseline_qasm
        selected_objective = tournament_objective if accepted else baseline_objective
        selected_beam = tournament_beam if accepted else baseline_beam
        equivalent = set(oracle[target]["equivalent_objectives"])
        rows.append(
            {
                "target": target,
                "train_targets": ",".join(train_targets),
                "baseline_objective": baseline_objective,
                "tournament_objective": tournament_objective,
                "selected_objective": selected_objective,
                "override_accepted": accepted,
                "oracle_objective": oracle[target]["oracle_objective"],
                "equivalent_objectives": ",".join(oracle[target]["equivalent_objectives"]),
                "exact_oracle_match": selected_objective == oracle[target]["oracle_objective"],
                "oracle_equivalent": selected_objective in equivalent,
                "baseline_qasm_depth": baseline_beam.get("qasm_depth", ""),
                "tournament_qasm_depth": tournament_beam.get("qasm_depth", ""),
                "selected_tcount": selected_beam.get("tcount", ""),
                "selected_primary_nc_depth_ratio": selected_beam.get("primary_nc_depth_ratio", ""),
                "selected_qasm_depth": selected_beam.get("qasm_depth", ""),
                "qasm_worse_than_baseline": tournament_qasm > baseline_qasm if accepted else False,
                "wins": format_scores(wins),
                "margins": format_scores(margins),
                "weights": format_weights(features, weights),
            }
        )
    return rows


def ok_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row.get("execution_status", "ok") == "ok"]


def complete_target_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    completed_targets = {
        target
        for target in {row["target"] for row in rows}
        if {
            row.get("objective_variant", "")
            for row in rows
            if row["target"] == target
        }
        >= REQUIRED_OBJECTIVES
    }
    return [row for row in rows if row["target"] in completed_targets]


def summary_rows(details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total = len(details)
    return [
        {
            "selector": "external_qasm_guarded_pairwise_transfer",
            "targets": total,
            "exact_oracle_matches": sum(bool(row["exact_oracle_match"]) for row in details),
            "oracle_equivalent_matches": sum(bool(row["oracle_equivalent"]) for row in details),
            "overrides_accepted": sum(bool(row["override_accepted"]) for row in details),
            "qasm_worse_than_baseline": sum(bool(row["qasm_worse_than_baseline"]) for row in details),
            "target_details": "; ".join(
                f"{row['target']}->{row['selected_objective']} exact={row['exact_oracle_match']} equiv={row['oracle_equivalent']}"
                for row in details
            ),
        }
    ]


def format_weights(features: tuple[str, ...], weights: tuple[int, ...]) -> str:
    return ",".join(f"{feature}:{weight}" for feature, weight in zip(features, weights) if weight != 0)


def format_scores(scores: dict[str, Any]) -> str:
    return ",".join(f"{key}:{value:.6g}" if isinstance(value, float) else f"{key}:{value}" for key, value in sorted(scores.items()))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_detail_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(
        path,
        rows,
        [
            "target",
            "train_targets",
            "baseline_objective",
            "tournament_objective",
            "selected_objective",
            "override_accepted",
            "oracle_objective",
            "equivalent_objectives",
            "exact_oracle_match",
            "oracle_equivalent",
            "baseline_qasm_depth",
            "tournament_qasm_depth",
            "selected_tcount",
            "selected_primary_nc_depth_ratio",
            "selected_qasm_depth",
            "qasm_worse_than_baseline",
            "wins",
            "margins",
            "weights",
        ],
    )


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(
        path,
        rows,
        [
            "selector",
            "targets",
            "exact_oracle_matches",
            "oracle_equivalent_matches",
            "overrides_accepted",
            "qasm_worse_than_baseline",
            "target_details",
        ],
    )


def write_report(path: Path, summaries: list[dict[str, Any]], details: list[dict[str, Any]], output_csv: Path, detail_csv: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = summaries[0] if summaries else None
    lines = [
        "# AlphaQ external selector transfer",
        "",
        f"Summary CSV: `{output_csv}`.",
        f"Detail CSV: `{detail_csv}`.",
        "",
        "This applies the AlphaQ-only pairwise selector trained on the current policy-grid targets to external targets, then uses the same QASM guard against each external target's `min_factor_count` baseline.",
        "",
        "## Bottom line",
        "",
    ]
    if row:
        lines.append(
            f"The guarded transfer reaches {row['exact_oracle_matches']}/{row['targets']} exact and "
            f"{row['oracle_equivalent_matches']}/{row['targets']} metric-equivalent oracle matches, with "
            f"{row['qasm_worse_than_baseline']}/{row['targets']} QASM regressions against the external baseline."
        )
    lines.extend(["", "## Details", ""])
    for item in details:
        lines.append(
            f"- `{item['target']}`: baseline `{item['baseline_objective']}`, tournament `{item['tournament_objective']}`, selected `{item['selected_objective']}`, oracle `{item['oracle_objective']}`, equivalent={item['oracle_equivalent']}."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    train_rows = [row for path in parse_paths(args.train_decomposition_csvs) for row in read_csv(path)]
    external_rows = read_csv(args.external_decomposition_csv)
    grid_rows = read_csv(args.grid_csv)
    details = detail_rows(train_decomp_rows=train_rows, external_decomp_rows=external_rows, grid_rows=grid_rows)
    summaries = summary_rows(details)
    write_detail_csv(args.detail_csv, details)
    write_summary_csv(args.output_csv, summaries)
    write_report(args.report_path, summaries, details, args.output_csv, args.detail_csv)
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.detail_csv}")
    print(f"Wrote {args.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
