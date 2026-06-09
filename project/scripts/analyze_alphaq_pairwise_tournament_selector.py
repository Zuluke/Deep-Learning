from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_alphaq_learned_objective_selector import normalize_rows
from scripts.analyze_alphaq_oracle_equivalence import oracle_info
from scripts.analyze_alphaq_pairwise_selector_separability import clear_pair_rows
from scripts.analyze_alphaq_pairwise_selector_separability import pairwise_margin
from scripts.analyze_alphaq_pairwise_selector_separability import selector_profiles
from scripts.analyze_alphaq_pairwise_selector_separability import train_weights
from scripts.run_best_objective_beam_ablation import parse_paths
from scripts.run_best_objective_beam_ablation import read_csv
from scripts.run_best_objective_beam_ablation import safe_ratio
from scripts.run_objective_beam_policy_grid import best_current_beams
from scripts.run_objective_beam_policy_grid import best_policy_beams
from scripts.structural_target import coerce_float


DEFAULT_DECOMP_CSVS = (
    PROJECT_ROOT / "results" / "csv" / "alphaq_decomposition_objective_ablation.csv",
    PROJECT_ROOT / "results" / "csv" / "alphaq_decomposition_objective_holdout_ablation.csv",
)
DEFAULT_GRID_CSV = PROJECT_ROOT / "results" / "csv" / "alphaq_objective_beam_policy_grid.csv"
DEFAULT_CURRENT_BEAM_CSVS = (
    PROJECT_ROOT / "results" / "csv" / "alphaq_beam_materializer_ablation.csv",
    PROJECT_ROOT / "results" / "csv" / "alphaq_beam_materializer_holdout_ablation.csv",
)
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "results" / "csv" / "alphaq_pairwise_tournament_selector_summary.csv"
DEFAULT_DETAIL_CSV = PROJECT_ROOT / "results" / "csv" / "alphaq_pairwise_tournament_selector_details.csv"
DEFAULT_REPORT = PROJECT_ROOT / "results" / "reports" / "alphaq_pairwise_tournament_selector.md"
DEFAULT_FIGURE = PROJECT_ROOT / "results" / "figures" / "alphaq_pairwise_tournament_selector.png"
OBJECTIVE_TIE_RANK = {
    "factor_count_pair_cap": 0,
    "factor_count": 1,
    "mixed_pair": 2,
    "depth_guarded_mixed_pair": 3,
    "t_preserving_frontier_pair": 4,
    "frontier_pair": 5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Turn pairwise AlphaQ-only objective decisions into leave-one-target-out tournament selectors."
    )
    parser.add_argument(
        "--decomposition-csvs",
        default=",".join(str(path) for path in DEFAULT_DECOMP_CSVS),
    )
    parser.add_argument("--grid-csv", type=Path, default=DEFAULT_GRID_CSV)
    parser.add_argument(
        "--current-beam-csvs",
        default=",".join(str(path) for path in DEFAULT_CURRENT_BEAM_CSVS),
    )
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--detail-csv", type=Path, default=DEFAULT_DETAIL_CSV)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--figure-path", type=Path, default=DEFAULT_FIGURE)
    return parser.parse_args()


def select_tournament_objective(
    *,
    target_rows: list[dict[str, Any]],
    features: tuple[str, ...],
    weights: tuple[int, ...],
) -> tuple[str, dict[str, int], dict[str, float]]:
    wins = {row["objective_variant"]: 0 for row in target_rows}
    margins = {row["objective_variant"]: 0.0 for row in target_rows}
    for left in target_rows:
        for right in target_rows:
            if left["objective_variant"] == right["objective_variant"]:
                continue
            margin = pairwise_margin(
                pair_delta(left, right, features),
                features,
                weights,
            )
            if margin < 0:
                objective = left["objective_variant"]
                wins[objective] += 1
                margins[objective] += -margin
    selected = max(
        target_rows,
        key=lambda row: (
            wins[row["objective_variant"]],
            margins[row["objective_variant"]],
            -numeric(row.get("factor_count")),
            -OBJECTIVE_TIE_RANK.get(row["objective_variant"], 99),
        ),
    )
    return selected["objective_variant"], wins, margins


def pair_delta(left: dict[str, Any], right: dict[str, Any], features: tuple[str, ...]) -> dict[str, Any]:
    return {
        f"delta_{feature}": float(left[f"norm_{feature}"]) - float(right[f"norm_{feature}"])
        for feature in features
    }


def numeric(value: Any) -> float:
    parsed = coerce_float(value)
    return float("inf") if parsed is None else parsed


def best_grid_row_for_objective(
    grid_rows: list[dict[str, str]],
    target: str,
    objective_variant: str,
) -> dict[str, str] | None:
    return best_policy_beams(grid_rows, objective_variant).get(target)


def detail_rows(
    *,
    decomp_rows: list[dict[str, str]],
    grid_rows: list[dict[str, str]],
    current_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    all_features = tuple(sorted({feature for profile in selector_profiles() for feature in profile["features"]}))
    normalized = normalize_rows(decomp_rows, all_features)
    clear_pairs = clear_pair_rows(normalized_rows=normalized, grid_rows=grid_rows, features=all_features)
    trainable_targets = sorted({pair["target"] for pair in clear_pairs})
    all_targets = sorted({row["target"] for row in normalized})
    oracle = oracle_info(grid_rows)
    current = best_current_beams(current_rows)
    details: list[dict[str, Any]] = []
    for profile in selector_profiles():
        features = tuple(profile["features"])
        for holdout in all_targets:
            weights = train_weights(
                pairs=clear_pairs,
                train_targets=[target for target in trainable_targets if target != holdout],
                profile=profile,
            )
            target_rows = [row for row in normalized if row["target"] == holdout]
            selected, wins, margins = select_tournament_objective(
                target_rows=target_rows,
                features=features,
                weights=weights,
            )
            beam = best_grid_row_for_objective(grid_rows, holdout, selected)
            baseline = current.get(holdout)
            if beam is None or baseline is None:
                continue
            details.append(
                {
                    "selector": profile["selector"].replace("loto_pairwise_", "loto_tournament_"),
                    "feature_scope": profile["feature_scope"],
                    "fold": holdout,
                    "target": holdout,
                    "selected_objective": selected,
                    "oracle_objective": oracle[holdout]["oracle_objective"],
                    "equivalent_objectives": ",".join(oracle[holdout]["equivalent_objectives"]),
                    "exact_oracle_match": selected == oracle[holdout]["oracle_objective"],
                    "oracle_equivalent": selected in oracle[holdout]["equivalent_objectives"],
                    "wins": format_scores(wins),
                    "margins": format_scores(margins),
                    "weights": format_weights(features, weights),
                    "tcount_ratio": safe_ratio(beam.get("tcount"), baseline.get("tcount")),
                    "primary_ratio": safe_ratio(beam.get("primary_nc_depth_ratio"), baseline.get("primary_nc_depth_ratio")),
                    "qasm_ratio": safe_ratio(beam.get("qasm_depth"), baseline.get("qasm_depth")),
                    "selected_beam_materializer": beam.get("materializer"),
                    "summary_path": beam.get("summary_path"),
                }
            )
    return details


def summary_rows(details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for selector in sorted({row["selector"] for row in details}):
        items = [row for row in details if row["selector"] == selector]
        total = len(items)
        rows.append(
            {
                "selector": selector,
                "feature_scope": items[0]["feature_scope"] if items else "",
                "targets": total,
                "exact_oracle_matches": sum(bool(row["exact_oracle_match"]) for row in items),
                "oracle_equivalent_matches": sum(bool(row["oracle_equivalent"]) for row in items),
                "tcount_nonworse": count(items, "tcount_ratio", strict=False),
                "tcount_wins": count(items, "tcount_ratio", strict=True),
                "primary_wins": count(items, "primary_ratio", strict=True),
                "qasm_wins": count(items, "qasm_ratio", strict=True),
                "joint_nonworse": sum(
                    metric_ok(row.get("tcount_ratio"), strict=False)
                    and metric_ok(row.get("primary_ratio"), strict=False)
                    and metric_ok(row.get("qasm_ratio"), strict=False)
                    for row in items
                ),
                "target_details": "; ".join(
                    f"{row['target']}->{row['selected_objective']} exact={row['exact_oracle_match']} equiv={row['oracle_equivalent']}"
                    for row in items
                ),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            -int(row["oracle_equivalent_matches"]),
            -int(row["exact_oracle_matches"]),
            -int(row["joint_nonworse"]),
            row["selector"],
        ),
    )


def count(rows: list[dict[str, Any]], key: str, *, strict: bool) -> int:
    return sum(metric_ok(row.get(key), strict=strict) for row in rows)


def metric_ok(value: Any, *, strict: bool) -> bool:
    numeric_value = coerce_float(value)
    if numeric_value is None:
        return False
    return numeric_value < 1.0 if strict else numeric_value <= 1.0


def format_weights(features: tuple[str, ...], weights: tuple[int, ...]) -> str:
    return ",".join(
        f"{feature}:{weight}"
        for feature, weight in zip(features, weights)
        if weight != 0
    )


def format_scores(scores: dict[str, Any]) -> str:
    return ",".join(f"{key}:{value:.6g}" if isinstance(value, float) else f"{key}:{value}" for key, value in sorted(scores.items()))


def write_detail_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(
        path,
        rows,
        [
            "selector",
            "feature_scope",
            "fold",
            "target",
            "selected_objective",
            "oracle_objective",
            "equivalent_objectives",
            "exact_oracle_match",
            "oracle_equivalent",
            "wins",
            "margins",
            "weights",
            "tcount_ratio",
            "primary_ratio",
            "qasm_ratio",
            "selected_beam_materializer",
            "summary_path",
        ],
    )


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(
        path,
        rows,
        [
            "selector",
            "feature_scope",
            "targets",
            "exact_oracle_matches",
            "oracle_equivalent_matches",
            "tcount_nonworse",
            "tcount_wins",
            "primary_wins",
            "qasm_wins",
            "joint_nonworse",
            "target_details",
        ],
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict[str, Any]], output_csv: Path, detail_csv: Path, figure_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    best = rows[0] if rows else None
    lines = [
        "# AlphaQ pairwise tournament selector",
        "",
        f"Summary CSV: `{output_csv}`.",
        f"Detail CSV: `{detail_csv}`.",
        f"Figure: `{figure_path}`.",
        "",
        "This selector trains pairwise rules on clear oracle-vs-non-oracle objective pairs, then selects a final objective by tournament wins on the held-out target.",
        "",
        "## Bottom line",
        "",
    ]
    if best is None:
        lines.append("No tournament rows were generated.")
    else:
        lines.append(
            f"The best tournament profile `{best['selector']}` reaches "
            f"{best['exact_oracle_matches']}/{best['targets']} exact matches and "
            f"{best['oracle_equivalent_matches']}/{best['targets']} metric-equivalent matches."
        )
        lines.append(
            "The pairwise tournament recovers the `barenco_tof_3` miss, but it does not improve the overall metric-equivalent frontier because it introduces a different held-out miss."
        )
    lines.extend(
        [
            "",
            "| selector | scope | exact oracle | oracle-equivalent | T <= current | T < current | primary < current | QASM < current | all non-worse |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {selector} | {scope} | {exact}/{n} | {equiv}/{n} | {tn}/{n} | {tw}/{n} | {p}/{n} | {q}/{n} | {j}/{n} |".format(
                selector=row["selector"],
                scope=row["feature_scope"],
                exact=row["exact_oracle_matches"],
                equiv=row["oracle_equivalent_matches"],
                n=row["targets"],
                tn=row["tcount_nonworse"],
                tw=row["tcount_wins"],
                p=row["primary_wins"],
                q=row["qasm_wins"],
                j=row["joint_nonworse"],
            )
        )
    lines.extend(["", "## Target Details", ""])
    for row in rows:
        lines.append(f"- `{row['selector']}`: {row['target_details']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figure(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [row["selector"].replace("loto_tournament_", "").replace("_", "\n") for row in rows]
    fields = [
        ("exact_oracle_matches", "exact"),
        ("oracle_equivalent_matches", "equivalent"),
        ("joint_nonworse", "all non-worse"),
    ]
    x = range(len(rows))
    width = 0.23
    fig, ax = plt.subplots(figsize=(9, 4.2), constrained_layout=True)
    colors = ["#4c78a8", "#1b9e77", "#d95f02"]
    for offset, (field, label) in enumerate(fields):
        positions = [item + (offset - 1) * width for item in x]
        ax.bar(positions, [int(row[field]) for row in rows], width=width, label=label, color=colors[offset])
    ax.set_title("Pairwise tournament objective selectors")
    ax.set_ylabel("targets")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, max(int(row["targets"]) for row in rows) + 1 if rows else 1)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=3)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def load_rows(args: argparse.Namespace) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    decomp_rows = [
        row
        for path in parse_paths(args.decomposition_csvs)
        for row in read_csv(path)
    ]
    current_rows = [
        row
        for path in parse_paths(args.current_beam_csvs)
        for row in read_csv(path)
    ]
    return decomp_rows, read_csv(args.grid_csv), current_rows


def main() -> int:
    args = parse_args()
    decomp_rows, grid_rows, current_rows = load_rows(args)
    details = detail_rows(decomp_rows=decomp_rows, grid_rows=grid_rows, current_rows=current_rows)
    summaries = summary_rows(details)
    write_detail_csv(args.detail_csv, details)
    write_summary_csv(args.output_csv, summaries)
    write_report(args.report_path, summaries, args.output_csv, args.detail_csv, args.figure_path)
    write_figure(args.figure_path, summaries)
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.detail_csv}")
    print(f"Wrote {args.report_path}")
    print(f"Wrote {args.figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
