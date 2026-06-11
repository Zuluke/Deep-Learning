from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_best_objective_beam_ablation import inf_if_none
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
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "results" / "csv" / "alphaq_objective_selector_summary.csv"
DEFAULT_DETAIL_CSV = PROJECT_ROOT / "results" / "csv" / "alphaq_objective_selector_details.csv"
DEFAULT_REPORT = PROJECT_ROOT / "results" / "reports" / "alphaq_objective_selector_summary.md"
DEFAULT_FIGURE = PROJECT_ROOT / "results" / "figures" / "alphaq_objective_selector_summary.png"

OBJECTIVE_TIE_RANK = {
    "factor_count_pair_cap": 0,
    "factor_count": 1,
    "mixed_pair": 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate AlphaQ-only objective selectors against current and oracle beam candidates."
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


def selector_specs() -> dict[str, Callable[[dict[str, str]], tuple[float, ...]]]:
    return {
        "min_factor_count": lambda row: (
            f(row, "factor_count"),
            objective_tie_rank(row),
        ),
        "min_factor_count_then_overlap": lambda row: (
            f(row, "factor_count"),
            f(row, "factor_pairwise_support_overlap_mean"),
            objective_tie_rank(row),
        ),
        "min_factor_count_then_jaccard": lambda row: (
            f(row, "factor_count"),
            f(row, "factor_pairwise_jaccard_mean"),
            objective_tie_rank(row),
        ),
        "min_factor_count_then_qubit_ci": lambda row: (
            f(row, "factor_count"),
            f(row, "factor_qubit_concentration_index"),
            objective_tie_rank(row),
        ),
        "min_overlap_then_factor_count": lambda row: (
            f(row, "factor_pairwise_support_overlap_mean"),
            f(row, "factor_count"),
            objective_tie_rank(row),
        ),
        "min_jaccard_then_factor_count": lambda row: (
            f(row, "factor_pairwise_jaccard_mean"),
            f(row, "factor_count"),
            objective_tie_rank(row),
        ),
        "min_support_weight_then_factor_count": lambda row: (
            f(row, "factor_support_weight_mean"),
            f(row, "factor_count"),
            objective_tie_rank(row),
        ),
    }


def f(row: dict[str, str], key: str) -> float:
    return inf_if_none(row.get(key))


def objective_tie_rank(row: dict[str, str]) -> float:
    return float(OBJECTIVE_TIE_RANK.get(row.get("objective_variant", ""), 99))


def beam_oracle_objective_by_target(grid_rows: list[dict[str, str]]) -> dict[str, str]:
    result = {}
    objectives = sorted({row.get("objective_variant", "") for row in grid_rows if row.get("objective_variant")})
    policy_beams = {
        objective: best_policy_beams(grid_rows, objective)
        for objective in objectives
    }
    targets = sorted(
        {
            target
            for beams in policy_beams.values()
            for target in beams
        }
    )
    for target in targets:
        candidates = [
            row
            for beams in policy_beams.values()
            if (row := beams.get(target)) is not None
        ]
        if not candidates:
            continue
        selected = min(
            candidates,
            key=lambda row: (
                f(row, "tcount"),
                f(row, "primary_nc_depth_ratio"),
                f(row, "qasm_depth"),
                row.get("objective_variant", ""),
            ),
        )
        result[target] = selected["objective_variant"]
    return result


def selected_rows_by_policy(
    rows: list[dict[str, str]],
    selector_name: str,
    key_fn: Callable[[dict[str, str]], tuple[float, ...]],
) -> list[dict[str, str]]:
    selected = []
    for target in sorted({row["target"] for row in rows}):
        candidates = [row for row in rows if row["target"] == target]
        row = min(candidates, key=key_fn)
        selected.append({**row, "selector": selector_name})
    return selected


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
    current = best_current_beams(current_rows)
    oracle = beam_oracle_objective_by_target(grid_rows)
    details: list[dict[str, Any]] = []
    for selector, key_fn in selector_specs().items():
        for selected in selected_rows_by_policy(decomp_rows, selector, key_fn):
            target = selected["target"]
            beam = best_grid_row_for_objective(grid_rows, target, selected["objective_variant"])
            baseline = current.get(target)
            if beam is None or baseline is None:
                continue
            details.append(
                {
                    "selector": selector,
                    "target": target,
                    "selected_objective": selected["objective_variant"],
                    "oracle_objective": oracle.get(target),
                    "matches_oracle": selected["objective_variant"] == oracle.get(target),
                    "tcount_ratio": safe_ratio(beam.get("tcount"), baseline.get("tcount")),
                    "primary_ratio": safe_ratio(
                        beam.get("primary_nc_depth_ratio"),
                        baseline.get("primary_nc_depth_ratio"),
                    ),
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
                "targets": total,
                "oracle_matches": sum(bool(row["matches_oracle"]) for row in items),
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
                "median_tcount_ratio": median_metric(items, "tcount_ratio"),
                "median_primary_ratio": median_metric(items, "primary_ratio"),
                "median_qasm_ratio": median_metric(items, "qasm_ratio"),
                "target_details": "; ".join(
                    f"{row['target']}->{row['selected_objective']}:T={fmt(row['tcount_ratio'])},P={fmt(row['primary_ratio'])},Q={fmt(row['qasm_ratio'])}"
                    for row in items
                ),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            -int(row["oracle_matches"]),
            -int(row["joint_nonworse"]),
            -int(row["tcount_wins"]),
            float(row["median_qasm_ratio"] or 99),
            row["selector"],
        ),
    )


def count(rows: list[dict[str, Any]], key: str, *, strict: bool) -> int:
    return sum(metric_ok(row.get(key), strict=strict) for row in rows)


def metric_ok(value: Any, *, strict: bool) -> bool:
    numeric = coerce_float(value)
    if numeric is None:
        return False
    return numeric < 1.0 if strict else numeric <= 1.0


def median_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = sorted(value for row in rows if (value := coerce_float(row.get(key))) is not None)
    if not values:
        return None
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2


def write_detail_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "selector",
        "target",
        "selected_objective",
        "oracle_objective",
        "matches_oracle",
        "tcount_ratio",
        "primary_ratio",
        "qasm_ratio",
        "selected_beam_materializer",
        "summary_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "selector",
        "targets",
        "oracle_matches",
        "tcount_nonworse",
        "tcount_wins",
        "primary_wins",
        "qasm_wins",
        "joint_nonworse",
        "median_tcount_ratio",
        "median_primary_ratio",
        "median_qasm_ratio",
        "target_details",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict[str, Any]], summary_csv: Path, detail_csv: Path, figure_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    best = rows[0] if rows else None
    lines = [
        "# AlphaQ-only objective selector audit",
        "",
        f"Summary CSV: `{summary_csv}`.",
        f"Detail CSV: `{detail_csv}`.",
        f"Figure: `{figure_path}`.",
        "",
        "This audit tests deterministic objective selectors using only tensor/factor metrics available before external structural evaluation.",
        "",
    ]
    if best is not None:
        lines.append(f"Best selector by oracle-match and non-worse counts: `{best['selector']}`.")
        lines.append("")
    lines.extend(
        [
            "| selector | oracle match | T <= current | T < current | primary < current | QASM < current | all non-worse | median T | median primary | median QASM |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {selector} | {oracle}/{n} | {tn}/{n} | {tw}/{n} | {p}/{n} | {q}/{n} | {j}/{n} | {mt} | {mp} | {mq} |".format(
                selector=row["selector"],
                oracle=row["oracle_matches"],
                n=row["targets"],
                tn=row["tcount_nonworse"],
                tw=row["tcount_wins"],
                p=row["primary_wins"],
                q=row["qasm_wins"],
                j=row["joint_nonworse"],
                mt=fmt(row.get("median_tcount_ratio")),
                mp=fmt(row.get("median_primary_ratio")),
                mq=fmt(row.get("median_qasm_ratio")),
            )
        )
    lines.extend(["", "## Target Details", ""])
    for row in rows:
        lines.append(f"- `{row['selector']}`: {row['target_details']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figure(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [row["selector"].replace("min_", "").replace("_then_", "\nthen ") for row in rows]
    x = range(len(rows))
    fields = [
        ("oracle_matches", "oracle match"),
        ("tcount_wins", "T < current"),
        ("primary_wins", "primary < current"),
        ("qasm_wins", "QASM < current"),
        ("joint_nonworse", "all non-worse"),
    ]
    width = 0.15
    colors = ["#1b9e77", "#4c78a8", "#d95f02", "#7570b3", "#66a61e"]
    fig, ax = plt.subplots(figsize=(12.5, 4.8), constrained_layout=True)
    for offset, (field, title) in enumerate(fields):
        positions = [item + (offset - 2) * width for item in x]
        ax.bar(positions, [int(row[field]) for row in rows], width=width, label=title, color=colors[offset])
    ax.set_title("AlphaQ-only objective selectors")
    ax.set_ylabel("targets")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=24, ha="right")
    ax.set_ylim(0, max(int(row["targets"]) for row in rows) + 0.8 if rows else 1)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def fmt(value: Any) -> str:
    numeric = coerce_float(value)
    return "" if numeric is None else f"{numeric:.3g}"


def main() -> int:
    args = parse_args()
    decomp_rows = [row for path in parse_paths(args.decomposition_csvs) for row in read_csv(path)]
    grid_rows = read_csv(args.grid_csv)
    current_rows = [row for path in parse_paths(args.current_beam_csvs) for row in read_csv(path)]
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
