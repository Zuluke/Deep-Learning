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

from scripts.run_best_objective_beam_ablation import parse_paths
from scripts.run_best_objective_beam_ablation import read_csv
from scripts.run_objective_beam_policy_grid import best_policy_beams
from scripts.structural_target import coerce_float


DEFAULT_GRID_CSV = PROJECT_ROOT / "results" / "csv" / "alphaq_objective_beam_policy_grid.csv"
DEFAULT_DETAIL_CSVS = (
    PROJECT_ROOT / "results" / "csv" / "alphaq_objective_selector_details.csv",
    PROJECT_ROOT / "results" / "csv" / "alphaq_selector_calibration_details.csv",
    PROJECT_ROOT / "results" / "csv" / "alphaq_budgeted_concentration_policy_details.csv",
)
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "results" / "csv" / "alphaq_oracle_equivalence_details.csv"
DEFAULT_SUMMARY_CSV = PROJECT_ROOT / "results" / "csv" / "alphaq_oracle_equivalence_summary.csv"
DEFAULT_REPORT = PROJECT_ROOT / "results" / "reports" / "alphaq_oracle_equivalence.md"
DEFAULT_FIGURE = PROJECT_ROOT / "results" / "figures" / "alphaq_oracle_equivalence.png"
METRIC_KEYS = ("tcount", "primary_nc_depth_ratio", "qasm_depth")
OBJECTIVES = (
    "factor_count",
    "factor_count_pair_cap",
    "mixed_pair",
    "frontier_pair",
    "depth_guarded_mixed_pair",
    "t_preserving_frontier_pair",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit selector oracle matches under metric-equivalent objective ties."
    )
    parser.add_argument("--grid-csv", type=Path, default=DEFAULT_GRID_CSV)
    parser.add_argument(
        "--detail-csvs",
        default=",".join(str(path) for path in DEFAULT_DETAIL_CSVS),
    )
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--figure-path", type=Path, default=DEFAULT_FIGURE)
    return parser.parse_args()


def objective_rows_by_target(grid_rows: list[dict[str, str]]) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for objective in OBJECTIVES:
        for target, row in best_policy_beams(grid_rows, objective).items():
            result.setdefault(target, {})[objective] = row
    return result


def metric_key(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        numeric(row.get("tcount")),
        numeric(row.get("primary_nc_depth_ratio")),
        numeric(row.get("qasm_depth")),
    )


def oracle_info(grid_rows: list[dict[str, str]], *, tolerance: float = 1e-9) -> dict[str, dict[str, Any]]:
    by_target = objective_rows_by_target(grid_rows)
    result: dict[str, dict[str, Any]] = {}
    for target, objective_rows in by_target.items():
        oracle_objective, oracle_row = min(
            objective_rows.items(),
            key=lambda item: (*metric_key(item[1]), item[0]),
        )
        oracle_metrics = metric_key(oracle_row)
        equivalent = sorted(
            objective
            for objective, row in objective_rows.items()
            if metrics_equal(metric_key(row), oracle_metrics, tolerance=tolerance)
        )
        result[target] = {
            "oracle_objective": oracle_objective,
            "oracle_metrics": oracle_metrics,
            "equivalent_objectives": equivalent,
        }
    return result


def metrics_equal(left: tuple[float, ...], right: tuple[float, ...], *, tolerance: float) -> bool:
    return all(abs(a - b) <= tolerance for a, b in zip(left, right))


def numeric(value: Any) -> float:
    parsed = coerce_float(value)
    if parsed is None:
        return float("inf")
    return parsed


def selector_rows(detail_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen = set()
    result = []
    for row in detail_rows:
        selector = row.get("selector", "")
        target = row.get("target", "")
        objective = row.get("selected_objective", "")
        if not selector or not target or not objective:
            continue
        key = (selector, target, objective)
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def equivalence_detail_rows(
    *,
    grid_rows: list[dict[str, str]],
    selector_detail_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    info = oracle_info(grid_rows)
    rows = []
    for row in selector_rows(selector_detail_rows):
        target = row["target"]
        target_info = info.get(target)
        if target_info is None:
            continue
        selected = row["selected_objective"]
        exact = selected == target_info["oracle_objective"]
        equivalent = selected in target_info["equivalent_objectives"]
        rows.append(
            {
                "selector": row["selector"],
                "target": target,
                "selected_objective": selected,
                "oracle_objective": target_info["oracle_objective"],
                "equivalent_objectives": ",".join(target_info["equivalent_objectives"]),
                "exact_oracle_match": exact,
                "oracle_equivalent": equivalent,
                "recovered_by_equivalence": equivalent and not exact,
                "tcount_ratio": row.get("tcount_ratio", ""),
                "primary_ratio": row.get("primary_ratio", ""),
                "qasm_ratio": row.get("qasm_ratio", ""),
            }
        )
    return rows


def summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for selector in sorted({row["selector"] for row in rows}):
        items = [row for row in rows if row["selector"] == selector]
        total = len(items)
        exact = sum(bool(row["exact_oracle_match"]) for row in items)
        equivalent = sum(bool(row["oracle_equivalent"]) for row in items)
        recovered = sum(bool(row["recovered_by_equivalence"]) for row in items)
        result.append(
            {
                "selector": selector,
                "targets": total,
                "exact_oracle_matches": exact,
                "oracle_equivalent_matches": equivalent,
                "recovered_by_equivalence": recovered,
                "tcount_nonworse": count(items, "tcount_ratio", strict=False),
                "tcount_wins": count(items, "tcount_ratio", strict=True),
                "primary_wins": count(items, "primary_ratio", strict=True),
                "qasm_wins": count(items, "qasm_ratio", strict=True),
                "target_details": "; ".join(
                    f"{row['target']}->{row['selected_objective']} exact={row['exact_oracle_match']} equivalent={row['oracle_equivalent']} equiv_set={row['equivalent_objectives']}"
                    for row in items
                ),
            }
        )
    return sorted(
        result,
        key=lambda row: (
            -int(row["oracle_equivalent_matches"]),
            -int(row["exact_oracle_matches"]),
            -int(row["tcount_wins"]),
            row["selector"],
        ),
    )


def count(rows: list[dict[str, Any]], key: str, *, strict: bool) -> int:
    values = [coerce_float(row.get(key)) for row in rows]
    if strict:
        return sum(value is not None and value < 1.0 for value in values)
    return sum(value is not None and value <= 1.0 for value in values)


def write_detail_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(
        path,
        rows,
        [
            "selector",
            "target",
            "selected_objective",
            "oracle_objective",
            "equivalent_objectives",
            "exact_oracle_match",
            "oracle_equivalent",
            "recovered_by_equivalence",
            "tcount_ratio",
            "primary_ratio",
            "qasm_ratio",
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
            "recovered_by_equivalence",
            "tcount_nonworse",
            "tcount_wins",
            "primary_wins",
            "qasm_wins",
            "target_details",
        ],
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict[str, Any]], detail_csv: Path, summary_csv: Path, figure_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    factor = next((row for row in rows if row["selector"] == "min_factor_count"), None)
    lines = [
        "# AlphaQ oracle equivalence audit",
        "",
        f"Detail CSV: `{detail_csv}`.",
        f"Summary CSV: `{summary_csv}`.",
        f"Figure: `{figure_path}`.",
        "",
        "This audit distinguishes exact oracle-objective matches from metric-equivalent matches. If two objectives produce the same best beam metrics, selecting either one should not be treated as a scientific failure.",
        "",
        "## Bottom line",
        "",
        bottom_line(factor),
        "",
        "| selector | exact oracle | oracle-equivalent | recovered by equivalence | T <= current | T < current | primary < current | QASM < current |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {selector} | {exact}/{n} | {equiv}/{n} | {recovered}/{n} | {tn}/{n} | {tw}/{n} | {p}/{n} | {q}/{n} |".format(
                selector=row["selector"],
                exact=row["exact_oracle_matches"],
                equiv=row["oracle_equivalent_matches"],
                recovered=row["recovered_by_equivalence"],
                n=row["targets"],
                tn=row["tcount_nonworse"],
                tw=row["tcount_wins"],
                p=row["primary_wins"],
                q=row["qasm_wins"],
            )
        )
    lines.extend(["", "## Target Details", ""])
    for row in rows:
        lines.append(f"- `{row['selector']}`: {row['target_details']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def bottom_line(factor: dict[str, Any] | None) -> str:
    if factor is None:
        return "No `min_factor_count` selector row was found."
    return (
        f"`min_factor_count` has {factor['exact_oracle_matches']}/{factor['targets']} exact oracle matches, "
        f"but {factor['oracle_equivalent_matches']}/{factor['targets']} metric-equivalent oracle matches. "
        f"{factor['recovered_by_equivalence']} apparent misses are therefore metric ties rather than real selector errors."
    )


def write_figure(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [row for row in rows if row["selector"] in important_selectors(rows)]
    labels = [row["selector"].replace("min_", "").replace("_", "\n") for row in rows]
    x = range(len(rows))
    width = 0.32
    fig, ax = plt.subplots(figsize=(9.8, 4.2), constrained_layout=True)
    ax.bar(
        [index - width / 2 for index in x],
        [int(row["exact_oracle_matches"]) for row in rows],
        width=width,
        label="exact",
        color="#4c78a8",
    )
    ax.bar(
        [index + width / 2 for index in x],
        [int(row["oracle_equivalent_matches"]) for row in rows],
        width=width,
        label="metric-equivalent",
        color="#1b9e77",
    )
    ax.set_title("Selector oracle matches after metric-equivalence audit")
    ax.set_ylabel("targets")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, max(int(row["targets"]) for row in rows) + 1 if rows else 1)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncols=2, loc="upper center")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def important_selectors(rows: list[dict[str, Any]]) -> set[str]:
    preferred = {
        "min_factor_count",
        "loto_calibrated",
        "loto_gated",
        "loto_budgeted_concentration",
        "min_factor_count_then_overlap",
    }
    available = {row["selector"] for row in rows}
    return preferred & available or available


def load_detail_rows(paths_text: str) -> list[dict[str, str]]:
    rows = []
    for path in parse_paths(paths_text):
        rows.extend(read_csv(path))
    return rows


def main() -> int:
    args = parse_args()
    grid_rows = read_csv(args.grid_csv)
    details = equivalence_detail_rows(
        grid_rows=grid_rows,
        selector_detail_rows=load_detail_rows(args.detail_csvs),
    )
    summaries = summary_rows(details)
    write_detail_csv(args.output_csv, details)
    write_summary_csv(args.summary_csv, summaries)
    write_report(args.report_path, summaries, args.output_csv, args.summary_csv, args.figure_path)
    write_figure(args.figure_path, summaries)
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.summary_csv}")
    print(f"Wrote {args.report_path}")
    print(f"Wrote {args.figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
