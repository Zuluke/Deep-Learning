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

from scripts.analyze_alphaq_objective_selectors import f
from scripts.analyze_alphaq_oracle_equivalence import oracle_info
from scripts.analyze_alphaq_pairwise_tournament_selector import detail_rows as tournament_detail_rows
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
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "results" / "csv" / "alphaq_guarded_pairwise_override_summary.csv"
DEFAULT_DETAIL_CSV = PROJECT_ROOT / "results" / "csv" / "alphaq_guarded_pairwise_override_details.csv"
DEFAULT_REPORT = PROJECT_ROOT / "results" / "reports" / "alphaq_guarded_pairwise_override.md"
DEFAULT_FIGURE = PROJECT_ROOT / "results" / "figures" / "alphaq_guarded_pairwise_override.png"
BASELINE_TIE_RANK = {
    "factor_count_pair_cap": 0,
    "factor_count": 1,
    "mixed_pair": 2,
    "depth_guarded_mixed_pair": 3,
    "t_preserving_frontier_pair": 4,
    "frontier_pair": 5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a QASM-guarded pairwise tournament override on top of min_factor_count."
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


def baseline_objective_by_target(decomp_rows: list[dict[str, str]]) -> dict[str, str]:
    result = {}
    for target in sorted({row["target"] for row in decomp_rows}):
        target_rows = [row for row in decomp_rows if row["target"] == target]
        selected = min(
            target_rows,
            key=lambda row: (
                f(row, "factor_count"),
                BASELINE_TIE_RANK.get(row.get("objective_variant", ""), 99),
            ),
        )
        result[target] = selected["objective_variant"]
    return result


def best_grid_row_for_objective(
    grid_rows: list[dict[str, str]],
    target: str,
    objective: str,
) -> dict[str, str] | None:
    return best_policy_beams(grid_rows, objective).get(target)


def tournament_linear_alphaq_by_target(
    *,
    decomp_rows: list[dict[str, str]],
    grid_rows: list[dict[str, str]],
    current_rows: list[dict[str, str]],
) -> dict[str, dict[str, Any]]:
    rows = tournament_detail_rows(
        decomp_rows=decomp_rows,
        grid_rows=grid_rows,
        current_rows=current_rows,
    )
    return {
        row["target"]: row
        for row in rows
        if row["selector"] == "loto_tournament_linear_alphaq"
    }


def detail_rows(
    *,
    decomp_rows: list[dict[str, str]],
    grid_rows: list[dict[str, str]],
    current_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    current = best_current_beams(current_rows)
    oracle = oracle_info(grid_rows)
    baseline = baseline_objective_by_target(decomp_rows)
    tournament = tournament_linear_alphaq_by_target(
        decomp_rows=decomp_rows,
        grid_rows=grid_rows,
        current_rows=current_rows,
    )
    rows: list[dict[str, Any]] = []
    for target in sorted(baseline):
        baseline_objective = baseline[target]
        tournament_objective = tournament[target]["selected_objective"]
        baseline_beam = best_grid_row_for_objective(grid_rows, target, baseline_objective)
        tournament_beam = best_grid_row_for_objective(grid_rows, target, tournament_objective)
        current_beam = current.get(target)
        if baseline_beam is None or tournament_beam is None or current_beam is None:
            continue
        baseline_qasm = numeric(baseline_beam.get("qasm_depth"))
        tournament_qasm = numeric(tournament_beam.get("qasm_depth"))
        accepted = tournament_objective != baseline_objective and tournament_qasm <= baseline_qasm
        selected_objective = tournament_objective if accepted else baseline_objective
        selected_beam = tournament_beam if accepted else baseline_beam
        equivalent = set(oracle[target]["equivalent_objectives"])
        rows.append(
            {
                "selector": "qasm_guarded_pairwise_override",
                "target": target,
                "baseline_objective": baseline_objective,
                "tournament_objective": tournament_objective,
                "selected_objective": selected_objective,
                "override_accepted": accepted,
                "override_reason": "qasm-nonworse" if accepted else reject_reason(tournament_objective, baseline_objective, tournament_qasm, baseline_qasm),
                "oracle_objective": oracle[target]["oracle_objective"],
                "equivalent_objectives": ",".join(oracle[target]["equivalent_objectives"]),
                "exact_oracle_match": selected_objective == oracle[target]["oracle_objective"],
                "oracle_equivalent": selected_objective in equivalent,
                "baseline_qasm_depth": baseline_beam.get("qasm_depth", ""),
                "tournament_qasm_depth": tournament_beam.get("qasm_depth", ""),
                "selected_beam_materializer": selected_beam.get("materializer", ""),
                "tcount_ratio": safe_ratio(selected_beam.get("tcount"), current_beam.get("tcount")),
                "primary_ratio": safe_ratio(
                    selected_beam.get("primary_nc_depth_ratio"),
                    current_beam.get("primary_nc_depth_ratio"),
                ),
                "qasm_ratio": safe_ratio(selected_beam.get("qasm_depth"), current_beam.get("qasm_depth")),
                "summary_path": selected_beam.get("summary_path", ""),
            }
        )
    return rows


def reject_reason(
    tournament_objective: str,
    baseline_objective: str,
    tournament_qasm: float,
    baseline_qasm: float,
) -> str:
    if tournament_objective == baseline_objective:
        return "same-objective"
    if tournament_qasm > baseline_qasm:
        return "qasm-worse"
    return "not-accepted"


def numeric(value: Any) -> float:
    parsed = coerce_float(value)
    return float("inf") if parsed is None else parsed


def summary_rows(details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total = len(details)
    return [
        {
            "selector": "qasm_guarded_pairwise_override",
            "targets": total,
            "exact_oracle_matches": sum(bool(row["exact_oracle_match"]) for row in details),
            "oracle_equivalent_matches": sum(bool(row["oracle_equivalent"]) for row in details),
            "overrides_accepted": sum(bool(row["override_accepted"]) for row in details),
            "tcount_nonworse": count(details, "tcount_ratio", strict=False),
            "tcount_wins": count(details, "tcount_ratio", strict=True),
            "primary_wins": count(details, "primary_ratio", strict=True),
            "qasm_wins": count(details, "qasm_ratio", strict=True),
            "joint_nonworse": sum(
                metric_ok(row.get("tcount_ratio"), strict=False)
                and metric_ok(row.get("primary_ratio"), strict=False)
                and metric_ok(row.get("qasm_ratio"), strict=False)
                for row in details
            ),
            "target_details": "; ".join(
                f"{row['target']}->{row['selected_objective']} accepted={row['override_accepted']} exact={row['exact_oracle_match']} equiv={row['oracle_equivalent']}"
                for row in details
            ),
        }
    ]


def count(rows: list[dict[str, Any]], key: str, *, strict: bool) -> int:
    return sum(metric_ok(row.get(key), strict=strict) for row in rows)


def metric_ok(value: Any, *, strict: bool) -> bool:
    parsed = coerce_float(value)
    if parsed is None:
        return False
    return parsed < 1.0 if strict else parsed <= 1.0


def write_detail_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(
        path,
        rows,
        [
            "selector",
            "target",
            "baseline_objective",
            "tournament_objective",
            "selected_objective",
            "override_accepted",
            "override_reason",
            "oracle_objective",
            "equivalent_objectives",
            "exact_oracle_match",
            "oracle_equivalent",
            "baseline_qasm_depth",
            "tournament_qasm_depth",
            "selected_beam_materializer",
            "tcount_ratio",
            "primary_ratio",
            "qasm_ratio",
            "summary_path",
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
    row = rows[0] if rows else None
    lines = [
        "# AlphaQ guarded pairwise override",
        "",
        f"Summary CSV: `{output_csv}`.",
        f"Detail CSV: `{detail_csv}`.",
        f"Figure: `{figure_path}`.",
        "",
        "This policy starts from the `min_factor_count` objective selector. It accepts the AlphaQ-only pairwise tournament objective only when the tournament objective's best beam materialization has QASM depth no worse than the baseline objective's best beam materialization. The guard uses QASM depth only; external structural metrics are used only for audit.",
        "",
        "## Bottom line",
        "",
    ]
    if row is None:
        lines.append("No guarded override rows were generated.")
    else:
        lines.append(
            f"The guarded override reaches {row['exact_oracle_matches']}/{row['targets']} exact oracle matches "
            f"and {row['oracle_equivalent_matches']}/{row['targets']} metric-equivalent matches, accepting "
            f"{row['overrides_accepted']}/{row['targets']} overrides."
        )
        lines.append(
            "This converts the local pairwise signal into a conservative deployable selector for the current grid: it accepts the barenco_tof_3 correction while rejecting the cuccaro_adder_n3 QASM-worse override."
        )
    lines.extend(
        [
            "",
            "| selector | exact oracle | oracle-equivalent | overrides | T <= current | T < current | primary < current | QASM < current | all non-worse |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in rows:
        lines.append(
            "| {selector} | {exact}/{n} | {equiv}/{n} | {ov}/{n} | {tn}/{n} | {tw}/{n} | {p}/{n} | {q}/{n} | {j}/{n} |".format(
                selector=item["selector"],
                exact=item["exact_oracle_matches"],
                equiv=item["oracle_equivalent_matches"],
                ov=item["overrides_accepted"],
                n=item["targets"],
                tn=item["tcount_nonworse"],
                tw=item["tcount_wins"],
                p=item["primary_wins"],
                q=item["qasm_wins"],
                j=item["joint_nonworse"],
            )
        )
    lines.extend(["", "## Target Details", ""])
    for item in rows:
        lines.append(f"- `{item['selector']}`: {item['target_details']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figure(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    row = rows[0]
    fields = [
        ("exact_oracle_matches", "exact"),
        ("oracle_equivalent_matches", "equivalent"),
        ("joint_nonworse", "all non-worse"),
        ("overrides_accepted", "overrides"),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    values = [int(row[field]) for field, _ in fields]
    labels = [label for _, label in fields]
    ax.bar(labels, values, color=["#4c78a8", "#1b9e77", "#d95f02", "#7570b3"])
    ax.set_title("QASM-guarded pairwise override")
    ax.set_ylabel("targets")
    ax.set_ylim(0, int(row["targets"]) + 1)
    ax.grid(axis="y", alpha=0.25)
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
