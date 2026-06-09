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

from scripts.run_best_objective_beam_ablation import collect_materialized_rows
from scripts.run_best_objective_beam_ablation import inf_if_none
from scripts.run_best_objective_beam_ablation import parse_paths
from scripts.run_best_objective_beam_ablation import parse_widths
from scripts.run_best_objective_beam_ablation import read_csv
from scripts.run_best_objective_beam_ablation import safe_ratio
from scripts.structural_target import coerce_float


DEFAULT_DECOMP_CSVS = (
    PROJECT_ROOT / "results" / "csv" / "alphaq_decomposition_objective_ablation.csv",
    PROJECT_ROOT / "results" / "csv" / "alphaq_decomposition_objective_holdout_ablation.csv",
)
DEFAULT_DECOMP_ROOTS = (
    PROJECT_ROOT / "results" / "alphaq_decomposition_objective_ablation",
    PROJECT_ROOT / "results" / "alphaq_decomposition_objective_holdout_ablation",
)
DEFAULT_CURRENT_BEAM_CSVS = (
    PROJECT_ROOT / "results" / "csv" / "alphaq_beam_materializer_ablation.csv",
    PROJECT_ROOT / "results" / "csv" / "alphaq_beam_materializer_holdout_ablation.csv",
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "alphaq_objective_beam_policy_grid"
DEFAULT_GRID_CSV = PROJECT_ROOT / "results" / "csv" / "alphaq_objective_beam_policy_grid.csv"
DEFAULT_POLICY_CSV = PROJECT_ROOT / "results" / "csv" / "alphaq_objective_beam_policy_summary.csv"
DEFAULT_REPORT = PROJECT_ROOT / "results" / "reports" / "alphaq_objective_beam_policy_summary.md"
DEFAULT_FIGURE = PROJECT_ROOT / "results" / "figures" / "alphaq_objective_beam_policy_summary.png"
DEFAULT_BEAM_WIDTHS = (4, 16)
DEFAULT_OBJECTIVE_POLICIES = (
    "factor_count",
    "factor_count_pair_cap",
    "mixed_pair",
    "frontier_pair",
    "depth_guarded_mixed_pair",
    "t_preserving_frontier_pair",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate fixed tensor-objective policies after beam materialization."
    )
    parser.add_argument(
        "--decomposition-csvs",
        default=",".join(str(path) for path in DEFAULT_DECOMP_CSVS),
    )
    parser.add_argument(
        "--decomposition-roots",
        default=",".join(str(path) for path in DEFAULT_DECOMP_ROOTS),
    )
    parser.add_argument(
        "--current-beam-csvs",
        default=",".join(str(path) for path in DEFAULT_CURRENT_BEAM_CSVS),
    )
    parser.add_argument(
        "--beam-widths",
        default=",".join(str(width) for width in DEFAULT_BEAM_WIDTHS),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--grid-csv", type=Path, default=DEFAULT_GRID_CSV)
    parser.add_argument("--policy-csv", type=Path, default=DEFAULT_POLICY_CSV)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--figure-path", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def decomp_rows_for_materialization(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "target": row["target"],
            "objective_variant": row["objective_variant"],
            "tcount": row.get("tcount", ""),
            "primary_nc_depth_ratio": row.get("primary_nc_depth_ratio", ""),
            "qasm_depth_ratio": row.get("qasm_depth_ratio", ""),
        }
        for row in rows
        if row.get("execution_status", "ok") == "ok"
    ]


def write_grid_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "target",
        "objective_variant",
        "materializer",
        "beam_width",
        "synthesis",
        "factor_order",
        "target_strategy",
        "tcount",
        "tdepth",
        "qasm_depth",
        "num_total_cnots",
        "primary_nc_depth_ratio",
        "qasm_depth_ratio",
        "structural_cost",
        "candidate_dir",
        "summary_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, lineterminator="\n", fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def best_current_beams(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return best_by_target(rows, prefix="beam-shared-parity")


def best_policy_beams(rows: list[dict[str, Any]], objective_variant: str) -> dict[str, dict[str, Any]]:
    return best_by_target(
        [row for row in rows if row.get("objective_variant") == objective_variant],
        prefix="selected-beam-shared-parity",
    )


def best_by_target(rows: list[dict[str, Any]], *, prefix: str) -> dict[str, dict[str, Any]]:
    result = {}
    for target in sorted({str(row["target"]) for row in rows}):
        candidates = [
            row
            for row in rows
            if row["target"] == target and str(row.get("materializer", "")).startswith(prefix)
        ]
        if not candidates:
            continue
        result[target] = min(candidates, key=beam_key)
    return result


def beam_key(row: dict[str, Any]) -> tuple[float, float, float, str]:
    return (
        inf_if_none(row.get("qasm_depth_ratio")),
        inf_if_none(row.get("primary_nc_depth_ratio")),
        inf_if_none(row.get("num_total_cnots")),
        str(row.get("materializer", "")),
    )


def policy_summary_rows(
    *,
    grid_rows: list[dict[str, Any]],
    current_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    current = best_current_beams(current_rows)
    rows = []
    for policy in objective_policies(grid_rows):
        policy_beams = best_policy_beams(grid_rows, policy)
        target_rows = []
        for target, selected in sorted(policy_beams.items()):
            baseline = current.get(target)
            if baseline is None:
                continue
            target_rows.append(compare_policy_target(policy, selected, baseline))
        rows.append(summarize_policy(policy, target_rows))
    return rows


def objective_policies(rows: list[dict[str, Any]]) -> tuple[str, ...]:
    present = {
        str(row.get("objective_variant", ""))
        for row in rows
        if row.get("objective_variant")
    }
    ordered = [policy for policy in DEFAULT_OBJECTIVE_POLICIES if policy in present]
    ordered.extend(sorted(present - set(ordered)))
    return tuple(ordered)


def compare_policy_target(
    policy: str,
    selected: dict[str, Any],
    current: dict[str, Any],
) -> dict[str, Any]:
    return {
        "policy": policy,
        "target": selected["target"],
        "tcount_ratio": safe_ratio(selected.get("tcount"), current.get("tcount")),
        "primary_ratio": safe_ratio(selected.get("primary_nc_depth_ratio"), current.get("primary_nc_depth_ratio")),
        "qasm_ratio": safe_ratio(selected.get("qasm_depth"), current.get("qasm_depth")),
        "tdepth_ratio": safe_ratio(selected.get("tdepth"), current.get("tdepth")),
        "cnot_ratio": safe_ratio(selected.get("num_total_cnots"), current.get("num_total_cnots")),
    }


def summarize_policy(policy: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    t_nonworse = count(rows, "tcount_ratio", strict=False)
    t_wins = count(rows, "tcount_ratio", strict=True)
    primary_wins = count(rows, "primary_ratio", strict=True)
    qasm_wins = count(rows, "qasm_ratio", strict=True)
    joint_wins = sum(
        metric_ok(row.get("tcount_ratio"), strict=False)
        and metric_ok(row.get("primary_ratio"), strict=False)
        and metric_ok(row.get("qasm_ratio"), strict=False)
        for row in rows
    )
    return {
        "policy": policy,
        "targets": total,
        "tcount_nonworse": t_nonworse,
        "tcount_wins": t_wins,
        "primary_wins": primary_wins,
        "qasm_wins": qasm_wins,
        "strict_improvement_score": t_wins + primary_wins + qasm_wins,
        "joint_nonworse": joint_wins,
        "median_tcount_ratio": median_metric(rows, "tcount_ratio"),
        "median_primary_ratio": median_metric(rows, "primary_ratio"),
        "median_qasm_ratio": median_metric(rows, "qasm_ratio"),
        "target_details": "; ".join(
            f"{row['target']}:T={fmt(row['tcount_ratio'])},P={fmt(row['primary_ratio'])},Q={fmt(row['qasm_ratio'])}"
            for row in rows
        ),
    }


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


def write_policy_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "policy",
        "targets",
        "tcount_nonworse",
        "tcount_wins",
        "primary_wins",
        "qasm_wins",
        "strict_improvement_score",
        "joint_nonworse",
        "median_tcount_ratio",
        "median_primary_ratio",
        "median_qasm_ratio",
        "target_details",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, lineterminator="\n", fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict[str, Any]], policy_csv: Path, figure_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    best = max(
        rows,
        key=lambda row: (
            int(row["strict_improvement_score"]),
            int(row["joint_nonworse"]),
            int(row["primary_wins"]),
            int(row["qasm_wins"]),
            -float(row["median_qasm_ratio"] or 99),
        ),
    )
    lines = [
        "# Objective beam policy grid",
        "",
        f"CSV: `{policy_csv}`.",
        f"Figure: `{figure_path}`.",
        "",
        "This grid asks whether a fixed tensor objective plus beam materialization can approximate the oracle best-objective result.",
        "",
        f"Best fixed policy by strict improvement score: `{best['policy']}`.",
        "",
        "| policy | T < current | T <= current | primary < current | QASM < current | all non-worse | median T | median primary | median QASM |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {policy} | {tw}/{n} | {t}/{n} | {p}/{n} | {q}/{n} | {j}/{n} | {mt} | {mp} | {mq} |".format(
                policy=row["policy"],
                tw=row["tcount_wins"],
                t=row["tcount_nonworse"],
                p=row["primary_wins"],
                q=row["qasm_wins"],
                j=row["joint_nonworse"],
                n=row["targets"],
                mt=fmt(row.get("median_tcount_ratio")),
                mp=fmt(row.get("median_primary_ratio")),
                mq=fmt(row.get("median_qasm_ratio")),
            )
        )
    lines.extend(["", "## Target Details", ""])
    for row in rows:
        lines.append(f"- `{row['policy']}`: {row['target_details']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figure(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    policies = [row["policy"] for row in rows]
    fields = [
        ("tcount_nonworse", "T <= current"),
        ("primary_wins", "primary < current"),
        ("qasm_wins", "QASM < current"),
        ("joint_nonworse", "all non-worse"),
    ]
    x = range(len(policies))
    width = 0.2
    fig, ax = plt.subplots(figsize=(9.6, 4.2), constrained_layout=True)
    colors = ["#1b9e77", "#4c78a8", "#d95f02", "#7570b3"]
    for offset, (field, label) in enumerate(fields):
        positions = [item + (offset - 1.5) * width for item in x]
        ax.bar(positions, [int(row[field]) for row in rows], width=width, label=label, color=colors[offset])
    ax.set_title("Fixed objective policies plus beam vs current checkpoint beam")
    ax.set_ylabel("targets")
    ax.set_xticks(list(x))
    ax.set_xticklabels(policies, rotation=20, ha="right")
    ax.set_ylim(0, max(int(row["targets"]) for row in rows) + 0.8)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def fmt(value: Any) -> str:
    numeric = coerce_float(value)
    return "" if numeric is None else f"{numeric:.3g}"


def main() -> int:
    args = parse_args()
    decomp_rows = [row for path in parse_paths(args.decomposition_csvs) for row in read_csv(path)]
    grid_rows = collect_materialized_rows(
        selected_rows=decomp_rows_for_materialization(decomp_rows),
        roots=parse_paths(args.decomposition_roots),
        beam_widths=parse_widths(args.beam_widths),
        output_root=args.output_root,
        force=args.force,
    )
    current_rows = [row for path in parse_paths(args.current_beam_csvs) for row in read_csv(path)]
    policy_rows = policy_summary_rows(grid_rows=grid_rows, current_rows=current_rows)
    write_grid_csv(args.grid_csv, grid_rows)
    write_policy_csv(args.policy_csv, policy_rows)
    write_report(args.report_path, policy_rows, args.policy_csv, args.figure_path)
    write_figure(args.figure_path, policy_rows)
    print(f"Wrote {args.grid_csv}")
    print(f"Wrote {args.policy_csv}")
    print(f"Wrote {args.report_path}")
    print(f"Wrote {args.figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
