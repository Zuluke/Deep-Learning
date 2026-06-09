from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_materialized_candidates import read_summary
from scripts.factor_concentration_metrics import factor_concentration_metrics
from scripts.materialize_shared_parity_candidate import canonicalize_factors
from scripts.materialize_shared_parity_candidate import load_manifest_row
from scripts.materialize_split_reward_candidate import resolve_project_path
from scripts.run_shared_parity_study import CORE_TARGETS
from scripts.run_shared_parity_study import STUDY_CASES
from scripts.structural_target import coerce_float


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "alphaq_decomposition_objective_ablation"
DEFAULT_CSV = PROJECT_ROOT / "results" / "csv" / "alphaq_decomposition_objective_ablation.csv"
DEFAULT_REPORT = PROJECT_ROOT / "results" / "reports" / "alphaq_decomposition_objective_ablation.md"
DEFAULT_FIGURE = PROJECT_ROOT / "results" / "figures" / "alphaq_decomposition_objective_ablation.png"
SPLIT_AWARE_OBJECTIVES = {
    "mixed-pair",
    "depth-guarded-mixed-pair",
    "frontier-pair",
    "t-preserving-frontier-pair",
}


@dataclass(frozen=True)
class ObjectiveVariant:
    name: str
    objective: str
    candidate_kind: str
    use_pair_cap: bool
    use_factor_count_cap: bool = False


OBJECTIVE_VARIANTS = (
    ObjectiveVariant(
        name="factor_count",
        objective="factor-count",
        candidate_kind="milp_span_factor_count_decomp",
        use_pair_cap=False,
    ),
    ObjectiveVariant(
        name="factor_count_pair_cap",
        objective="factor-count",
        candidate_kind="milp_span_factor_count_pair_cap_decomp",
        use_pair_cap=True,
    ),
    ObjectiveVariant(
        name="mixed_pair",
        objective="mixed-pair",
        candidate_kind="milp_span_mixed_pair_decomp",
        use_pair_cap=True,
    ),
    ObjectiveVariant(
        name="frontier_pair",
        objective="frontier-pair",
        candidate_kind="milp_span_frontier_pair_decomp",
        use_pair_cap=False,
    ),
    ObjectiveVariant(
        name="depth_guarded_mixed_pair",
        objective="depth-guarded-mixed-pair",
        candidate_kind="milp_span_depth_guarded_mixed_pair_decomp",
        use_pair_cap=True,
    ),
    ObjectiveVariant(
        name="t_preserving_frontier_pair",
        objective="t-preserving-frontier-pair",
        candidate_kind="milp_span_t_preserving_frontier_pair_decomp",
        use_pair_cap=False,
        use_factor_count_cap=True,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare factor-count and concentration-aware tensor decompositions "
            "under a fixed shared-parity materializer."
        )
    )
    parser.add_argument("--targets", default=",".join(CORE_TARGETS))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--figure-path", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--time-limit-sec", type=float, default=120.0)
    parser.add_argument(
        "--objective-variants",
        default=",".join(variant.name for variant in OBJECTIVE_VARIANTS),
        help=(
            "Comma-separated objective variants to run. "
            f"Available: {','.join(variant.name for variant in OBJECTIVE_VARIANTS)}."
        ),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help=(
            "Record objective-level failures and continue the batch. "
            "No replacement candidate is materialized for failed objectives."
        ),
    )
    return parser.parse_args()


def parse_targets(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_objective_variants(value: str) -> tuple[ObjectiveVariant, ...]:
    requested = [item.strip() for item in value.split(",") if item.strip()]
    by_name = {variant.name: variant for variant in OBJECTIVE_VARIANTS}
    unknown = [name for name in requested if name not in by_name]
    if unknown:
        raise ValueError(
            f"Unknown objective variant(s): {','.join(unknown)}. "
            f"Expected one of {','.join(by_name)}."
        )
    if not requested:
        raise ValueError("At least one objective variant must be requested.")
    return tuple(by_name[name] for name in requested)


def optimization_dir(linear_root: Path, target: str, max_weight: int, objective: str) -> Path:
    return linear_root / f"{target}_low-weight_w{max_weight}_k175_{objective}"


def factor_count_cap(output_root: Path, target: str) -> int:
    case = STUDY_CASES[target]
    factor_variant = next(variant for variant in OBJECTIVE_VARIANTS if variant.name == "factor_count")
    manifest = (
        output_root
        / "linear_span"
        / factor_variant.name
        / f"{target}_low-weight_w{case.max_action_weight}_k175_{factor_variant.objective}"
        / "candidate_factors_manifest.csv"
    )
    if not manifest.exists():
        raise RuntimeError(
            f"`t_preserving_frontier_pair` requires a solved factor_count baseline for {target}; "
            f"missing {manifest}."
        )
    row = load_manifest_row(manifest, target=target, candidate_kind=factor_variant.candidate_kind)
    value = coerce_float(row.get("effective_t_cost") or row.get("num_moves"))
    if value is None:
        raise RuntimeError(f"Cannot read factor_count cap for {target} from {manifest}.")
    return int(value)


def run_optimization(
    *,
    target: str,
    variant: ObjectiveVariant,
    output_root: Path,
    time_limit_sec: float,
    force: bool,
) -> Path:
    case = STUDY_CASES[target]
    linear_root = output_root / "linear_span" / variant.name
    manifest = optimization_dir(linear_root, target, case.max_action_weight, variant.objective) / "candidate_factors_manifest.csv"
    if manifest.exists() and not force:
        return manifest
    cmd = [
        sys.executable,
        "scripts/optimize_linear_span_candidate.py",
        "--target",
        target,
        "--action-dictionary",
        "low-weight",
        "--max-action-weight",
        str(case.max_action_weight),
        "--objective",
        variant.objective,
        "--mixed-weight-scale",
        str(case.mixed_weight_scale if variant.objective in SPLIT_AWARE_OBJECTIVES else 1.0),
        "--support-weight-scale",
        "0.25",
        "--pair-weight-scale",
        str(case.pair_weight_scale if variant.objective in SPLIT_AWARE_OBJECTIVES else 0.0),
        "--candidate-kind",
        variant.candidate_kind,
        "--output-root",
        str(linear_root),
        "--time-limit-sec",
        str(time_limit_sec),
    ]
    if variant.use_pair_cap and case.max_pair_overlap is not None:
        cmd.extend(["--max-pair-overlap", str(case.max_pair_overlap)])
    if variant.use_factor_count_cap:
        cmd.extend(["--max-factors", str(factor_count_cap(output_root, target))])
    print(f"+ optimize {target} {variant.name}", flush=True)
    wall_timeout = max(float(time_limit_sec) + 120.0, float(time_limit_sec) * 1.2)
    started = time.monotonic()
    completed = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        check=False,
        timeout=wall_timeout,
    )
    elapsed = time.monotonic() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"Optimization failed for {target} {variant.name} "
            f"with return code {completed.returncode} after {elapsed:.1f}s."
        )
    print(f"+ optimized {target} {variant.name} in {elapsed:.1f}s", flush=True)
    return manifest


def run_materialization(
    *,
    target: str,
    variant: ObjectiveVariant,
    manifest: Path,
    output_root: Path,
    force: bool,
) -> tuple[Path, float]:
    case = STUDY_CASES[target]
    destination = output_root / target / variant.name / "shared-parity"
    summary_path = destination / "summary.json"
    if summary_path.exists() and not force:
        return summary_path, 0.0
    destination.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "scripts/materialize_shared_parity_candidate.py",
        "--target",
        target,
        "--manifest-csv",
        str(manifest),
        "--candidate-kind",
        variant.candidate_kind,
        "--factor-order",
        case.factor_order,
        "--target-strategy",
        case.target_strategy,
        "--synthesis",
        "shared-parity",
        "--output-root",
        str(destination),
    ]
    print(f"+ materialize {target} {variant.name}", flush=True)
    started = time.monotonic()
    completed = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        check=False,
    )
    elapsed = time.monotonic() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"Materialization failed for {target} {variant.name} "
            f"with return code {completed.returncode} after {elapsed:.1f}s."
        )
    print(f"+ materialized {target} {variant.name} in {elapsed:.1f}s", flush=True)
    return summary_path, elapsed


def optimization_summary_from_manifest(manifest: Path) -> dict[str, Any]:
    summary_path = manifest.parent / "summary.json"
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def elapsed_sum(*values: Any) -> float | str:
    total = 0.0
    seen = False
    for value in values:
        parsed = coerce_float(value)
        if parsed is None:
            continue
        total += parsed
        seen = True
    return total if seen else ""


def factor_metrics_from_manifest(manifest: Path, *, target: str, variant: ObjectiveVariant) -> dict[str, Any]:
    row = load_manifest_row(manifest, target=target, candidate_kind=variant.candidate_kind)
    raw_factors = np.load(resolve_project_path(row["factor_path"]))
    change_of_basis_path = row.get("change_of_basis_path") or ""
    change_of_basis = (
        np.load(resolve_project_path(change_of_basis_path))
        if change_of_basis_path
        else None
    )
    factors = canonicalize_factors(raw_factors, change_of_basis).astype(np.uint8)
    metrics = factor_concentration_metrics(factors)
    prefixed = {f"factor_{key}": value for key, value in metrics.items()}
    prefixed["factor_count"] = metrics["factor_count"]
    return prefixed


def collect_rows(
    targets: list[str],
    output_root: Path,
    time_limit_sec: float,
    force: bool,
    *,
    objective_variants: tuple[ObjectiveVariant, ...] = OBJECTIVE_VARIANTS,
    continue_on_error: bool = False,
    checkpoint_csv: Path | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for target in targets:
        for variant in objective_variants:
            try:
                manifest = run_optimization(
                    target=target,
                    variant=variant,
                    output_root=output_root,
                    time_limit_sec=time_limit_sec,
                    force=force,
                )
                optimization_summary = optimization_summary_from_manifest(manifest)
                summary_path, materialization_elapsed_sec = run_materialization(
                    target=target,
                    variant=variant,
                    manifest=manifest,
                    output_root=output_root,
                    force=force,
                )
                materialized = read_summary(summary_path)
                rows.append(
                    {
                        **materialized,
                        **factor_metrics_from_manifest(manifest, target=target, variant=variant),
                        "objective_variant": variant.name,
                        "span_objective": variant.objective,
                        "pair_cap_enabled": variant.use_pair_cap,
                        "factor_count_cap_enabled": variant.use_factor_count_cap,
                        "optimization_elapsed_sec": optimization_summary.get("elapsed_sec", ""),
                        "materialization_elapsed_sec": materialization_elapsed_sec,
                        "objective_elapsed_sec": elapsed_sum(
                            optimization_summary.get("elapsed_sec", ""),
                            materialization_elapsed_sec,
                        ),
                        "execution_status": "ok",
                        "error_message": "",
                    }
                )
            except Exception as exc:
                if not continue_on_error:
                    raise
                rows.append(error_row(target, variant, exc))
            if checkpoint_csv is not None:
                write_csv(checkpoint_csv, rows)
    return rows


def error_row(target: str, variant: ObjectiveVariant, exc: Exception) -> dict[str, Any]:
    message = " ".join(str(exc).split())
    return {
        "target": target,
        "objective_variant": variant.name,
        "span_objective": variant.objective,
        "pair_cap_enabled": variant.use_pair_cap,
        "factor_count_cap_enabled": variant.use_factor_count_cap,
        "candidate_kind": variant.candidate_kind,
        "optimization_elapsed_sec": "",
        "materialization_elapsed_sec": "",
        "objective_elapsed_sec": "",
        "structural_target_status": "not-materialized",
        "summary_path": "",
        "execution_status": "failed",
        "error_message": message[:500],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "target",
        "objective_variant",
        "span_objective",
        "pair_cap_enabled",
        "factor_count_cap_enabled",
        "candidate_kind",
        "factor_count",
        "factor_unique_parity_count",
        "factor_parity_reuse_ratio",
        "factor_parity_concentration_index",
        "factor_qubit_concentration_index",
        "factor_support_weight_mean",
        "factor_pairwise_support_overlap_mean",
        "factor_pairwise_jaccard_mean",
        "tcount",
        "tdepth",
        "qasm_depth",
        "primary_nc_depth_ratio",
        "qasm_depth_ratio",
        "structural_cost",
        "optimization_elapsed_sec",
        "materialization_elapsed_sec",
        "objective_elapsed_sec",
        "structural_target_status",
        "summary_path",
        "execution_status",
        "error_message",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, lineterminator="\n", fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def pairwise_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for target in ordered_targets(rows):
        target_rows = {
            row["objective_variant"]: row
            for row in rows
            if row["target"] == target and row_ok(row)
        }
        factor = target_rows.get("factor_count")
        matched = target_rows.get("factor_count_pair_cap")
        mixed = target_rows.get("mixed_pair")
        frontier = target_rows.get("frontier_pair")
        if factor is None:
            continue
        if mixed is not None:
            summaries.append(compare_rows(target, "mixed_pair_vs_factor_count", mixed, factor))
            if matched is not None:
                summaries.append(compare_rows(target, "mixed_pair_vs_factor_count_pair_cap", mixed, matched))
        if frontier is not None:
            summaries.append(compare_rows(target, "frontier_pair_vs_factor_count", frontier, factor))
            if matched is not None:
                summaries.append(compare_rows(target, "frontier_pair_vs_factor_count_pair_cap", frontier, matched))
    return summaries


def row_ok(row: dict[str, Any]) -> bool:
    return row.get("execution_status", "ok") == "ok"


def compare_rows(target: str, comparison: str, numerator: dict[str, Any], denominator: dict[str, Any]) -> dict[str, Any]:
    return {
        "target": target,
        "comparison": comparison,
        "factor_count_ratio": safe_ratio(numerator.get("factor_count"), denominator.get("factor_count")),
        "parity_reuse_delta": safe_delta(
            numerator.get("factor_parity_reuse_ratio"),
            denominator.get("factor_parity_reuse_ratio"),
        ),
        "parity_ci_ratio": safe_ratio(
            numerator.get("factor_parity_concentration_index"),
            denominator.get("factor_parity_concentration_index"),
        ),
        "pairwise_overlap_ratio": safe_ratio(
            numerator.get("factor_pairwise_support_overlap_mean"),
            denominator.get("factor_pairwise_support_overlap_mean"),
        ),
        "primary_ratio": safe_ratio(
            numerator.get("primary_nc_depth_ratio"),
            denominator.get("primary_nc_depth_ratio"),
        ),
        "qasm_depth_ratio": safe_ratio(
            numerator.get("qasm_depth_ratio"),
            denominator.get("qasm_depth_ratio"),
        ),
        "tdepth_ratio": safe_ratio(numerator.get("tdepth"), denominator.get("tdepth")),
        "mixed_tcount": coerce_float(numerator.get("tcount")),
        "baseline_tcount": coerce_float(denominator.get("tcount")),
    }


def safe_ratio(numerator: Any, denominator: Any) -> float | None:
    numerator_f = coerce_float(numerator)
    denominator_f = coerce_float(denominator)
    if numerator_f is None or denominator_f is None or denominator_f <= 0:
        return None
    return numerator_f / denominator_f


def safe_delta(numerator: Any, denominator: Any) -> float | None:
    numerator_f = coerce_float(numerator)
    denominator_f = coerce_float(denominator)
    if numerator_f is None or denominator_f is None:
        return None
    return numerator_f - denominator_f


def write_report(path: Path, rows: list[dict[str, Any]], csv_path: Path, figure_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summaries = pairwise_summary(rows)
    failures = [row for row in rows if not row_ok(row)]
    mixed_matched = [row for row in summaries if row["comparison"] == "mixed_pair_vs_factor_count_pair_cap"]
    frontier_matched = [row for row in summaries if row["comparison"] == "frontier_pair_vs_factor_count_pair_cap"]
    primary_wins = [row for row in frontier_matched if is_win(row.get("primary_ratio"))]
    qasm_wins = [row for row in frontier_matched if is_win(row.get("qasm_depth_ratio"))]
    tcount_wins_or_ties = [
        row for row in frontier_matched
        if row.get("mixed_tcount") is not None
        and row.get("baseline_tcount") is not None
        and float(row["mixed_tcount"]) <= float(row["baseline_tcount"])
    ]
    lines = [
        "# Decomposition objective ablation",
        "",
        f"CSV: `{csv_path}`.",
        f"Figure: `{figure_path}`.",
        "",
        "This ablation changes the tensor decomposition objective while keeping the "
        "shared-parity materializer, factor order, and target strategy fixed per target.",
        "",
        f"Completed objective rows: {sum(row_ok(row) for row in rows)}/{len(rows)}.",
        f"Failed objective rows: {len(failures)}/{len(rows)}.",
        "",
        "`frontier_pair` is the article-inspired AlphaQ-only proxy: it penalizes "
        "cross-partition support pairs and off-target tensor mass while rewarding "
        "target coverage. `mixed_pair` is retained as the previous proxy.",
        "",
        f"- `frontier_pair` has T-count <= matched factor-count in {len(tcount_wins_or_ties)}/{len(frontier_matched)} targets.",
        f"- `frontier_pair` improves primary NC ratio in {len(primary_wins)}/{len(frontier_matched)} matched comparisons.",
        f"- `frontier_pair` improves QASM depth ratio in {len(qasm_wins)}/{len(frontier_matched)} matched comparisons.",
        f"- `mixed_pair` remains available for direct comparison in {len(mixed_matched)} matched rows.",
        "",
        "| target | comparison | factor count ratio | parity reuse delta | pairwise overlap ratio | parity CI ratio | primary ratio | QASM ratio | T-depth ratio |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {target} | {comparison} | {factor_count} | {reuse_delta} | {overlap} | {parity_ci} | {primary} | {qasm} | {tdepth} |".format(
                target=row["target"],
                comparison=row["comparison"],
                factor_count=fmt(row.get("factor_count_ratio")),
                reuse_delta=fmt(row.get("parity_reuse_delta")),
                overlap=fmt(row.get("pairwise_overlap_ratio")),
                parity_ci=fmt(row.get("parity_ci_ratio")),
                primary=fmt(row.get("primary_ratio")),
                qasm=fmt(row.get("qasm_depth_ratio")),
                tdepth=fmt(row.get("tdepth_ratio")),
            )
        )
    if failures:
        lines.extend(
            [
                "",
                "## Failed objective rows",
                "",
                "These rows are explicit optimization/materialization failures. They are not replaced by fallback candidates.",
                "",
                "| target | objective | status | error |",
                "|---|---|---|---|",
            ]
        )
        for row in failures:
            lines.append(
                "| {target} | {objective} | {status} | {error} |".format(
                    target=row["target"],
                    objective=row["objective_variant"],
                    status=row.get("execution_status", "failed"),
                    error=str(row.get("error_message", "")).replace("|", "/"),
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figure(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    targets = ordered_targets(rows)
    variants = [variant.name for variant in OBJECTIVE_VARIANTS]
    x = list(range(len(targets)))
    width = min(0.8 / max(len(variants), 1), 0.24)
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0), constrained_layout=True)
    fields = [
        ("factor_pairwise_support_overlap_mean", "Mean pairwise support overlap"),
        ("primary_nc_depth_ratio", "primary NC depth ratio"),
        ("qasm_depth_ratio", "QASM depth ratio"),
    ]
    colors = {
        "factor_count": "#7f7f7f",
        "factor_count_pair_cap": "#4c78a8",
        "mixed_pair": "#1b9e77",
        "frontier_pair": "#d95f02",
        "depth_guarded_mixed_pair": "#984ea3",
        "t_preserving_frontier_pair": "#e7298a",
    }
    by_target_variant = {
        (row["target"], row["objective_variant"]): row
        for row in rows
    }
    for ax, (field, title) in zip(axes, fields):
        for offset, variant in enumerate(variants):
            values = [
                coerce_float(by_target_variant.get((target, variant), {}).get(field)) or 0.0
                for target in targets
            ]
            center_offset = offset - (len(variants) - 1) / 2
            positions = [item + center_offset * width for item in x]
            ax.bar(positions, values, width=width, label=variant, color=colors.get(variant, "#999999"))
        if "ratio" in field:
            ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1.0)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([short_label(target) for target in targets], rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("Tensor objective ablation with fixed shared-parity materializer", fontsize=12)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def is_win(value: Any) -> bool:
    numeric = coerce_float(value)
    return numeric is not None and numeric < 1.0


def ordered_targets(rows: list[dict[str, Any]]) -> list[str]:
    seen = {str(row["target"]) for row in rows}
    ordered = [target for target in CORE_TARGETS if target in seen]
    ordered.extend(sorted(seen - set(ordered)))
    return ordered


def short_label(target: str) -> str:
    return target.replace("hamming_weight_", "hw ").replace("gf_2pow", "gf2^").replace("_mult", " mult")


def fmt(value: Any) -> str:
    numeric = coerce_float(value)
    return "" if numeric is None else f"{numeric:.3g}"


def main() -> int:
    args = parse_args()
    rows = collect_rows(
        parse_targets(args.targets),
        args.output_root,
        args.time_limit_sec,
        args.force,
        objective_variants=parse_objective_variants(args.objective_variants),
        continue_on_error=args.continue_on_error,
        checkpoint_csv=args.output_csv,
    )
    write_csv(args.output_csv, rows)
    write_figure(args.figure_path, rows)
    write_report(args.report_path, rows, args.output_csv, args.figure_path)
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.report_path}")
    print(f"Wrote {args.figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
