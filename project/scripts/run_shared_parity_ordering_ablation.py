from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from statistics import median
from typing import Any

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_materialized_candidates import read_summary
from scripts.run_shared_parity_study import CORE_TARGETS
from scripts.run_shared_parity_study import STUDY_CASES
from scripts.structural_target import coerce_float


ORDER_STRATEGIES = (
    "given",
    "reverse",
    "lex",
    "support-ascending",
    "support-descending",
    "greedy-cnot",
)
TARGET_STRATEGIES = (
    "min-change",
    "max-change",
    "min-row-weight",
    "max-row-weight",
)
DEFAULT_SEARCH_ROOTS = (
    PROJECT_ROOT / "results" / "alphaq_shared_parity_study",
    PROJECT_ROOT / "results" / "alphaq_shared_parity_study_expansion_pilot",
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "alphaq_shared_parity_ordering_ablation"
DEFAULT_CSV = PROJECT_ROOT / "results" / "csv" / "alphaq_shared_parity_ordering_ablation.csv"
DEFAULT_REPORT = PROJECT_ROOT / "results" / "reports" / "alphaq_shared_parity_ordering_ablation.md"
DEFAULT_FIGURE = PROJECT_ROOT / "results" / "figures" / "alphaq_shared_parity_ordering_ablation.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize existing shared-parity factors across ordering strategies."
    )
    parser.add_argument("--targets", default=",".join(CORE_TARGETS))
    parser.add_argument("--orders", default=",".join(ORDER_STRATEGIES))
    parser.add_argument("--target-strategies", default=",".join(TARGET_STRATEGIES))
    parser.add_argument(
        "--random-orders",
        type=int,
        default=0,
        help="Append N deterministic random factor orders named random-seed-N.",
    )
    parser.add_argument(
        "--random-seed-start",
        type=int,
        default=0,
        help="First seed used when --random-orders is positive.",
    )
    parser.add_argument(
        "--search-root",
        action="append",
        type=Path,
        default=[],
        help="Study roots containing linear_span/*/candidate_factors_manifest.csv.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--figure-path", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def expand_orders(orders: list[str], *, random_orders: int, random_seed_start: int) -> list[str]:
    if random_orders < 0:
        raise ValueError("--random-orders must be non-negative.")
    random_labels = [
        f"random-seed-{seed}"
        for seed in range(random_seed_start, random_seed_start + random_orders)
    ]
    seen: set[str] = set()
    expanded: list[str] = []
    for order in [*orders, *random_labels]:
        if order not in seen:
            expanded.append(order)
            seen.add(order)
    return expanded


def materialization_grid(
    *,
    targets: list[str],
    orders: list[str],
    target_strategies: list[str],
) -> list[tuple[str, str, str]]:
    return [
        (target, order, target_strategy)
        for target in targets
        for order in orders
        for target_strategy in target_strategies
    ]


def find_manifest(target: str, candidate_kind: str, search_roots: list[Path]) -> Path:
    for root in search_roots:
        if not root.exists():
            continue
        for manifest in sorted((root / "linear_span").glob("**/candidate_factors_manifest.csv")):
            with manifest.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    if row.get("target") == target and row.get("candidate_kind") == candidate_kind:
                        return manifest
    raise FileNotFoundError(
        f"Could not find manifest for target={target!r}, candidate_kind={candidate_kind!r}."
    )


def output_dir(output_root: Path, target: str, order: str, target_strategy: str) -> Path:
    return output_root / target / f"{order}_{target_strategy}"


def run_materialization(
    *,
    target: str,
    manifest: Path,
    candidate_kind: str,
    order: str,
    target_strategy: str,
    output_root: Path,
    force: bool,
) -> Path:
    destination = output_dir(output_root, target, order, target_strategy)
    summary_path = destination / "summary.json"
    if summary_path.exists() and not force:
        return summary_path
    destination.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "scripts/materialize_shared_parity_candidate.py",
        "--target",
        target,
        "--manifest-csv",
        str(manifest),
        "--candidate-kind",
        candidate_kind,
        "--factor-order",
        order,
        "--target-strategy",
        target_strategy,
        "--output-root",
        str(destination),
    ]
    print(f"+ materialize {target} {order}/{target_strategy}", flush=True)
    completed = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr, file=sys.stderr)
        completed.check_returncode()
    return summary_path


def collect_row(summary_path: Path, *, checkpoint_order: str, checkpoint_target_strategy: str) -> dict[str, Any]:
    row = read_summary(summary_path)
    row["checkpoint_selected"] = (
        row.get("factor_order") == checkpoint_order
        and row.get("target_strategy") == checkpoint_target_strategy
    )
    row["order_class"] = "random" if str(row.get("factor_order", "")).startswith("random-seed-") else "named"
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "target",
        "candidate_kind",
        "factor_order",
        "target_strategy",
        "order_class",
        "checkpoint_selected",
        "tcount",
        "tdepth",
        "qasm_depth",
        "tcount_ratio",
        "primary_nc_depth_ratio",
        "qasm_depth_ratio",
        "structural_cost",
        "structural_target_status",
        "candidate_dir",
        "summary_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize_target_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for target in ordered_targets(rows):
        target_rows = [row for row in rows if row.get("target") == target]
        random_rows = [row for row in target_rows if row.get("order_class") == "random"]
        qasm_values = finite_values(target_rows, "qasm_depth_ratio")
        primary_values = finite_values(target_rows, "primary_nc_depth_ratio")
        random_qasm_values = finite_values(random_rows, "qasm_depth_ratio")
        random_primary_values = finite_values(random_rows, "primary_nc_depth_ratio")
        selected = next((row for row in target_rows if row.get("checkpoint_selected")), None)
        best_qasm = min(target_rows, key=lambda row: finite_or_inf(row.get("qasm_depth_ratio")))
        best_primary = min(target_rows, key=lambda row: finite_or_inf(row.get("primary_nc_depth_ratio")))
        selected_qasm = None if selected is None else coerce_float(selected.get("qasm_depth_ratio"))
        selected_primary = None if selected is None else coerce_float(selected.get("primary_nc_depth_ratio"))
        summaries.append(
            {
                "target": target,
                "n_variants": len(target_rows),
                "n_random_variants": len(random_rows),
                "selected_order": "" if selected is None else selected.get("factor_order", ""),
                "selected_target_strategy": "" if selected is None else selected.get("target_strategy", ""),
                "selected_qasm_depth_ratio": selected_qasm,
                "median_qasm_depth_ratio": None if not qasm_values else median(qasm_values),
                "qasm_iqr_low": percentile(qasm_values, 25),
                "qasm_iqr_high": percentile(qasm_values, 75),
                "best_qasm_depth_ratio": coerce_float(best_qasm.get("qasm_depth_ratio")),
                "best_qasm_order": best_qasm.get("factor_order", ""),
                "best_qasm_target_strategy": best_qasm.get("target_strategy", ""),
                "selected_primary_nc_depth_ratio": selected_primary,
                "median_primary_nc_depth_ratio": None if not primary_values else median(primary_values),
                "primary_iqr_low": percentile(primary_values, 25),
                "primary_iqr_high": percentile(primary_values, 75),
                "best_primary_nc_depth_ratio": coerce_float(best_primary.get("primary_nc_depth_ratio")),
                "best_primary_order": best_primary.get("factor_order", ""),
                "best_primary_target_strategy": best_primary.get("target_strategy", ""),
                "selected_qasm_rank": rank_value(target_rows, selected, "qasm_depth_ratio"),
                "selected_primary_rank": rank_value(target_rows, selected, "primary_nc_depth_ratio"),
                "random_median_qasm_depth_ratio": None if not random_qasm_values else median(random_qasm_values),
                "random_best_qasm_depth_ratio": None if not random_qasm_values else min(random_qasm_values),
                "selected_beats_random_qasm_fraction": beats_fraction(
                    selected_qasm,
                    random_qasm_values,
                ),
                "random_median_primary_nc_depth_ratio": None if not random_primary_values else median(random_primary_values),
                "random_best_primary_nc_depth_ratio": None if not random_primary_values else min(random_primary_values),
                "selected_beats_random_primary_fraction": beats_fraction(
                    selected_primary,
                    random_primary_values,
                ),
            }
        )
    return summaries


def finite_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [
        numeric
        for row in rows
        if (numeric := coerce_float(row.get(key))) is not None
    ]


def finite_or_inf(value: Any) -> float:
    numeric = coerce_float(value)
    return float("inf") if numeric is None else numeric


def percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percent / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def beats_fraction(selected_value: float | None, values: list[float]) -> float | None:
    if selected_value is None or not values:
        return None
    return sum(1 for value in values if selected_value <= value) / len(values)


def rank_value(rows: list[dict[str, Any]], selected: dict[str, Any] | None, key: str) -> int | None:
    if selected is None:
        return None
    selected_value = coerce_float(selected.get(key))
    if selected_value is None:
        return None
    sorted_values = sorted(finite_values(rows, key))
    return sorted_values.index(selected_value) + 1


def write_report(path: Path, rows: list[dict[str, Any]], csv_path: Path, figure_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summaries = summarize_target_rows(rows)
    lines = [
        "# Shared-parity ordering ablation",
        "",
        f"CSV: `{csv_path}`.",
        f"Figure: `{figure_path}`.",
        "",
        "This ablation keeps the tensor factors fixed and varies only the shared-parity "
        "materialization order and target-selection strategy.",
        "",
        "| target | variants | random variants | selected QASM | median QASM | QASM IQR | best QASM | selected primary | median primary | primary IQR | best primary | selected QASM rank | selected primary rank |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            "| {target} | {n_variants} | {n_random_variants} | {selected_qasm} | {median_qasm} | {qasm_iqr} | {best_qasm} | {selected_primary} | {median_primary} | {primary_iqr} | {best_primary} | {selected_qasm_rank} | {selected_primary_rank} |".format(
                target=summary["target"],
                n_variants=summary["n_variants"],
                n_random_variants=summary["n_random_variants"],
                selected_qasm=fmt(summary.get("selected_qasm_depth_ratio")),
                median_qasm=fmt(summary.get("median_qasm_depth_ratio")),
                qasm_iqr=f"{fmt(summary.get('qasm_iqr_low'))}-{fmt(summary.get('qasm_iqr_high'))}",
                best_qasm=fmt(summary.get("best_qasm_depth_ratio")),
                selected_primary=fmt(summary.get("selected_primary_nc_depth_ratio")),
                median_primary=fmt(summary.get("median_primary_nc_depth_ratio")),
                primary_iqr=f"{fmt(summary.get('primary_iqr_low'))}-{fmt(summary.get('primary_iqr_high'))}",
                best_primary=fmt(summary.get("best_primary_nc_depth_ratio")),
                selected_qasm_rank=summary.get("selected_qasm_rank") or "",
                selected_primary_rank=summary.get("selected_primary_rank") or "",
            )
        )
    random_summaries = [summary for summary in summaries if summary["n_random_variants"]]
    if random_summaries:
        lines.extend(
            [
                "",
                "## Random-order controls",
                "",
                "The fractions below are the share of random-order variants that the selected checkpoint matches or beats. Higher is better because both metrics are minimized.",
                "",
                "| target | random median QASM | selected beats random QASM | random median primary | selected beats random primary |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for summary in random_summaries:
            lines.append(
                "| {target} | {random_qasm} | {qasm_fraction} | {random_primary} | {primary_fraction} |".format(
                    target=summary["target"],
                    random_qasm=fmt(summary.get("random_median_qasm_depth_ratio")),
                    qasm_fraction=fmt_fraction(summary.get("selected_beats_random_qasm_fraction")),
                    random_primary=fmt(summary.get("random_median_primary_nc_depth_ratio")),
                    primary_fraction=fmt_fraction(summary.get("selected_beats_random_primary_fraction")),
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figure(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    targets = ordered_targets(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), constrained_layout=True)
    for ax, key, title in [
        (axes[0], "primary_nc_depth_ratio", "primary NC depth ratio"),
        (axes[1], "qasm_depth_ratio", "QASM depth ratio"),
    ]:
        data = [finite_values([row for row in rows if row.get("target") == target], key) for target in targets]
        ax.boxplot(data, tick_labels=[short_label(target) for target in targets], showfliers=True)
        for index, target in enumerate(targets, start=1):
            selected = next(
                (
                    row
                    for row in rows
                    if row.get("target") == target and row.get("checkpoint_selected")
                ),
                None,
            )
            value = None if selected is None else coerce_float(selected.get(key))
            if value is not None:
                ax.scatter([index], [value], color="#d95f02", s=55, zorder=3, label="checkpoint" if index == 1 else None)
        ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1.0)
        ax.set_title(title)
        ax.set_xticklabels([short_label(target) for target in targets], rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("Ordering/target-strategy sensitivity with fixed tensor factors", fontsize=12)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def short_label(target: str) -> str:
    return target.replace("hamming_weight_", "hw ").replace("gf_2pow", "gf2^").replace("_mult", " mult")


def ordered_targets(rows: list[dict[str, Any]]) -> list[str]:
    seen = {str(row["target"]) for row in rows}
    ordered = [target for target in CORE_TARGETS if target in seen]
    ordered.extend(sorted(seen - set(ordered)))
    return ordered


def fmt(value: Any) -> str:
    numeric = coerce_float(value)
    return "" if numeric is None else f"{numeric:.3g}"


def fmt_fraction(value: Any) -> str:
    numeric = coerce_float(value)
    return "" if numeric is None else f"{100 * numeric:.1f}%"


def main() -> int:
    args = parse_args()
    targets = parse_list(args.targets)
    orders = expand_orders(
        parse_list(args.orders),
        random_orders=args.random_orders,
        random_seed_start=args.random_seed_start,
    )
    target_strategies = parse_list(args.target_strategies)
    search_roots = [PROJECT_ROOT / root for root in args.search_root] if args.search_root else list(DEFAULT_SEARCH_ROOTS)
    rows: list[dict[str, Any]] = []
    manifest_cache: dict[str, Path] = {}
    for target, order, target_strategy in materialization_grid(
        targets=targets,
        orders=orders,
        target_strategies=target_strategies,
    ):
        case = STUDY_CASES[target]
        manifest = manifest_cache.get(target)
        if manifest is None:
            manifest = find_manifest(target, case.candidate_kind, search_roots)
            manifest_cache[target] = manifest
        summary_path = run_materialization(
            target=target,
            manifest=manifest,
            candidate_kind=case.candidate_kind,
            order=order,
            target_strategy=target_strategy,
            output_root=args.output_root,
            force=args.force,
        )
        rows.append(
            collect_row(
                summary_path,
                checkpoint_order=case.factor_order,
                checkpoint_target_strategy=case.target_strategy,
            )
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
