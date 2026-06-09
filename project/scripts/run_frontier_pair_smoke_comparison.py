from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_decomposition_objective_ablation import collect_rows
from scripts.run_decomposition_objective_ablation import parse_objective_variants
from scripts.run_decomposition_objective_ablation import parse_targets
from scripts.structural_target import coerce_float
from scripts.zx_splitting import compute_paper_zx_splitting_metrics_from_qasm


DEFAULT_TARGETS = "mod_5_4,gf_2pow2_mult"
DEFAULT_OBJECTIVES = "factor_count,mixed_pair,frontier_pair"
DEFAULT_OUTPUT_ROOT = Path("/tmp/alphaq_frontier_pair_smoke")
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "results" / "csv" / "alphaq_frontier_pair_smoke.csv"
DEFAULT_REPORT = PROJECT_ROOT / "results" / "reports" / "alphaq_frontier_pair_smoke.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a small AlphaQ-only frontier-pair smoke comparison and audit it "
            "with the paper-style ZX border detector."
        )
    )
    parser.add_argument("--targets", default=DEFAULT_TARGETS)
    parser.add_argument("--objective-variants", default=DEFAULT_OBJECTIVES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--time-limit-sec", type=float, default=90.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def comparison_rows(
    *,
    targets: list[str],
    objective_variants_text: str,
    output_root: Path,
    time_limit_sec: float,
    force: bool,
) -> list[dict[str, Any]]:
    rows = collect_rows(
        targets,
        output_root,
        time_limit_sec,
        force,
        objective_variants=parse_objective_variants(objective_variants_text),
        continue_on_error=True,
    )
    enriched = [enrich_row(row) for row in rows]
    baselines = {
        row["target"]: row
        for row in enriched
        if row.get("objective_variant") == "factor_count"
    }
    for row in enriched:
        baseline = baselines.get(row["target"])
        row["paper_primary_ratio_vs_factor_count"] = safe_ratio(
            row.get("paper_primary_nc_depth_ratio"),
            None if baseline is None else baseline.get("paper_primary_nc_depth_ratio"),
        )
        row["paper_nc_depth_ratio_vs_factor_count"] = safe_ratio(
            row.get("paper_zx_best_nonclifford_depth"),
            None if baseline is None else baseline.get("paper_zx_best_nonclifford_depth"),
        )
    return enriched


def enrich_row(row: dict[str, Any]) -> dict[str, Any]:
    if row.get("execution_status", "ok") != "ok":
        return {
            **row,
            "paper_zx_split_status": "not-materialized",
            "paper_structural_target_status": "not-materialized",
        }
    summary_path = Path(str(row.get("summary_path", "")))
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            **row,
            "paper_zx_split_status": "failed",
            "paper_zx_split_error": f"summary: {exc}",
            "paper_structural_target_status": "missing-summary",
        }

    qasm_path = Path(str(summary.get("assembled_qasm") or ""))
    benchmark_dir = Path(str(summary.get("benchmark_dir") or ""))
    original_qasm = benchmark_dir / f"{row['target']}.qasm"
    candidate_metrics = compute_paper_zx_splitting_metrics_from_qasm(qasm_path)
    original_metrics = (
        compute_paper_zx_splitting_metrics_from_qasm(original_qasm)
        if original_qasm.exists()
        else {"paper_zx_split_status": "missing-original"}
    )
    numerator = coerce_float(candidate_metrics.get("paper_zx_best_nonclifford_depth"))
    denominator = coerce_float(original_metrics.get("paper_zx_total_depth"))
    if candidate_metrics.get("paper_zx_split_status") != "ok":
        status = "missing-zx"
        paper_primary = None
    elif original_metrics.get("paper_zx_split_status") != "ok":
        status = "missing-original"
        paper_primary = None
    elif numerator is None or denominator is None or denominator <= 0:
        status = "invalid-depth"
        paper_primary = None
    else:
        status = "ok"
        paper_primary = numerator / max(denominator, 1.0)

    return {
        **row,
        "candidate_qasm_path": str(qasm_path),
        "original_qasm_path": str(original_qasm) if original_qasm.exists() else "",
        "paper_zx_split_status": candidate_metrics.get("paper_zx_split_status", ""),
        "paper_zx_total_depth": candidate_metrics.get("paper_zx_total_depth", ""),
        "paper_zx_best_nonclifford_depth": candidate_metrics.get(
            "paper_zx_best_nonclifford_depth", ""
        ),
        "paper_zx_best_clifford_fraction": candidate_metrics.get(
            "paper_zx_best_clifford_fraction", ""
        ),
        "paper_primary_nc_depth_ratio": paper_primary,
        "paper_structural_target_status": status,
        "paper_zx_split_error": candidate_metrics.get("paper_zx_split_error", ""),
    }


def safe_ratio(numerator: Any, denominator: Any) -> float | None:
    numerator_f = coerce_float(numerator)
    denominator_f = coerce_float(denominator)
    if numerator_f is None or denominator_f is None or denominator_f <= 0:
        return None
    return numerator_f / denominator_f


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "target",
        "objective_variant",
        "execution_status",
        "factor_count",
        "tcount",
        "tdepth",
        "qasm_depth",
        "qasm_depth_ratio",
        "primary_nc_depth_ratio",
        "paper_zx_split_status",
        "paper_zx_total_depth",
        "paper_zx_best_nonclifford_depth",
        "paper_zx_best_clifford_fraction",
        "paper_primary_nc_depth_ratio",
        "paper_primary_ratio_vs_factor_count",
        "paper_nc_depth_ratio_vs_factor_count",
        "paper_structural_target_status",
        "summary_path",
        "candidate_qasm_path",
        "original_qasm_path",
        "error_message",
        "paper_zx_split_error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict[str, Any]], csv_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# AlphaQ Frontier-Pair Smoke Comparison",
        "",
        f"CSV: `{display_path(csv_path)}`.",
        "",
        "`frontier_pair` is an AlphaQ-only objective. ZX is used here only as an external paper-style audit.",
        "",
        "| target | objective | T-count | QASM ratio | legacy primary | paper primary | paper vs factor_count | paper NC-depth vs factor_count |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {target} | {objective_variant} | {tcount} | {qasm} | {legacy} | {paper} | {paper_vs_factor} | {paper_nc_vs_factor} |".format(
                target=row.get("target", ""),
                objective_variant=row.get("objective_variant", ""),
                tcount=fmt(row.get("tcount")),
                qasm=fmt(row.get("qasm_depth_ratio")),
                legacy=fmt(row.get("primary_nc_depth_ratio")),
                paper=fmt(row.get("paper_primary_nc_depth_ratio")),
                paper_vs_factor=fmt(row.get("paper_primary_ratio_vs_factor_count")),
                paper_nc_vs_factor=fmt(row.get("paper_nc_depth_ratio_vs_factor_count")),
            )
        )

    lines.extend(["", "## Reading", ""])
    for target in sorted({str(row.get("target", "")) for row in rows}):
        items = [row for row in rows if row.get("target") == target]
        frontier = next((row for row in items if row.get("objective_variant") == "frontier_pair"), None)
        mixed = next((row for row in items if row.get("objective_variant") == "mixed_pair"), None)
        factor = next((row for row in items if row.get("objective_variant") == "factor_count"), None)
        if factor is None or frontier is None:
            continue
        lines.append(
            "- `{target}`: `frontier_pair` paper-primary={frontier_primary}, factor_count paper-primary={factor_primary}, mixed_pair paper-primary={mixed_primary}.".format(
                target=target,
                frontier_primary=fmt(frontier.get("paper_primary_nc_depth_ratio")),
                factor_primary=fmt(factor.get("paper_primary_nc_depth_ratio")),
                mixed_primary="n/a" if mixed is None else fmt(mixed.get("paper_primary_nc_depth_ratio")),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def fmt(value: Any) -> str:
    numeric = coerce_float(value)
    return "" if numeric is None else f"{numeric:.4g}"


def main() -> int:
    args = parse_args()
    rows = comparison_rows(
        targets=parse_targets(args.targets),
        objective_variants_text=args.objective_variants,
        output_root=args.output_root,
        time_limit_sec=args.time_limit_sec,
        force=args.force,
    )
    write_csv(args.output_csv, rows)
    write_report(args.report_path, rows, args.output_csv)
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
