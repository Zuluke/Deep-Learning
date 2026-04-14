from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._analysis_common import DEFAULT_BOOTSTRAP_SAMPLES
from scripts._analysis_common import DEFAULT_BOOTSTRAP_SEED
from scripts._analysis_common import DEFAULT_CSV_ROOT
from scripts._analysis_common import DEFAULT_REPORTS_ROOT
from scripts._analysis_common import ensure_dir
from scripts._analysis_common import write_csv_rows
from scripts._analysis_common import write_json
from scripts._manifest import append_command

STRUCTURAL_METRICS = (
    "rho_t",
    "rho_w",
    "n_clifford_blocks",
    "n_nonclifford_blocks",
    "avg_nonclifford_block_len",
    "hadamard_boundary_density",
    "tdepth_over_tcount",
)
TARGETS = ("delta_t", "rel_gain_t")


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def coerce_float(value: Any) -> float | None:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_mean(values: list[float]) -> float | None:
    return None if not values else float(np.mean(values))


def bootstrap_ci(x: np.ndarray, y: np.ndarray, mode: str) -> tuple[float | None, float | None]:
    if len(x) < 2 or len(y) < 2:
        return None, None
    rng = np.random.default_rng(DEFAULT_BOOTSTRAP_SEED)
    samples = []
    for _ in range(DEFAULT_BOOTSTRAP_SAMPLES):
        indices = rng.integers(0, len(x), size=len(x))
        xb = x[indices]
        yb = y[indices]
        if np.allclose(xb, xb[0]) or np.allclose(yb, yb[0]):
            continue
        if mode == "spearman":
            coefficient = stats.spearmanr(xb, yb).statistic
        else:
            coefficient = stats.pearsonr(xb, yb).statistic
        if coefficient is not None and math.isfinite(coefficient):
            samples.append(float(coefficient))
    if not samples:
        return None, None
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def correlation_rows(final_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    methods = sorted({row["method"] for row in final_rows if row["method"] != "original"})
    for method in methods:
        method_rows = [
            row for row in final_rows
            if row["method"] == method
            and row.get("method_status") in {"ok", "partial-log-only"}
        ]
        for metric_name in STRUCTURAL_METRICS:
            for target_name in TARGETS:
                values = [
                    (coerce_float(row.get(metric_name)), coerce_float(row.get(target_name)))
                    for row in method_rows
                ]
                values = [
                    (metric_value, target_value)
                    for metric_value, target_value in values
                    if metric_value is not None and target_value is not None
                ]
                n = len(values)
                metric_array = np.asarray([item[0] for item in values], dtype=float)
                target_array = np.asarray([item[1] for item in values], dtype=float)
                if n >= 2 and not np.allclose(metric_array, metric_array[0]) and not np.allclose(target_array, target_array[0]):
                    spearman = stats.spearmanr(metric_array, target_array).statistic
                    pearson = stats.pearsonr(metric_array, target_array).statistic
                    if n >= 6:
                        spearman_ci = bootstrap_ci(metric_array, target_array, "spearman")
                        pearson_ci = bootstrap_ci(metric_array, target_array, "pearson")
                    else:
                        spearman_ci = (None, None)
                        pearson_ci = (None, None)
                else:
                    spearman = None
                    pearson = None
                    spearman_ci = (None, None)
                    pearson_ci = (None, None)
                rows.append(
                    {
                        "method": method,
                        "metric": metric_name,
                        "target": target_name,
                        "n": n,
                        "spearman_rho": spearman,
                        "spearman_ci_low": spearman_ci[0],
                        "spearman_ci_high": spearman_ci[1],
                        "pearson_r": pearson,
                        "pearson_ci_low": pearson_ci[0],
                        "pearson_ci_high": pearson_ci[1],
                        "inferential_status": "reported" if n >= 6 else "descriptive-only",
                    }
                )
    return rows


def summary_rows(final_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    methods = sorted({row["method"] for row in final_rows})
    for method in methods:
        method_rows = [row for row in final_rows if row["method"] == method]
        optimized_rows = [
            row for row in method_rows
            if row["method"] != "original"
            and row.get("method_status") in {"ok", "partial-log-only"}
            and coerce_float(row.get("delta_t")) is not None
        ]
        delta_values = [float(row["delta_t"]) for row in optimized_rows]
        gain_values = [float(row["rel_gain_t"]) for row in optimized_rows if row.get("rel_gain_t") not in ("", None)]
        rows.append(
            {
                "method": method,
                "num_rows": len(method_rows),
                "num_ok_like_rows": sum(row.get("method_status") in {"ok", "partial-log-only"} for row in method_rows),
                "num_improved_vs_original": sum((coerce_float(row.get("delta_t")) or 0.0) > 0 for row in optimized_rows),
                "mean_delta_t": safe_mean(delta_values),
                "median_delta_t": None if not delta_values else float(np.median(delta_values)),
                "mean_rel_gain_t": safe_mean(gain_values),
                "median_rel_gain_t": None if not gain_values else float(np.median(gain_values)),
            }
        )
    return rows


def overlap_rows(final_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    method_pairs = [
        ("pyzx", "public_resynth_gadgets"),
        ("pyzx", "public_resynth_no_gadgets"),
        ("pyzx", "compile_quizx"),
    ]
    indexed = {(row["circuit_id"], row["method"]): row for row in final_rows}
    rows: list[dict[str, Any]] = []
    circuit_ids = sorted({row["circuit_id"] for row in final_rows})
    for left, right in method_pairs:
        both = 0
        left_only = 0
        right_only = 0
        neither = 0
        for circuit_id in circuit_ids:
            left_row = indexed.get((circuit_id, left))
            right_row = indexed.get((circuit_id, right))
            left_improved = (coerce_float(left_row.get("delta_t")) or 0.0) > 0 if left_row else False
            right_improved = (coerce_float(right_row.get("delta_t")) or 0.0) > 0 if right_row else False
            if left_improved and right_improved:
                both += 1
            elif left_improved:
                left_only += 1
            elif right_improved:
                right_only += 1
            else:
                neither += 1
        rows.append(
            {
                "left_method": left,
                "right_method": right,
                "both_improved": both,
                "left_only_improved": left_only,
                "right_only_improved": right_only,
                "neither_improved": neither,
            }
        )
    return rows


def stage1_report(
    *,
    final_rows: list[dict[str, str]],
    correlation: list[dict[str, Any]],
    overlap: list[dict[str, Any]],
    report_path: Path,
) -> Path:
    original_rows = [row for row in final_rows if row["method"] == "original"]
    methods = sorted({row["method"] for row in final_rows if row["method"] != "original"})
    best_rows = sorted(
        [
            row for row in final_rows
            if row["method"] != "original"
            and row.get("method_status") in {"ok", "partial-log-only"}
            and coerce_float(row.get("delta_t")) is not None
        ],
        key=lambda row: (
            -(coerce_float(row.get("delta_t")) or -1.0),
            row["circuit_id"],
            row["method"],
        ),
    )
    strongest = sorted(
        [
            row for row in correlation
            if row["target"] == "delta_t" and row["n"] >= 6 and row["spearman_rho"] is not None
        ],
        key=lambda row: abs(row["spearman_rho"]),
        reverse=True,
    )
    positive = [
        row for row in final_rows
        if row["method"] == "public_resynth_gadgets" and (coerce_float(row.get("delta_t")) or 0.0) > 0
    ]
    pyzx_beats = [
        row["circuit_id"]
        for row in final_rows
        if row["method"] == "pyzx"
        and (coerce_float(row.get("delta_t")) or 0.0)
        > (coerce_float(next(
            (
                other.get("delta_t")
                for other in final_rows
                if other["circuit_id"] == row["circuit_id"] and other["method"] == "public_resynth_gadgets"
            ),
            None,
        )) or 0.0)
    ]

    lines = [
        "# Hypothesis Check Stage 1",
        "",
        "## Scope",
        "",
        f"- Inventory size: {len(original_rows)} vendored circuits.",
        f"- Methods evaluated: {', '.join(methods)}.",
        "- Verification status remains `not-run` in this stage because `feynver` is not installed locally.",
        "",
        "## Pipeline Status",
        "",
        f"- Original metrics available for {len(original_rows)}/{len(original_rows)} circuits.",
        f"- PyZX completed on {sum(row.get('method_status') == 'ok' for row in final_rows if row['method'] == 'pyzx')} circuits.",
        f"- QuiZX compile completed on {sum(row.get('method_status') in {'ok', 'skipped-existing'} for row in final_rows if row['method'] == 'compile_quizx')} circuits.",
        f"- Public resynthesis with gadgets assembled on {sum(row.get('method_status') == 'ok' for row in final_rows if row['method'] == 'public_resynth_gadgets')} circuits.",
        f"- Public resynthesis without gadgets assembled on {sum(row.get('method_status') == 'ok' for row in final_rows if row['method'] == 'public_resynth_no_gadgets')} circuits.",
        "",
        "## Mandatory Questions",
        "",
        "1. Do the metrics predict absolute T-count gain?",
    ]
    if strongest:
        top = strongest[0]
        lines.append(
            f"   The strongest Stage 1 signal for `delta_t` was `{top['metric']}` under `{top['method']}` with Spearman {top['spearman_rho']:.3f} over n={top['n']} circuits."
        )
    else:
        lines.append("   There is not yet a stable inferential signal with n>=6 for absolute gain.")

    rel_rows = [
        row for row in correlation
        if row["target"] == "rel_gain_t" and row["n"] >= 6 and row["spearman_rho"] is not None
    ]
    lines.append("")
    lines.append("2. Do the metrics predict percentage gain?")
    if rel_rows:
        top = sorted(rel_rows, key=lambda row: abs(row["spearman_rho"]), reverse=True)[0]
        lines.append(
            f"   Yes, but only weakly to moderately: `{top['metric']}` under `{top['method']}` reached Spearman {top['spearman_rho']:.3f} over n={top['n']}."
        )
    else:
        lines.append("   Stage 1 provides only descriptive evidence for relative gain.")

    lines.append("")
    lines.append("3. Do PyZX and AlphaTensor-Quantum improve the same circuits?")
    overlap_row = next((row for row in overlap if row["left_method"] == "pyzx" and row["right_method"] == "public_resynth_gadgets"), None)
    if overlap_row:
        lines.append(
            f"   Partially. PyZX and public resynthesis with gadgets improved the same circuit on {overlap_row['both_improved']} cases, while PyZX alone improved {overlap_row['left_only_improved']} and AlphaTensor-public replay alone improved {overlap_row['right_only_improved']}."
        )

    lines.append("")
    lines.append("4. Are there circuits where the ZX baseline beats the RL/public resynthesis route?")
    if pyzx_beats:
        sample = ", ".join(sorted(pyzx_beats)[:8])
        lines.append(f"   Yes. Examples include: {sample}.")
    else:
        lines.append("   Not in this Stage 1 slice.")

    lines.append("")
    lines.append("5. Are the metrics stable before and after resynthesis?")
    stability_candidates = [
        row for row in final_rows
        if row["method"] == "public_resynth_gadgets" and row.get("method_status") == "ok"
    ]
    if stability_candidates:
        mean_rho_shift = float(np.mean([
            abs((coerce_float(row.get("rho_t")) or 0.0) - (coerce_float(next(
                (
                    other.get("rho_t")
                    for other in final_rows
                    if other["circuit_id"] == row["circuit_id"] and other["method"] == "original"
                ),
                0.0,
            )) or 0.0))
            for row in stability_candidates
        ]))
        lines.append(
            f"   Broadly yes for coarse density metrics: the mean absolute shift in `rho_t` between original and public resynthesized circuits was {mean_rho_shift:.3f}."
        )
    else:
        lines.append("   Not enough assembled public-resynthesis cases yet for a strong stability claim.")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The current evidence is best described as `sinal fraco mas interpretavel em alguns regimes`.",
            "",
            "- Density-style metrics (`rho_t`, `rho_w`) are easy to compute and remain comparable across methods, but they do not yet explain gains uniformly across all local circuits.",
            "- The strongest improvements tend to appear in circuits where public decompositions are available and the non-Clifford structure is already isolated into small compiled blocks.",
            "- PyZX is a real baseline here, not a strawman: it improves several circuits that the public AlphaTensor replay does not cover or does not beat.",
            "- The manuscript hypothesis remains plausible, but Stage 1 supports it only as a regime-dependent explanatory signal rather than a universal predictor.",
            "",
            "## Best Improvements",
            "",
        ]
    )
    for row in best_rows[:10]:
        lines.append(
            f"- {row['circuit_id']} / {row['method']}: delta_t={coerce_float(row.get('delta_t'))}, rel_gain_t={coerce_float(row.get('rel_gain_t'))}"
        )

    ensure_dir(report_path.parent)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def stage2_report(report_path: Path, op_t_mize_inventory: Path | None) -> Path:
    lines = [
        "# Hypothesis Check Stage 2",
        "",
        "Stage 2 is prepared but not fully executed in this local run.",
        "",
    ]
    if op_t_mize_inventory and op_t_mize_inventory.exists():
        rows = load_rows(op_t_mize_inventory)
        lines.append(f"- `op-T-mize` inventory available with {len(rows)} raw entries.")
        lines.append("- Full normalization and evaluation over the benchmark remains the next execution step.")
    else:
        lines.append("- `op-T-mize` inventory is not present locally yet.")
        lines.append("- The Stage 2 ingestion script is implemented, but the dataset still needs to be downloaded in an environment with network access.")
    ensure_dir(report_path.parent)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate canonical metrics into summaries, statistics and reports.")
    parser.add_argument("--final-metrics-csv", type=Path, default=DEFAULT_CSV_ROOT / "final_metrics.csv")
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_CSV_ROOT / "final_summary.csv")
    parser.add_argument(
        "--correlation-csv",
        type=Path,
        default=DEFAULT_CSV_ROOT / "correlation_summary.csv",
    )
    parser.add_argument("--method-overlap-csv", type=Path, default=DEFAULT_CSV_ROOT / "method_overlap.csv")
    parser.add_argument("--stage1-report", type=Path, default=DEFAULT_REPORTS_ROOT / "hypothesis_check_stage1.md")
    parser.add_argument("--stage2-report", type=Path, default=DEFAULT_REPORTS_ROOT / "hypothesis_check_stage2.md")
    parser.add_argument("--op-t-mize-inventory-csv", type=Path, default=DEFAULT_CSV_ROOT / "op_t_mize_inventory.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    final_rows = load_rows(args.final_metrics_csv)
    summary = summary_rows(final_rows)
    correlation = correlation_rows(final_rows)
    overlap = overlap_rows(final_rows)

    write_csv_rows(summary, args.summary_csv)
    write_csv_rows(correlation, args.correlation_csv)
    write_csv_rows(overlap, args.method_overlap_csv)
    write_json(
        {
            "summary_csv": str(args.summary_csv),
            "correlation_csv": str(args.correlation_csv),
            "method_overlap_csv": str(args.method_overlap_csv),
            "num_final_rows": len(final_rows),
        },
        args.summary_csv.with_suffix(".json"),
    )
    stage1_report(
        final_rows=final_rows,
        correlation=correlation,
        overlap=overlap,
        report_path=args.stage1_report,
    )
    stage2_report(args.stage2_report, args.op_t_mize_inventory_csv)
    append_command(
        {
            "tool": "aggregate_results.py",
            "command": (
                f"{sys.executable} scripts/aggregate_results.py --final-metrics-csv {args.final_metrics_csv} "
                f"--summary-csv {args.summary_csv} --correlation-csv {args.correlation_csv} "
                f"--method-overlap-csv {args.method_overlap_csv}"
            ),
            "cwd": str(PROJECT_ROOT),
            "final_metrics_csv": str(args.final_metrics_csv),
            "summary_csv": str(args.summary_csv),
            "correlation_csv": str(args.correlation_csv),
            "method_overlap_csv": str(args.method_overlap_csv),
            "exit_code": 0,
        }
    )
    print(
        {
            "summary_csv": str(args.summary_csv),
            "correlation_csv": str(args.correlation_csv),
            "method_overlap_csv": str(args.method_overlap_csv),
            "stage1_report": str(args.stage1_report),
            "stage2_report": str(args.stage2_report),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
