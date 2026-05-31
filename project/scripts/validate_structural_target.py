from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._analysis_common import DEFAULT_CSV_ROOT
from scripts._analysis_common import DEFAULT_FIGURES_ROOT
from scripts._analysis_common import DEFAULT_REPORTS_ROOT
from scripts._analysis_common import ensure_dir
from scripts._analysis_common import natural_sort_key
from scripts._analysis_common import write_csv_rows
from scripts.structural_target import PRIMARY_RATIO_KEY
from scripts.structural_target import STRUCTURAL_STATUS_KEY
from scripts.structural_target import coerce_float


VALIDATION_COLUMNS = (
    "circuit_id",
    "method",
    "method_label",
    "tcount_after",
    "delta_t",
    "depth_after",
    "zx_total_depth",
    "zx_best_nonclifford_depth",
    "zx_best_clifford_fraction",
    PRIMARY_RATIO_KEY,
    "primary_nc_depth_delta_vs_original",
    "zx_total_depth_ratio",
    "qasm_depth_ratio",
    "tcount_improves_primary_worsens",
    "clifford_fraction_misleading",
    "zx_depth_inflation",
    "qasm_depth_inflation",
    STRUCTURAL_STATUS_KEY,
    "structural_target_error",
)

METHOD_LABELS = {
    "original": "Original",
    "pyzx": "PyZX",
    "alphatensor_public": "AlphaTensor-public",
}

METHOD_COLORS = {
    "original": "#4D4D4D",
    "pyzx": "#0072B2",
    "alphatensor_public": "#009E73",
}


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def validation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {column: row.get(column) for column in VALIDATION_COLUMNS}
        for row in sorted(
            rows,
            key=lambda row: (
                natural_sort_key(row.get("circuit_id", "")),
                ("original", "pyzx", "alphatensor_public").index(
                    row.get("method", "alphatensor_public")
                    if row.get("method") in METHOD_LABELS
                    else "alphatensor_public"
                ),
            ),
        )
    ]


def save_structural_target_plot(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=(8.2, 5.4), constrained_layout=True)
    for method in ("original", "pyzx", "alphatensor_public"):
        points = [
            row
            for row in rows
            if row.get("method") == method
            and row.get(STRUCTURAL_STATUS_KEY) == "ok"
            and coerce_float(row.get(PRIMARY_RATIO_KEY)) is not None
            and coerce_float(row.get("tcount_after")) is not None
        ]
        if not points:
            continue
        ax.scatter(
            [coerce_float(row.get(PRIMARY_RATIO_KEY)) for row in points],
            [coerce_float(row.get("tcount_after")) for row in points],
            s=62,
            alpha=0.86,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
        for row in points:
            ax.annotate(
                row["circuit_id"],
                (
                    coerce_float(row.get(PRIMARY_RATIO_KEY)) or 0.0,
                    coerce_float(row.get("tcount_after")) or 0.0,
                ),
                fontsize=7,
                alpha=0.72,
                xytext=(4, 3),
                textcoords="offset points",
            )

    ax.set_title("T-count vs primary ZX splitting target")
    ax.set_xlabel("primary_nc_depth_ratio")
    ax.set_ylabel("T-count")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    png_path = output_dir / "structural_target_vs_tcount.png"
    fig.savefig(png_path, dpi=300)
    fig.savefig(output_dir / "structural_target_vs_tcount.pdf")
    plt.close(fig)
    return png_path


def write_validation_report(
    rows: list[dict[str, Any]],
    output_path: Path,
    validation_csv_path: Path,
    figure_path: Path,
) -> Path:
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            natural_sort_key(row.get("circuit_id", "")),
            row.get("method", ""),
        ),
    )
    non_ok_rows = [
        row
        for row in sorted_rows
        if row.get(STRUCTURAL_STATUS_KEY) != "ok"
    ]
    tcount_conflicts = [
        row
        for row in sorted_rows
        if _truthy(row.get("tcount_improves_primary_worsens"))
    ]
    clifford_fraction_conflicts = [
        row
        for row in sorted_rows
        if _truthy(row.get("clifford_fraction_misleading"))
    ]
    depth_inflation = [
        row
        for row in sorted_rows
        if _truthy(row.get("zx_depth_inflation"))
        or _truthy(row.get("qasm_depth_inflation"))
    ]

    text = [
        "# Structural Target Validation",
        "",
        "## Artefacts",
        "",
        f"- Validation CSV: `{validation_csv_path}`.",
        f"- Figure: `{figure_path}`.",
        "",
        "## Status",
        "",
        _status_line(rows, non_ok_rows),
        "",
        "## T-count improves but primary target worsens",
        "",
        *_case_lines(
            tcount_conflicts,
            "No retained row improved T-count while worsening primary_nc_depth_ratio.",
        ),
        "",
        "## Clifford fraction can mislead",
        "",
        *_case_lines(
            clifford_fraction_conflicts,
            "No retained row increased zx_best_clifford_fraction while worsening primary_nc_depth_ratio.",
        ),
        "",
        "## Depth inflation diagnostics",
        "",
        *_case_lines(
            depth_inflation,
            "No retained row crossed the 1.25x depth-inflation diagnostic threshold.",
        ),
        "",
        "## Interpretation",
        "",
        (
            "`primary_nc_depth_ratio` is the primary target for this sprint. "
            "`zx_best_clifford_fraction`, `heuristic_splitting_score`, `T-span`, "
            "`T-qubits`, and `area_T` remain interpretive or historical diagnostics."
        ),
        "",
    ]
    ensure_dir(output_path.parent)
    output_path.write_text("\n".join(text), encoding="utf-8")
    return output_path


def _status_line(rows: list[dict[str, Any]], non_ok_rows: list[dict[str, Any]]) -> str:
    circuit_ids = sorted({row.get("circuit_id", "") for row in rows}, key=natural_sort_key)
    if not non_ok_rows:
        return (
            f"- All retained rows have `{STRUCTURAL_STATUS_KEY}=ok` "
            f"across {len(circuit_ids)} benchmarks: "
            f"{', '.join(f'`{circuit_id}`' for circuit_id in circuit_ids)}."
        )
    return (
        f"- {len(non_ok_rows)} retained rows do not have "
        f"`{STRUCTURAL_STATUS_KEY}=ok`."
    )


def _case_lines(rows: list[dict[str, Any]], empty_message: str) -> list[str]:
    if not rows:
        return [f"- {empty_message}"]
    return [_format_case_line(row) for row in rows]


def _format_case_line(row: dict[str, Any]) -> str:
    return (
        f"- `{row.get('circuit_id')}` / {row.get('method_label') or row.get('method')}: "
        f"T-count={_fmt(row.get('tcount_after'))}, "
        f"primary={_fmt(row.get(PRIMARY_RATIO_KEY))}, "
        f"primary_delta={_fmt(row.get('primary_nc_depth_delta_vs_original'))}, "
        f"ZX-depth-ratio={_fmt(row.get('zx_total_depth_ratio'))}, "
        f"QASM-depth-ratio={_fmt(row.get('qasm_depth_ratio'))}, "
        f"Clifford-fraction={_fmt(row.get('zx_best_clifford_fraction'))}."
    )


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes"}


def _fmt(value: Any) -> str:
    numeric = coerce_float(value)
    if numeric is None:
        return "NA"
    if abs(numeric - round(numeric)) < 1e-9:
        return str(int(round(numeric)))
    return f"{numeric:.3f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the primary ZX structural target for Entrega 1."
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=DEFAULT_CSV_ROOT / "entrega1_metrics_formally_verified.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_CSV_ROOT / "structural_target_validation.csv",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORTS_ROOT / "structural_target_validation.md",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=DEFAULT_FIGURES_ROOT / "entrega1_formal",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_csv_rows(args.metrics_csv)
    output_rows = validation_rows(rows)
    write_csv_rows(output_rows, args.output_csv)
    figure_path = save_structural_target_plot(rows, args.figure_dir)
    report_path = write_validation_report(
        rows,
        args.report_path,
        args.output_csv,
        figure_path,
    )
    print(
        {
            "validation_csv": str(args.output_csv),
            "report_path": str(report_path),
            "figure_path": str(figure_path),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
