from __future__ import annotations

import argparse
import csv
import re
import shutil
from collections import Counter
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._analysis_common import DEFAULT_CSV_ROOT
from scripts._analysis_common import DEFAULT_FIGURES_ROOT
from scripts._analysis_common import DEFAULT_RESULTS_ROOT
from scripts._analysis_common import ensure_dir


PAPER_IMG_DIR = REPO_ROOT / "paper" / "imgs"
REPRO_FIGURE_DIR = DEFAULT_RESULTS_ROOT / "reproducibility" / "paper" / "figures"
TENSOR_V3_FIGURE_DIR = DEFAULT_FIGURES_ROOT / "entrega1_tensor_v3_phase_slack_formal"

SOURCE_LABELS = {
    "paper_table_benchmark_no_gadgets": "Benchmarks\n(no gadgets)",
    "paper_table_benchmark_gadgets": "Benchmarks\n(with gadgets)",
    "paper_fig4_gf_no_gadgets": "GF(2^m)\n(no gadgets)",
    "paper_fig4_gf_gadgets": "GF(2^m)\n(with gadgets)",
    "paper_fig4_binary_addition": "Binary\naddition",
}

SOURCE_COLORS = {
    "paper_table_benchmark_no_gadgets": "#D55E00",
    "paper_table_benchmark_gadgets": "#0072B2",
    "paper_fig4_gf_no_gadgets": "#E69F00",
    "paper_fig4_gf_gadgets": "#009E73",
    "paper_fig4_binary_addition": "#CC79A7",
}

TENSOR_V3_CIRCUITS = (
    "gf_2pow2_mult",
    "hamming_weight_n4",
    "hamming_weight_n5",
    "mod_5_4",
    "qft_4",
)

TENSOR_V3_METHODS = (
    ("alphatensor_public", "AlphaTensor-public", "#009E73"),
    ("alphaq_tensor_v3_phase_slack", "AlphaQ tensor-v3", "#D55E00"),
)


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
        }
    )


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def coerce_float(value: Any) -> float:
    if value in (None, "", "None"):
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def copy_to_paper(output_path: Path, paper_filename: str) -> None:
    ensure_dir(PAPER_IMG_DIR)
    for suffix in (".pdf", ".png"):
        source = output_path.with_suffix(suffix)
        if source.exists():
            shutil.copyfile(source, PAPER_IMG_DIR / f"{paper_filename}{suffix}")


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, *, paper_copy: bool = False) -> Path:
    ensure_dir(output_dir)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    if paper_copy:
        copy_to_paper(pdf_path, stem)
    return png_path


def export_baseline_comparison(comparison_csv: Path) -> Path:
    rows = load_csv_rows(comparison_csv)
    counts = Counter(row["source"] for row in rows)
    matches = Counter(row["source"] for row in rows if row.get("match") == "true")
    ordered_sources = tuple(SOURCE_LABELS)

    fig, (ax_cov, ax_scatter) = plt.subplots(
        1,
        2,
        figsize=(11.8, 4.6),
        gridspec_kw={"width_ratios": [1.05, 1.0], "wspace": 0.34},
        constrained_layout=True,
    )

    y = np.arange(len(ordered_sources))
    totals = [counts[source] for source in ordered_sources]
    matched = [matches[source] for source in ordered_sources]
    ax_cov.barh(y, totals, color="#E6E6E6", edgecolor="#666666", linewidth=0.6, label="Compared")
    ax_cov.barh(
        y,
        matched,
        color=[SOURCE_COLORS[source] for source in ordered_sources],
        label="Exact match",
    )
    for y_pos, match_count, total_count in zip(y, matched, totals, strict=True):
        ax_cov.text(match_count + 0.35, y_pos, f"{match_count}/{total_count}", va="center", fontsize=7)
    ax_cov.set_yticks(y, [SOURCE_LABELS[source] for source in ordered_sources])
    ax_cov.invert_yaxis()
    ax_cov.set_xlabel("reproduced rows")
    ax_cov.set_title("A. Reproduction coverage", loc="left")
    ax_cov.legend(frameon=False, loc="lower right")
    ax_cov.spines["top"].set_visible(False)
    ax_cov.spines["right"].set_visible(False)

    for source in ordered_sources:
        subset = [row for row in rows if row["source"] == source]
        ax_scatter.scatter(
            [coerce_float(row["paper_effective_tcount"]) for row in subset],
            [coerce_float(row["reproduced_effective_tcount"]) for row in subset],
            s=18,
            color=SOURCE_COLORS[source],
            label=SOURCE_LABELS[source].replace("\n", " "),
            alpha=0.88,
            edgecolor="white",
            linewidth=0.25,
        )
    ax_scatter.plot([2, 1000], [2, 1000], color="#444444", linestyle="--", linewidth=0.8)
    ax_scatter.set_xscale("log")
    ax_scatter.set_yscale("log")
    ax_scatter.set_xlim(2, 1000)
    ax_scatter.set_ylim(2, 1000)
    ax_scatter.set_xlabel("paper effective T-count")
    ax_scatter.set_ylabel("reproduced effective T-count")
    ax_scatter.set_title("B. Value-by-value agreement", loc="left")
    ax_scatter.legend(frameon=False, loc="lower right", fontsize=6)
    ax_scatter.spines["top"].set_visible(False)
    ax_scatter.spines["right"].set_visible(False)

    return save_figure(fig, REPRO_FIGURE_DIR, "paper_benchmark_comparison", paper_copy=True)


def gf_exponent(circuit_id: str) -> int:
    match = re.search(r"gf_2pow(\d+)_mult", circuit_id)
    if not match:
        raise ValueError(f"Could not parse GF exponent from {circuit_id!r}")
    return int(match.group(1))


def export_gf_multiplication(aggregate_csv: Path) -> Path:
    rows = [
        row
        for row in load_csv_rows(aggregate_csv)
        if row.get("family_label") == "finite_field_multiplication"
    ]
    by_method: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_method.setdefault(row["method"], []).append(row)

    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    for method, label, color, marker in [
        ("paper_no_gadgets", "AT-Q without gadgets", "#D55E00", "o"),
        ("paper_gadgets", "AT-Q with gadgets", "#009E73", "s"),
    ]:
        subset = sorted(by_method.get(method, []), key=lambda row: gf_exponent(row["circuit_id"]))
        ax.plot(
            [gf_exponent(row["circuit_id"]) for row in subset],
            [coerce_float(row["best_effective_tcount"]) for row in subset],
            marker=marker,
            markersize=4,
            linewidth=1.2,
            color=color,
            label=label,
        )
    ax.set_title("Finite-field multiplication")
    ax.set_xlabel("m in GF(2^m)")
    ax.set_ylabel("reproduced effective T-count")
    ax.legend(frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return save_figure(fig, REPRO_FIGURE_DIR, "gf_multiplication_effective_tcount", paper_copy=True)


def indexed_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(row["circuit_id"], row["method"]): row for row in rows}


def row_value(
    rows_by_key: dict[tuple[str, str], dict[str, str]],
    circuit_id: str,
    method: str,
    field: str,
) -> float:
    return coerce_float(rows_by_key.get((circuit_id, method), {}).get(field))


def export_tensor_v3_bar(
    rows: list[dict[str, str]],
    *,
    field: str,
    title: str,
    ylabel: str,
    stem: str,
    value_format: str,
) -> Path:
    rows_by_key = indexed_rows(rows)
    x = np.arange(len(TENSOR_V3_CIRCUITS))
    width = 0.34
    finite_values = [
        value
        for circuit_id in TENSOR_V3_CIRCUITS
        for method, _, _ in TENSOR_V3_METHODS
        if np.isfinite(value := row_value(rows_by_key, circuit_id, method, field))
    ]
    y_max = max(finite_values) if finite_values else 1.0

    fig, ax = plt.subplots(figsize=(9.4, 4.4), constrained_layout=True)
    for offset, (method, label, color) in enumerate(TENSOR_V3_METHODS):
        values = [row_value(rows_by_key, circuit_id, method, field) for circuit_id in TENSOR_V3_CIRCUITS]
        positions = x + (offset - (len(TENSOR_V3_METHODS) - 1) / 2) * width
        bars = ax.bar(positions, values, width, label=label, color=color)
        bar_labels = []
        for value in values:
            if np.isnan(value):
                bar_labels.append("")
            elif value_format == "int":
                bar_labels.append(f"{value:.0f}")
            else:
                bar_labels.append(f"{value:.3f}")
        ax.bar_label(bars, labels=bar_labels, fontsize=7, padding=2)

    ax.set_title(title, pad=12)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x, TENSOR_V3_CIRCUITS, rotation=22, ha="right")
    ax.set_ylim(0, y_max * (1.28 if field == "primary_nc_depth_ratio" else 1.14))
    ax.legend(frameon=False, ncols=len(TENSOR_V3_METHODS), loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if field == "primary_nc_depth_ratio":
        ax.text(
            0.99,
            0.92,
            "lower is better",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            color="#555555",
        )
    return save_figure(fig, TENSOR_V3_FIGURE_DIR, stem)


def export_tensor_v3_figures(metrics_csv: Path) -> tuple[Path, Path]:
    rows = load_csv_rows(metrics_csv)
    tcount_path = export_tensor_v3_bar(
        rows,
        field="tcount_after",
        title="T-count: public baseline vs tensor-v3 selection",
        ylabel="T-count",
        stem="tcount_comparison",
        value_format="int",
    )
    primary_path = export_tensor_v3_bar(
        rows,
        field="primary_nc_depth_ratio",
        title="Primary non-Clifford-depth ratio",
        ylabel=r"$r_{\mathrm{NC}}$",
        stem="zx_splitting_comparison",
        value_format="float",
    )
    return tcount_path, primary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the figures used by the paper draft.")
    parser.add_argument(
        "--paper-comparison-csv",
        type=Path,
        default=DEFAULT_RESULTS_ROOT / "reproducibility" / "paper" / "paper_benchmark_comparison.csv",
    )
    parser.add_argument(
        "--paper-aggregate-csv",
        type=Path,
        default=DEFAULT_RESULTS_ROOT / "reproducibility" / "paper" / "paper_aggregate_best.csv",
    )
    parser.add_argument(
        "--formal-metrics-csv",
        type=Path,
        default=DEFAULT_CSV_ROOT / "entrega1_metrics_formally_verified.csv",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_matplotlib()
    baseline_path = export_baseline_comparison(args.paper_comparison_csv)
    gf_path = export_gf_multiplication(args.paper_aggregate_csv)
    tcount_path, primary_path = export_tensor_v3_figures(args.formal_metrics_csv)
    print(
        {
            "baseline": str(baseline_path),
            "gf_multiplication": str(gf_path),
            "tensor_v3_tcount": str(tcount_path),
            "tensor_v3_primary_nc": str(primary_path),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
