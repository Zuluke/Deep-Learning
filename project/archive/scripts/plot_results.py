from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._analysis_common import DEFAULT_CSV_ROOT
from scripts._analysis_common import DEFAULT_FIGURES_ROOT
from scripts._analysis_common import ensure_dir
from scripts._manifest import append_command

METRICS = (
    "rho_t",
    "rho_w",
    "n_clifford_blocks",
    "n_nonclifford_blocks",
    "avg_nonclifford_block_len",
    "hadamard_boundary_density",
    "tdepth_over_tcount",
)


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


def scatter_grid(final_rows: list[dict[str, str]], output_root: Path) -> None:
    methods = ["pyzx", "compile_quizx", "public_resynth_gadgets"]
    filtered = [
        row for row in final_rows
        if row["method"] in methods and row.get("method_status") in {"ok", "partial-log-only"}
    ]
    fig, axes = plt.subplots(4, 2, figsize=(10, 12), constrained_layout=True)
    colors = {"pyzx": "#0072B2", "compile_quizx": "#009E73", "public_resynth_gadgets": "#D55E00"}
    targets = ["delta_t", "rel_gain_t"]
    for index, metric in enumerate(METRICS):
        ax = axes[index // 2, index % 2]
        for method in methods:
            method_rows = [row for row in filtered if row["method"] == method]
            x = [coerce_float(row.get(metric)) for row in method_rows]
            y = [coerce_float(row.get("delta_t")) for row in method_rows]
            pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
            if not pairs:
                continue
            ax.scatter(
                [pair[0] for pair in pairs],
                [pair[1] for pair in pairs],
                s=28,
                alpha=0.8,
                label=method.replace("_", " "),
                color=colors[method],
            )
        ax.set_title(metric)
        ax.set_xlabel(metric)
        ax.set_ylabel(targets[0])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.savefig(output_root / "metric_scatter_delta_t.png", dpi=300)
    fig.savefig(output_root / "metric_scatter_delta_t.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(4, 2, figsize=(10, 12), constrained_layout=True)
    for index, metric in enumerate(METRICS):
        ax = axes[index // 2, index % 2]
        for method in methods:
            method_rows = [row for row in filtered if row["method"] == method]
            x = [coerce_float(row.get(metric)) for row in method_rows]
            y = [coerce_float(row.get("rel_gain_t")) for row in method_rows]
            pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
            if not pairs:
                continue
            ax.scatter(
                [pair[0] for pair in pairs],
                [pair[1] for pair in pairs],
                s=28,
                alpha=0.8,
                label=method.replace("_", " "),
                color=colors[method],
            )
        ax.set_title(metric)
        ax.set_xlabel(metric)
        ax.set_ylabel(targets[1])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.savefig(output_root / "metric_scatter_rel_gain_t.png", dpi=300)
    fig.savefig(output_root / "metric_scatter_rel_gain_t.pdf")
    plt.close(fig)


def heatmap(correlation_rows: list[dict[str, str]], output_root: Path) -> None:
    method_order = ["pyzx", "compile_quizx", "public_resynth_gadgets", "public_resynth_no_gadgets"]
    target = "delta_t"
    metric_order = list(METRICS)
    matrix = np.full((len(method_order), len(metric_order)), np.nan, dtype=float)
    for i, method in enumerate(method_order):
        for j, metric in enumerate(metric_order):
            row = next(
                (
                    item for item in correlation_rows
                    if item["method"] == method and item["metric"] == metric and item["target"] == target
                ),
                None,
            )
            if row is None:
                continue
            value = coerce_float(row.get("spearman_rho"))
            if value is not None:
                matrix[i, j] = value
    fig, ax = plt.subplots(figsize=(9, 3.5), constrained_layout=True)
    image = ax.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(metric_order)), metric_order, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(method_order)), [name.replace("_", " ") for name in method_order])
    for i in range(len(method_order)):
        for j in range(len(metric_order)):
            value = matrix[i, j]
            if np.isnan(value):
                continue
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="black", fontsize=8)
    ax.set_title("Spearman correlations vs delta_t")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(output_root / "correlation_heatmap.png", dpi=300)
    fig.savefig(output_root / "correlation_heatmap.pdf")
    plt.close(fig)


def overlap_plot(overlap_rows: list[dict[str, str]], output_root: Path) -> None:
    labels = [f"{row['left_method']} vs {row['right_method']}" for row in overlap_rows]
    both = [int(row["both_improved"]) for row in overlap_rows]
    left_only = [int(row["left_only_improved"]) for row in overlap_rows]
    right_only = [int(row["right_only_improved"]) for row in overlap_rows]
    neither = [int(row["neither_improved"]) for row in overlap_rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    ax.bar(x, both, label="Both", color="#009E73")
    ax.bar(x, left_only, bottom=both, label="Left only", color="#0072B2")
    ax.bar(x, right_only, bottom=np.asarray(both) + np.asarray(left_only), label="Right only", color="#D55E00")
    ax.bar(
        x,
        neither,
        bottom=np.asarray(both) + np.asarray(left_only) + np.asarray(right_only),
        label="Neither",
        color="#999999",
    )
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("Number of circuits")
    ax.set_title("Method overlap on T-count improvement")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(output_root / "method_overlap.png", dpi=300)
    fig.savefig(output_root / "method_overlap.pdf")
    plt.close(fig)


def stability_plot(final_rows: list[dict[str, str]], output_root: Path) -> None:
    original = {row["circuit_id"]: row for row in final_rows if row["method"] == "original"}
    public_rows = [
        row for row in final_rows
        if row["method"] == "public_resynth_gadgets" and row.get("method_status") == "ok"
    ]
    if not public_rows:
        return
    x = []
    y = []
    labels = []
    for row in public_rows:
        base = original.get(row["circuit_id"])
        if base is None:
            continue
        rho_before = coerce_float(base.get("rho_t"))
        rho_after = coerce_float(row.get("rho_t"))
        if rho_before is None or rho_after is None:
            continue
        x.append(rho_before)
        y.append(rho_after)
        labels.append(row["circuit_id"])
    if not x:
        return
    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
    ax.scatter(x, y, color="#D55E00", s=35, alpha=0.85)
    limit = max(max(x), max(y), 1.0)
    ax.plot([0, limit], [0, limit], linestyle="--", color="#666666", linewidth=1)
    for xi, yi, label in zip(x, y, labels):
        ax.annotate(label, (xi, yi), fontsize=7, alpha=0.8)
    ax.set_xlabel("rho_t before resynthesis")
    ax.set_ylabel("rho_t after resynthesis")
    ax.set_title("Metric stability under public resynthesis")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(output_root / "resynthesis_stability.png", dpi=300)
    fig.savefig(output_root / "resynthesis_stability.pdf")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate publication-ready plots for the hypothesis check.")
    parser.add_argument("--final-metrics-csv", type=Path, default=DEFAULT_CSV_ROOT / "final_metrics.csv")
    parser.add_argument(
        "--correlation-csv",
        type=Path,
        default=DEFAULT_CSV_ROOT / "correlation_summary.csv",
    )
    parser.add_argument("--method-overlap-csv", type=Path, default=DEFAULT_CSV_ROOT / "method_overlap.csv")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_FIGURES_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_dir(args.output_root)
    final_rows = load_rows(args.final_metrics_csv)
    correlation_rows = load_rows(args.correlation_csv)
    overlap_rows_data = load_rows(args.method_overlap_csv)
    scatter_grid(final_rows, args.output_root)
    heatmap(correlation_rows, args.output_root)
    overlap_plot(overlap_rows_data, args.output_root)
    stability_plot(final_rows, args.output_root)
    append_command(
        {
            "tool": "plot_results.py",
            "command": (
                f"{sys.executable} scripts/plot_results.py --final-metrics-csv {args.final_metrics_csv} "
                f"--correlation-csv {args.correlation_csv} --method-overlap-csv {args.method_overlap_csv} "
                f"--output-root {args.output_root}"
            ),
            "cwd": str(PROJECT_ROOT),
            "output_root": str(args.output_root),
            "exit_code": 0,
        }
    )
    print({"output_root": str(args.output_root)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
