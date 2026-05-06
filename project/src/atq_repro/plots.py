from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from atq_repro.paths import ensure_dir


COLORS = {
    "paper": "#4D4D4D",
    "reproduced": "#0072B2",
    "gadgets": "#009E73",
    "no_gadgets": "#D55E00",
    "binary": "#CC79A7",
    "benchmark": "#0072B2",
    "gf": "#009E73",
}

SOURCE_LABELS = {
    "paper_table_benchmark_no_gadgets": "Benchmarks\nsem gadgets",
    "paper_table_benchmark_gadgets": "Benchmarks\ncom gadgets",
    "paper_fig4_gf_no_gadgets": "GF(2^m)\nsem gadgets",
    "paper_fig4_gf_gadgets": "GF(2^m)\ncom gadgets",
    "paper_fig4_binary_addition": "Binary\naddition",
}

SOURCE_COLORS = {
    "paper_table_benchmark_no_gadgets": COLORS["no_gadgets"],
    "paper_table_benchmark_gadgets": COLORS["benchmark"],
    "paper_fig4_gf_no_gadgets": "#E69F00",
    "paper_fig4_gf_gadgets": COLORS["gf"],
    "paper_fig4_binary_addition": COLORS["binary"],
}


def _save(fig: Any, output_dir: Path, stem: str) -> Path:
    ensure_dir(output_dir)
    png_path = output_dir / f"{stem}.png"
    fig.savefig(png_path, dpi=300)
    fig.savefig(output_dir / f"{stem}.pdf")
    plt.close(fig)
    return png_path


def _coerce_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def plot_benchmark_matches(comparison_rows: list[dict[str, Any]], output_dir: Path) -> Path:
    rows = [
        row for row in comparison_rows
        if row["source"] in SOURCE_LABELS
    ]
    source_order = [
        "paper_table_benchmark_no_gadgets",
        "paper_table_benchmark_gadgets",
        "paper_fig4_gf_no_gadgets",
        "paper_fig4_gf_gadgets",
        "paper_fig4_binary_addition",
    ]
    totals = [sum(row["source"] == source for row in rows) for source in source_order]
    matches = [
        sum(row["source"] == source and str(row["match"]).lower() == "true" for row in rows)
        for source in source_order
    ]

    fig, (ax_bar, ax_parity) = plt.subplots(
        1,
        2,
        figsize=(11.8, 4.6),
        gridspec_kw={"width_ratios": [1.05, 1.35]},
        constrained_layout=True,
    )

    y = np.arange(len(source_order))
    ax_bar.barh(
        y,
        totals,
        color="#E6E6E6",
        edgecolor="#777777",
        linewidth=0.7,
        label="Total comparado",
    )
    ax_bar.barh(
        y,
        matches,
        color=[SOURCE_COLORS[source] for source in source_order],
        label="Match exato",
    )
    for index, (match_count, total_count) in enumerate(zip(matches, totals)):
        ax_bar.text(
            total_count + 0.8,
            index,
            f"{match_count}/{total_count}",
            va="center",
            ha="left",
            fontsize=8,
        )
    ax_bar.set_yticks(y, [SOURCE_LABELS[source] for source in source_order])
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("linhas reproduzidas")
    ax_bar.set_title("A. Cobertura da reproducao")
    ax_bar.set_xlim(0, max(totals) + 8)
    ax_bar.legend(frameon=False, loc="lower right", fontsize=8)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    for source in source_order:
        subset = [row for row in rows if row["source"] == source]
        paper = [_coerce_int(row["paper_effective_tcount"]) for row in subset]
        reproduced = [_coerce_int(row["reproduced_effective_tcount"]) for row in subset]
        ax_parity.scatter(
            paper,
            reproduced,
            s=34,
            alpha=0.84,
            color=SOURCE_COLORS[source],
            edgecolor="white",
            linewidth=0.35,
            label=SOURCE_LABELS[source].replace("\n", " "),
        )

    finite_values = [
        value
        for row in rows
        for value in (
            _coerce_int(row["paper_effective_tcount"]),
            _coerce_int(row["reproduced_effective_tcount"]),
        )
        if value is not None and value > 0
    ]
    lower = max(min(finite_values) * 0.75, 1)
    upper = max(finite_values) * 1.35
    ax_parity.plot([lower, upper], [lower, upper], color="#4D4D4D", linewidth=1.0, linestyle="--")
    ax_parity.set_xscale("log")
    ax_parity.set_yscale("log")
    ax_parity.set_xlim(lower, upper)
    ax_parity.set_ylim(lower, upper)
    ax_parity.set_aspect("equal", adjustable="box")
    ax_parity.set_xlabel("T-count efetivo no artigo")
    ax_parity.set_ylabel("T-count efetivo reproduzido")
    ax_parity.set_title("B. Concordancia valor-a-valor")
    ax_parity.legend(frameon=False, fontsize=7, loc="lower right")
    ax_parity.spines["top"].set_visible(False)
    ax_parity.spines["right"].set_visible(False)

    fig.suptitle("Reproducao local dos resultados publicos do AlphaTensor-Quantum", fontsize=12)
    return _save(fig, output_dir, "paper_benchmark_comparison")


def plot_gf_multiplication(comparison_rows: list[dict[str, Any]], output_dir: Path) -> Path:
    rows = [
        row for row in comparison_rows
        if row["source"] in {"paper_fig4_gf_no_gadgets", "paper_fig4_gf_gadgets"}
    ]
    fig, ax = plt.subplots(figsize=(7.5, 5.2), constrained_layout=True)
    for source, label, color in [
        ("paper_fig4_gf_no_gadgets", "AT-Q sem gadgets", COLORS["no_gadgets"]),
        ("paper_fig4_gf_gadgets", "AT-Q com gadgets", COLORS["gadgets"]),
    ]:
        subset = sorted(
            [row for row in rows if row["source"] == source],
            key=lambda row: int(row["circuit_id"].split("gf_2pow", 1)[1].split("_", 1)[0]),
        )
        x = [int(row["circuit_id"].split("gf_2pow", 1)[1].split("_", 1)[0]) for row in subset]
        y = [_coerce_int(row["reproduced_effective_tcount"]) for row in subset]
        ax.plot(x, y, marker="o", color=color, label=label)
    ax.set_xlabel("m em GF(2^m)")
    ax.set_ylabel("T-count efetivo reproduzido")
    ax.set_title("Multiplicacao em corpos finitos")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return _save(fig, output_dir, "gf_multiplication_effective_tcount")


def plot_binary_addition(comparison_rows: list[dict[str, Any]], output_dir: Path) -> Path:
    rows = sorted(
        [row for row in comparison_rows if row["source"] == "paper_fig4_binary_addition"],
        key=lambda row: int(row["circuit_id"].rsplit("_n", 1)[1]),
    )
    x = [int(row["circuit_id"].rsplit("_n", 1)[1]) for row in rows]
    reproduced = [_coerce_int(row["reproduced_effective_tcount"]) for row in rows]
    paper = [_coerce_int(row["paper_effective_tcount"]) for row in rows]

    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    ax.plot(x, paper, marker="o", color=COLORS["paper"], label="Paper")
    ax.plot(x, reproduced, marker="x", color=COLORS["reproduced"], label="Reproduzido")
    ax.set_xlabel("bits do somador Cuccaro")
    ax.set_ylabel("T-count efetivo")
    ax.set_title("Binary addition: reproducao local")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return _save(fig, output_dir, "binary_addition_effective_tcount")


def plot_family_coverage(aggregate_rows: list[dict[str, Any]], output_dir: Path) -> Path:
    families = sorted({row["family_label"] for row in aggregate_rows})
    counts = [sum(row["family_label"] == family for row in aggregate_rows) for family in families]
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    ax.bar(families, counts, color="#0072B2")
    ax.set_ylabel("circuitos/blocos agregados")
    ax.set_title("Cobertura das decomposicoes oficiais")
    ax.tick_params(axis="x", rotation=25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return _save(fig, output_dir, "paper_family_coverage")


def generate_paper_figures(
    *,
    comparison_rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Path]:
    ensure_dir(output_dir)
    return {
        "benchmark_comparison": plot_benchmark_matches(comparison_rows, output_dir),
        "gf_multiplication": plot_gf_multiplication(comparison_rows, output_dir),
        "binary_addition": plot_binary_addition(comparison_rows, output_dir),
        "family_coverage": plot_family_coverage(aggregate_rows, output_dir),
    }
