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
from scripts._analysis_common import DEFAULT_REPORTS_ROOT
from scripts._analysis_common import DEFAULT_RESULTS_ROOT
from scripts._analysis_common import ensure_dir
from scripts._analysis_common import natural_sort_key
from scripts._analysis_common import write_csv_rows
from scripts.alphatensor_reranker import DEFAULT_PREDICTION_TOLERANCE
from scripts.alphatensor_reranker import FEATURE_COLUMNS
from scripts.alphatensor_reranker import LABEL_COLUMN
from scripts.alphatensor_reranker import coerce_float
from scripts.alphatensor_reranker import comparison_rows
from scripts.alphatensor_reranker import leave_one_circuit_out_eval
from scripts.alphatensor_reranker import predictions_rows
from scripts.alphatensor_reranker import train_mlp
from scripts.alphatensor_reranker import valid_candidate_rows


DEFAULT_FRONTIER_CSV = (
    DEFAULT_RESULTS_ROOT / "public_resynth_structural" / "candidate_frontier.csv"
)
DEFAULT_DIAGNOSTICS_CSV = DEFAULT_CSV_ROOT / "splitting_candidate_diagnostics.csv"
DEFAULT_CORRELATIONS_CSV = DEFAULT_CSV_ROOT / "splitting_feature_correlations.csv"
DEFAULT_TOLERANCE_SWEEP_CSV = DEFAULT_CSV_ROOT / "splitting_tolerance_sweep.csv"
DEFAULT_REPORT_PATH = DEFAULT_REPORTS_ROOT / "splitting_phenomenon_analysis.md"
DEFAULT_FIGURE_DIR = DEFAULT_FIGURES_ROOT / "splitting_phenomenon"
DEFAULT_TOLERANCES = (0.0, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3)


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def finite_pairs(rows: list[dict[str, Any]], x_column: str, y_column: str) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for row in rows:
        x = coerce_float(row.get(x_column))
        y = coerce_float(row.get(y_column))
        if x is not None and y is not None:
            xs.append(x)
            ys.append(y)
    return np.array(xs, dtype=float), np.array(ys, dtype=float)


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        if stop - start > 1:
            ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3:
        return None
    if float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3:
        return None
    return pearson(rankdata(x), rankdata(y))


def centered_pairs(
    rows: list[dict[str, Any]],
    x_column: str,
    y_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for circuit_id in sorted({row["circuit_id"] for row in rows}, key=natural_sort_key):
        subset = [row for row in rows if row["circuit_id"] == circuit_id]
        local_x, local_y = finite_pairs(subset, x_column, y_column)
        if len(local_x) < 2:
            continue
        xs.extend((local_x - np.mean(local_x)).tolist())
        ys.extend((local_y - np.mean(local_y)).tolist())
    return np.array(xs, dtype=float), np.array(ys, dtype=float)


def value_or_none(row: dict[str, Any], column: str) -> float | None:
    return coerce_float(row.get(column))


def rank_value(row: dict[str, Any], column: str) -> float:
    value = value_or_none(row, column)
    return float("inf") if value is None else value


def candidate_diagnostics(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    diagnostics = []
    for circuit_id in sorted({row["circuit_id"] for row in rows}, key=natural_sort_key):
        subset = [row for row in rows if row["circuit_id"] == circuit_id]
        structural_best = min(
            subset,
            key=lambda row: (
                rank_value(row, LABEL_COLUMN),
                rank_value(row, "tcount_after"),
            ),
        )
        tcount_best = min(
            subset,
            key=lambda row: (
                rank_value(row, "tcount_after"),
                rank_value(row, LABEL_COLUMN),
            ),
        )
        primary_values = sorted(
            (value_or_none(row, LABEL_COLUMN), row["candidate_id"])
            for row in subset
            if value_or_none(row, LABEL_COLUMN) is not None
        )
        tcount_values = sorted(
            (value_or_none(row, "tcount_after"), row["candidate_id"])
            for row in subset
            if value_or_none(row, "tcount_after") is not None
        )
        primary_rank = {
            candidate_id: rank + 1
            for rank, (_, candidate_id) in enumerate(primary_values)
        }
        tcount_rank = {
            candidate_id: rank + 1 for rank, (_, candidate_id) in enumerate(tcount_values)
        }
        structural_primary = value_or_none(structural_best, LABEL_COLUMN)
        structural_tcount = value_or_none(structural_best, "tcount_after")
        tcount_primary = value_or_none(tcount_best, LABEL_COLUMN)
        tcount_tcount = value_or_none(tcount_best, "tcount_after")
        for row in subset:
            primary = value_or_none(row, LABEL_COLUMN)
            tcount = value_or_none(row, "tcount_after")
            qasm_depth_ratio = value_or_none(row, "qasm_depth_ratio")
            zx_total_depth_ratio = value_or_none(row, "zx_total_depth_ratio")
            diagnostics.append(
                {
                    **row,
                    "primary_rank_within_circuit": primary_rank.get(row["candidate_id"]),
                    "tcount_rank_within_circuit": tcount_rank.get(row["candidate_id"]),
                    "primary_regret_vs_structural_best": (
                        None if primary is None or structural_primary is None else primary - structural_primary
                    ),
                    "tcount_delta_vs_structural_best": (
                        None if tcount is None or structural_tcount is None else tcount - structural_tcount
                    ),
                    "primary_gain_vs_tcount_best": (
                        None if primary is None or tcount_primary is None else tcount_primary - primary
                    ),
                    "tcount_delta_vs_tcount_best": (
                        None if tcount is None or tcount_tcount is None else tcount - tcount_tcount
                    ),
                    "is_structural_best": row["candidate_id"] == structural_best["candidate_id"],
                    "is_tcount_best": row["candidate_id"] == tcount_best["candidate_id"],
                    "structural_inflation_flag": bool(
                        (qasm_depth_ratio is not None and qasm_depth_ratio > 2.0)
                        or (
                            zx_total_depth_ratio is not None
                            and zx_total_depth_ratio > 2.0
                        )
                    ),
                }
            )
    return diagnostics


def feature_correlations(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    candidates = list(FEATURE_COLUMNS) + [
        "zx_total_depth_ratio",
        "qasm_depth_ratio",
        "tcount_ratio",
        "tdepth_ratio",
        "gate_count_ratio",
    ]
    result = []
    seen = set()
    for feature in candidates:
        if feature in seen or feature == LABEL_COLUMN:
            continue
        seen.add(feature)
        x, y = finite_pairs(rows, feature, LABEL_COLUMN)
        centered_x, centered_y = centered_pairs(rows, feature, LABEL_COLUMN)
        global_pearson = pearson(x, y)
        global_spearman = spearman(x, y)
        within_pearson = pearson(centered_x, centered_y)
        within_spearman = spearman(centered_x, centered_y)
        result.append(
            {
                "feature": feature,
                "n": len(x),
                "global_pearson": global_pearson,
                "global_spearman": global_spearman,
                "within_circuit_centered_pearson": within_pearson,
                "within_circuit_centered_spearman": within_spearman,
                "abs_within_circuit_centered_pearson": (
                    None if within_pearson is None else abs(within_pearson)
                ),
            }
        )
    return sorted(
        result,
        key=lambda row: -(
            row["abs_within_circuit_centered_pearson"]
            if row["abs_within_circuit_centered_pearson"] is not None
            else -1.0
        ),
    )


def summarize_comparison(rows: list[dict[str, Any]]) -> dict[str, Any]:
    regrets = [coerce_float(row.get("primary_regret_vs_structural_best")) for row in rows]
    gains = [coerce_float(row.get("primary_gain_vs_tcount_best")) for row in rows]
    tcount_delta_structural = [
        coerce_float(row.get("tcount_delta_vs_structural_best")) for row in rows
    ]
    tcount_delta_tbest = [
        coerce_float(row.get("tcount_delta_vs_tcount_best")) for row in rows
    ]
    regrets = [item for item in regrets if item is not None]
    gains = [item for item in gains if item is not None]
    tcount_delta_structural = [
        item for item in tcount_delta_structural if item is not None
    ]
    tcount_delta_tbest = [item for item in tcount_delta_tbest if item is not None]
    return {
        "mean_primary_regret_vs_structural_best": float(np.mean(regrets)) if regrets else None,
        "max_primary_regret_vs_structural_best": float(np.max(regrets)) if regrets else None,
        "mean_primary_gain_vs_tcount_best": float(np.mean(gains)) if gains else None,
        "total_tcount_delta_vs_structural_best": float(np.sum(tcount_delta_structural))
        if tcount_delta_structural
        else None,
        "total_tcount_delta_vs_tcount_best": float(np.sum(tcount_delta_tbest))
        if tcount_delta_tbest
        else None,
    }


def tolerance_sweep(
    rows: list[dict[str, str]],
    tolerances: tuple[float, ...],
    *,
    hidden_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> list[dict[str, Any]]:
    model = train_mlp(
        rows,
        hidden_size=hidden_size,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
    )
    sweep_rows = []
    for tolerance in tolerances:
        full_predictions = predictions_rows(
            rows,
            model,
            prediction_tolerance=tolerance,
        )
        full_comparison = comparison_rows(full_predictions)
        full_summary = summarize_comparison(full_comparison)
        eval_rows = leave_one_circuit_out_eval(
            rows,
            hidden_size=hidden_size,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            seed=seed,
            prediction_tolerance=tolerance,
        )
        regrets = [
            coerce_float(row.get("primary_regret_vs_true_best")) for row in eval_rows
        ]
        gains = [
            coerce_float(row.get("primary_gain_vs_tcount_best")) for row in eval_rows
        ]
        regrets = [item for item in regrets if item is not None]
        gains = [item for item in gains if item is not None]
        qft = next(
            (row for row in full_comparison if row["circuit_id"] == "qft_4"),
            None,
        )
        sweep_rows.append(
            {
                "prediction_tolerance": tolerance,
                "loco_hit_rate": (
                    sum(row.get("hit_true_best") is True for row in eval_rows)
                    / max(len(eval_rows), 1)
                ),
                "loco_mean_primary_regret_vs_true_best": (
                    float(np.mean(regrets)) if regrets else None
                ),
                "loco_mean_primary_gain_vs_tcount_best": (
                    float(np.mean(gains)) if gains else None
                ),
                "full_mean_primary_regret_vs_structural_best": full_summary[
                    "mean_primary_regret_vs_structural_best"
                ],
                "full_max_primary_regret_vs_structural_best": full_summary[
                    "max_primary_regret_vs_structural_best"
                ],
                "full_mean_primary_gain_vs_tcount_best": full_summary[
                    "mean_primary_gain_vs_tcount_best"
                ],
                "full_total_tcount_delta_vs_structural_best": full_summary[
                    "total_tcount_delta_vs_structural_best"
                ],
                "full_total_tcount_delta_vs_tcount_best": full_summary[
                    "total_tcount_delta_vs_tcount_best"
                ],
                "qft_4_selected_candidate": None if qft is None else qft["reranker_candidate_id"],
                "qft_4_primary_regret": None
                if qft is None
                else qft["primary_regret_vs_structural_best"],
                "qft_4_tcount_delta_vs_structural": None
                if qft is None
                else qft["tcount_delta_vs_structural_best"],
            }
        )
    return sweep_rows


def top_correlation_lines(correlations: list[dict[str, Any]], *, limit: int = 6) -> list[str]:
    lines = []
    for row in correlations[:limit]:
        value = row["within_circuit_centered_pearson"]
        if value is None:
            continue
        global_spearman = row["global_spearman"]
        global_text = "NA" if global_spearman is None else f"{float(global_spearman):.3f}"
        lines.append(
            f"- `{row['feature']}`: within-circuit Pearson {float(value):.3f} "
            f"(global Spearman {global_text})."
        )
    return lines


def fmt(value: Any, digits: int = 3) -> str:
    numeric = coerce_float(value)
    return "NA" if numeric is None else f"{numeric:.{digits}f}"


def make_tradeoff_figure(diagnostics: list[dict[str, Any]], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    circuits = sorted({row["circuit_id"] for row in diagnostics}, key=natural_sort_key)
    colors = plt.get_cmap("tab10")
    for index, circuit_id in enumerate(circuits):
        subset = [row for row in diagnostics if row["circuit_id"] == circuit_id]
        x = [coerce_float(row.get("tcount_after")) for row in subset]
        y = [coerce_float(row.get(LABEL_COLUMN)) for row in subset]
        ax.scatter(
            x,
            y,
            s=55,
            alpha=0.8,
            label=circuit_id,
            color=colors(index % 10),
            edgecolor="white",
            linewidth=0.6,
        )
        for row in subset:
            if row["is_structural_best"] or row["is_tcount_best"]:
                marker = "*" if row["is_structural_best"] else "x"
                ax.scatter(
                    [coerce_float(row.get("tcount_after"))],
                    [coerce_float(row.get(LABEL_COLUMN))],
                    s=130,
                    marker=marker,
                    color=colors(index % 10),
                    edgecolor=None if marker == "x" else "black",
                    linewidth=0.8,
                )
    ax.set_xlabel("T-count after resynthesis")
    ax.set_ylabel("primary_nc_depth_ratio")
    ax.set_title("Candidate tradeoff: T-count vs structural non-Clifford core")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncols=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def make_tolerance_figure(sweep_rows: list[dict[str, Any]], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    tolerances = [coerce_float(row["prediction_tolerance"]) for row in sweep_rows]
    full_regret = [
        coerce_float(row["full_mean_primary_regret_vs_structural_best"])
        for row in sweep_rows
    ]
    total_tcount_delta = [
        coerce_float(row["full_total_tcount_delta_vs_structural_best"])
        for row in sweep_rows
    ]
    fig, ax1 = plt.subplots(figsize=(8.0, 4.8))
    ax1.plot(tolerances, full_regret, marker="o", color="#1f77b4", label="Mean primary regret")
    ax1.set_xlabel("prediction_tolerance")
    ax1.set_ylabel("Mean primary regret", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(True, alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(
        tolerances,
        total_tcount_delta,
        marker="s",
        color="#d62728",
        label="Total T-count delta",
    )
    ax2.set_ylabel("Total T-count delta vs structural best", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    fig.suptitle("Tolerance trades structural regret for T-count reduction")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_report(
    *,
    output_path: Path,
    rows: list[dict[str, str]],
    diagnostics: list[dict[str, Any]],
    correlations: list[dict[str, Any]],
    sweep_rows: list[dict[str, Any]],
    tradeoff_figure: Path,
    tolerance_figure: Path,
) -> Path:
    circuits = sorted({row["circuit_id"] for row in rows}, key=natural_sort_key)
    full_tolerance_row = min(
        sweep_rows,
        key=lambda row: abs(
            float(row["prediction_tolerance"]) - DEFAULT_PREDICTION_TOLERANCE
        ),
    )
    inflation_count = sum(bool(row["structural_inflation_flag"]) for row in diagnostics)
    structural_best_rows = [row for row in diagnostics if row["is_structural_best"]]
    tcount_best_rows = [row for row in diagnostics if row["is_tcount_best"]]
    separation_lines = []
    for structural_row in structural_best_rows:
        circuit_id = structural_row["circuit_id"]
        tcount_row = next(row for row in tcount_best_rows if row["circuit_id"] == circuit_id)
        primary_gain = coerce_float(structural_row["primary_gain_vs_tcount_best"])
        t_delta = coerce_float(structural_row["tcount_delta_vs_tcount_best"])
        if primary_gain is None or t_delta is None:
            continue
        if abs(primary_gain) > 1e-9 or abs(t_delta) > 1e-9:
            separation_lines.append(
                f"- `{circuit_id}`: structural best improves primary by "
                f"{primary_gain:.3f} relative to T-count best while changing T-count by "
                f"{t_delta:+.0f}."
            )
    qft_row = next(
        (row for row in diagnostics if row["circuit_id"] == "qft_4" and row["candidate_id"] == "qft_4:combo0"),
        None,
    )
    qft_structural = next(
        (row for row in diagnostics if row["circuit_id"] == "qft_4" and row["is_structural_best"]),
        None,
    )
    qft_note = ""
    if qft_row and qft_structural:
        qft_note = (
            f"For `qft_4`, the tolerance-aware reranker can choose a candidate with "
            f"T-count {fmt(qft_row.get('tcount_after'), 0)} instead of "
            f"{fmt(qft_structural.get('tcount_after'), 0)}, paying only "
            f"{fmt(qft_row.get('primary_regret_vs_structural_best'))} primary regret."
        )

    text = [
        "# Splitting Phenomenon Analysis",
        "",
        "## Scope",
        "",
        f"- Candidate rows: {len(rows)}.",
        f"- Circuits: {', '.join(f'`{item}`' for item in circuits)}.",
        f"- Primary target: `{LABEL_COLUMN}`.",
        f"- Tradeoff figure: `{tradeoff_figure}`.",
        f"- Tolerance figure: `{tolerance_figure}`.",
        "",
        "## Main empirical picture",
        "",
        (
            "The current public-resynthesis frontier is not a simple T-count frontier. "
            "Several candidates reduce T-count while expanding the ZX non-Clifford core, "
            "and several structural winners accept more T gates because they keep the "
            "non-Clifford portion more compact after Clifford closure."
        ),
        "",
        *separation_lines,
        (
            f"- {inflation_count}/{len(diagnostics)} candidates have either "
            "`qasm_depth_ratio > 2` or `zx_total_depth_ratio > 2`, so Clifford scaffolding "
            "inflation is common and should remain a diagnostic rather than be hidden by "
            "T-count alone."
        ),
        f"- {qft_note}" if qft_note else "",
        "",
        "## Reranker and tolerance",
        "",
        (
            "The tolerance should be interpreted as an uncertainty band around the learned "
            "structural prediction, not as a fallback to the old objective. Within that band, "
            "using T-count as a tie-breaker makes the policy less brittle while still asking "
            "the model to find candidates near the structural frontier."
        ),
        "",
        (
            f"At tolerance {DEFAULT_PREDICTION_TOLERANCE:.3f}, the full-frontier selection has "
            f"mean primary regret {fmt(full_tolerance_row['full_mean_primary_regret_vs_structural_best'])} "
            "against the exact structural best and total T-count delta "
            f"{fmt(full_tolerance_row['full_total_tcount_delta_vs_structural_best'], 0)} "
            "against the exact structural best."
        ),
        "",
        "## Feature evidence",
        "",
        (
            "The strongest correlations should be read as mechanistic hints, not as stable "
            "feature importance: there are only 19 candidates and 5 circuits. The within-circuit "
            "centered correlations are more informative than global correlations because they "
            "remove most benchmark-size effects."
        ),
        "",
        *top_correlation_lines(correlations),
        "",
        "## Scientific interpretation",
        "",
        (
            "`primary_nc_depth_ratio` is probing whether Clifford regions can be peeled away "
            "without leaving a long residual non-Clifford kernel. T-count is only a count of "
            "phase resources; it does not say whether those resources are geometrically or "
            "causally isolated after CNOT/Hadamard interactions. A candidate can therefore "
            "have fewer T gates but a worse splitting target if the remaining phase gates are "
            "spread across a deeper entangled core."
        ),
        "",
        (
            "This explains the observed tension: the AlphaTensor-public decompositions often "
            "lower T-count by inserting Clifford-heavy scaffolding. Sometimes that scaffolding "
            "also isolates the non-Clifford kernel; sometimes it increases total depth and "
            "keeps the non-Clifford part coupled to a large closure. The structural target is "
            "detecting the latter case."
        ),
        "",
        "## Recommended refinements",
        "",
        (
            "1. Treat tolerance as calibrated model uncertainty: sweep it and select the "
            "smallest value that gives material T-count relief without large structural regret."
        ),
        (
            "2. Move from point regression to pairwise/listwise ranking once the frontier is "
            "larger; the scientific question is candidate dominance on a structural/T-count "
            "frontier, not absolute prediction of a scalar."
        ),
        (
            "3. Add cheap features that approximate non-Clifford closure: T-span, CNOTs crossing "
            "the first/last T boundary, Hadamard-boundary counts near T layers, Clifford scaffold "
            "ratio, and interaction width of the T support."
        ),
        (
            "4. Expand candidate generation deliberately around frontier diversity. The current "
            "dataset is too small to distinguish a real law from benchmark-specific coincidences."
        ),
        "",
        "## Validity cautions",
        "",
        (
            "The evidence is promising but weak: n=19 candidates, singleton circuit cases, and "
            "full-fit selections are optimistic. Leave-one-circuit-out is the right sanity check, "
            "but it is high variance at this scale. Claims should be phrased as a mechanistic "
            "hypothesis supported by early evidence, not as a settled optimizer result."
        ),
        "",
    ]
    ensure_dir(output_path.parent)
    output_path.write_text("\n".join(line for line in text if line is not None), encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze the structural Clifford-splitting phenomenon."
    )
    parser.add_argument("--frontier-csv", type=Path, default=DEFAULT_FRONTIER_CSV)
    parser.add_argument("--diagnostics-csv", type=Path, default=DEFAULT_DIAGNOSTICS_CSV)
    parser.add_argument("--correlations-csv", type=Path, default=DEFAULT_CORRELATIONS_CSV)
    parser.add_argument("--tolerance-sweep-csv", type=Path, default=DEFAULT_TOLERANCE_SWEEP_CSV)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--hidden-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2_000)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = valid_candidate_rows(load_csv_rows(args.frontier_csv))
    if not rows:
        raise SystemExit(f"No valid candidates found in {args.frontier_csv}")

    diagnostics = candidate_diagnostics(rows)
    correlations = feature_correlations(rows)
    sweep_rows = tolerance_sweep(
        rows,
        DEFAULT_TOLERANCES,
        hidden_size=args.hidden_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )

    tradeoff_figure = args.figure_dir / "candidate_tradeoff.png"
    tolerance_figure = args.figure_dir / "tolerance_sweep.png"
    make_tradeoff_figure(diagnostics, tradeoff_figure)
    make_tolerance_figure(sweep_rows, tolerance_figure)

    write_csv_rows(diagnostics, args.diagnostics_csv)
    write_csv_rows(correlations, args.correlations_csv)
    write_csv_rows(sweep_rows, args.tolerance_sweep_csv)
    report_path = write_report(
        output_path=args.report_path,
        rows=rows,
        diagnostics=diagnostics,
        correlations=correlations,
        sweep_rows=sweep_rows,
        tradeoff_figure=tradeoff_figure,
        tolerance_figure=tolerance_figure,
    )
    print(
        {
            "diagnostics_csv": str(args.diagnostics_csv),
            "correlations_csv": str(args.correlations_csv),
            "tolerance_sweep_csv": str(args.tolerance_sweep_csv),
            "report_path": str(report_path),
            "tradeoff_figure": str(tradeoff_figure),
            "tolerance_figure": str(tolerance_figure),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
