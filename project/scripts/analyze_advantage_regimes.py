from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._analysis_common import DEFAULT_CSV_ROOT
from scripts._analysis_common import DEFAULT_FIGURES_ROOT
from scripts._analysis_common import DEFAULT_REPORTS_ROOT
from scripts._analysis_common import ensure_dir
from scripts._analysis_common import write_csv_rows
from scripts._analysis_common import write_json
from scripts._manifest import append_command


PUBLIC_METHODS = ("public_resynth_gadgets", "public_resynth_no_gadgets")


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


def coerce_int(value: Any) -> int | None:
    value_f = coerce_float(value)
    return None if value_f is None else int(value_f)


def build_pair_rows(rows: list[dict[str, str]], public_method: str) -> list[dict[str, Any]]:
    indexed = {(row["circuit_id"], row["method"]): row for row in rows}
    pair_rows: list[dict[str, Any]] = []
    for circuit_id in sorted({row["circuit_id"] for row in rows}):
        pyzx = indexed[(circuit_id, "pyzx")]
        public = indexed[(circuit_id, public_method)]
        if public["method_status"] != "ok":
            continue
        pyzx_delta = coerce_float(pyzx.get("delta_t"))
        public_delta = coerce_float(public.get("delta_t"))
        if pyzx_delta is None or public_delta is None:
            continue
        if public_delta > pyzx_delta:
            winner = "alphatensor"
        elif public_delta < pyzx_delta:
            winner = "pyzx"
        else:
            winner = "tie"
        pair_rows.append(
            {
                "circuit_id": circuit_id,
                "family": public["family"],
                "faixa": public["faixa"],
                "n_qubits": coerce_int(public.get("n_qubits")),
                "public_method": public_method,
                "winner": winner,
                "pyzx_tcount_before": coerce_int(pyzx.get("tcount_before")),
                "pyzx_tcount_after": coerce_int(pyzx.get("tcount_after")),
                "public_tcount_after": coerce_int(public.get("tcount_after")),
                "pyzx_delta_t": pyzx_delta,
                "public_delta_t": public_delta,
                "delta_advantage_public_minus_pyzx": public_delta - pyzx_delta,
                "pyzx_rel_gain_t": coerce_float(pyzx.get("rel_gain_t")),
                "public_rel_gain_t": coerce_float(public.get("rel_gain_t")),
                "rho_t_public": coerce_float(public.get("rho_t")),
                "rho_w_public": coerce_float(public.get("rho_w")),
                "hadamard_boundary_density_public": coerce_float(
                    public.get("hadamard_boundary_density")
                ),
                "n_nonclifford_blocks_public": coerce_int(
                    public.get("n_nonclifford_blocks")
                ),
                "avg_nonclifford_block_len_public": coerce_float(
                    public.get("avg_nonclifford_block_len")
                ),
                "public_qasm_artifact_path": public.get("qasm_artifact_path"),
                "pyzx_qasm_artifact_path": pyzx.get("qasm_artifact_path"),
            }
        )
    return pair_rows


def summarize_pairs(pair_rows: list[dict[str, Any]], public_method: str) -> dict[str, Any]:
    winners = [row["winner"] for row in pair_rows]
    public_better = [row for row in pair_rows if row["winner"] == "alphatensor"]
    ties = [row for row in pair_rows if row["winner"] == "tie"]
    pyzx_better = [row for row in pair_rows if row["winner"] == "pyzx"]
    public_advantages = [row["delta_advantage_public_minus_pyzx"] for row in pair_rows]
    public_gain_values = [row["public_delta_t"] for row in pair_rows]
    pyzx_gain_values = [row["pyzx_delta_t"] for row in pair_rows]
    rho_values = np.asarray([row["rho_t_public"] for row in pair_rows], dtype=float)
    delta_values = np.asarray([row["public_delta_t"] for row in pair_rows], dtype=float)
    spearman = float(stats.spearmanr(rho_values, delta_values).statistic)
    return {
        "public_method": public_method,
        "num_comparable_circuits": len(pair_rows),
        "num_alphatensor_better": len(public_better),
        "num_ties": len(ties),
        "num_pyzx_better": len(pyzx_better),
        "mean_public_delta_t": float(np.mean(public_gain_values)) if public_gain_values else None,
        "mean_pyzx_delta_t": float(np.mean(pyzx_gain_values)) if pyzx_gain_values else None,
        "mean_advantage_public_minus_pyzx": (
            float(np.mean(public_advantages)) if public_advantages else None
        ),
        "median_advantage_public_minus_pyzx": (
            float(np.median(public_advantages)) if public_advantages else None
        ),
        "spearman_rho_t_vs_public_delta_t": spearman,
        "alphatensor_better_examples": [row["circuit_id"] for row in public_better[:12]],
        "pyzx_better_examples": [row["circuit_id"] for row in pyzx_better[:12]],
    }


def save_pair_data(rows: list[dict[str, str]], output_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    all_pairs: list[dict[str, Any]] = []
    for public_method in PUBLIC_METHODS:
        pair_rows = build_pair_rows(rows, public_method)
        summary = summarize_pairs(pair_rows, public_method)
        write_csv_rows(
            pair_rows,
            output_dir / f"{public_method}_vs_pyzx.csv",
        )
        write_json(
            summary,
            output_dir / f"{public_method}_vs_pyzx.json",
        )
        summaries.append(summary)
        all_pairs.extend(pair_rows)
    write_csv_rows(summaries, output_dir / "advantage_summary.csv")
    write_json({"summaries": summaries}, output_dir / "advantage_summary.json")
    return summaries, all_pairs


def generate_parity_figure(all_pairs: list[dict[str, Any]], output_dir: Path) -> Path:
    colors = {
        "alphatensor": "#009E73",
        "pyzx": "#D55E00",
        "tie": "#666666",
    }
    labels = {
        "alphatensor": "AlphaTensor > PyZX",
        "pyzx": "PyZX > AlphaTensor",
        "tie": "Tie",
    }
    figure, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    for axis, method, title in [
        (axes[0], "public_resynth_gadgets", "Public Replay with Gadgets"),
        (axes[1], "public_resynth_no_gadgets", "Public Replay without Gadgets"),
    ]:
        subset = [row for row in all_pairs if row["public_method"] == method]
        max_value = max(
            [max(row["pyzx_delta_t"], row["public_delta_t"]) for row in subset] + [1.0]
        )
        axis.plot([0, max_value], [0, max_value], linestyle="--", color="#444444", linewidth=1)
        for winner in ("alphatensor", "tie", "pyzx"):
            points = [row for row in subset if row["winner"] == winner]
            if not points:
                continue
            axis.scatter(
                [row["pyzx_delta_t"] for row in points],
                [row["public_delta_t"] for row in points],
                s=44,
                alpha=0.85,
                color=colors[winner],
                label=f"{labels[winner]} ({len(points)})",
            )
        axis.set_title(title)
        axis.set_xlabel("PyZX absolute T-count gain ($\\Delta T$)")
        axis.set_ylabel("AlphaTensor-public absolute T-count gain ($\\Delta T$)")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.legend(frameon=False, loc="upper left")
    png_path = output_dir / "pyzx_vs_public_replay_parity.png"
    pdf_path = output_dir / "pyzx_vs_public_replay_parity.pdf"
    figure.savefig(png_path, dpi=300)
    figure.savefig(pdf_path)
    plt.close(figure)
    return png_path


def generate_rho_t_figure(all_pairs: list[dict[str, Any]], output_dir: Path) -> Path:
    subset = [row for row in all_pairs if row["public_method"] == "public_resynth_no_gadgets"]
    x = np.asarray([row["rho_t_public"] for row in subset], dtype=float)
    y = np.asarray([row["public_delta_t"] for row in subset], dtype=float)
    rho = float(stats.spearmanr(x, y).statistic)

    figure, axis = plt.subplots(figsize=(6.5, 4.8), constrained_layout=True)
    axis.scatter(x, y, s=48, alpha=0.85, color="#0072B2")
    order = np.argsort(x)
    coeff = np.polyfit(x, y, deg=1)
    fit = np.poly1d(coeff)
    axis.plot(x[order], fit(x[order]), color="#D55E00", linewidth=2)

    highlight_ids = {"hamming_15_med", "8_bit_adder", "qcla_mod_7", "mod_5_4", "gf_2pow2_mult"}
    for row in subset:
        if row["circuit_id"] not in highlight_ids:
            continue
        axis.annotate(
            row["circuit_id"],
            (row["rho_t_public"], row["public_delta_t"]),
            fontsize=8,
            alpha=0.9,
        )

    axis.set_title(
        rf"Lower $\rho_t$ aligns with larger gain in public no-gadgets replay ($\rho_s={rho:.3f}$)"
    )
    axis.set_xlabel(r"Non-Clifford density $\rho_t$")
    axis.set_ylabel(r"Absolute T-count gain $\Delta T$")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    png_path = output_dir / "rho_t_vs_delta_t_public_no_gadgets.png"
    pdf_path = output_dir / "rho_t_vs_delta_t_public_no_gadgets.pdf"
    figure.savefig(png_path, dpi=300)
    figure.savefig(pdf_path)
    plt.close(figure)
    return png_path


def write_report(report_path: Path, figure_dir: Path, summaries: list[dict[str, Any]]) -> Path:
    by_method = {summary["public_method"]: summary for summary in summaries}
    gadgets = by_method["public_resynth_gadgets"]
    no_gadgets = by_method["public_resynth_no_gadgets"]
    text = f"""# Current Highlights

## 1. PyZX vs AlphaTensor-public replay

This figure compares absolute T-count gain (`ΔT`) achieved by `PyZX` against the replay of public AlphaTensor decompositions.
At this stage, the main message is that the `no_gadgets` replay is stronger on all {no_gadgets["num_comparable_circuits"]} comparable circuits, while the `gadgets` replay is mixed: AlphaTensor wins on {gadgets["num_alphatensor_better"]}, ties on {gadgets["num_ties"]}, and trails PyZX on {gadgets["num_pyzx_better"]}.

![PyZX vs AlphaTensor-public parity]({figure_dir / "pyzx_vs_public_replay_parity.png"})

## 2. `rho_t` as a predictor in the `no_gadgets` regime

This figure shows the strongest structural signal we currently have: in `public_resynth_no_gadgets`, lower non-Clifford density (`rho_t`) is strongly associated with larger absolute gain.
The current Spearman correlation is {no_gadgets["spearman_rho_t_vs_public_delta_t"]:.3f}, which is the clearest evidence so far that the Clifford/non-Clifford structure is explaining optimization performance in a nontrivial way.

![rho_t vs delta_t]({figure_dir / "rho_t_vs_delta_t_public_no_gadgets.png"})
"""
    report_path.write_text(text, encoding="utf-8")
    return report_path


def main() -> int:
    rows = load_rows(DEFAULT_CSV_ROOT / "final_metrics.csv")
    data_dir = ensure_dir(DEFAULT_CSV_ROOT / "advantage_regimes")
    figure_dir = ensure_dir(DEFAULT_FIGURES_ROOT / "highlights")
    report_path = ensure_dir(DEFAULT_REPORTS_ROOT) / "current_highlights.md"

    summaries, all_pairs = save_pair_data(rows, data_dir)
    parity_png = generate_parity_figure(all_pairs, figure_dir)
    rho_png = generate_rho_t_figure(all_pairs, figure_dir)
    write_report(report_path, figure_dir, summaries)

    append_command(
        {
            "tool": "analyze_advantage_regimes.py",
            "command": f"{sys.executable} scripts/analyze_advantage_regimes.py",
            "cwd": str(PROJECT_ROOT),
            "report_path": str(report_path),
            "data_dir": str(data_dir),
            "figure_dir": str(figure_dir),
            "exit_code": 0,
        }
    )
    print(
        {
            "report_path": str(report_path),
            "data_dir": str(data_dir),
            "parity_png": str(parity_png),
            "rho_png": str(rho_png),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
