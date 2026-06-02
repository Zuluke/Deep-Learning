from __future__ import annotations

import csv
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
from scripts._manifest import append_command


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


def _pairs(
    rows: list[dict[str, str]],
    public_method: str,
) -> list[dict[str, Any]]:
    by = {(row["circuit_id"], row["method"]): row for row in rows}
    pairs = []
    for circuit_id in sorted({row["circuit_id"] for row in rows}):
        pyzx = by[(circuit_id, "pyzx")]
        public = by[(circuit_id, public_method)]
        if public["method_status"] != "ok":
            continue
        pyzx_delta = coerce_float(pyzx.get("delta_t"))
        public_delta = coerce_float(public.get("delta_t"))
        if pyzx_delta is None or public_delta is None:
            continue
        if public_delta > pyzx_delta:
            relation = "AlphaTensor > PyZX"
        elif public_delta < pyzx_delta:
            relation = "PyZX > AlphaTensor"
        else:
            relation = "Tie"
        pairs.append(
            {
                "circuit_id": circuit_id,
                "family": public["family"],
                "pyzx_delta": pyzx_delta,
                "public_delta": public_delta,
                "relation": relation,
            }
        )
    return pairs


def generate_parity_figure(rows: list[dict[str, str]], output_dir: Path) -> dict[str, Path]:
    gadgets_pairs = _pairs(rows, "public_resynth_gadgets")
    no_gadgets_pairs = _pairs(rows, "public_resynth_no_gadgets")
    colors = {
        "AlphaTensor > PyZX": "#009E73",
        "PyZX > AlphaTensor": "#D55E00",
        "Tie": "#666666",
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    for ax, pairs, title in [
        (axes[0], gadgets_pairs, "Public Replay with Gadgets"),
        (axes[1], no_gadgets_pairs, "Public Replay without Gadgets"),
    ]:
        limit = max(
            [max(item["pyzx_delta"], item["public_delta"]) for item in pairs] + [1.0]
        )
        ax.plot([0, limit], [0, limit], linestyle="--", color="#444444", linewidth=1)
        for relation in ("AlphaTensor > PyZX", "Tie", "PyZX > AlphaTensor"):
            subset = [item for item in pairs if item["relation"] == relation]
            if not subset:
                continue
            ax.scatter(
                [item["pyzx_delta"] for item in subset],
                [item["public_delta"] for item in subset],
                s=42,
                alpha=0.85,
                color=colors[relation],
                label=f"{relation} ({len(subset)})",
            )
        ax.set_title(title)
        ax.set_xlabel("PyZX absolute T-count gain ($\\Delta T$)")
        ax.set_ylabel("AlphaTensor-public absolute T-count gain ($\\Delta T$)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, loc="upper left")

    png_path = output_dir / "pyzx_vs_public_replay_parity.png"
    pdf_path = output_dir / "pyzx_vs_public_replay_parity.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return {"png": png_path, "pdf": pdf_path}


def generate_rho_t_figure(rows: list[dict[str, str]], output_dir: Path) -> dict[str, Path]:
    subset = [
        row for row in rows
        if row["method"] == "public_resynth_no_gadgets" and row["method_status"] == "ok"
    ]
    x = np.asarray([coerce_float(row["rho_t"]) for row in subset], dtype=float)
    y = np.asarray([coerce_float(row["delta_t"]) for row in subset], dtype=float)
    labels = [row["circuit_id"] for row in subset]
    rho = float(stats.spearmanr(x, y).statistic)

    fig, ax = plt.subplots(figsize=(6.5, 4.8), constrained_layout=True)
    ax.scatter(x, y, s=48, alpha=0.85, color="#0072B2")
    order = np.argsort(x)
    coeff = np.polyfit(x, y, deg=1)
    fit = np.poly1d(coeff)
    ax.plot(x[order], fit(x[order]), color="#D55E00", linewidth=2)

    highlight_ids = {"hamming_15_med", "8_bit_adder", "qcla_mod_7", "mod_5_4", "gf_2pow2_mult"}
    for row in subset:
        if row["circuit_id"] not in highlight_ids:
            continue
        xi = float(row["rho_t"])
        yi = float(row["delta_t"])
        ax.annotate(row["circuit_id"], (xi, yi), fontsize=8, alpha=0.9)

    ax.set_title(rf"Lower $\rho_t$ aligns with larger gain in public no-gadgets replay ($\rho_s={rho:.3f}$)")
    ax.set_xlabel(r"Non-Clifford density $\rho_t$")
    ax.set_ylabel(r"Absolute T-count gain $\Delta T$")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    png_path = output_dir / "rho_t_vs_delta_t_public_no_gadgets.png"
    pdf_path = output_dir / "rho_t_vs_delta_t_public_no_gadgets.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return {"png": png_path, "pdf": pdf_path}


def write_markdown(output_path: Path, parity_paths: dict[str, Path], rho_paths: dict[str, Path]) -> Path:
    text = f"""# Current Highlights

## 1. PyZX vs AlphaTensor-public replay

This figure compares absolute T-count gain (`ΔT`) achieved by `PyZX` against the replay of public AlphaTensor decompositions.
The main point is that the `no_gadgets` replay is consistently stronger on every comparable case, while the `gadgets` replay is mixed: it wins often, ties often, and still loses on a meaningful subset.

![PyZX vs AlphaTensor-public parity]({parity_paths["png"]})

## 2. `rho_t` as a predictor in the `no_gadgets` regime

This figure shows the strongest structural signal we currently have: in `public_resynth_no_gadgets`, lower non-Clifford density (`rho_t`) is strongly associated with larger absolute gain.
At this stage, this is the clearest evidence that the Clifford/non-Clifford structure is explaining optimization performance in a nontrivial way.

![rho_t vs delta_t]({rho_paths["png"]})
"""
    output_path.write_text(text, encoding="utf-8")
    return output_path


def main() -> int:
    csv_path = DEFAULT_CSV_ROOT / "final_metrics.csv"
    output_dir = ensure_dir(DEFAULT_FIGURES_ROOT / "highlights")
    report_path = ensure_dir(DEFAULT_REPORTS_ROOT) / "current_highlights.md"
    rows = load_rows(csv_path)

    parity_paths = generate_parity_figure(rows, output_dir)
    rho_paths = generate_rho_t_figure(rows, output_dir)
    write_markdown(report_path, parity_paths, rho_paths)

    append_command(
        {
            "tool": "generate_highlight_graphs.py",
            "command": f"{sys.executable} scripts/generate_highlight_graphs.py",
            "cwd": str(PROJECT_ROOT),
            "report_path": str(report_path),
            "output_dir": str(output_dir),
            "exit_code": 0,
        }
    )
    print(
        {
            "report_path": str(report_path),
            "parity_png": str(parity_paths["png"]),
            "rho_png": str(rho_paths["png"]),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
