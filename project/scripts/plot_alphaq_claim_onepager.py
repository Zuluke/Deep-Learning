"""Generate the AlphaQ claim one-pager bar chart.

Shows T-count reduction per benchmark circuit under guarded top-2 portfolio
selection vs the factor_count baseline. Only improved circuits get a label.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DETAILS_CSV = PROJECT_ROOT / "results" / "csv" / "alphaq_portfolio_budget_details.csv"
OUT_PNG = PROJECT_ROOT / "results" / "figures" / "alphaq_claim_onepager.png"

PROVEN = {
    "barenco_tof_4", "hamming_weight_n7", "mod_mult_55",
    "nc_tof_4", "gf_2pow4_mult", "cuccaro_adder_n4",
    "vbe_adder_3",
}

TARGET_LABELS = {
    "barenco_tof_4": "barenco_tof_4",
    "vbe_adder_3": "vbe_adder_3",
    "hamming_weight_n7": "hamming_weight_n7",
    "nc_tof_4": "nc_tof_4",
    "gf_2pow4_mult": "gf_2pow4_mult",
    "mod_mult_55": "mod_mult_55",
    "cuccaro_adder_n4": "cuccaro_adder_n4",
    "gf_2pow3_mult": "gf_2pow3_mult",
    "nc_tof_3": "nc_tof_3",
    "cuccaro_adder_n3": "cuccaro_adder_n3",
    "nc_tof_5": "nc_tof_5",
    "gf_2pow5_mult": "gf_2pow5_mult",
    "hamming_weight_n6": "hamming_weight_n6",
    "hamming_weight_n5": "hamming_weight_n5",
    "hamming_weight_n4": "hamming_weight_n4",
    "mod_5_4": "mod_5_4",
    "barenco_tof_3": "barenco_tof_3",
    "gf_2pow2_mult": "gf_2pow2_mult",
}


def main() -> None:
    df = pd.read_csv(DETAILS_CSV)
    rows = df[(df["policy"] == "guarded_top2") & (df["scope"] == "targets")].copy()

    # Compute T-count reduction (positive = improvement)
    rows["t_reduction"] = rows["baseline_tcount"] - rows["final_tcount"]
    rows["t_ratio"] = rows["final_tcount"] / rows["baseline_tcount"]

    # Sort: largest improvement first, then alphabetical
    rows = rows.sort_values(["t_reduction", "target"], ascending=[False, True])

    targets = rows["target"].tolist()
    reductions = rows["t_reduction"].tolist()
    baselines = rows["baseline_tcount"].tolist()
    finals = rows["final_tcount"].tolist()
    objectives = rows["final_objective"].tolist()

    n = len(targets)
    fig, ax = plt.subplots(figsize=(9, 0.48 * n + 1.2))

    bar_colors = ["#2e7d32" if r > 0 else "#cccccc" for r in reductions]

    bars = ax.barh(
        range(n),
        [max(r, 0) for r in reductions],
        color=bar_colors,
        height=0.65,
        zorder=2,
    )

    for i, (r, tgt, bl, fin, obj) in enumerate(
        zip(reductions, targets, baselines, finals, objectives)
    ):
        if r > 0:
            proven_mark = "  ✓" if tgt in PROVEN else ""
            label = f"{int(bl)} → {int(fin)} T  ({obj}){proven_mark}"
            ax.text(r + 0.4, i, label, va="center", fontsize=8.5, color="#1a5c36")

    ax.set_yticks(range(n))
    ax.set_yticklabels(targets, fontsize=9)
    ax.set_xlabel("T-count reduction vs baseline", fontsize=10)
    ax.set_title("Adaptive objective selection for AlphaTensor-Quantum", fontsize=12, pad=10)
    ax.set_xlim(0, max(reductions) * 1.55 if max(reductions) > 0 else 5)
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", alpha=0.4, zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
