"""Generate the column-width T-count reduction figure for the CBCTQ paper.

Same data as plot_alphaq_claim_onepager.py, sized for a two-column A4 paper:
single-column width, fonts that stay legible at ~3.4in, no title (the LaTeX
caption carries it).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DETAILS_CSV = PROJECT_ROOT / "results" / "csv" / "alphaq_portfolio_budget_details.csv"
OUT_PNG = PROJECT_ROOT / "paper" / "cbctq2026" / "fig_tcount_reductions.png"

PROVEN = {
    "barenco_tof_4", "hamming_weight_n7", "mod_mult_55",
    "nc_tof_4", "gf_2pow4_mult", "cuccaro_adder_n4",
    "vbe_adder_3",
}

OBJECTIVE_SHORT = {
    "factor_count": "baseline",
    "factor_count_pair_cap": "pair cap",
    "mixed_pair": "mixed pair",
    "frontier_pair": "frontier",
    "depth_guarded_mixed_pair": "depth guarded",
    "t_preserving_frontier_pair": "T-preserving",
}


def main() -> None:
    df = pd.read_csv(DETAILS_CSV)
    rows = df[(df["policy"] == "guarded_top2") & (df["scope"] == "targets")].copy()
    rows["t_reduction"] = rows["baseline_tcount"] - rows["final_tcount"]
    improved = rows[rows["t_reduction"] > 0].sort_values(
        ["t_reduction", "target"],
        ascending=[False, True],
    )
    unchanged = rows[rows["t_reduction"] <= 0]

    labels = [target.replace("_", " ") for target in improved["target"]]
    reductions = improved["t_reduction"].tolist()
    labels.append(f"{len(unchanged)} further targets")
    reductions.append(0.35)

    n = len(labels)
    fig, ax = plt.subplots(figsize=(4.6, 2.45))

    bar_colors = ["#2e7d32"] * len(improved) + ["#bdbdbd"]
    ax.barh(range(n), reductions, color=bar_colors, height=0.58, zorder=2)

    for i, row in enumerate(improved.itertuples(index=False)):
        mark = " ✓" if row.target in PROVEN else ""
        short = OBJECTIVE_SHORT.get(str(row.final_objective), str(row.final_objective))
        label = f"{int(row.baseline_tcount)}→{int(row.final_tcount)} ({short}){mark}"
        ax.text(row.t_reduction + 0.45, i, label, va="center", fontsize=7.6, color="#1a5c36")
    ax.text(
        0.55,
        n - 1,
        "baseline kept, no regression",
        va="center",
        fontsize=7.4,
        color="#555555",
    )

    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=7.8)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_xlabel("T-count reduction vs baseline", fontsize=9)
    ax.set_xlim(0, max(reductions) * 1.62)
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", alpha=0.4, zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=0.3)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
