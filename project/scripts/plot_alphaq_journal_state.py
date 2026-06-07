from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_GATES = PROJECT_ROOT / "results" / "csv" / "alphaq_journal_evidence_gates.csv"
DEFAULT_EXTERNAL = PROJECT_ROOT / "results" / "csv" / "alphaq_external_runs_consolidated.csv"
DEFAULT_SPLIT_SELECT = PROJECT_ROOT / "results" / "csv" / "alphaq_split_select_summary.csv"
DEFAULT_FIGURE = PROJECT_ROOT / "results" / "figures" / "alphaq_journal_current_state.png"
DEFAULT_REPORT = PROJECT_ROOT / "results" / "reports" / "alphaq_journal_current_state.md"

OBJECTIVE_LABELS = {
    "factor_count": "Factor-count",
    "factor_count_pair_cap": "Pair-cap",
    "mixed_pair": "Mixed-pair",
}
STATUS_COLORS = {
    "pass": "#2F9E44",
    "partial": "#E9A126",
    "not-yet-journal-ready": "#D9480F",
    "planned": "#5C677D",
    "fail": "#C92A2A",
    "missing": "#868E96",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the current AlphaQ journal evidence state.")
    parser.add_argument("--gates-csv", type=Path, default=DEFAULT_GATES)
    parser.add_argument("--external-csv", type=Path, default=DEFAULT_EXTERNAL)
    parser.add_argument("--split-select-csv", type=Path, default=DEFAULT_SPLIT_SELECT)
    parser.add_argument("--figure-path", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def target_status(external: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, group in external.groupby("target", sort=True):
        statuses = group.set_index("objective_variant")["best_status"].to_dict()
        ok_count = sum(status == "ok" for status in statuses.values())
        if ok_count == 3:
            status = "complete"
        elif ok_count > 0:
            status = "partial"
        else:
            status = "failed"
        rows.append({"target": target, "status": status, "ok_objectives": ok_count})
    return pd.DataFrame(rows)


def objective_effects(external: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, group in external.groupby("target", sort=True):
        ok = group[group["best_status"] == "ok"].copy()
        factor = ok[ok["objective_variant"] == "factor_count"]
        if ok.empty or factor.empty:
            continue
        factor_row = factor.iloc[0]
        best = ok.sort_values(
            by=["best_tcount", "best_beam_qasm_depth", "objective_variant"],
            ascending=[True, True, True],
            kind="mergesort",
        ).iloc[0]
        factor_t = safe_float(factor_row["best_tcount"])
        factor_qasm = safe_float(factor_row["best_beam_qasm_depth"])
        best_t = safe_float(best["best_tcount"])
        best_qasm = safe_float(best["best_beam_qasm_depth"])
        if factor_t <= 0 or factor_qasm <= 0 or not math.isfinite(best_t) or not math.isfinite(best_qasm):
            continue
        rows.append(
            {
                "target": target,
                "status": target_status(group).iloc[0]["status"],
                "best_objective": best["objective_variant"],
                "factor_tcount": factor_t,
                "best_tcount": best_t,
                "tcount_ratio": best_t / factor_t,
                "factor_qasm_depth": factor_qasm,
                "best_qasm_depth": best_qasm,
                "qasm_ratio": best_qasm / factor_qasm,
            }
        )
    return pd.DataFrame(rows)


def safe_float(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number


def plot_state(gates: pd.DataFrame, external: pd.DataFrame, split_select: pd.DataFrame, figure_path: Path) -> None:
    effects = objective_effects(external)
    status_df = target_status(external)
    fig, ax = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    plot_effects(ax, effects)
    complete = int((status_df["status"] == "complete").sum())
    partial = ", ".join(status_df.loc[status_df["status"] == "partial", "target"])
    failed = ", ".join(status_df.loc[status_df["status"] == "failed", "target"])
    subtitle_parts = [f"{complete} complete external targets"]
    if partial:
        subtitle_parts.append(f"partial: {partial}")
    if failed:
        subtitle_parts.append(f"failed: {failed}")
    ax.set_title("Best AlphaQ Split-Select objective vs factor-count baseline", fontsize=13, fontweight="bold", pad=16)
    ax.text(
        0.0,
        1.02,
        " | ".join(subtitle_parts),
        transform=ax.transAxes,
        fontsize=9,
        color="#495057",
        va="bottom",
    )
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_gates(ax: plt.Axes, gates: pd.DataFrame) -> None:
    display = gates[gates["gate"] != "next_decisive_battery"].copy()
    display["label"] = display["gate"].map(short_gate_label)
    display = display.iloc[::-1]
    colors = [STATUS_COLORS.get(status, "#868E96") for status in display["status"]]
    ax.barh(display["label"], display["score"], color=colors, edgecolor="#343A40", linewidth=0.4)
    ax.set_xlim(0, 1.05)
    ax.axvline(1.0, color="#212529", linewidth=0.8, alpha=0.35)
    ax.set_xlabel("gate score")
    ax.set_title("Evidence gates")
    ax.grid(axis="x", alpha=0.25)
    for y, (_, row) in enumerate(display.iterrows()):
        ax.text(min(row["score"] + 0.03, 1.02), y, row["status"], va="center", fontsize=8)


def plot_policy(ax: plt.Axes, split_select: pd.DataFrame) -> None:
    policies = split_select[
        split_select["policy"].isin(["baseline_factor_count", "split_select_linear_alphaq", "oracle_posthoc"])
    ].copy()
    policies["label"] = policies["policy"].map(
        {
            "baseline_factor_count": "Factor-count baseline",
            "split_select_linear_alphaq": "Split-Select",
            "oracle_posthoc": "Oracle",
        }
    )
    order = ["Factor-count baseline", "Split-Select", "Oracle"]
    policies["label"] = pd.Categorical(policies["label"], categories=order, ordered=True)
    policies = policies.sort_values("label")

    x = range(len(policies))
    width = 0.26
    ax.bar([i - width for i in x], policies["exact_oracle_matches"], width=width, label="oracle matches", color="#4263EB")
    ax.bar([i for i in x], policies["tcount_wins_vs_baseline"], width=width, label="T wins", color="#2F9E44")
    ax.bar([i + width for i in x], policies["qasm_wins_vs_baseline"], width=width, label="QASM wins", color="#E67700")
    ax.set_xticks(list(x))
    ax.set_xticklabels(policies["label"], rotation=0)
    ax.set_ylabel("count over evaluated groups")
    ax.set_title("Policy behavior")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=3, fontsize=8, loc="upper left")
    for container in ax.containers:
        ax.bar_label(container, fontsize=8, padding=2)


def plot_external_status(ax: plt.Axes, status_df: pd.DataFrame) -> None:
    counts = status_df["status"].value_counts().reindex(["complete", "partial", "failed"], fill_value=0)
    colors = ["#2F9E44", "#E9A126", "#C92A2A"]
    ax.bar(counts.index, counts.values, color=colors, edgecolor="#343A40", linewidth=0.4)
    ax.set_ylabel("targets")
    ax.set_title("External target coverage")
    ax.grid(axis="y", alpha=0.25)
    for idx, value in enumerate(counts.values):
        ax.text(idx, value + 0.05, str(value), ha="center", va="bottom", fontsize=9)
    partial = ", ".join(status_df.loc[status_df["status"] == "partial", "target"])
    failed = ", ".join(status_df.loc[status_df["status"] == "failed", "target"])
    note = []
    if partial:
        note.append(f"partial: {partial}")
    if failed:
        note.append(f"failed: {failed}")
    if note:
        ax.text(0.02, -0.24, "\n".join(note), transform=ax.transAxes, fontsize=8, va="top")


def plot_effects(ax: plt.Axes, effects: pd.DataFrame) -> None:
    if effects.empty:
        ax.text(0.5, 0.5, "No external effects available", ha="center", va="center")
        return
    effects = effects.sort_values(["tcount_ratio", "qasm_ratio", "target"], ascending=[True, True, True])
    y = range(len(effects))
    colors = [objective_color(obj) for obj in effects["best_objective"]]
    ax.barh(list(y), effects["tcount_ratio"], color=colors, edgecolor="#343A40", linewidth=0.4, alpha=0.9)
    ax.scatter(effects["qasm_ratio"], list(y), marker="D", s=35, color="#212529", label="QASM depth ratio")
    ax.axvline(1.0, color="#212529", linewidth=0.9, linestyle="--", alpha=0.75)
    ax.set_yticks(list(y))
    ax.set_yticklabels(effects["target"])
    ax.set_xlabel("ratio vs factor-count baseline")
    ax.set_title("Best external objective per target")
    ax.grid(axis="x", alpha=0.25)
    xmax = max(1.22, max(effects["tcount_ratio"].max(), effects["qasm_ratio"].max()) + 0.12)
    ax.set_xlim(0, xmax)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=objective_color("factor_count"), label="Factor-count baseline"),
        plt.Rectangle((0, 0), 1, 1, color=objective_color("factor_count_pair_cap"), label="Pair-cap"),
        plt.Rectangle((0, 0), 1, 1, color=objective_color("mixed_pair"), label="Mixed-pair"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="#212529", label="diamond: QASM depth ratio", markersize=6),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="lower right", ncols=2)


def objective_color(objective: str) -> str:
    return {
        "factor_count": "#868E96",
        "factor_count_pair_cap": "#339AF0",
        "mixed_pair": "#E67700",
    }.get(objective, "#ADB5BD")


def short_gate_label(gate: str) -> str:
    return {
        "selector_loto": "selector",
        "dataset_scale_and_label_diversity": "dataset",
        "external_generalization_coverage": "coverage",
        "external_nonbaseline_effect": "nonbaseline wins",
        "formal_verification_coverage": "verification",
        "depth_control": "depth control",
        "overall_journal_readiness": "overall",
    }.get(gate, gate)


def write_report(report_path: Path, gates: pd.DataFrame, external: pd.DataFrame, split_select: pd.DataFrame, figure_path: Path) -> None:
    effects = objective_effects(external)
    status_df = target_status(external)
    split = split_select[split_select["policy"] == "split_select_linear_alphaq"].iloc[0]
    overall = gates[gates["gate"] == "overall_journal_readiness"].iloc[0]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# AlphaQ Journal Current State",
        "",
        f"Figure: `{figure_path}`.",
        "",
        f"Decision: `{overall['status']}`.",
        "",
        "## Main Readout",
        "",
        (
            f"Split-Select currently matches the post-hoc oracle in {int(split['exact_oracle_matches'])}/"
            f"{int(split['ok_groups'])} evaluated groups, versus 8/{int(split['ok_groups'])} for the "
            "factor-count-only baseline."
        ),
        "",
        (
            f"The external battery has {int((status_df['status'] == 'complete').sum())} complete targets, "
            f"{int((status_df['status'] == 'partial').sum())} partial target, and "
            f"{int((status_df['status'] == 'failed').sum())} failed targets."
        ),
        "",
        "## Best External Objective Relative to Factor-count",
        "",
        "| target | best objective | T-count ratio | QASM-depth ratio |",
        "|---|---|---:|---:|",
    ]
    for row in effects.sort_values(["tcount_ratio", "qasm_ratio", "target"]).itertuples(index=False):
        lines.append(
            f"| {row.target} | {OBJECTIVE_LABELS.get(row.best_objective, row.best_objective)} | "
            f"{row.tcount_ratio:.3f} | {row.qasm_ratio:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Gate Status",
            "",
            "| gate | status | score |",
            "|---|---|---:|",
        ]
    )
    for row in gates.itertuples(index=False):
        lines.append(f"| {row.gate} | {row.status} | {row.score:.3f} |")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    gates = read_csv(args.gates_csv)
    external = read_csv(args.external_csv)
    split_select = read_csv(args.split_select_csv)
    plot_state(gates, external, split_select, args.figure_path)
    write_report(args.report_path, gates, external, split_select, args.figure_path)
    print(f"Wrote {args.figure_path}")
    print(f"Wrote {args.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
