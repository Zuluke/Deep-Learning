"""Budget-regret analysis for guarded AlphaQ portfolio selection.

This script complements ``analyze_alphaq_portfolio_budget.py`` by reporting
the T-count regret, in gates, against the available-candidate oracle:

    regret = T_selected - T_oracle.

It evaluates budgets m=1..4 for guarded and unguarded rankings, at both group
and deduplicated-target scope, under LOTO and LOFO folds.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_alphaq_portfolio_budget import (
    PORTFOLIO_OBJECTIVES,
    dedupe_groups,
    evaluate_policy_on_groups,
    loto_rankings,
    portfolio_rows,
)
from scripts.analyze_alphaq_split_select import (
    grouped,
    read_csv,
    train_ready_rows,
    write_csv,
)
from scripts.structural_target import coerce_float

DEFAULT_DATASET = PROJECT_ROOT / "results" / "csv" / "alphaq_objective_selection_dataset.csv"
DEFAULT_SUMMARY = PROJECT_ROOT / "results" / "csv" / "alphaq_budget_regret_summary.csv"
DEFAULT_DETAILS = PROJECT_ROOT / "results" / "csv" / "alphaq_budget_regret_details.csv"
DEFAULT_MISSES = PROJECT_ROOT / "results" / "csv" / "alphaq_budget_regret_misses.csv"
DEFAULT_REPORT = PROJECT_ROOT / "results" / "reports" / "alphaq_budget_regret.md"
DEFAULT_FIGURE = PROJECT_ROOT / "paper" / "cbctq2026" / "fig_budget_regret.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute oracle-recovery and T-count regret by materialization budget."
    )
    parser.add_argument("--dataset-csv", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--detail-csv", type=Path, default=DEFAULT_DETAILS)
    parser.add_argument("--miss-csv", type=Path, default=DEFAULT_MISSES)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--figure-path", type=Path, default=DEFAULT_FIGURE)
    return parser.parse_args()


def tcount_regret(row: dict[str, Any]) -> float | None:
    selected = coerce_float(row.get("final_tcount"))
    oracle = coerce_float(row.get("oracle_tcount"))
    if selected is None or oracle is None:
        return None
    return selected - oracle


def tcount_delta_vs_baseline(row: dict[str, Any]) -> float | None:
    selected = coerce_float(row.get("final_tcount"))
    baseline = coerce_float(row.get("baseline_tcount"))
    if selected is None or baseline is None:
        return None
    return selected - baseline


def summarize_details(
    protocol: str,
    scope: str,
    policy: str,
    budget: int,
    details: list[dict[str, Any]],
) -> dict[str, Any]:
    regrets = [value for row in details if (value := tcount_regret(row)) is not None]
    deltas = [value for row in details if (value := tcount_delta_vs_baseline(row)) is not None]
    return {
        "protocol": protocol,
        "scope": scope,
        "policy": policy,
        "budget": budget,
        "groups": len(details),
        "oracle_t_recovered": sum(bool(row["oracle_t_recovered"]) for row in details),
        "oracle_recovery_fraction": ""
        if not details
        else sum(bool(row["oracle_t_recovered"]) for row in details) / len(details),
        "misses": sum(not bool(row["oracle_t_recovered"]) for row in details),
        "mean_regret": "" if not regrets else statistics.fmean(regrets),
        "median_regret": "" if not regrets else statistics.median(regrets),
        "max_regret": "" if not regrets else max(regrets),
        "wins_vs_baseline": sum(value < 0 for value in deltas),
        "losses_vs_baseline": sum(value > 0 for value in deltas),
        "max_regression_vs_baseline": "" if not deltas else max(deltas),
    }


def evaluate_budget_regret(
    rows: list[dict[str, str]],
    *,
    protocol: str,
    scope: str,
    fold_attr: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rankings, _weights = loto_rankings(rows, fold_attr=fold_attr)
    groups = {key: rankings[key] for key in rankings}
    details_out: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    misses: list[dict[str, Any]] = []
    for guarded in (False, True):
        policy_prefix = "guarded_top" if guarded else "top"
        for budget in range(1, len(PORTFOLIO_OBJECTIVES) + 1):
            policy = f"{policy_prefix}{budget}"
            details = evaluate_policy_on_groups(groups, rankings, budget, guarded=guarded)
            for row in details:
                regret = tcount_regret(row)
                delta = tcount_delta_vs_baseline(row)
                enriched = {
                    "protocol": protocol,
                    "scope": scope,
                    "policy": policy,
                    **row,
                    "tcount_regret": "" if regret is None else regret,
                    "tcount_delta_vs_baseline": "" if delta is None else delta,
                }
                details_out.append(enriched)
                if not bool(row["oracle_t_recovered"]):
                    misses.append(enriched)
            summaries.append(summarize_details(protocol, scope, policy, budget, details))
    return details_out, summaries, misses


def run_all(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    all_details: list[dict[str, Any]] = []
    all_summaries: list[dict[str, Any]] = []
    all_misses: list[dict[str, Any]] = []
    protocols = (("LOTO", "target"), ("LOFO", "functional_family"))
    scopes = (("groups", rows), ("targets", dedupe_groups(rows)))
    for protocol, fold_attr in protocols:
        for scope, scope_rows in scopes:
            details, summaries, misses = evaluate_budget_regret(
                scope_rows,
                protocol=protocol,
                scope=scope,
                fold_attr=fold_attr,
            )
            all_details.extend(details)
            all_summaries.extend(summaries)
            all_misses.extend(misses)
    return all_details, all_summaries, all_misses


def write_figure(path: Path, summaries: list[dict[str, Any]]) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 300,
        }
    )
    colors = {"LOTO": "#0072B2", "LOFO": "#D55E00"}
    fig, axes = plt.subplots(2, 2, figsize=(4.6, 3.2), sharex=True, sharey=True)
    for row_index, scope in enumerate(("groups", "targets")):
        for col_index, protocol in enumerate(("LOTO", "LOFO")):
            ax = axes[row_index][col_index]
            for guarded, linestyle, marker, label in (
                (False, "--", "o", "top-m"),
                (True, "-", "s", "guarded"),
            ):
                prefix = "guarded_top" if guarded else "top"
                items = sorted(
                    (
                        int(row["budget"]),
                        float(row["oracle_recovery_fraction"]),
                    )
                    for row in summaries
                    if row["scope"] == scope
                    and row["protocol"] == protocol
                    and str(row["policy"]).startswith(prefix)
                    and (guarded or not str(row["policy"]).startswith("guarded_top"))
                )
                ax.plot(
                    [item[0] for item in items],
                    [item[1] for item in items],
                    linestyle=linestyle,
                    marker=marker,
                    markersize=3,
                    linewidth=1.1,
                    color=colors[protocol],
                    label=label,
                )
            ax.set_ylim(0.55, 1.02)
            ax.set_xticks(range(1, len(PORTFOLIO_OBJECTIVES) + 1))
            ax.grid(axis="y", alpha=0.25, linewidth=0.5)
            ax.text(
                0.03,
                0.12,
                f"{scope}, {protocol}",
                transform=ax.transAxes,
                fontsize=7,
                ha="left",
                va="bottom",
            )
            if row_index == 1:
                ax.set_xlabel("budget m")
            if col_index == 0:
                ax.set_ylabel("oracle recovery")
            if row_index == 0 and col_index == 0:
                ax.legend(frameon=False, loc="lower right", handlelength=1.8)
    fig.tight_layout(pad=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def write_report(
    path: Path,
    summaries: list[dict[str, Any]],
    misses: list[dict[str, Any]],
    *,
    dataset_csv: Path,
    figure_path: Path,
) -> None:
    lines = [
        "# AlphaQ Budget-Regret Analysis",
        "",
        f"Dataset: `{dataset_csv}`.",
        f"Figure: `{figure_path}`.",
        "",
        "Regret is measured as `T_selected - T_oracle`, where the oracle is the "
        "best available candidate within the deployed objective portfolio for that group.",
        "",
        "| protocol | scope | policy | m | oracle | misses | max regret | wins/losses vs baseline | max regression |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {protocol} | {scope} | {policy} | {budget} | {oracle_t_recovered}/{groups} | {misses} | {max_regret} | {wins_vs_baseline}/{losses_vs_baseline} | {max_regression_vs_baseline} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Misses",
            "",
            "| protocol | scope | policy | source split | target | selected | oracle | regret |",
            "|---|---|---|---|---|---|---|---:|",
        ]
    )
    for row in misses:
        lines.append(
            "| {protocol} | {scope} | {policy} | {source_split} | {target} | {final_objective} ({final_tcount}) | oracle ({oracle_tcount}) | {tcount_regret} |".format(
                **row
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


SUMMARY_FIELDS = [
    "protocol",
    "scope",
    "policy",
    "budget",
    "groups",
    "oracle_t_recovered",
    "oracle_recovery_fraction",
    "misses",
    "mean_regret",
    "median_regret",
    "max_regret",
    "wins_vs_baseline",
    "losses_vs_baseline",
    "max_regression_vs_baseline",
]

DETAIL_FIELDS = [
    "protocol",
    "scope",
    "policy",
    "source_split",
    "target",
    "budget",
    "available_candidates",
    "materialized_objectives",
    "final_objective",
    "final_tcount",
    "oracle_tcount",
    "baseline_tcount",
    "oracle_t_recovered",
    "tcount_regret",
    "tcount_delta_vs_baseline",
]

MISS_FIELDS = DETAIL_FIELDS


def main() -> int:
    args = parse_args()
    rows = portfolio_rows(train_ready_rows(read_csv(args.dataset_csv)))
    details, summaries, misses = run_all(rows)
    write_csv(args.detail_csv, details, DETAIL_FIELDS)
    write_csv(args.summary_csv, summaries, SUMMARY_FIELDS)
    write_csv(args.miss_csv, misses, MISS_FIELDS)
    write_figure(args.figure_path, summaries)
    write_report(
        args.report_path,
        summaries,
        misses,
        dataset_csv=args.dataset_csv,
        figure_path=args.figure_path,
    )
    print(f"Wrote {args.summary_csv}")
    print(f"Wrote {args.detail_csv}")
    print(f"Wrote {args.miss_csv}")
    print(f"Wrote {args.report_path}")
    print(f"Wrote {args.figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
