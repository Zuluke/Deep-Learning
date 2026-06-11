"""Audit whether the AlphaQ selector needs more than rank information.

The main guarded policy ranks candidates with the five cheap AlphaQ
factor-structure features. This audit compares it with two simpler guarded
top-2 policies under the same leave-one-target-out protocol:

- the learned five-feature scorer used by the portfolio-budget analysis;
- a learned one-feature scorer using only candidate ``factor_count``;
- a deterministic alphabetical ranking control.

The report explicitly lists groups where the materialized sets differ between
the five-feature and rank-only policies.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")

from scripts.analyze_alphaq_portfolio_budget import (
    PORTFOLIO_OBJECTIVES,
    SELECTOR_LEVELS,
    best_materialized,
    loto_rankings,
    materialized_set,
    portfolio_rows,
    sign_test_one_sided,
)
from scripts.analyze_alphaq_split_select import (
    ALPHAQ_FEATURES,
    BASELINE_OBJECTIVE,
    grouped,
    normalize_rows,
    oracle_row,
    read_csv,
    target_folds,
    train_linear_weights,
    train_ready_rows,
    write_csv,
)
from scripts.analyze_alphaq_portfolio_budget import ranked_candidates
from scripts.structural_target import coerce_float

DEFAULT_DATASET = PROJECT_ROOT / "results" / "csv" / "alphaq_objective_selection_dataset.csv"
DEFAULT_SUMMARY = PROJECT_ROOT / "results" / "csv" / "alphaq_rank_audit_summary.csv"
DEFAULT_DETAILS = PROJECT_ROOT / "results" / "csv" / "alphaq_rank_audit_details.csv"
DEFAULT_DIFFS = PROJECT_ROOT / "results" / "csv" / "alphaq_rank_audit_diffs.csv"
DEFAULT_REPORT = PROJECT_ROOT / "results" / "reports" / "alphaq_rank_audit.md"

RANK_FEATURES = ("factor_count",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare five-feature, rank-only, and alphabetical guarded top-2 policies."
    )
    parser.add_argument("--dataset-csv", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--detail-csv", type=Path, default=DEFAULT_DETAILS)
    parser.add_argument("--diff-csv", type=Path, default=DEFAULT_DIFFS)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def rank_only_loto_rankings(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    normalized = normalize_rows(rows, RANK_FEATURES)
    normalized_by_group = grouped(normalized)
    targets = target_folds(rows)
    rankings: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for holdout in targets:
        train_targets = set(targets) - {holdout}
        weights = train_linear_weights(
            rows=normalized,
            train_targets=train_targets,
            features=RANK_FEATURES,
            levels=SELECTOR_LEVELS,
        )
        for key, items in normalized_by_group.items():
            if key[1] != holdout:
                continue
            rankings[key] = ranked_candidates(items, RANK_FEATURES, weights)
    return rankings


def alphabetical_rankings(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    return {
        key: sorted(items, key=lambda row: row["objective_variant"])
        for key, items in grouped(rows).items()
    }


def evaluate_rankings(
    policy: str,
    rankings: dict[tuple[str, str], list[dict[str, Any]]],
    groups: dict[tuple[str, str], list[dict[str, str]]],
) -> list[dict[str, Any]]:
    details = []
    for key in sorted(rankings):
        ranking = rankings[key]
        chosen = materialized_set(ranking, 2, guarded=True)
        final = best_materialized(chosen)
        oracle = oracle_row(groups[key])
        baseline = next(
            row for row in groups[key] if row["objective_variant"] == BASELINE_OBJECTIVE
        )
        final_t = coerce_float(final.get("best_beam_tcount"))
        oracle_t = coerce_float(oracle.get("best_beam_tcount"))
        baseline_t = coerce_float(baseline.get("best_beam_tcount"))
        details.append(
            {
                "policy": policy,
                "source_split": key[0],
                "target": key[1],
                "available_candidates": len(groups[key]),
                "materialized_objectives": ",".join(row["objective_variant"] for row in chosen),
                "final_objective": final["objective_variant"],
                "final_tcount": final.get("best_beam_tcount"),
                "oracle_objective": oracle["objective_variant"],
                "oracle_tcount": oracle.get("best_beam_tcount"),
                "baseline_tcount": baseline.get("best_beam_tcount"),
                "oracle_t_recovered": final_t is not None
                and oracle_t is not None
                and final_t == oracle_t,
                "tcount_delta_vs_baseline": ""
                if final_t is None or baseline_t is None
                else final_t - baseline_t,
                "tcount_regret": ""
                if final_t is None or oracle_t is None
                else final_t - oracle_t,
            }
        )
    return details


def summarize(policy: str, details: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = [
        value
        for row in details
        if (value := coerce_float(row.get("tcount_delta_vs_baseline"))) is not None
    ]
    wins = sum(value < 0 for value in deltas)
    losses = sum(value > 0 for value in deltas)
    return {
        "policy": policy,
        "groups": len(details),
        "oracle_t_recovered": sum(bool(row["oracle_t_recovered"]) for row in details),
        "tcount_wins_vs_baseline": wins,
        "tcount_losses_vs_baseline": losses,
        "sign_test_p_one_sided": "" if sign_test_one_sided(wins, losses) is None else sign_test_one_sided(wins, losses),
        "max_tcount_regret": max(
            (coerce_float(row.get("tcount_regret")) or 0.0 for row in details),
            default=0.0,
        ),
    }


def diff_rows(
    learned_details: list[dict[str, Any]],
    rank_details: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_key = {
        (row["source_split"], row["target"]): row
        for row in rank_details
    }
    diffs = []
    for learned in learned_details:
        key = (learned["source_split"], learned["target"])
        rank = by_key[key]
        if learned["materialized_objectives"] == rank["materialized_objectives"]:
            continue
        diffs.append(
            {
                "source_split": key[0],
                "target": key[1],
                "learned_materialized": learned["materialized_objectives"],
                "rank_only_materialized": rank["materialized_objectives"],
                "learned_final_objective": learned["final_objective"],
                "rank_only_final_objective": rank["final_objective"],
                "learned_final_tcount": learned["final_tcount"],
                "rank_only_final_tcount": rank["final_tcount"],
                "oracle_objective": learned["oracle_objective"],
                "oracle_tcount": learned["oracle_tcount"],
                "baseline_tcount": learned["baseline_tcount"],
                "learned_regret": learned["tcount_regret"],
                "rank_only_regret": rank["tcount_regret"],
            }
        )
    return diffs


def write_report(
    path: Path,
    summary_rows: list[dict[str, Any]],
    diffs: list[dict[str, Any]],
    *,
    dataset_csv: Path,
) -> None:
    lines = [
        "# AlphaQ Rank Audit",
        "",
        f"Dataset: `{dataset_csv}`.",
        "",
        "Policy: guarded top-2, leave-one-target-out. The learned policy uses "
        f"{len(ALPHAQ_FEATURES)} factor-structure features over the deployed "
        f"{len(PORTFOLIO_OBJECTIVES)}-objective portfolio; the rank-only policy "
        "uses only candidate `factor_count`; the alphabetical control ignores features.",
        "",
        "| policy | groups | oracle T recovered | W/L | sign p | max regret |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {policy} | {groups} | {oracle} | {wins}/{losses} | {p} | {regret} |".format(
                policy=row["policy"],
                groups=row["groups"],
                oracle=row["oracle_t_recovered"],
                wins=row["tcount_wins_vs_baseline"],
                losses=row["tcount_losses_vs_baseline"],
                p=row["sign_test_p_one_sided"],
                regret=row["max_tcount_regret"],
            )
        )
    lines.extend(
        [
            "",
            "## Learned vs Rank-Only Materialization Differences",
            "",
        ]
    )
    if not diffs:
        lines.append("No groups differ: the learned five-feature and rank-only policies materialize the same objective sets on this dataset.")
    else:
        lines.extend(
            [
                "| source split | target | learned materialized | rank-only materialized | learned final | rank-only final | oracle |",
                "|---|---|---|---|---|---|---|",
            ]
        )
        for row in diffs:
            lines.append(
                "| {source_split} | {target} | {learned_materialized} | {rank_only_materialized} | {learned_final_objective} ({learned_final_tcount}) | {rank_only_final_objective} ({rank_only_final_tcount}) | {oracle_objective} ({oracle_tcount}) |".format(
                    **row
                )
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


SUMMARY_FIELDS = [
    "policy",
    "groups",
    "oracle_t_recovered",
    "tcount_wins_vs_baseline",
    "tcount_losses_vs_baseline",
    "sign_test_p_one_sided",
    "max_tcount_regret",
]

DETAIL_FIELDS = [
    "policy",
    "source_split",
    "target",
    "available_candidates",
    "materialized_objectives",
    "final_objective",
    "final_tcount",
    "oracle_objective",
    "oracle_tcount",
    "baseline_tcount",
    "oracle_t_recovered",
    "tcount_delta_vs_baseline",
    "tcount_regret",
]

DIFF_FIELDS = [
    "source_split",
    "target",
    "learned_materialized",
    "rank_only_materialized",
    "learned_final_objective",
    "rank_only_final_objective",
    "learned_final_tcount",
    "rank_only_final_tcount",
    "oracle_objective",
    "oracle_tcount",
    "baseline_tcount",
    "learned_regret",
    "rank_only_regret",
]


def main() -> int:
    args = parse_args()
    rows = portfolio_rows(train_ready_rows(read_csv(args.dataset_csv)))
    groups = grouped(rows)
    learned_rankings, _weights = loto_rankings(rows, fold_attr="target")
    rank_rankings = rank_only_loto_rankings(rows)
    alpha_rankings = alphabetical_rankings(rows)

    detail_rows = []
    policy_details = {}
    for policy, rankings in (
        ("learned_five_feature", learned_rankings),
        ("rank_only_factor_count", rank_rankings),
        ("alphabetical_control", alpha_rankings),
    ):
        details = evaluate_rankings(policy, rankings, groups)
        policy_details[policy] = details
        detail_rows.extend(details)

    summary_rows = [
        summarize(policy, policy_details[policy])
        for policy in ("learned_five_feature", "rank_only_factor_count", "alphabetical_control")
    ]
    diffs = diff_rows(
        policy_details["learned_five_feature"],
        policy_details["rank_only_factor_count"],
    )

    write_csv(args.summary_csv, summary_rows, SUMMARY_FIELDS)
    write_csv(args.detail_csv, detail_rows, DETAIL_FIELDS)
    write_csv(args.diff_csv, diffs, DIFF_FIELDS)
    write_report(args.report_path, summary_rows, diffs, dataset_csv=args.dataset_csv)
    print(f"Wrote {args.summary_csv}")
    print(f"Wrote {args.detail_csv}")
    print(f"Wrote {args.diff_csv}")
    print(f"Wrote {args.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
