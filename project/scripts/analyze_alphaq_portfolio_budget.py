"""Evaluate guarded objective-portfolio selection under a materialization budget.

This is the algorithm-level evaluation for the AlphaQ Split-Select line. The
method under test ("guarded portfolio selection") is:

1. Generate tensor decompositions under the K deployed AlphaQ objectives
   (the portfolio).
2. Rank the candidate decompositions with the cheap factor-structure linear
   selector trained leave-one-target-out (no ZX or materialization features).
3. Materialize only ``m`` candidates (the budget). The guarded variant always
   spends one slot of the budget on the ``factor_count`` baseline candidate,
   so for any ``m >= 2`` the final circuit is never worse than the baseline
   pipeline in T-count by construction.
4. Keep the best materialized circuit by (T-count, QASM depth, primary
   non-Clifford depth ratio).

The evaluation reports, per budget ``m``:

- oracle T-count recovery (final T-count equals the post-hoc best over all
  portfolio candidates);
- wins/losses vs the ``factor_count`` baseline with an exact one-sided sign
  test;
- bootstrap confidence intervals for the median T-count ratio vs baseline;
- a random-ranking permutation control quantifying how much of the recovery
  is due to the learned selector rather than the budget itself.

Group-level rows treat each (source_split, target) pair as one observation;
the deduplicated target-level rows keep one group per target to avoid
pseudo-replication of targets that were run under several batteries.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_alphaq_split_select import (
    ALPHAQ_FEATURES,
    BASELINE_OBJECTIVE,
    grouped,
    inf_if_missing,
    linear_score,
    median_metric,
    normalize_rows,
    oracle_key,
    oracle_row,
    read_csv,
    target_folds,
    train_linear_weights,
    train_ready_rows,
    write_csv,
)
from scripts.run_best_objective_beam_ablation import safe_ratio
from scripts.structural_target import coerce_float

DEFAULT_DATASET = PROJECT_ROOT / "results" / "csv" / "alphaq_objective_selection_dataset.csv"
DEFAULT_SUMMARY = PROJECT_ROOT / "results" / "csv" / "alphaq_portfolio_budget_summary.csv"
DEFAULT_DETAILS = PROJECT_ROOT / "results" / "csv" / "alphaq_portfolio_budget_details.csv"
DEFAULT_DEDUPE_AUDIT = PROJECT_ROOT / "results" / "csv" / "alphaq_portfolio_budget_dedupe_audit.csv"
DEFAULT_REPORT = PROJECT_ROOT / "results" / "reports" / "alphaq_portfolio_budget.md"
DEFAULT_FIGURE = PROJECT_ROOT / "results" / "figures" / "alphaq_portfolio_budget.png"

# The deployed evaluation portfolio: objectives that have actually been run
# end-to-end in the internal and external batteries.
PORTFOLIO_OBJECTIVES = (
    "factor_count",
    "factor_count_pair_cap",
    "mixed_pair",
    "frontier_pair",
    "depth_guarded_mixed_pair",
    "t_preserving_frontier_pair",
)
SELECTOR_LEVELS = (-2, -1, 0, 1, 2)
BOOTSTRAP_SAMPLES = 10_000
PERMUTATION_SAMPLES = 2_000
RANDOM_SEED = 20260609


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate guarded AlphaQ objective-portfolio selection under a "
            "materialization budget, leave-one-target-out."
        )
    )
    parser.add_argument("--dataset-csv", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--detail-csv", type=Path, default=DEFAULT_DETAILS)
    parser.add_argument("--dedupe-audit-csv", type=Path, default=DEFAULT_DEDUPE_AUDIT)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--figure-path", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--permutation-samples", type=int, default=PERMUTATION_SAMPLES)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--fold-attr",
        choices=("target", "functional_family"),
        default="target",
        help=(
            "Cross-validation unit: leave-one-target-out (default) or "
            "leave-one-functional-family-out for a stricter generalization test."
        ),
    )
    return parser.parse_args()


def portfolio_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row["objective_variant"] in PORTFOLIO_OBJECTIVES]


def dedupe_groups(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Keep one group per target: most ok objectives, external preferred."""
    by_target: dict[str, list[tuple[tuple[Any, ...], tuple[str, str], list[dict[str, str]]]]] = {}
    for key, items in grouped(rows).items():
        preference = (
            len(items),
            key[0].startswith("external"),
            key[0],
        )
        by_target.setdefault(key[1], []).append((preference, key, items))
    kept: list[dict[str, str]] = []
    for target in sorted(by_target):
        _preference, _key, items = max(by_target[target], key=lambda entry: entry[0])
        kept.extend(items)
    return kept


def ranked_candidates(
    items: list[dict[str, Any]],
    features: tuple[str, ...],
    weights: tuple[int, ...],
) -> list[dict[str, Any]]:
    return sorted(
        items,
        key=lambda row: (
            linear_score(row, features, weights),
            row["objective_variant"],
        ),
    )


def materialized_set(
    ranking: Sequence[dict[str, Any]],
    budget: int,
    *,
    guarded: bool,
) -> list[dict[str, Any]]:
    if budget <= 0:
        return []
    if not guarded:
        return list(ranking[:budget])
    baseline = next(
        (row for row in ranking if row["objective_variant"] == BASELINE_OBJECTIVE),
        None,
    )
    if baseline is None:
        raise ValueError(
            "Guarded portfolio selection requires a factor_count baseline candidate."
        )
    chosen: list[dict[str, Any]] = [baseline]
    for row in ranking:
        if len(chosen) >= budget:
            break
        if row is baseline:
            continue
        chosen.append(row)
    return chosen


def best_materialized(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return min(rows, key=oracle_key)


def sign_test_one_sided(wins: int, losses: int) -> float | None:
    """Exact one-sided binomial sign test, H1: wins more likely than losses."""
    n = wins + losses
    if n == 0:
        return None
    return sum(math.comb(n, k) for k in range(wins, n + 1)) / 2.0**n


def bootstrap_median_ci(
    values: Sequence[float],
    rng: random.Random,
    samples: int,
    *,
    level: float = 0.95,
) -> tuple[float, float] | None:
    if not values:
        return None
    medians = []
    for _ in range(samples):
        resample = [rng.choice(values) for _ in values]
        medians.append(sorted(resample)[len(resample) // 2])
    medians.sort()
    lower = medians[int((1 - level) / 2 * (samples - 1))]
    upper = medians[int((1 + level) / 2 * (samples - 1))]
    return lower, upper


def evaluate_policy_on_groups(
    groups: dict[tuple[str, str], list[dict[str, Any]]],
    rankings: dict[tuple[str, str], list[dict[str, Any]]],
    budget: int,
    *,
    guarded: bool,
) -> list[dict[str, Any]]:
    rows = []
    for key, items in sorted(groups.items()):
        ranking = rankings[key]
        chosen_set = materialized_set(ranking, budget, guarded=guarded)
        if not chosen_set:
            continue
        final = best_materialized(chosen_set)
        oracle = oracle_row(items)
        baseline = next(
            (row for row in items if row["objective_variant"] == BASELINE_OBJECTIVE),
            None,
        )
        rows.append(
            {
                "source_split": key[0],
                "target": key[1],
                "budget": budget,
                "available_candidates": len(items),
                "materialized_objectives": ",".join(row["objective_variant"] for row in chosen_set),
                "final_objective": final["objective_variant"],
                "final_tcount": final.get("best_beam_tcount"),
                "final_qasm_depth": final.get("best_beam_qasm_depth"),
                "oracle_tcount": oracle.get("best_beam_tcount"),
                "oracle_qasm_depth": oracle.get("best_beam_qasm_depth"),
                "baseline_tcount": "" if baseline is None else baseline.get("best_beam_tcount"),
                "baseline_qasm_depth": "" if baseline is None else baseline.get("best_beam_qasm_depth"),
                "oracle_t_recovered": metric_eq(final.get("best_beam_tcount"), oracle.get("best_beam_tcount")),
                "tcount_ratio_vs_baseline": "" if baseline is None else safe_ratio(final.get("best_beam_tcount"), baseline.get("best_beam_tcount")),
                "qasm_ratio_vs_baseline": "" if baseline is None else safe_ratio(final.get("best_beam_qasm_depth"), baseline.get("best_beam_qasm_depth")),
                "tcount_ratio_vs_oracle": safe_ratio(final.get("best_beam_tcount"), oracle.get("best_beam_tcount")),
            }
        )
    return rows


def metric_eq(left: Any, right: Any) -> bool:
    left_num = coerce_float(left)
    right_num = coerce_float(right)
    return left_num is not None and right_num is not None and left_num == right_num


def fold_of_target(rows: list[dict[str, str]], fold_attr: str) -> dict[str, str]:
    if fold_attr == "target":
        return {target: target for target in target_folds(rows)}
    folds: dict[str, str] = {}
    for row in rows:
        folds[row["target"]] = row.get(fold_attr) or "unknown"
    return folds


def loto_rankings(
    rows: list[dict[str, str]],
    fold_attr: str = "target",
) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], dict[str, str]]:
    """Rank each group's candidates with weights trained without its fold.

    With ``fold_attr='target'`` this is leave-one-target-out; with
    ``fold_attr='functional_family'`` the selector never sees any target of
    the held-out construction family during training.
    """
    normalized = normalize_rows(rows, ALPHAQ_FEATURES)
    normalized_by_group = grouped(normalized)
    target_fold = fold_of_target(rows, fold_attr)
    rankings: dict[tuple[str, str], list[dict[str, Any]]] = {}
    weights_by_fold: dict[str, str] = {}
    for holdout in sorted(set(target_fold.values())):
        train_targets = {
            target for target, fold in target_fold.items() if fold != holdout
        }
        weights = train_linear_weights(
            rows=normalized,
            train_targets=train_targets,
            features=ALPHAQ_FEATURES,
            levels=SELECTOR_LEVELS,
        )
        weights_by_fold[holdout] = ",".join(
            f"{feature}:{weight}"
            for feature, weight in zip(ALPHAQ_FEATURES, weights)
            if weight != 0
        )
        for key, items in normalized_by_group.items():
            if target_fold.get(key[1]) != holdout:
                continue
            rankings[key] = ranked_candidates(items, ALPHAQ_FEATURES, weights)
    return rankings, weights_by_fold


def random_rankings(
    groups: dict[tuple[str, str], list[dict[str, Any]]],
    rng: random.Random,
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    out = {}
    for key, items in groups.items():
        shuffled = list(items)
        rng.shuffle(shuffled)
        out[key] = shuffled
    return out


def policy_summary(
    policy: str,
    scope: str,
    budget: int,
    details: list[dict[str, Any]],
    rng: random.Random,
    bootstrap_samples: int,
) -> dict[str, Any]:
    ratios = [
        value
        for row in details
        if (value := coerce_float(row.get("tcount_ratio_vs_baseline"))) is not None
    ]
    qasm_ratios = [
        value
        for row in details
        if (value := coerce_float(row.get("qasm_ratio_vs_baseline"))) is not None
    ]
    wins = sum(value < 1.0 for value in ratios)
    losses = sum(value > 1.0 for value in ratios)
    ci = bootstrap_median_ci(ratios, rng, bootstrap_samples)
    return {
        "policy": policy,
        "scope": scope,
        "budget": budget,
        "groups": len(details),
        "oracle_t_recovered": sum(bool(row["oracle_t_recovered"]) for row in details),
        "tcount_wins_vs_baseline": wins,
        "tcount_losses_vs_baseline": losses,
        "sign_test_p_one_sided": fmt_float(sign_test_one_sided(wins, losses)),
        "median_tcount_ratio_vs_baseline": fmt_float(median_metric(details, "tcount_ratio_vs_baseline")),
        "median_tcount_ci_low": fmt_float(ci[0] if ci else None),
        "median_tcount_ci_high": fmt_float(ci[1] if ci else None),
        "qasm_wins_vs_baseline": sum(value < 1.0 for value in qasm_ratios),
        "qasm_losses_vs_baseline": sum(value > 1.0 for value in qasm_ratios),
        "median_qasm_ratio_vs_baseline": fmt_float(median_metric(details, "qasm_ratio_vs_baseline")),
        "mean_log_tcount_ratio": fmt_float(mean_log(ratios)),
    }


def mean_log(values: Sequence[float]) -> float | None:
    positives = [value for value in values if value > 0]
    if not positives:
        return None
    return math.exp(sum(math.log(value) for value in positives) / len(positives))


def fmt_float(value: float | None) -> str:
    return "" if value is None else f"{value:.6g}"


def permutation_control(
    groups: dict[tuple[str, str], list[dict[str, Any]]],
    budget: int,
    learned_recovery: int,
    rng: random.Random,
    samples: int,
    *,
    guarded: bool,
) -> tuple[float, float]:
    """Return (mean random oracle-T recovery, p-value learned <= random)."""
    at_least = 0
    total = 0
    for _ in range(samples):
        rankings = random_rankings(groups, rng)
        details = evaluate_policy_on_groups(groups, rankings, budget, guarded=guarded)
        recovered = sum(bool(row["oracle_t_recovered"]) for row in details)
        total += recovered
        if recovered >= learned_recovery:
            at_least += 1
    return total / samples, at_least / samples


def best_oracle_t_dedupe_groups(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_target: dict[str, list[tuple[tuple[Any, ...], tuple[str, str], list[dict[str, str]]]]] = {}
    for key, items in grouped(rows).items():
        oracle = oracle_row(items)
        preference = (
            -inf_if_missing(oracle.get("best_beam_tcount")),
            len(items),
            key[0].startswith("external"),
            key[0],
        )
        by_target.setdefault(key[1], []).append((preference, key, items))
    kept: list[dict[str, str]] = []
    for target in sorted(by_target):
        _preference, _key, items = max(by_target[target], key=lambda entry: entry[0])
        kept.extend(items)
    return kept


def current_dedupe_choice_by_target(
    rows: list[dict[str, str]],
) -> dict[str, tuple[tuple[str, str], list[dict[str, str]]]]:
    return {
        items[0]["target"]: (key, items)
        for key, items in grouped(dedupe_groups(rows)).items()
    }


def best_oracle_t_choice_by_target(
    rows: list[dict[str, str]],
) -> dict[str, tuple[tuple[str, str], list[dict[str, str]]]]:
    return {
        items[0]["target"]: (key, items)
        for key, items in grouped(best_oracle_t_dedupe_groups(rows)).items()
    }


def dedupe_group_audit_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    repeated_targets = sorted(
        target
        for target, split_count in target_group_counts(rows).items()
        if split_count > 1
    )
    current = current_dedupe_choice_by_target(rows)
    best_oracle_t = best_oracle_t_choice_by_target(rows)
    out = []
    for target in repeated_targets:
        for key, items in sorted(
            (key, items)
            for key, items in grouped(rows).items()
            if key[1] == target
        ):
            oracle = oracle_row(items)
            baseline = next(
                (row for row in items if row["objective_variant"] == BASELINE_OBJECTIVE),
                None,
            )
            current_key = current[target][0]
            best_key = best_oracle_t[target][0]
            out.append(
                {
                    "row_type": "group",
                    "target": target,
                    "source_split": key[0],
                    "dedupe_policy": "",
                    "chosen_by_current_dedupe": key == current_key,
                    "chosen_by_best_oracle_t_dedupe": key == best_key,
                    "available_candidates": len(items),
                    "oracle_objective": oracle["objective_variant"],
                    "oracle_tcount": oracle.get("best_beam_tcount"),
                    "baseline_tcount": "" if baseline is None else baseline.get("best_beam_tcount"),
                    "budget": "",
                    "groups": "",
                    "oracle_t_recovered": "",
                    "tcount_wins_vs_baseline": "",
                    "tcount_losses_vs_baseline": "",
                }
            )
    return out


def target_group_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for _source, target in grouped(rows):
        counts[target] = counts.get(target, 0) + 1
    return counts


def dedupe_policy_summary_rows(
    rows: list[dict[str, str]],
    *,
    fold_attr: str,
) -> list[dict[str, Any]]:
    out = []
    policies = (
        ("current_dedupe", dedupe_groups(rows)),
        ("best_oracle_t_dedupe", best_oracle_t_dedupe_groups(rows)),
    )
    for policy, policy_rows in policies:
        groups_raw = grouped(policy_rows)
        rankings, _weights = loto_rankings(policy_rows, fold_attr)
        for budget in (2, 3):
            details = evaluate_policy_on_groups(groups_raw, rankings, budget, guarded=True)
            ratios = [
                value
                for row in details
                if (value := coerce_float(row.get("tcount_ratio_vs_baseline"))) is not None
            ]
            out.append(
                {
                    "row_type": "summary",
                    "target": "",
                    "source_split": "",
                    "dedupe_policy": policy,
                    "chosen_by_current_dedupe": "",
                    "chosen_by_best_oracle_t_dedupe": "",
                    "available_candidates": "",
                    "oracle_objective": "",
                    "oracle_tcount": "",
                    "baseline_tcount": "",
                    "budget": budget,
                    "groups": len(details),
                    "oracle_t_recovered": sum(bool(row["oracle_t_recovered"]) for row in details),
                    "tcount_wins_vs_baseline": sum(value < 1.0 for value in ratios),
                    "tcount_losses_vs_baseline": sum(value > 1.0 for value in ratios),
                }
            )
    return out


def dedupe_audit_rows(
    rows: list[dict[str, str]],
    *,
    fold_attr: str,
) -> list[dict[str, Any]]:
    return [
        *dedupe_group_audit_rows(rows),
        *dedupe_policy_summary_rows(rows, fold_attr=fold_attr),
    ]


def evaluate_scope(
    scope: str,
    rows: list[dict[str, str]],
    rng: random.Random,
    *,
    bootstrap_samples: int,
    permutation_samples: int,
    fold_attr: str = "target",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups_raw = grouped(rows)
    rankings, _weights = loto_rankings(rows, fold_attr)
    groups = {key: rankings[key] for key in rankings}
    details_out: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    max_budget = len(PORTFOLIO_OBJECTIVES)
    for guarded in (False, True):
        prefix = "guarded_top" if guarded else "top"
        for budget in range(1, max_budget + 1):
            details = evaluate_policy_on_groups(groups, rankings, budget, guarded=guarded)
            policy = f"{prefix}{budget}"
            for row in details:
                details_out.append({"policy": policy, "scope": scope, **row})
            summary = policy_summary(policy, scope, budget, details, rng, bootstrap_samples)
            if budget in (1, 2):
                mean_random, p_value = permutation_control(
                    groups,
                    budget,
                    int(summary["oracle_t_recovered"]),
                    rng,
                    permutation_samples,
                    guarded=guarded,
                )
                summary["random_mean_oracle_t_recovered"] = f"{mean_random:.3f}"
                summary["permutation_p_value"] = f"{p_value:.4g}"
            else:
                summary["random_mean_oracle_t_recovered"] = ""
                summary["permutation_p_value"] = ""
            summaries.append(summary)
    # Reference rows: baseline alone and the full-portfolio oracle.
    oracle_details = evaluate_policy_on_groups(groups, rankings, max_budget, guarded=False)
    summaries.append(policy_summary("oracle_full_portfolio", scope, max_budget, oracle_details, rng, bootstrap_samples))
    _ = groups_raw
    return details_out, summaries


def write_report(
    path: Path,
    summaries: list[dict[str, Any]],
    *,
    summary_csv: Path,
    detail_csv: Path,
    dedupe_audit_csv: Path,
    figure_path: Path,
    dedupe_summaries: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# AlphaQ Guarded Portfolio Selection Under A Materialization Budget",
        "",
        f"Summary CSV: `{summary_csv}`.",
        f"Detail CSV: `{detail_csv}`.",
        f"Dedupe audit CSV: `{dedupe_audit_csv}`.",
        f"Figure: `{figure_path}`.",
        "",
        "The evaluation portfolio is the six deployed AlphaQ objectives: "
        + ", ".join(f"`{objective}`" for objective in PORTFOLIO_OBJECTIVES)
        + ".",
        "",
        "Policies `top-m` materialize the m candidates ranked best by the "
        "leave-one-target-out cheap-feature selector. Policies `guarded_top-m` "
        "always include the `factor_count` baseline candidate in the budget, so "
        "for m >= 2 they are never worse than the baseline pipeline in T-count "
        "by construction. `scope=groups` treats each (split, target) pair as one "
        "observation; `scope=targets` deduplicates to one group per target.",
        "",
        "| scope | policy | budget | groups | oracle-T recovered | T wins | T losses | sign-test p | median T ratio | 95% CI | median QASM ratio | random recovery | perm. p |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in summaries:
        ci = (
            f"[{row['median_tcount_ci_low']}, {row['median_tcount_ci_high']}]"
            if row.get("median_tcount_ci_low")
            else ""
        )
        lines.append(
            "| {scope} | {policy} | {budget} | {groups} | {rec} | {wins} | {losses} | {p} | {medt} | {ci} | {medq} | {rnd} | {perm} |".format(
                scope=row["scope"],
                policy=row["policy"],
                budget=row["budget"],
                groups=row["groups"],
                rec=row["oracle_t_recovered"],
                wins=row["tcount_wins_vs_baseline"],
                losses=row["tcount_losses_vs_baseline"],
                p=row["sign_test_p_one_sided"],
                medt=row["median_tcount_ratio_vs_baseline"],
                ci=ci,
                medq=row["median_qasm_ratio_vs_baseline"],
                rnd=row.get("random_mean_oracle_t_recovered", ""),
                perm=row.get("permutation_p_value", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Dedupe Sensitivity",
            "",
            "Default target-level dedupe keeps the most complete group, preferring external splits. "
            "The audit CSV lists every repeated target and compares that rule with an alternate "
            "best-oracle-T dedupe rule.",
            "",
            "| dedupe policy | budget | targets | oracle-T recovered | T wins | T losses |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in dedupe_summaries:
        lines.append(
            "| {policy} | {budget} | {groups} | {rec} | {wins} | {losses} |".format(
                policy=row["dedupe_policy"],
                budget=row["budget"],
                groups=row["groups"],
                rec=row["oracle_t_recovered"],
                wins=row["tcount_wins_vs_baseline"],
                losses=row["tcount_losses_vs_baseline"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figure(path: Path, summaries: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for scope, ax in zip(("groups", "targets"), axes):
        rows = [row for row in summaries if row["scope"] == scope and row["policy"] != "oracle_full_portfolio"]
        for guarded, style, label in ((False, "o--", "top-m"), (True, "s-", "guarded top-m")):
            prefix = "guarded_top" if guarded else "top"
            points = sorted(
                (
                    (int(row["budget"]), int(row["oracle_t_recovered"]), int(row["groups"]))
                    for row in rows
                    if row["policy"].startswith(prefix)
                    and (guarded or not row["policy"].startswith("guarded_top"))
                ),
            )
            if not points:
                continue
            budgets = [item[0] for item in points]
            fractions = [item[1] / item[2] if item[2] else 0.0 for item in points]
            ax.plot(budgets, fractions, style, label=label)
        ax.set_title(f"oracle T-count recovery ({scope})")
        ax.set_xlabel("materialization budget m")
        ax.set_ylabel("fraction of groups at oracle T-count")
        ax.set_xticks(range(1, len(PORTFOLIO_OBJECTIVES) + 1))
        ax.set_ylim(0.0, 1.02)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    fig.savefig(path, dpi=220)
    plt.close(fig)


DETAIL_FIELDS = [
    "policy",
    "scope",
    "source_split",
    "target",
    "budget",
    "available_candidates",
    "materialized_objectives",
    "final_objective",
    "final_tcount",
    "final_qasm_depth",
    "oracle_tcount",
    "oracle_qasm_depth",
    "baseline_tcount",
    "baseline_qasm_depth",
    "oracle_t_recovered",
    "tcount_ratio_vs_baseline",
    "qasm_ratio_vs_baseline",
    "tcount_ratio_vs_oracle",
]

SUMMARY_FIELDS = [
    "policy",
    "scope",
    "budget",
    "groups",
    "oracle_t_recovered",
    "tcount_wins_vs_baseline",
    "tcount_losses_vs_baseline",
    "sign_test_p_one_sided",
    "median_tcount_ratio_vs_baseline",
    "median_tcount_ci_low",
    "median_tcount_ci_high",
    "qasm_wins_vs_baseline",
    "qasm_losses_vs_baseline",
    "median_qasm_ratio_vs_baseline",
    "mean_log_tcount_ratio",
    "random_mean_oracle_t_recovered",
    "permutation_p_value",
]

DEDUPE_AUDIT_FIELDS = [
    "row_type",
    "target",
    "source_split",
    "dedupe_policy",
    "chosen_by_current_dedupe",
    "chosen_by_best_oracle_t_dedupe",
    "available_candidates",
    "oracle_objective",
    "oracle_tcount",
    "baseline_tcount",
    "budget",
    "groups",
    "oracle_t_recovered",
    "tcount_wins_vs_baseline",
    "tcount_losses_vs_baseline",
]


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    dataset = read_csv(args.dataset_csv)
    rows = portfolio_rows(train_ready_rows(dataset))
    all_details: list[dict[str, Any]] = []
    all_summaries: list[dict[str, Any]] = []
    for scope, scope_rows in (
        ("groups", rows),
        ("targets", dedupe_groups(rows)),
    ):
        details, summaries = evaluate_scope(
            scope,
            scope_rows,
            rng,
            bootstrap_samples=args.bootstrap_samples,
            permutation_samples=args.permutation_samples,
            fold_attr=args.fold_attr,
        )
        all_details.extend(details)
        all_summaries.extend(summaries)
    write_csv(args.detail_csv, all_details, DETAIL_FIELDS)
    write_csv(args.summary_csv, all_summaries, SUMMARY_FIELDS)
    dedupe_rows = dedupe_audit_rows(rows, fold_attr=args.fold_attr)
    write_csv(args.dedupe_audit_csv, dedupe_rows, DEDUPE_AUDIT_FIELDS)
    write_report(
        args.report_path,
        all_summaries,
        summary_csv=args.summary_csv,
        detail_csv=args.detail_csv,
        dedupe_audit_csv=args.dedupe_audit_csv,
        figure_path=args.figure_path,
        dedupe_summaries=[row for row in dedupe_rows if row["row_type"] == "summary"],
    )
    write_figure(args.figure_path, all_summaries)
    print(f"Wrote {args.summary_csv}")
    print(f"Wrote {args.detail_csv}")
    print(f"Wrote {args.dedupe_audit_csv}")
    print(f"Wrote {args.report_path}")
    print(f"Wrote {args.figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
