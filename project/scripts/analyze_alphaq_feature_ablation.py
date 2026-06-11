"""Feature ablation for the AlphaQ guarded portfolio selector.

For each feature subset (all five, leave-one-out, and single-feature), retrain
the sparse linear selector leave-one-target-out and evaluate the guarded
top-2 policy. This isolates which factor-structure features carry the ranking
signal. Pure local compute: it re-ranks already-generated candidates from the
objective-selection dataset; no MILP or synthesis is involved.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_alphaq_split_select import (
    ALPHAQ_FEATURES,
    BASELINE_OBJECTIVE,
    grouped,
    normalize_rows,
    oracle_key,
    oracle_row,
    read_csv,
    train_linear_weights,
    train_ready_rows,
)
from scripts.analyze_alphaq_portfolio_budget import (
    PORTFOLIO_OBJECTIVES,
    SELECTOR_LEVELS,
    best_materialized,
    dedupe_groups,
    materialized_set,
    portfolio_rows,
    ranked_candidates,
    sign_test_one_sided,
)
from scripts.structural_target import coerce_float

DEFAULT_DATASET = PROJECT_ROOT / "results" / "csv" / "alphaq_objective_selection_dataset.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "csv" / "alphaq_feature_ablation.csv"
DEFAULT_REPORT = PROJECT_ROOT / "results" / "reports" / "alphaq_feature_ablation.md"

FEATURE_SHORT = {
    "factor_count": "R",
    "factor_qubit_concentration_index": "C",
    "factor_support_weight_mean": "W",
    "factor_pairwise_support_overlap_mean": "O",
    "factor_pairwise_jaccard_mean": "J",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leave-one-feature-out ablation of the AlphaQ selector."
    )
    parser.add_argument("--dataset-csv", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--budget", type=int, default=2)
    return parser.parse_args()


def loto_rankings_for_features(
    rows: list[dict[str, str]],
    features: tuple[str, ...],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    normalized = normalize_rows(rows, features)
    normalized_by_group = grouped(normalized)
    targets = sorted({row["target"] for row in rows})
    rankings: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for holdout in targets:
        train_targets = set(targets) - {holdout}
        weights = train_linear_weights(
            rows=normalized,
            train_targets=train_targets,
            features=features,
            levels=SELECTOR_LEVELS,
        )
        for key, items in normalized_by_group.items():
            if key[1] != holdout:
                continue
            rankings[key] = ranked_candidates(items, features, weights)
    return rankings


def evaluate(
    rankings: dict[tuple[str, str], list[dict[str, Any]]],
    budget: int,
) -> dict[str, Any]:
    oracle_hits = 0
    wins = 0
    losses = 0
    total = 0
    for _key, ranking in sorted(rankings.items()):
        baseline = next(
            (row for row in ranking if row["objective_variant"] == BASELINE_OBJECTIVE),
            None,
        )
        if baseline is None:
            continue
        total += 1
        chosen = materialized_set(ranking, budget, guarded=True)
        best = best_materialized(chosen)
        oracle = oracle_row(ranking)
        best_t = coerce_float(best.get("best_beam_tcount"))
        oracle_t = coerce_float(oracle.get("best_beam_tcount"))
        baseline_t = coerce_float(baseline.get("best_beam_tcount"))
        if best_t is not None and oracle_t is not None and best_t <= oracle_t:
            oracle_hits += 1
        if best_t is not None and baseline_t is not None:
            if best_t < baseline_t:
                wins += 1
            elif best_t > baseline_t:
                losses += 1
    return {
        "groups": total,
        "oracle_recovered": oracle_hits,
        "wins": wins,
        "losses": losses,
        "sign_p": sign_test_one_sided(wins, losses),
    }


def subset_label(features: tuple[str, ...]) -> str:
    return "".join(FEATURE_SHORT[f] for f in features)


def main() -> int:
    args = parse_args()
    rows = train_ready_rows(portfolio_rows(read_csv(args.dataset_csv)))

    subsets: list[tuple[str, tuple[str, ...]]] = [("all", ALPHAQ_FEATURES)]
    for dropped in ALPHAQ_FEATURES:
        kept = tuple(f for f in ALPHAQ_FEATURES if f != dropped)
        subsets.append((f"drop {FEATURE_SHORT[dropped]}", kept))
    for single in ALPHAQ_FEATURES:
        subsets.append((f"only {FEATURE_SHORT[single]}", (single,)))

    results = []
    for name, features in subsets:
        rankings = loto_rankings_for_features(rows, features)
        for scope, scoped in (
            ("groups", rankings),
            ("targets", grouped(dedupe_groups([r for rk in rankings.values() for r in rk]))),
        ):
            if scope == "targets":
                # Re-rank deduped groups using the same per-fold protocol.
                deduped_rows = [r for rk in scoped.values() for r in rk]
                scoped = loto_rankings_for_features(deduped_rows, features)
            stats = evaluate(scoped, args.budget)
            results.append(
                {
                    "subset": name,
                    "features": subset_label(features),
                    "scope": scope,
                    **stats,
                }
            )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {args.output_csv}")

    lines = [
        "# AlphaQ Selector Feature Ablation",
        "",
        f"Dataset: `{args.dataset_csv}`. Policy: guarded top-{args.budget}, LOTO.",
        "",
        "| subset | features | scope | groups | oracle | W/L | sign p |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in results:
        p = row["sign_p"]
        lines.append(
            f"| {row['subset']} | {row['features']} | {row['scope']} | "
            f"{row['groups']} | {row['oracle_recovered']} | "
            f"{row['wins']}/{row['losses']} | "
            f"{'' if p is None else f'{p:.2e}'} |"
        )
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
