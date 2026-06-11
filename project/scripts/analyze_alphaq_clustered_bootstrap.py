"""Clustered bootstrap and cross-battery reproducibility for AlphaQ selection."""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")

from scripts.analyze_alphaq_portfolio_budget import (
    best_materialized,
    loto_rankings,
    materialized_set,
    portfolio_rows,
)
from scripts.analyze_alphaq_split_select import (
    BASELINE_OBJECTIVE,
    grouped,
    oracle_row,
    read_csv,
    train_ready_rows,
    write_csv,
)
from scripts.structural_target import coerce_float

DEFAULT_DATASET = PROJECT_ROOT / "results" / "csv" / "alphaq_objective_selection_dataset.csv"
DEFAULT_BOOTSTRAP = PROJECT_ROOT / "results" / "csv" / "alphaq_clustered_bootstrap.csv"
DEFAULT_REPRO = PROJECT_ROOT / "results" / "csv" / "alphaq_cross_battery_reproducibility.csv"
DEFAULT_REPORT = PROJECT_ROOT / "results" / "reports" / "alphaq_clustered_bootstrap.md"
BOOTSTRAP_SAMPLES = 10_000
RANDOM_SEED = 20260611


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap AlphaQ policy metrics over target clusters."
    )
    parser.add_argument("--dataset-csv", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--bootstrap-csv", type=Path, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--repro-csv", type=Path, default=DEFAULT_REPRO)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def policy_details(
    rows: list[dict[str, str]],
    *,
    policy: str,
    rankings: dict[tuple[str, str], list[dict[str, Any]]] | None = None,
    groups_raw: dict[tuple[str, str], list[dict[str, str]]] | None = None,
) -> list[dict[str, Any]]:
    if rankings is None:
        rankings, _weights = loto_rankings(rows, fold_attr="target")
    if groups_raw is None:
        groups_raw = grouped(rows)
    details = []
    guarded = policy == "guarded_top2"
    budget = 2 if guarded else 1
    for key, ranking in sorted(rankings.items()):
        chosen = materialized_set(ranking, budget, guarded=guarded)
        final = best_materialized(chosen)
        oracle = oracle_row(groups_raw[key])
        baseline = next(
            row for row in groups_raw[key] if row["objective_variant"] == BASELINE_OBJECTIVE
        )
        final_t = coerce_float(final.get("best_beam_tcount"))
        baseline_t = coerce_float(baseline.get("best_beam_tcount"))
        oracle_t = coerce_float(oracle.get("best_beam_tcount"))
        ratio = None if final_t is None or baseline_t in (None, 0) else final_t / baseline_t
        regression = None if final_t is None or baseline_t is None else final_t - baseline_t
        details.append(
            {
                "policy": policy,
                "source_split": key[0],
                "target": key[1],
                "materialized_objectives": ",".join(row["objective_variant"] for row in chosen),
                "final_objective": final["objective_variant"],
                "final_tcount": final.get("best_beam_tcount"),
                "oracle_objective": oracle["objective_variant"],
                "oracle_tcount": oracle.get("best_beam_tcount"),
                "baseline_tcount": baseline.get("best_beam_tcount"),
                "tcount_ratio_vs_baseline": "" if ratio is None else ratio,
                "tcount_regression_vs_baseline": "" if regression is None else regression,
                "oracle_t_recovered": final_t is not None
                and oracle_t is not None
                and final_t == oracle_t,
            }
        )
    return details


def geometric_mean(values: list[float]) -> float:
    positives = [value for value in values if value > 0]
    if not positives:
        return float("nan")
    return math.exp(sum(math.log(value) for value in positives) / len(positives))


def metric_values(rows: list[dict[str, Any]]) -> dict[str, float]:
    ratios = [
        value
        for row in rows
        if (value := coerce_float(row.get("tcount_ratio_vs_baseline"))) is not None
    ]
    regressions = [
        value
        for row in rows
        if (value := coerce_float(row.get("tcount_regression_vs_baseline"))) is not None
    ]
    return {
        "geomean_t_ratio": geometric_mean(ratios),
        "win_count": float(sum(value < 1.0 for value in ratios)),
        "max_regression": 0.0 if not regressions else max(regressions),
    }


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - index) + ordered[high] * (index - low)


def bootstrap_cluster_metrics(
    rows: list[dict[str, Any]],
    *,
    samples: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_target[row["target"]].append(row)
    targets = sorted(by_target)
    rng = random.Random(seed)
    draws = {
        "geomean_t_ratio": [],
        "win_count": [],
        "max_regression": [],
    }
    for _ in range(samples):
        sample_rows: list[dict[str, Any]] = []
        for target in (rng.choice(targets) for _ in targets):
            sample_rows.extend(by_target[target])
        metrics = metric_values(sample_rows)
        for key in draws:
            draws[key].append(metrics[key])
    return {
        key: {
            "ci_low": percentile(values, 0.025),
            "ci_high": percentile(values, 0.975),
        }
        for key, values in draws.items()
    }


def bootstrap_rows(
    details_by_policy: dict[str, list[dict[str, Any]]],
    *,
    samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows = []
    for offset, (policy, details) in enumerate(details_by_policy.items()):
        point = metric_values(details)
        intervals = bootstrap_cluster_metrics(details, samples=samples, seed=seed + offset)
        for metric, point_value in point.items():
            rows.append(
                {
                    "policy": policy,
                    "metric": metric,
                    "point": point_value,
                    "ci_low": intervals[metric]["ci_low"],
                    "ci_high": intervals[metric]["ci_high"],
                    "samples": samples,
                    "clusters": len({row["target"] for row in details}),
                    "rows": len(details),
                }
            )
    return rows


def cross_battery_rows(
    dataset_rows: list[dict[str, str]],
    guarded_details: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    oracle_by_group = {}
    for key, items in grouped(dataset_rows).items():
        oracle = oracle_row(items)
        oracle_by_group[key] = {
            "oracle_objective": oracle["objective_variant"],
            "oracle_tcount": oracle.get("best_beam_tcount"),
        }
    final_by_group = {
        (row["source_split"], row["target"]): row
        for row in guarded_details
    }
    by_target: dict[str, list[tuple[tuple[str, str], dict[str, Any]]]] = defaultdict(list)
    for key, oracle in oracle_by_group.items():
        if key in final_by_group:
            by_target[key[1]].append((key, oracle))
    rows = []
    for target, entries in sorted(by_target.items()):
        if len(entries) < 2:
            continue
        source_splits = []
        oracle_objectives = []
        oracle_tcounts = []
        final_tcounts = []
        final_objectives = []
        for key, oracle in entries:
            detail = final_by_group[key]
            source_splits.append(key[0])
            oracle_objectives.append(str(oracle["oracle_objective"]))
            oracle_tcounts.append(str(oracle["oracle_tcount"]))
            final_tcounts.append(str(detail["final_tcount"]))
            final_objectives.append(str(detail["final_objective"]))
        rows.append(
            {
                "target": target,
                "runs": len(entries),
                "source_splits": ";".join(source_splits),
                "oracle_objectives": ";".join(oracle_objectives),
                "oracle_tcounts": ";".join(oracle_tcounts),
                "final_objectives": ";".join(final_objectives),
                "final_tcounts": ";".join(final_tcounts),
                "oracle_objective_agrees": len(set(oracle_objectives)) == 1,
                "final_tcount_agrees": len(set(final_tcounts)) == 1,
            }
        )
    return rows


def write_report(
    path: Path,
    bootstrap: list[dict[str, Any]],
    reproducibility: list[dict[str, Any]],
    *,
    dataset_csv: Path,
) -> None:
    lines = [
        "# AlphaQ Clustered Bootstrap",
        "",
        f"Dataset: `{dataset_csv}`.",
        "",
        "Bootstrap clusters are targets; all source-split rows for a sampled target are retained together.",
        "",
        "| policy | metric | point | 95% CI | clusters | rows |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in bootstrap:
        lines.append(
            "| {policy} | {metric} | {point:.6g} | [{ci_low:.6g}, {ci_high:.6g}] | {clusters} | {rows} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Cross-Battery Reproducibility",
            "",
            "| target | runs | oracle objectives agree | final T-counts agree | oracle objectives | final T-counts |",
            "|---|---:|---|---|---|---|",
        ]
    )
    for row in reproducibility:
        lines.append(
            "| {target} | {runs} | {oracle_objective_agrees} | {final_tcount_agrees} | {oracle_objectives} | {final_tcounts} |".format(
                **row
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


BOOTSTRAP_FIELDS = [
    "policy",
    "metric",
    "point",
    "ci_low",
    "ci_high",
    "samples",
    "clusters",
    "rows",
]

REPRO_FIELDS = [
    "target",
    "runs",
    "source_splits",
    "oracle_objectives",
    "oracle_tcounts",
    "final_objectives",
    "final_tcounts",
    "oracle_objective_agrees",
    "final_tcount_agrees",
]


def main() -> int:
    args = parse_args()
    rows = portfolio_rows(train_ready_rows(read_csv(args.dataset_csv)))
    rankings, _weights = loto_rankings(rows, fold_attr="target")
    groups_raw = grouped(rows)
    details_by_policy = {
        "top1": policy_details(rows, policy="top1", rankings=rankings, groups_raw=groups_raw),
        "guarded_top2": policy_details(
            rows,
            policy="guarded_top2",
            rankings=rankings,
            groups_raw=groups_raw,
        ),
    }
    bootstrap = bootstrap_rows(
        details_by_policy,
        samples=args.samples,
        seed=args.seed,
    )
    reproducibility = cross_battery_rows(rows, details_by_policy["guarded_top2"])
    write_csv(args.bootstrap_csv, bootstrap, BOOTSTRAP_FIELDS)
    write_csv(args.repro_csv, reproducibility, REPRO_FIELDS)
    write_report(
        args.report_path,
        bootstrap,
        reproducibility,
        dataset_csv=args.dataset_csv,
    )
    print(f"Wrote {args.bootstrap_csv}")
    print(f"Wrote {args.repro_csv}")
    print(f"Wrote {args.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
