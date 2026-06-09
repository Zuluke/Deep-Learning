from __future__ import annotations

import argparse
import csv
import itertools
import math
import sys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_best_objective_beam_ablation import safe_ratio
from scripts.structural_target import coerce_float


DEFAULT_DATASET = PROJECT_ROOT / "results" / "csv" / "alphaq_objective_selection_dataset.csv"
DEFAULT_SUMMARY = PROJECT_ROOT / "results" / "csv" / "alphaq_split_select_summary.csv"
DEFAULT_DETAILS = PROJECT_ROOT / "results" / "csv" / "alphaq_split_select_details.csv"
DEFAULT_REPORT = PROJECT_ROOT / "results" / "reports" / "alphaq_split_select.md"
DEFAULT_FIGURE = PROJECT_ROOT / "results" / "figures" / "alphaq_split_select.png"

OBJECTIVES = (
    "factor_count",
    "factor_count_pair_cap",
    "mixed_pair",
    "frontier_pair",
    "depth_guarded_mixed_pair",
    "t_preserving_frontier_pair",
)
BASELINE_OBJECTIVE = "factor_count"
ALPHAQ_FEATURES = (
    "factor_count",
    "factor_qubit_concentration_index",
    "factor_support_weight_mean",
    "factor_pairwise_support_overlap_mean",
    "factor_pairwise_jaccard_mean",
)
ALPHAQ_DECOMP_FEATURES = (
    *ALPHAQ_FEATURES,
    "decomp_tcount",
    "decomp_tdepth",
)
ALPHAQ_QASM_FEATURES = (
    *ALPHAQ_DECOMP_FEATURES,
    "decomp_qasm_depth_ratio",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate AlphaQuantum Split-Select objective selectors "
            "on the consolidated objective-selection dataset."
        )
    )
    parser.add_argument("--dataset-csv", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--detail-csv", type=Path, default=DEFAULT_DETAILS)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--figure-path", type=Path, default=DEFAULT_FIGURE)
    return parser.parse_args()


def selector_profiles() -> list[dict[str, Any]]:
    return [
        {
            "policy": "split_select_linear_alphaq",
            "features": ALPHAQ_FEATURES,
            "levels": (-2, -1, 0, 1, 2),
            "feature_scope": "alphaq-factor",
        },
        {
            "policy": "split_select_linear_alphaq_decomp",
            "features": ALPHAQ_DECOMP_FEATURES,
            "levels": (-2, -1, 0, 1, 2),
            "feature_scope": "alphaq-decomp",
        },
        {
            "policy": "split_select_linear_alphaq_qasm",
            "features": ALPHAQ_QASM_FEATURES,
            "levels": (-2, -1, 0, 1, 2),
            "feature_scope": "alphaq-qasm",
        },
    ]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def train_ready_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if boolish(row.get("train_ready"))
        and row.get("execution_status") == "ok"
        and boolish(row.get("has_beam_candidate"))
    ]


def boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def group_key(row: dict[str, str]) -> tuple[str, str]:
    return row["source_split"], row["target"]


def grouped(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(group_key(row), []).append(row)
    return groups


def target_folds(rows: list[dict[str, str]]) -> list[str]:
    return sorted({row["target"] for row in rows})


def objective_row(items: list[dict[str, str]], objective: str) -> dict[str, str] | None:
    return next((row for row in items if row["objective_variant"] == objective), None)


def oracle_row(items: list[dict[str, str]]) -> dict[str, str]:
    labeled = next(
        (
            row
            for row in items
            if row.get("objective_variant") == row.get("oracle_objective")
        ),
        None,
    )
    if labeled is not None:
        return labeled
    return min(items, key=oracle_key)


def oracle_key(row: dict[str, str]) -> tuple[float, float, float, str]:
    return (
        inf_if_missing(row.get("best_beam_tcount")),
        inf_if_missing(row.get("best_beam_qasm_depth")),
        inf_if_missing(row.get("best_beam_primary_nc_depth_ratio")),
        row.get("objective_variant", ""),
    )


def inf_if_missing(value: Any) -> float:
    numeric = coerce_float(value)
    return float("inf") if numeric is None else numeric


def normalize_rows(rows: list[dict[str, str]], features: Iterable[str]) -> list[dict[str, Any]]:
    features = tuple(features)
    out: list[dict[str, Any]] = []
    for key, items in grouped(rows).items():
        ranges = {feature: feature_range(items, feature) for feature in features}
        for row in items:
            normalized = dict(row)
            normalized["group_id"] = "::".join(key)
            for feature in features:
                low, high = ranges[feature]
                normalized[f"norm_{feature}"] = normalized_feature(row, feature, low, high)
            out.append(normalized)
    return out


def feature_range(rows: list[dict[str, str]], feature: str) -> tuple[float, float]:
    values = [
        value
        for row in rows
        if (value := coerce_float(row.get(feature))) is not None and math.isfinite(value)
    ]
    if not values:
        return 0.0, 0.0
    return min(values), max(values)


def normalized_feature(row: dict[str, str], feature: str, low: float, high: float) -> float:
    numeric = coerce_float(row.get(feature))
    if numeric is None or not math.isfinite(numeric):
        return 1.0
    if high == low:
        return 0.0
    value = (numeric - low) / (high - low)
    if not math.isfinite(value):
        return 1.0
    return max(0.0, min(1.0, value))


def linear_score(row: dict[str, Any], features: tuple[str, ...], weights: tuple[int, ...]) -> float:
    return sum(weight * float(row[f"norm_{feature}"]) for feature, weight in zip(features, weights))


def select_by_weights(
    items: list[dict[str, Any]],
    features: tuple[str, ...],
    weights: tuple[int, ...],
) -> dict[str, Any]:
    return min(
        items,
        key=lambda row: (
            linear_score(row, features, weights),
            row["objective_variant"],
        ),
    )


def candidate_weights(levels: tuple[int, ...], size: int) -> Iterable[tuple[int, ...]]:
    nonzero_levels = tuple(level for level in levels if level != 0)
    if size > 5:
        # Keep the selector trainable and auditable on small datasets: for
        # larger feature sets, test sparse one- and two-feature rules instead
        # of an exponential dense grid.
        for first in range(size):
            for level in nonzero_levels:
                weights = [0] * size
                weights[first] = level
                yield tuple(weights)
        for first, second in itertools.combinations(range(size), 2):
            for first_level, second_level in itertools.product(nonzero_levels, repeat=2):
                weights = [0] * size
                weights[first] = first_level
                weights[second] = second_level
                yield tuple(weights)
        return
    for weights in itertools.product(levels, repeat=size):
        if any(weight != 0 for weight in weights):
            yield weights


def train_linear_weights(
    *,
    rows: list[dict[str, Any]],
    train_targets: set[str],
    features: tuple[str, ...],
    levels: tuple[int, ...],
) -> tuple[int, ...]:
    train_groups = [
        items
        for (_split, target), items in grouped(rows).items()
        if target in train_targets and len(items) >= 2
    ]
    if not train_groups:
        raise ValueError("No train groups available for Split-Select.")
    best_weights: tuple[int, ...] | None = None
    best_score: tuple[Any, ...] | None = None
    for weights in candidate_weights(levels, len(features)):
        selected = [select_by_weights(items, features, weights) for items in train_groups]
        score = policy_training_score(selected, train_groups, weights)
        if best_score is None or score > best_score:
            best_score = score
            best_weights = weights
    if best_weights is None:
        raise ValueError("No non-zero Split-Select weights available.")
    return best_weights


def policy_training_score(
    selected: list[dict[str, Any]],
    train_groups: list[list[dict[str, Any]]],
    weights: tuple[int, ...],
) -> tuple[Any, ...]:
    oracles = [oracle_row(items) for items in train_groups]
    exact = sum(row["objective_variant"] == oracle["objective_variant"] for row, oracle in zip(selected, oracles))
    t_nonworse = sum(metric_leq(row["best_beam_tcount"], oracle["best_beam_tcount"]) for row, oracle in zip(selected, oracles))
    qasm_nonworse = sum(metric_leq(row["best_beam_qasm_depth"], oracle["best_beam_qasm_depth"]) for row, oracle in zip(selected, oracles))
    return (
        exact,
        t_nonworse,
        qasm_nonworse,
        -sum(weight != 0 for weight in weights),
        -sum(abs(weight) for weight in weights),
        tuple(-weight for weight in weights),
    )


def metric_leq(left: Any, right: Any) -> bool:
    left_num = coerce_float(left)
    right_num = coerce_float(right)
    return left_num is not None and right_num is not None and left_num <= right_num


def best_fixed_objective(rows: list[dict[str, str]], train_targets: set[str]) -> str:
    best_objective = BASELINE_OBJECTIVE
    best_score: tuple[Any, ...] | None = None
    for objective in OBJECTIVES:
        selected: list[dict[str, str]] = []
        train_groups: list[list[dict[str, str]]] = []
        for (_split, target), items in grouped(rows).items():
            if target not in train_targets:
                continue
            row = objective_row(items, objective)
            if row is None:
                continue
            selected.append(row)
            train_groups.append(items)
        if not selected:
            continue
        oracles = [oracle_row(items) for items in train_groups]
        score = (
            sum(row["objective_variant"] == oracle["objective_variant"] for row, oracle in zip(selected, oracles)),
            sum(metric_leq(row["best_beam_tcount"], oracle["best_beam_tcount"]) for row, oracle in zip(selected, oracles)),
            len(selected),
            -OBJECTIVES.index(objective),
        )
        if best_score is None or score > best_score:
            best_score = score
            best_objective = objective
    return best_objective


def evaluation_rows(dataset_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows = train_ready_rows(dataset_rows)
    all_features = tuple(sorted({feature for profile in selector_profiles() for feature in profile["features"]}))
    normalized = normalize_rows(rows, all_features)
    normalized_by_group = grouped(normalized)
    raw_by_group = grouped(rows)
    details: list[dict[str, Any]] = []
    for holdout in target_folds(rows):
        train_targets = {target for target in target_folds(rows) if target != holdout}
        fold_groups = [
            (key, items)
            for key, items in raw_by_group.items()
            if key[1] == holdout
        ]
        fixed = best_fixed_objective(rows, train_targets)
        for key, items in fold_groups:
            details.append(policy_row("baseline_factor_count", "baseline", holdout, key, objective_row(items, BASELINE_OBJECTIVE), items))
            details.append(policy_row("best_fixed_loto", "fixed-objective", holdout, key, objective_row(items, fixed), items, fixed_objective=fixed))
            details.append(policy_row("oracle_posthoc", "oracle", holdout, key, oracle_row(items), items))
            for profile in selector_profiles():
                try:
                    weights = train_linear_weights(
                        rows=normalized,
                        train_targets=train_targets,
                        features=tuple(profile["features"]),
                        levels=tuple(profile["levels"]),
                    )
                    selected = select_by_weights(normalized_by_group[key], tuple(profile["features"]), weights)
                    raw_selected = objective_row(items, selected["objective_variant"])
                    weights_text = format_weights(tuple(profile["features"]), weights)
                    missing_status = "missing-selected-objective"
                except ValueError:
                    raw_selected = None
                    weights_text = ""
                    missing_status = "no-training-data"
                details.append(
                    policy_row(
                        profile["policy"],
                        profile["feature_scope"],
                        holdout,
                        key,
                        raw_selected,
                        items,
                        weights=weights_text,
                        missing_status=missing_status,
                    )
                )
    return details


def policy_row(
    policy: str,
    policy_type: str,
    fold_target: str,
    key: tuple[str, str],
    selected: dict[str, str] | None,
    items: list[dict[str, str]],
    *,
    weights: str = "",
    fixed_objective: str = "",
    missing_status: str = "missing-selected-objective",
) -> dict[str, Any]:
    oracle = oracle_row(items)
    baseline = objective_row(items, BASELINE_OBJECTIVE)
    if selected is None:
        return {
            "policy": policy,
            "policy_type": policy_type,
            "fold_target": fold_target,
            "source_split": key[0],
            "target": key[1],
            "selection_status": missing_status,
            "fixed_objective": fixed_objective,
            "weights": weights,
            "oracle_objective": oracle["objective_variant"],
        }
    return {
        "policy": policy,
        "policy_type": policy_type,
        "fold_target": fold_target,
        "source_split": key[0],
        "target": key[1],
        "selection_status": "ok",
        "selected_objective": selected["objective_variant"],
        "fixed_objective": fixed_objective,
        "oracle_objective": oracle["objective_variant"],
        "exact_oracle_match": selected["objective_variant"] == oracle["objective_variant"],
        "weights": weights,
        "selected_tcount": selected.get("best_beam_tcount"),
        "selected_primary_nc_depth_ratio": selected.get("best_beam_primary_nc_depth_ratio"),
        "selected_qasm_depth": selected.get("best_beam_qasm_depth"),
        "oracle_tcount": oracle.get("best_beam_tcount"),
        "oracle_primary_nc_depth_ratio": oracle.get("best_beam_primary_nc_depth_ratio"),
        "oracle_qasm_depth": oracle.get("best_beam_qasm_depth"),
        "baseline_tcount": "" if baseline is None else baseline.get("best_beam_tcount"),
        "baseline_primary_nc_depth_ratio": "" if baseline is None else baseline.get("best_beam_primary_nc_depth_ratio"),
        "baseline_qasm_depth": "" if baseline is None else baseline.get("best_beam_qasm_depth"),
        "tcount_ratio_vs_baseline": "" if baseline is None else safe_ratio(selected.get("best_beam_tcount"), baseline.get("best_beam_tcount")),
        "primary_ratio_vs_baseline": "" if baseline is None else safe_ratio(selected.get("best_beam_primary_nc_depth_ratio"), baseline.get("best_beam_primary_nc_depth_ratio")),
        "qasm_ratio_vs_baseline": "" if baseline is None else safe_ratio(selected.get("best_beam_qasm_depth"), baseline.get("best_beam_qasm_depth")),
        "tcount_ratio_vs_oracle": safe_ratio(selected.get("best_beam_tcount"), oracle.get("best_beam_tcount")),
        "qasm_ratio_vs_oracle": safe_ratio(selected.get("best_beam_qasm_depth"), oracle.get("best_beam_qasm_depth")),
    }


def summary_rows(details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for policy in sorted({row["policy"] for row in details}):
        items = [row for row in details if row["policy"] == policy]
        ok = [row for row in items if row["selection_status"] == "ok"]
        rows.append(
            {
                "policy": policy,
                "policy_type": items[0]["policy_type"] if items else "",
                "evaluated_groups": len(items),
                "ok_groups": len(ok),
                "missing_groups": len(items) - len(ok),
                "exact_oracle_matches": count_bool(ok, "exact_oracle_match"),
                "tcount_nonworse_vs_baseline": count_metric(ok, "tcount_ratio_vs_baseline", strict=False),
                "tcount_wins_vs_baseline": count_metric(ok, "tcount_ratio_vs_baseline", strict=True),
                "primary_nonworse_vs_baseline": count_metric(ok, "primary_ratio_vs_baseline", strict=False),
                "primary_wins_vs_baseline": count_metric(ok, "primary_ratio_vs_baseline", strict=True),
                "qasm_nonworse_vs_baseline": count_metric(ok, "qasm_ratio_vs_baseline", strict=False),
                "qasm_wins_vs_baseline": count_metric(ok, "qasm_ratio_vs_baseline", strict=True),
                "joint_nonworse_vs_baseline": count_joint_nonworse(ok),
                "median_tcount_ratio_vs_baseline": median_metric(ok, "tcount_ratio_vs_baseline"),
                "median_primary_ratio_vs_baseline": median_metric(ok, "primary_ratio_vs_baseline"),
                "median_qasm_ratio_vs_baseline": median_metric(ok, "qasm_ratio_vs_baseline"),
                "median_tcount_ratio_vs_oracle": median_metric(ok, "tcount_ratio_vs_oracle"),
                "median_qasm_ratio_vs_oracle": median_metric(ok, "qasm_ratio_vs_oracle"),
                "selected_objective_counts": format_counts(Counter(row.get("selected_objective", "") for row in ok)),
                "target_details": "; ".join(
                    f"{row['source_split']}/{row['target']}->{row.get('selected_objective', '-')}"
                    for row in items
                ),
            }
        )
    return sorted(rows, key=summary_sort_key)


def summary_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["policy"] != "oracle_posthoc",
        row["policy"] != "baseline_factor_count",
        -int(row["exact_oracle_matches"]),
        -int(row["joint_nonworse_vs_baseline"]),
        inf_if_missing(row["median_tcount_ratio_vs_baseline"]),
        inf_if_missing(row["median_qasm_ratio_vs_baseline"]),
        row["policy"],
    )


def count_bool(rows: list[dict[str, Any]], key: str) -> int:
    return sum(bool(row.get(key)) for row in rows)


def count_metric(rows: list[dict[str, Any]], key: str, *, strict: bool) -> int:
    return sum(metric_ok(row.get(key), strict=strict) for row in rows)


def count_joint_nonworse(rows: list[dict[str, Any]]) -> int:
    return sum(
        metric_ok(row.get("tcount_ratio_vs_baseline"), strict=False)
        and metric_ok(row.get("primary_ratio_vs_baseline"), strict=False)
        and metric_ok(row.get("qasm_ratio_vs_baseline"), strict=False)
        for row in rows
    )


def metric_ok(value: Any, *, strict: bool) -> bool:
    numeric = coerce_float(value)
    if numeric is None:
        return False
    return numeric < 1.0 if strict else numeric <= 1.0


def median_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = sorted(value for row in rows if (value := coerce_float(row.get(key))) is not None)
    if not values:
        return None
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2


def format_counts(counts: Counter[str]) -> str:
    clean = {key: value for key, value in counts.items() if key}
    if not clean:
        return "-"
    return ", ".join(f"{key}={value}" for key, value in sorted(clean.items()))


def format_weights(features: tuple[str, ...], weights: tuple[int, ...]) -> str:
    return ",".join(
        f"{feature}:{weight}"
        for feature, weight in zip(features, weights)
        if weight != 0
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, lineterminator="\n", fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_detail_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(
        path,
        rows,
        [
            "policy",
            "policy_type",
            "fold_target",
            "source_split",
            "target",
            "selection_status",
            "selected_objective",
            "fixed_objective",
            "oracle_objective",
            "exact_oracle_match",
            "weights",
            "selected_tcount",
            "selected_primary_nc_depth_ratio",
            "selected_qasm_depth",
            "baseline_tcount",
            "baseline_primary_nc_depth_ratio",
            "baseline_qasm_depth",
            "oracle_tcount",
            "oracle_primary_nc_depth_ratio",
            "oracle_qasm_depth",
            "tcount_ratio_vs_baseline",
            "primary_ratio_vs_baseline",
            "qasm_ratio_vs_baseline",
            "tcount_ratio_vs_oracle",
            "qasm_ratio_vs_oracle",
        ],
    )


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(
        path,
        rows,
        [
            "policy",
            "policy_type",
            "evaluated_groups",
            "ok_groups",
            "missing_groups",
            "exact_oracle_matches",
            "tcount_nonworse_vs_baseline",
            "tcount_wins_vs_baseline",
            "primary_nonworse_vs_baseline",
            "primary_wins_vs_baseline",
            "qasm_nonworse_vs_baseline",
            "qasm_wins_vs_baseline",
            "joint_nonworse_vs_baseline",
            "median_tcount_ratio_vs_baseline",
            "median_primary_ratio_vs_baseline",
            "median_qasm_ratio_vs_baseline",
            "median_tcount_ratio_vs_oracle",
            "median_qasm_ratio_vs_oracle",
            "selected_objective_counts",
            "target_details",
        ],
    )


def decision(summary: list[dict[str, Any]]) -> tuple[str, str]:
    split_rows = [row for row in summary if row["policy"].startswith("split_select_")]
    fixed = next((row for row in summary if row["policy"] == "best_fixed_loto"), None)
    baseline = next((row for row in summary if row["policy"] == "baseline_factor_count"), None)
    if not split_rows or baseline is None:
        return "not-ready", "The evaluation did not produce comparable Split-Select and baseline rows."
    best_split = max(
        split_rows,
        key=lambda row: (
            int(row["exact_oracle_matches"]),
            int(row["joint_nonworse_vs_baseline"]),
            -inf_if_missing(row["median_tcount_ratio_vs_baseline"]),
            -inf_if_missing(row["median_qasm_ratio_vs_baseline"]),
        ),
    )
    if fixed is not None and int(best_split["exact_oracle_matches"]) < int(fixed["exact_oracle_matches"]):
        return (
            "selector-not-yet-robust",
            f"The best Split-Select policy (`{best_split['policy']}`) tracks fewer oracle choices than the learned fixed-policy baseline.",
        )
    if int(best_split["tcount_wins_vs_baseline"]) == 0:
        return (
            "selector-not-yet-robust",
            f"The best Split-Select policy (`{best_split['policy']}`) does not produce T-count wins over the AlphaQuantum baseline.",
        )
    if inf_if_missing(best_split["median_qasm_ratio_vs_baseline"]) > 1.05:
        return (
            "selector-not-yet-robust",
            f"The best Split-Select policy (`{best_split['policy']}`) still has a median QASM-depth overhead above 5%.",
        )
    return (
        "prototype-integration-ready",
        f"The best Split-Select policy (`{best_split['policy']}`) is competitive with fixed-policy selection and improves T-count without a large median QASM-depth overhead.",
    )


def write_report(
    path: Path,
    summary: list[dict[str, Any]],
    summary_csv: Path,
    detail_csv: Path,
    figure_path: Path,
) -> None:
    status, reason = decision(summary)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# AlphaQuantum Split-Select Evaluation",
        "",
        f"Summary CSV: `{summary_csv}`.",
        f"Detail CSV: `{detail_csv}`.",
        f"Figure: `{figure_path}`.",
        "",
        "This report evaluates whether the consolidated objective-selection dataset is strong enough to train a deployable AlphaQuantum Split-Select policy. The selector chooses among the available AlphaQ objectives, including `factor_count`, `factor_count_pair_cap`, `mixed_pair`, and the article-inspired `frontier_pair`, using only AlphaQuantum/QASM-side candidate descriptors; ZX/feynver metrics are labels/audit targets only, not selector inputs.",
        "",
        "## Decision",
        "",
        f"Decision: `{status}`.",
        "",
        reason,
        "",
        "A positive decision here means the selector is ready for a prototype integration experiment. It does not yet mean the result is journal-ready; that still requires more external targets and a larger held-out battery.",
        "",
        "## Policy Comparison",
        "",
        "| policy | type | oracle matches | T wins | primary wins | QASM wins | all non-worse | median T | median primary | median QASM | selected objectives |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary:
        lines.append(
            "| {policy} | {policy_type} | {exact}/{ok} | {tw}/{ok} | {pw}/{ok} | {qw}/{ok} | {joint}/{ok} | {mt} | {mp} | {mq} | {counts} |".format(
                policy=row["policy"],
                policy_type=row["policy_type"],
                exact=row["exact_oracle_matches"],
                ok=row["ok_groups"],
                tw=row["tcount_wins_vs_baseline"],
                pw=row["primary_wins_vs_baseline"],
                qw=row["qasm_wins_vs_baseline"],
                joint=row["joint_nonworse_vs_baseline"],
                mt=fmt(row["median_tcount_ratio_vs_baseline"]),
                mp=fmt(row["median_primary_ratio_vs_baseline"]),
                mq=fmt(row["median_qasm_ratio_vs_baseline"]),
                counts=row["selected_objective_counts"],
            )
        )
    lines.extend(["", "## Target Details", ""])
    for row in summary:
        lines.append(f"- `{row['policy']}`: {row['target_details']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figure(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plot_rows = [
        row
        for row in rows
        if row["policy"] in {"baseline_factor_count", "best_fixed_loto", "oracle_posthoc"}
        or row["policy"].startswith("split_select_")
    ]
    labels = [row["policy"].replace("split_select_", "split_").replace("_", "\n") for row in plot_rows]
    x = range(len(plot_rows))
    width = 0.24
    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    specs = [
        ("exact_oracle_matches", "oracle matches", "#4c78a8"),
        ("tcount_wins_vs_baseline", "T wins", "#1b9e77"),
        ("qasm_wins_vs_baseline", "QASM wins", "#d95f02"),
    ]
    for offset, (field, label, color) in enumerate(specs):
        positions = [item + (offset - 1) * width for item in x]
        ax.bar(positions, [int(row[field]) for row in plot_rows], width=width, label=label, color=color)
    ax.set_title("AlphaQuantum Split-Select leave-one-target-out evaluation")
    ax.set_ylabel("target/run groups")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, max(int(row["ok_groups"]) for row in plot_rows) + 1 if plot_rows else 1)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=3)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def fmt(value: Any) -> str:
    numeric = coerce_float(value)
    return "" if numeric is None else f"{numeric:.3g}"


def main() -> int:
    args = parse_args()
    details = evaluation_rows(read_csv(args.dataset_csv))
    summaries = summary_rows(details)
    write_detail_csv(args.detail_csv, details)
    write_summary_csv(args.summary_csv, summaries)
    write_report(args.report_path, summaries, args.summary_csv, args.detail_csv, args.figure_path)
    write_figure(args.figure_path, summaries)
    print(f"Wrote {args.summary_csv}")
    print(f"Wrote {args.detail_csv}")
    print(f"Wrote {args.report_path}")
    print(f"Wrote {args.figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
