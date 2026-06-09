from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from collections import Counter
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_alphaq_objective_selection_dataset import OBJECTIVES
from scripts.run_best_objective_beam_ablation import safe_ratio
from scripts.structural_target import coerce_float


DEFAULT_DATASET = PROJECT_ROOT / "results" / "csv" / "alphaq_objective_selection_dataset.csv"
DEFAULT_DETAIL_CSV = PROJECT_ROOT / "results" / "csv" / "alphaq_circuit_objective_selector_details.csv"
DEFAULT_SUMMARY_CSV = PROJECT_ROOT / "results" / "csv" / "alphaq_circuit_objective_selector_summary.csv"
DEFAULT_REPORT = PROJECT_ROOT / "results" / "reports" / "alphaq_circuit_objective_selector.md"

CIRCUIT_FEATURES = (
    "n_qubits",
    "tensor_size",
    "original_tcount",
    "tensor_weight",
    "tensor_density",
    "tensor_index_degree_mean",
    "tensor_index_degree_max",
    "tensor_index_degree_std",
    "tensor_pair_graph_edges",
    "tensor_pair_graph_density",
    "tensor_pair_graph_degree_mean",
    "tensor_pair_graph_degree_max",
    "tensor_pair_graph_degree_std",
)
BASELINE_OBJECTIVE = "factor_count"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train an AlphaQ-only circuit-conditioned objective selector from the "
            "consolidated objective-selection dataset."
        )
    )
    parser.add_argument("--dataset-csv", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--detail-csv", type=Path, default=DEFAULT_DETAIL_CSV)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--shuffle-seed", type=int, default=17)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def group_key(row: dict[str, str]) -> tuple[str, str]:
    return row.get("source_split", ""), row.get("target", "")


def grouped_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        if row.get("target"):
            grouped.setdefault(group_key(row), []).append(row)
    return grouped


def train_ready_groups(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    return {
        key: items
        for key, items in grouped_rows(rows).items()
        if any(boolish(row.get("train_ready")) for row in items)
        and len([row for row in items if row.get("execution_status") == "ok" and boolish(row.get("has_beam_candidate"))]) >= 2
    }


def oracle_row(items: list[dict[str, str]]) -> dict[str, str]:
    labeled = next(
        (
            row
            for row in items
            if row.get("objective_variant") == row.get("oracle_objective")
            and row.get("execution_status") == "ok"
            and boolish(row.get("has_beam_candidate"))
        ),
        None,
    )
    if labeled is not None:
        return labeled
    candidates = [
        row
        for row in items
        if row.get("execution_status") == "ok" and boolish(row.get("has_beam_candidate"))
    ]
    return min(candidates, key=oracle_key)


def oracle_key(row: dict[str, str]) -> tuple[float, float, float, float, str]:
    return (
        inf_if_missing(row.get("best_beam_tcount")),
        inf_if_missing(row.get("best_beam_primary_nc_depth_ratio")),
        inf_if_missing(row.get("best_beam_qasm_depth")),
        inf_if_missing(row.get("objective_elapsed_sec")),
        row.get("objective_variant", ""),
    )


def inf_if_missing(value: Any) -> float:
    parsed = coerce_float(value)
    return float("inf") if parsed is None else parsed


def row_for_objective(items: list[dict[str, str]], objective: str) -> dict[str, str] | None:
    return next((row for row in items if row.get("objective_variant") == objective), None)


def tensor_features(target: str) -> dict[str, float]:
    from scripts.optimize_linear_span_candidate import load_target_tensor

    tensor = load_target_tensor(target).astype(np.uint8)
    tensor_size = int(tensor.shape[0])
    tensor_weight = int(np.count_nonzero(tensor))
    density = tensor_weight / max(tensor_size**3, 1)
    degrees = np.zeros((tensor_size,), dtype=float)
    edges: set[tuple[int, int]] = set()
    for triple in np.argwhere(tensor):
        unique = sorted({int(index) for index in triple})
        for index in unique:
            degrees[index] += 1.0
        for left_pos, left in enumerate(unique):
            for right in unique[left_pos + 1 :]:
                edges.add((left, right))
    pair_degrees = np.zeros((tensor_size,), dtype=float)
    for left, right in edges:
        pair_degrees[left] += 1.0
        pair_degrees[right] += 1.0
    max_edges = max(tensor_size * (tensor_size - 1) / 2, 1.0)
    return {
        "tensor_size": float(tensor_size),
        "tensor_weight": float(tensor_weight),
        "tensor_density": float(density),
        "tensor_index_degree_mean": float(np.mean(degrees)) if degrees.size else 0.0,
        "tensor_index_degree_max": float(np.max(degrees)) if degrees.size else 0.0,
        "tensor_index_degree_std": float(np.std(degrees)) if degrees.size else 0.0,
        "tensor_pair_graph_edges": float(len(edges)),
        "tensor_pair_graph_density": float(len(edges) / max_edges),
        "tensor_pair_graph_degree_mean": float(np.mean(pair_degrees)) if pair_degrees.size else 0.0,
        "tensor_pair_graph_degree_max": float(np.max(pair_degrees)) if pair_degrees.size else 0.0,
        "tensor_pair_graph_degree_std": float(np.std(pair_degrees)) if pair_degrees.size else 0.0,
    }


def group_features(items: list[dict[str, str]]) -> dict[str, float]:
    first = items[0]
    features = tensor_features(first["target"])
    for key in ("n_qubits", "original_tcount"):
        parsed = first_value(items, key)
        features[key] = 0.0 if parsed is None else parsed
    features["tensor_size"] = first_value(items, "tensor_size") or features["tensor_size"]
    return features


def first_value(items: list[dict[str, str]], key: str) -> float | None:
    for row in items:
        parsed = coerce_float(row.get(key))
        if parsed is not None and math.isfinite(parsed):
            return parsed
    return None


def feature_ranges(groups: dict[tuple[str, str], list[dict[str, str]]]) -> dict[str, tuple[float, float]]:
    values: dict[str, list[float]] = defaultdict(list)
    for items in groups.values():
        features = group_features(items)
        for key in CIRCUIT_FEATURES:
            value = features.get(key)
            if value is not None and math.isfinite(value):
                values[key].append(float(value))
    return {
        key: (min(vals), max(vals)) if vals else (0.0, 0.0)
        for key, vals in values.items()
    }


def normalized_vector(items: list[dict[str, str]], ranges: dict[str, tuple[float, float]]) -> tuple[float, ...]:
    features = group_features(items)
    out = []
    for key in CIRCUIT_FEATURES:
        value = features.get(key, 0.0)
        low, high = ranges.get(key, (0.0, 0.0))
        if high == low:
            out.append(0.0)
        else:
            out.append(max(0.0, min(1.0, (float(value) - low) / (high - low))))
    return tuple(out)


def euclidean(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def best_fixed_objective(train_items: list[list[dict[str, str]]]) -> str:
    counts = Counter(oracle_row(items)["objective_variant"] for items in train_items)
    if not counts:
        return BASELINE_OBJECTIVE
    return min(
        counts,
        key=lambda objective: (-counts[objective], OBJECTIVES.index(objective) if objective in OBJECTIVES else 99),
    )


def centroid_selector(
    *,
    train_items: list[list[dict[str, str]]],
    holdout_items: list[dict[str, str]],
    ranges: dict[str, tuple[float, float]],
    shuffled: bool = False,
    rng: random.Random | None = None,
) -> str:
    labels = [oracle_row(items)["objective_variant"] for items in train_items]
    if shuffled and rng is not None:
        labels = labels[:]
        rng.shuffle(labels)
    vectors_by_label: dict[str, list[tuple[float, ...]]] = defaultdict(list)
    for items, label in zip(train_items, labels):
        vectors_by_label[label].append(normalized_vector(items, ranges))
    if not vectors_by_label:
        return BASELINE_OBJECTIVE
    centroids = {
        label: tuple(float(np.mean(np.asarray(vectors), axis=0)[index]) for index in range(len(CIRCUIT_FEATURES)))
        for label, vectors in vectors_by_label.items()
    }
    holdout_vector = normalized_vector(holdout_items, ranges)
    return min(
        centroids,
        key=lambda label: (
            euclidean(holdout_vector, centroids[label]),
            OBJECTIVES.index(label) if label in OBJECTIVES else 99,
        ),
    )


def nearest_neighbor_selector(
    *,
    train_items: list[list[dict[str, str]]],
    holdout_items: list[dict[str, str]],
    ranges: dict[str, tuple[float, float]],
) -> str:
    if not train_items:
        return BASELINE_OBJECTIVE
    holdout_vector = normalized_vector(holdout_items, ranges)
    nearest = min(
        train_items,
        key=lambda items: euclidean(holdout_vector, normalized_vector(items, ranges)),
    )
    return oracle_row(nearest)["objective_variant"]


def knn_selector(
    *,
    train_items: list[list[dict[str, str]]],
    holdout_items: list[dict[str, str]],
    ranges: dict[str, tuple[float, float]],
    k: int = 3,
) -> str:
    if not train_items:
        return BASELINE_OBJECTIVE
    holdout_vector = normalized_vector(holdout_items, ranges)
    neighbors = sorted(
        (
            (
                euclidean(holdout_vector, normalized_vector(items, ranges)),
                oracle_row(items)["objective_variant"],
            )
            for items in train_items
        ),
        key=lambda item: (item[0], OBJECTIVES.index(item[1]) if item[1] in OBJECTIVES else 99),
    )[: max(1, k)]
    scores: dict[str, float] = defaultdict(float)
    for distance, label in neighbors:
        scores[label] += 1.0 / (distance + 1e-6)
    return min(
        scores,
        key=lambda label: (-scores[label], OBJECTIVES.index(label) if label in OBJECTIVES else 99),
    )


def softmax_selector(
    *,
    train_items: list[list[dict[str, str]]],
    holdout_items: list[dict[str, str]],
    ranges: dict[str, tuple[float, float]],
    seed: int,
    steps: int = 600,
    learning_rate: float = 0.35,
    l2: float = 0.05,
) -> str:
    labels = sorted(
        {oracle_row(items)["objective_variant"] for items in train_items},
        key=lambda objective: OBJECTIVES.index(objective) if objective in OBJECTIVES else 99,
    )
    if len(labels) <= 1:
        return labels[0] if labels else BASELINE_OBJECTIVE

    x_train = np.asarray([normalized_vector(items, ranges) for items in train_items], dtype=float)
    x_train = np.column_stack([x_train, np.ones((x_train.shape[0],), dtype=float)])
    y = np.asarray([labels.index(oracle_row(items)["objective_variant"]) for items in train_items], dtype=int)
    rng = np.random.default_rng(seed)
    weights = rng.normal(loc=0.0, scale=0.01, size=(x_train.shape[1], len(labels)))
    counts = Counter(int(label) for label in y)
    sample_weights = np.asarray([1.0 / math.sqrt(counts[int(label)]) for label in y], dtype=float)
    sample_weights /= max(float(np.mean(sample_weights)), 1e-12)

    for _step in range(steps):
        logits = x_train @ weights
        logits -= np.max(logits, axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= np.sum(probs, axis=1, keepdims=True)
        probs[np.arange(len(y)), y] -= 1.0
        probs *= sample_weights[:, None] / max(len(y), 1)
        grad = x_train.T @ probs
        grad[:-1, :] += l2 * weights[:-1, :]
        weights -= learning_rate * grad

    holdout = np.asarray(normalized_vector(holdout_items, ranges), dtype=float)
    holdout = np.concatenate([holdout, np.ones((1,), dtype=float)])
    logits = holdout @ weights
    return labels[int(np.argmax(logits))]


def stable_seed(label: str, base_seed: int) -> int:
    total = base_seed
    for char in label:
        total = (total * 131 + ord(char)) % (2**32 - 1)
    return total


def evaluation_rows(groups: dict[tuple[str, str], list[dict[str, str]]], *, shuffle_seed: int) -> list[dict[str, Any]]:
    ranges = feature_ranges(groups)
    targets = sorted({target for _split, target in groups})
    rows: list[dict[str, Any]] = []
    rng = random.Random(shuffle_seed)
    for holdout in targets:
        train_items = [items for (_split, target), items in groups.items() if target != holdout]
        fixed = best_fixed_objective(train_items)
        for key, items in groups.items():
            if key[1] != holdout:
                continue
            policies = {
                "baseline_factor_count": BASELINE_OBJECTIVE,
                "best_fixed_loto": fixed,
                "circuit_1nn": nearest_neighbor_selector(
                    train_items=train_items,
                    holdout_items=items,
                    ranges=ranges,
                ),
                "circuit_knn3": knn_selector(
                    train_items=train_items,
                    holdout_items=items,
                    ranges=ranges,
                    k=3,
                ),
                "circuit_nearest_centroid": centroid_selector(
                    train_items=train_items,
                    holdout_items=items,
                    ranges=ranges,
                ),
                "circuit_softmax": softmax_selector(
                    train_items=train_items,
                    holdout_items=items,
                    ranges=ranges,
                    seed=stable_seed(holdout, shuffle_seed),
                ),
                "shuffled_centroid_control": centroid_selector(
                    train_items=train_items,
                    holdout_items=items,
                    ranges=ranges,
                    shuffled=True,
                    rng=rng,
                ),
                "oracle_posthoc": oracle_row(items)["objective_variant"],
            }
            for policy, objective in policies.items():
                rows.append(policy_row(policy, key, objective, items))
    return rows


def policy_row(
    policy: str,
    key: tuple[str, str],
    objective: str,
    items: list[dict[str, str]],
) -> dict[str, Any]:
    selected = row_for_objective(items, objective)
    oracle = oracle_row(items)
    baseline = row_for_objective(items, BASELINE_OBJECTIVE)
    if selected is None or selected.get("execution_status") != "ok" or not boolish(selected.get("has_beam_candidate")):
        return {
            "policy": policy,
            "source_split": key[0],
            "target": key[1],
            "selection_status": "missing-selected-objective",
            "selected_objective": objective,
            "oracle_objective": oracle.get("objective_variant", ""),
        }
    return {
        "policy": policy,
        "source_split": key[0],
        "target": key[1],
        "selection_status": "ok",
        "selected_objective": objective,
        "oracle_objective": oracle["objective_variant"],
        "exact_oracle_match": objective == oracle["objective_variant"],
        "selected_tcount": selected.get("best_beam_tcount", ""),
        "selected_primary_nc_depth_ratio": selected.get("best_beam_primary_nc_depth_ratio", ""),
        "selected_qasm_depth": selected.get("best_beam_qasm_depth", ""),
        "selected_runtime_sec": selected.get("objective_elapsed_sec", ""),
        "baseline_tcount": "" if baseline is None else baseline.get("best_beam_tcount", ""),
        "baseline_primary_nc_depth_ratio": "" if baseline is None else baseline.get("best_beam_primary_nc_depth_ratio", ""),
        "baseline_qasm_depth": "" if baseline is None else baseline.get("best_beam_qasm_depth", ""),
        "baseline_runtime_sec": "" if baseline is None else baseline.get("objective_elapsed_sec", ""),
        "oracle_tcount": oracle.get("best_beam_tcount", ""),
        "oracle_primary_nc_depth_ratio": oracle.get("best_beam_primary_nc_depth_ratio", ""),
        "oracle_qasm_depth": oracle.get("best_beam_qasm_depth", ""),
        "oracle_runtime_sec": oracle.get("objective_elapsed_sec", ""),
        "tcount_ratio_vs_baseline": "" if baseline is None else safe_ratio(selected.get("best_beam_tcount"), baseline.get("best_beam_tcount")),
        "primary_ratio_vs_baseline": "" if baseline is None else safe_ratio(selected.get("best_beam_primary_nc_depth_ratio"), baseline.get("best_beam_primary_nc_depth_ratio")),
        "qasm_ratio_vs_baseline": "" if baseline is None else safe_ratio(selected.get("best_beam_qasm_depth"), baseline.get("best_beam_qasm_depth")),
        "runtime_ratio_vs_baseline": "" if baseline is None else safe_ratio(selected.get("objective_elapsed_sec"), baseline.get("objective_elapsed_sec")),
        "tcount_ratio_vs_oracle": safe_ratio(selected.get("best_beam_tcount"), oracle.get("best_beam_tcount")),
        "qasm_ratio_vs_oracle": safe_ratio(selected.get("best_beam_qasm_depth"), oracle.get("best_beam_qasm_depth")),
    }


def metric_ok(value: Any, *, strict: bool) -> bool:
    parsed = coerce_float(value)
    if parsed is None:
        return False
    return parsed < 1.0 if strict else parsed <= 1.0


def median(values: list[float]) -> float | str:
    values = sorted(value for value in values if math.isfinite(value))
    if not values:
        return ""
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2


def summary_rows(details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for policy in sorted({row["policy"] for row in details}):
        items = [row for row in details if row["policy"] == policy]
        ok = [row for row in items if row.get("selection_status") == "ok"]
        rows.append(
            {
                "policy": policy,
                "evaluated_groups": len(items),
                "ok_groups": len(ok),
                "exact_oracle_matches": sum(bool(row.get("exact_oracle_match")) for row in ok),
                "tcount_nonworse_vs_baseline": sum(metric_ok(row.get("tcount_ratio_vs_baseline"), strict=False) for row in ok),
                "tcount_wins_vs_baseline": sum(metric_ok(row.get("tcount_ratio_vs_baseline"), strict=True) for row in ok),
                "primary_wins_vs_baseline": sum(metric_ok(row.get("primary_ratio_vs_baseline"), strict=True) for row in ok),
                "qasm_wins_vs_baseline": sum(metric_ok(row.get("qasm_ratio_vs_baseline"), strict=True) for row in ok),
                "runtime_wins_vs_baseline": sum(metric_ok(row.get("runtime_ratio_vs_baseline"), strict=True) for row in ok),
                "joint_nonworse_vs_baseline": sum(
                    metric_ok(row.get("tcount_ratio_vs_baseline"), strict=False)
                    and metric_ok(row.get("qasm_ratio_vs_baseline"), strict=False)
                    for row in ok
                ),
                "median_tcount_ratio_vs_baseline": median_metric(ok, "tcount_ratio_vs_baseline"),
                "median_qasm_ratio_vs_baseline": median_metric(ok, "qasm_ratio_vs_baseline"),
                "median_runtime_ratio_vs_baseline": median_metric(ok, "runtime_ratio_vs_baseline"),
                "selected_objective_counts": format_counts(Counter(row.get("selected_objective", "") for row in ok)),
            }
        )
    return sorted(rows, key=lambda row: (row["policy"] != "oracle_posthoc", -int(row["exact_oracle_matches"]), row["policy"]))


def median_metric(rows: list[dict[str, Any]], key: str) -> float | str:
    return median([value for row in rows if (value := coerce_float(row.get(key))) is not None])


def format_counts(counts: Counter[str]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()) if key) or "-"


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
            "source_split",
            "target",
            "selection_status",
            "selected_objective",
            "oracle_objective",
            "exact_oracle_match",
            "selected_tcount",
            "selected_primary_nc_depth_ratio",
            "selected_qasm_depth",
            "selected_runtime_sec",
            "baseline_tcount",
            "baseline_primary_nc_depth_ratio",
            "baseline_qasm_depth",
            "baseline_runtime_sec",
            "oracle_tcount",
            "oracle_primary_nc_depth_ratio",
            "oracle_qasm_depth",
            "oracle_runtime_sec",
            "tcount_ratio_vs_baseline",
            "primary_ratio_vs_baseline",
            "qasm_ratio_vs_baseline",
            "runtime_ratio_vs_baseline",
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
            "evaluated_groups",
            "ok_groups",
            "exact_oracle_matches",
            "tcount_nonworse_vs_baseline",
            "tcount_wins_vs_baseline",
            "primary_wins_vs_baseline",
            "qasm_wins_vs_baseline",
            "runtime_wins_vs_baseline",
            "joint_nonworse_vs_baseline",
            "median_tcount_ratio_vs_baseline",
            "median_qasm_ratio_vs_baseline",
            "median_runtime_ratio_vs_baseline",
            "selected_objective_counts",
        ],
    )


def write_report(path: Path, summary: list[dict[str, Any]], detail_csv: Path, summary_csv: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# AlphaQ circuit-conditioned objective selector",
        "",
        f"Summary CSV: `{summary_csv}`.",
        f"Detail CSV: `{detail_csv}`.",
        "",
        "This selector uses only pre-run AlphaQ/circuit tensor features. ZX and materialized metrics are labels/evaluation targets only.",
        "",
        "| policy | groups | exact oracle | T non-worse | T wins | QASM wins | runtime wins | median T ratio | median QASM ratio | selected objectives |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary:
        lines.append(
            "| {policy} | {groups} | {exact} | {tnw} | {tw} | {qw} | {rw} | {mt} | {mq} | {counts} |".format(
                policy=row["policy"],
                groups=row["ok_groups"],
                exact=row["exact_oracle_matches"],
                tnw=row["tcount_nonworse_vs_baseline"],
                tw=row["tcount_wins_vs_baseline"],
                qw=row["qasm_wins_vs_baseline"],
                rw=row["runtime_wins_vs_baseline"],
                mt=fmt(row["median_tcount_ratio_vs_baseline"]),
                mq=fmt(row["median_qasm_ratio_vs_baseline"]),
                counts=row["selected_objective_counts"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: Any) -> str:
    parsed = coerce_float(value)
    return "" if parsed is None else f"{parsed:.3g}"


def main() -> int:
    args = parse_args()
    groups = train_ready_groups(read_csv(args.dataset_csv))
    details = evaluation_rows(groups, shuffle_seed=args.shuffle_seed)
    summary = summary_rows(details)
    write_detail_csv(args.detail_csv, details)
    write_summary_csv(args.summary_csv, summary)
    write_report(args.report_path, summary, args.detail_csv, args.summary_csv)
    print(f"Wrote {args.detail_csv}")
    print(f"Wrote {args.summary_csv}")
    print(f"Wrote {args.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
