from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import hashlib
import itertools
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._analysis_common import DEFAULT_CSV_ROOT
from scripts._analysis_common import DEFAULT_FIGURES_ROOT
from scripts._analysis_common import DEFAULT_REPORTS_ROOT
from scripts._analysis_common import DECOMPOSITIONS_ROOT
from scripts._analysis_common import ensure_dir
from scripts._analysis_common import natural_sort_key
from scripts._analysis_common import resolve_artifact_stem
from scripts._analysis_common import write_csv_rows
from scripts._manifest import append_command
from scripts.tensor_split_core import default_partitions
from scripts.tensor_split_core import group_multiset_gadgets
from scripts.tensor_split_core import mixed_auc
from scripts.tensor_split_core import mixed_weight
from scripts.tensor_split_core import raw_bridge_count
from scripts.tensor_split_core import semantic_partitions_from_qasm
from scripts.tensor_split_core import tensor_from_factors
from scripts.tensor_split_core import tensor_split_v3_stats


GADGET_METHOD = "public_resynth_gadgets"
NO_GADGET_METHOD = "public_resynth_no_gadgets"
GADGET_FAMILIES = (
    "benchmarks_gadgets.npz",
    "binary_addition.npz",
    "hamming_weight_phase_gradient.npz",
    "multiplication_finite_fields_gadgets.npz",
    "quantum_chemistry.npz",
    "unary_iteration_gadgets.npz",
)
NO_GADGET_FAMILIES = (
    "benchmarks_no_gadgets.npz",
    "multiplication_finite_fields_no_gadgets.npz",
    "unary_iteration_no_gadgets.npz",
)

DEFAULT_CANDIDATE_CSVS = (
    DEFAULT_CSV_ROOT / "splitting_candidate_diagnostics.csv",
    DEFAULT_CSV_ROOT / "alphaq_final_model_predictions.csv",
)
DEFAULT_INVENTORY_CSV = DEFAULT_CSV_ROOT / "circuit_inventory.csv"
DEFAULT_OUTPUT_CSV = DEFAULT_CSV_ROOT / "tensor_split_candidate_metrics.csv"
DEFAULT_REPORT_PATH = DEFAULT_REPORTS_ROOT / "tensor_split_evidence_report.md"
DEFAULT_FIGURE_PATH = (
    DEFAULT_FIGURES_ROOT
    / "tensor_split_evidence"
    / "tensor_split_v3_metric_ablation.png"
)
DEFAULT_RANDOM_SEEDS = tuple(range(50))
SCORE_V3_RANK_LAMBDA = 0.25
SCORE_V3_RANK_MU = 0.05
METRIC_SPECS: dict[str, tuple[str, ...]] = {
    "gadget_aware_mixed_cost": ("gadget_aware_mixed_cost",),
    "gadget_mixed_weight": ("gadget_mixed_weight",),
    "mixed_excess_norm": ("mixed_excess_norm",),
    "mixed_auc_original_norm": ("mixed_auc_original_norm",),
    "mixed_auc_greedy_norm": ("mixed_auc_greedy_norm",),
    "score_v3_lex": (
        "mixed_excess_norm",
        "mixed_auc_greedy_norm",
        "singleton_bridge_count_norm",
    ),
    "score_v3_rank": ("score_v3_rank",),
}
REPORT_METRICS = (
    "gadget_aware_mixed_cost",
    "gadget_mixed_weight",
    "mixed_excess_norm",
    "mixed_auc_original_norm",
    "mixed_auc_greedy_norm",
    "score_v3_lex",
    "score_v3_rank",
)
FORMAL_CORE_CIRCUITS = (
    "gf_2pow2_mult",
    "mod_5_4",
    "qft_4",
    "hamming_weight_n4",
    "hamming_weight_n5",
)


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def merged_candidate_rows(paths: list[Path]) -> list[dict[str, str]]:
    merged: dict[str, dict[str, str]] = {}
    for path in paths:
        for row in load_csv_rows(path):
            candidate_id = row.get("candidate_id")
            if not candidate_id:
                continue
            current = merged.setdefault(candidate_id, {})
            for key, value in row.items():
                if value not in (None, "") or key not in current:
                    current[key] = value
    return sorted(
        merged.values(),
        key=lambda row: (natural_sort_key(row.get("circuit_id", "")), row.get("candidate_id", "")),
    )


def load_inventory_index(path: Path) -> dict[str, dict[str, str]]:
    return {row["circuit_id"]: row for row in load_csv_rows(path) if row.get("circuit_id")}


def load_decomposition_index() -> dict[tuple[str, str], Path]:
    index: dict[tuple[str, str], Path] = {}
    for family_file in NO_GADGET_FAMILIES:
        path = DECOMPOSITIONS_ROOT / family_file
        if not path.exists():
            continue
        with np.load(path, allow_pickle=True) as data:
            for key in data.files:
                index[(NO_GADGET_METHOD, key)] = path
    for family_file in GADGET_FAMILIES:
        path = DECOMPOSITIONS_ROOT / family_file
        if not path.exists():
            continue
        with np.load(path, allow_pickle=True) as data:
            for key in data.files:
                index[(GADGET_METHOD, key)] = path
    return index


def tensor_split_rows(
    candidate_rows: list[dict[str, str]],
    *,
    inventory_index: dict[str, dict[str, str]],
    decomposition_index: dict[tuple[str, str], Path],
    random_seeds: tuple[int, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate_row in candidate_rows:
        block_rows = rows_for_candidate(
            candidate_row,
            inventory_index=inventory_index,
            decomposition_index=decomposition_index,
            random_seeds=random_seeds,
        )
        rows.extend(block_rows)
        rows.extend(aggregate_candidate_rows(block_rows))
    return rows


def rows_for_candidate(
    candidate_row: dict[str, str],
    *,
    inventory_index: dict[str, dict[str, str]],
    decomposition_index: dict[tuple[str, str], Path],
    random_seeds: tuple[int, ...],
) -> list[dict[str, Any]]:
    circuit_id = candidate_row.get("circuit_id", "")
    compile_dir = resolve_compile_dir(circuit_id, inventory_index)
    qasm_path = resolve_qasm_path(circuit_id, inventory_index)
    if compile_dir is None:
        return [
            base_output_row(candidate_row)
            | {
                "metric_scope": "candidate",
                "tensor_split_status": "missing-compile-dir",
                "tensor_split_error": f"No compile directory for {circuit_id}",
            }
        ]

    keys = split_field(candidate_row.get("decomposition_keys"))
    indices = [int(value) for value in split_field(candidate_row.get("candidate_indices"))]
    methods = split_field(candidate_row.get("source_methods"))
    if len(methods) == 1 and len(keys) > 1:
        methods = methods * len(keys)
    if not (len(keys) == len(indices) == len(methods)):
        return [
            base_output_row(candidate_row)
            | {
                "metric_scope": "candidate",
                "tensor_split_status": "invalid-candidate-fields",
                "tensor_split_error": (
                    f"keys={len(keys)}, indices={len(indices)}, methods={len(methods)}"
                ),
            }
        ]

    rows: list[dict[str, Any]] = []
    for block_index, (key, candidate_index, source_method) in enumerate(
        zip(keys, indices, methods)
    ):
        block_result = rows_for_block(
            candidate_row,
            compile_dir=compile_dir,
            qasm_path=qasm_path,
            block_index=block_index,
            decomposition_key=key,
            candidate_index=candidate_index,
            source_method=source_method,
            decomposition_index=decomposition_index,
            random_seeds=random_seeds,
        )
        rows.extend(block_result)
    return rows


def rows_for_block(
    candidate_row: dict[str, str],
    *,
    compile_dir: Path,
    qasm_path: Path | None,
    block_index: int,
    decomposition_key: str,
    candidate_index: int,
    source_method: str,
    decomposition_index: dict[tuple[str, str], Path],
    random_seeds: tuple[int, ...],
) -> list[dict[str, Any]]:
    artifact_stem = resolve_artifact_stem(compile_dir, decomposition_key)
    if artifact_stem is None:
        return [
            base_output_row(candidate_row)
            | {
                "metric_scope": "block",
                "block_index": block_index,
                "decomposition_key": decomposition_key,
                "source_method": source_method,
                "tensor_split_status": "missing-artifact-stem",
                "tensor_split_error": f"No tensor artifact for {decomposition_key}",
            }
        ]

    tensor_path = compile_dir / f"{artifact_stem}.tensor.npy"
    if not tensor_path.exists():
        return [
            base_output_row(candidate_row)
            | {
                "metric_scope": "block",
                "block_index": block_index,
                "artifact_stem": artifact_stem,
                "decomposition_key": decomposition_key,
                "source_method": source_method,
                "tensor_split_status": "missing-tensor",
                "tensor_split_error": str(tensor_path),
            }
        ]

    npz_path = decomposition_index.get((source_method, decomposition_key))
    if npz_path is None:
        return [
            base_output_row(candidate_row)
            | {
                "metric_scope": "block",
                "block_index": block_index,
                "artifact_stem": artifact_stem,
                "decomposition_key": decomposition_key,
                "source_method": source_method,
                "tensor_split_status": "missing-decomposition",
                "tensor_split_error": f"{source_method}:{decomposition_key}",
            }
        ]

    try:
        tensor = np.load(tensor_path, allow_pickle=True).astype(np.uint8) % 2
        with np.load(npz_path, allow_pickle=True) as data:
            factors = np.asarray(data[decomposition_key][candidate_index], dtype=np.uint8) % 2
    except Exception as exc:
        return [
            base_output_row(candidate_row)
            | {
                "metric_scope": "block",
                "block_index": block_index,
                "artifact_stem": artifact_stem,
                "decomposition_key": decomposition_key,
                "source_method": source_method,
                "tensor_split_status": "load-failed",
                "tensor_split_error": str(exc),
            }
        ]

    semantic_partitions = (
        semantic_partitions_from_qasm(
            circuit_id=str(candidate_row.get("circuit_id") or ""),
            qasm_path=qasm_path,
            mapping_path=compile_dir / f"{artifact_stem}.mapping.txt",
            tensor_size=tensor.shape[0],
        )
        if qasm_path is not None and qasm_path.exists()
        else []
    )
    partitions = [
        *semantic_partitions,
        *default_partitions(tensor, random_seeds=random_seeds),
    ]
    groups = group_multiset_gadgets(factors)
    reconstructed = tensor_from_factors(factors, tensor_size=tensor.shape[0])
    residual = tensor ^ reconstructed
    output_rows: list[dict[str, Any]] = []
    for partition in partitions:
        v3_stats = tensor_split_v3_stats(tensor, factors, partition, groups=groups)
        effective_mixed_cost = v3_stats["gadget_aware_effective_mixed_cost"]
        output_rows.append(
            base_output_row(candidate_row)
            | {
                "metric_scope": "block",
                "tensor_split_status": "ok",
                "tensor_split_error": None,
                "block_index": block_index,
                "artifact_stem": artifact_stem,
                "decomposition_key": decomposition_key,
                "source_method": source_method,
                "candidate_index": candidate_index,
                "npz_path": str(npz_path.relative_to(PROJECT_ROOT)),
                "tensor_path": str(tensor_path.relative_to(PROJECT_ROOT)),
                "partition_id": partition.partition_id,
                "partition_kind": partition.kind,
                "semantic_partition_status": partition.semantic_partition_status,
                "semantic_partition_error": partition.semantic_partition_error,
                "partition_block_sizes": ";".join(map(str, partition.block_sizes)),
                "tensor_size": tensor.shape[0],
                "factor_count": factors.shape[0],
                "raw_bridge_count": raw_bridge_count(factors, partition),
                "gadget_aware_mixed_cost": effective_mixed_cost,
                **v3_stats,
                "mixed_auc": mixed_auc(tensor, factors, partition),
                "initial_mixed_weight": mixed_weight(tensor, partition),
                "final_mixed_weight": mixed_weight(residual, partition),
                "reconstruction_residual_weight": int(np.count_nonzero(residual)),
                "num_factor_groups": len(groups),
                "num_toffoli_groups": sum(group.group_type == "toffoli" for group in groups),
                "num_cs_groups": sum(group.group_type == "cs" for group in groups),
                "num_single_factor_groups": sum(group.group_type == "factor" for group in groups),
            }
        )
    return output_rows


def aggregate_candidate_rows(block_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ok_rows = [row for row in block_rows if row.get("tensor_split_status") == "ok"]
    if not ok_rows:
        return block_rows if block_rows and block_rows[0].get("metric_scope") == "candidate" else []
    by_partition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ok_rows:
        by_partition[str(row["partition_id"])].append(row)
    aggregate_rows: list[dict[str, Any]] = []
    for partition_id, rows in sorted(by_partition.items(), key=lambda item: natural_sort_key(item[0])):
        first = rows[0]
        tensor_size = sum_number(rows, "tensor_size")
        factor_count = sum_number(rows, "factor_count")
        target_mixed_weight = sum_number(rows, "target_mixed_weight")
        gadget_mixed_weight = sum_number(rows, "gadget_mixed_weight")
        num_factor_groups = sum_number(rows, "num_factor_groups")
        mixed_group_count = sum_number(rows, "mixed_group_count")
        singleton_bridge_count = sum_number(rows, "singleton_bridge_count")
        mixed_auc_original = sum_number(rows, "mixed_auc_original")
        mixed_auc_greedy = sum_number(rows, "mixed_auc_greedy")
        off_target_mixed_weight = sum_number(rows, "off_target_mixed_weight")
        target_denominator = max(target_mixed_weight, 1)
        auc_denominator = (num_factor_groups + len(rows)) * target_denominator
        stable_hash = stable_hash_from_parts(
            str(row.get("stable_factor_hash") or "") for row in rows
        )
        mixed_excess_norm = (
            (gadget_mixed_weight - target_mixed_weight) / target_denominator
        )
        mixed_auc_greedy_norm = mixed_auc_greedy / auc_denominator
        singleton_bridge_count_norm = singleton_bridge_count / max(factor_count, 1)
        aggregate_rows.append(
            base_output_row(first)
            | {
                "metric_scope": "candidate",
                "tensor_split_status": "ok",
                "tensor_split_error": None,
                "partition_id": partition_id,
                "partition_kind": first.get("partition_kind"),
                "semantic_partition_status": first.get("semantic_partition_status"),
                "semantic_partition_error": first.get("semantic_partition_error"),
                "num_blocks": len(rows),
                "tensor_size": tensor_size,
                "factor_count": factor_count,
                "raw_bridge_count": sum_number(rows, "raw_bridge_count"),
                "gadget_aware_mixed_cost": sum_number(rows, "gadget_aware_mixed_cost"),
                "gadget_aware_effective_mixed_cost": sum_number(
                    rows, "gadget_aware_effective_mixed_cost"
                ),
                "gadget_mixed_weight": gadget_mixed_weight,
                "mixed_group_count": mixed_group_count,
                "mixed_block_span": sum_number(rows, "mixed_block_span"),
                "mixed_auc": sum_number(rows, "mixed_auc"),
                "target_mixed_weight": target_mixed_weight,
                "gadget_mixed_weight_norm": gadget_mixed_weight
                / target_denominator,
                "mixed_excess_norm": mixed_excess_norm,
                "mixed_auc_original": mixed_auc_original,
                "mixed_auc_original_norm": mixed_auc_original / auc_denominator,
                "mixed_auc_greedy": mixed_auc_greedy,
                "mixed_auc_greedy_norm": mixed_auc_greedy_norm,
                "off_target_mixed_weight": off_target_mixed_weight,
                "off_target_mixed_weight_norm": off_target_mixed_weight
                / target_denominator,
                "mixed_group_count_norm": mixed_group_count
                / max(num_factor_groups, 1),
                "singleton_bridge_count": singleton_bridge_count,
                "singleton_bridge_count_norm": singleton_bridge_count_norm,
                "stable_factor_hash": stable_hash,
                "score_v3_lex": (
                    f"{mixed_excess_norm:.12g};"
                    f"{mixed_auc_greedy_norm:.12g};"
                    f"{singleton_bridge_count_norm:.12g};"
                    f"{stable_hash}"
                ),
                "initial_mixed_weight": sum_number(rows, "initial_mixed_weight"),
                "final_mixed_weight": sum_number(rows, "final_mixed_weight"),
                "reconstruction_residual_weight": sum_number(
                    rows, "reconstruction_residual_weight"
                ),
                "num_factor_groups": num_factor_groups,
                "num_toffoli_groups": sum_number(rows, "num_toffoli_groups"),
                "num_cs_groups": sum_number(rows, "num_cs_groups"),
                "num_single_factor_groups": sum_number(rows, "num_single_factor_groups"),
            }
        )
    return aggregate_rows


def base_output_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "circuit_id": row.get("circuit_id"),
        "candidate_id": row.get("candidate_id"),
        "combo_index": row.get("combo_index"),
        "selection_status": row.get("selection_status"),
        "frontier_verification_status": row.get("frontier_verification_status"),
        "source_methods": row.get("source_methods"),
        "decomposition_keys": row.get("decomposition_keys"),
        "candidate_indices": row.get("candidate_indices"),
        "primary_nc_depth_ratio": row.get("primary_nc_depth_ratio"),
        "zx_total_depth_ratio": row.get("zx_total_depth_ratio"),
        "qasm_depth_ratio": row.get("qasm_depth_ratio"),
        "tcount_after": row.get("tcount_after"),
        "tdepth_after": row.get("tdepth_after"),
        "structural_cost": row.get("structural_cost"),
        "alphaq_dependency_core_area_ratio": row.get(
            "alphaq_dependency_core_area_ratio"
        ),
    }


def resolve_compile_dir(
    circuit_id: str,
    inventory_index: dict[str, dict[str, str]],
) -> Path | None:
    inventory_row = inventory_index.get(circuit_id)
    if inventory_row and inventory_row.get("vendored_compile_dir"):
        candidate = PROJECT_ROOT / inventory_row["vendored_compile_dir"]
        if candidate.exists():
            return candidate
    fallback = PROJECT_ROOT / "results" / "compile_stage1" / "quizx" / circuit_id
    return fallback if fallback.exists() else None


def resolve_qasm_path(
    circuit_id: str,
    inventory_index: dict[str, dict[str, str]],
) -> Path | None:
    inventory_row = inventory_index.get(circuit_id)
    if inventory_row and inventory_row.get("qasm_path"):
        candidate = PROJECT_ROOT / inventory_row["qasm_path"]
        if candidate.exists():
            return candidate
    fallback = PROJECT_ROOT / "results" / "compile_stage1" / "quizx" / circuit_id / f"{circuit_id}.hopt.qasm"
    return fallback if fallback.exists() else None


def split_field(value: str | None) -> list[str]:
    if value is None or value == "":
        return []
    return [part for part in str(value).split(";") if part != ""]


def sum_number(rows: list[dict[str, Any]], key: str) -> int:
    return int(sum(int(float(row.get(key) or 0)) for row in rows))


def coerce_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def stable_hash_from_parts(parts: Any) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def add_score_v3_rank(rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("metric_scope") != "candidate" or row.get("tensor_split_status") != "ok":
            continue
        tcount = coerce_float(row.get("tcount_after"))
        if tcount is None:
            continue
        grouped[(str(row["partition_id"]), str(row["circuit_id"]), int(tcount))].append(row)

    for bucket in grouped.values():
        usable = [
            row
            for row in bucket
            if coerce_float(row.get("mixed_excess_norm")) is not None
            and coerce_float(row.get("mixed_auc_greedy_norm")) is not None
            and coerce_float(row.get("singleton_bridge_count_norm")) is not None
        ]
        if not usable:
            continue
        excess_ranks = rank_values(
            [coerce_float(row["mixed_excess_norm"]) for row in usable]  # type: ignore[list-item]
        )
        auc_ranks = rank_values(
            [coerce_float(row["mixed_auc_greedy_norm"]) for row in usable]  # type: ignore[list-item]
        )
        singleton_ranks = rank_values(
            [
                coerce_float(row["singleton_bridge_count_norm"])  # type: ignore[list-item]
                for row in usable
            ]
        )
        for row, excess_rank, auc_rank, singleton_rank in zip(
            usable, excess_ranks, auc_ranks, singleton_ranks
        ):
            row["score_v3_rank"] = (
                excess_rank
                + SCORE_V3_RANK_LAMBDA * auc_rank
                + SCORE_V3_RANK_MU * singleton_rank
            )


def metric_score_tuple(row: dict[str, Any], metric_id: str) -> tuple[float, ...] | None:
    fields = METRIC_SPECS[metric_id]
    values = tuple(coerce_float(row.get(field)) for field in fields)
    if any(value is None for value in values):
        return None
    return values  # type: ignore[return-value]


def metric_scalar(row: dict[str, Any], metric_id: str) -> float | None:
    score = metric_score_tuple(row, metric_id)
    return None if score is None else score[0]


def blind_sort_key(row: dict[str, Any], metric_id: str) -> tuple[Any, ...]:
    score = metric_score_tuple(row, metric_id)
    if score is None:
        return (float("inf"), str(row.get("stable_factor_hash") or ""), str(row.get("candidate_id") or ""))
    return (
        *score,
        str(row.get("stable_factor_hash") or ""),
        str(row.get("candidate_id") or ""),
    )


def metric_key_string(row: dict[str, Any], metric_id: str) -> str:
    score = metric_score_tuple(row, metric_id)
    if score is None:
        return ""
    return ";".join(f"{value:.12g}" for value in score)


def evidence_summaries(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], str]:
    candidate_rows = [
        row
        for row in rows
        if row.get("metric_scope") == "candidate"
        and row.get("tensor_split_status") == "ok"
        and coerce_float(row.get("primary_nc_depth_ratio")) is not None
        and any(
            metric_score_tuple(row, metric_id) is not None
            for metric_id in METRIC_SPECS
        )
    ]
    by_partition_metric: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        for metric_id in METRIC_SPECS:
            if metric_score_tuple(row, metric_id) is not None:
                by_partition_metric[(metric_id, str(row["partition_id"]))].append(row)

    bucket_summaries = controlled_tcount_buckets(candidate_rows)
    bucket_rows_by_partition_metric: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in bucket_summaries:
        bucket_rows_by_partition_metric[
            (str(row["metric_id"]), str(row["partition_id"]))
        ].append(row)

    partition_summaries: list[dict[str, Any]] = []
    for (metric_id, partition_id), partition_rows in sorted(
        by_partition_metric.items(),
        key=lambda item: (natural_sort_key(item[0][0]), natural_sort_key(item[0][1])),
    ):
        costs = [metric_scalar(row, metric_id) for row in partition_rows]
        primary = [coerce_float(row.get("primary_nc_depth_ratio")) for row in partition_rows]
        tcounts = [coerce_float(row.get("tcount_after")) for row in partition_rows]
        pairs = [
            (cost, ratio)
            for cost, ratio in zip(costs, primary)
            if cost is not None and ratio is not None
        ]
        partition_bucket_rows = bucket_rows_by_partition_metric.get(
            (metric_id, partition_id), []
        )
        partition_summaries.append(
            {
                "metric_id": metric_id,
                "partition_id": partition_id,
                "partition_kind": partition_rows[0].get("partition_kind"),
                "num_candidates": len(partition_rows),
                "num_circuits": len({row.get("circuit_id") for row in partition_rows}),
                "num_tcount_buckets": len(partition_bucket_rows),
                "mean_pairwise_win_rate": mean(
                    row.get("within_tcount_pairwise_win_rate")
                    for row in partition_bucket_rows
                ),
                "mean_topk_reranking_gain": mean(
                    row.get("topk_reranking_gain") for row in partition_bucket_rows
                ),
                "mean_tie_rate": mean(row.get("tie_rate") for row in partition_bucket_rows),
                "mean_coverage": mean(row.get("coverage") for row in partition_bucket_rows),
                "topk_hit_rate": mean(
                    row.get("best_metric_hits_best_primary")
                    for row in partition_bucket_rows
                ),
                "mean_regret": mean(row.get("topk_regret") for row in partition_bucket_rows),
                "max_regret": max_float(
                    row.get("topk_regret") for row in partition_bucket_rows
                ),
                "pearson_cost_vs_primary": pearson([p[0] for p in pairs], [p[1] for p in pairs]),
                "spearman_cost_vs_primary": spearman([p[0] for p in pairs], [p[1] for p in pairs]),
                "pearson_cost_vs_tcount": pearson(
                    [cost for cost in costs if cost is not None],
                    [tcount for tcount in tcounts if tcount is not None],
                ),
            }
        )

    semantic_random_summaries = semantic_vs_random_rows(bucket_summaries)
    signal_status = classify_signal(semantic_random_summaries)
    return partition_summaries, bucket_summaries, semantic_random_summaries, signal_status


def controlled_tcount_buckets(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        tcount = coerce_float(row.get("tcount_after"))
        if tcount is None:
            continue
        grouped[(str(row["partition_id"]), str(row["circuit_id"]), int(tcount))].append(row)

    summaries: list[dict[str, Any]] = []
    for (partition_id, circuit_id, tcount), bucket in sorted(
        grouped.items(), key=lambda item: (natural_sort_key(item[0][0]), natural_sort_key(item[0][1]), item[0][2])
    ):
        if len(bucket) < 2:
            continue
        for metric_id in METRIC_SPECS:
            metric_bucket = [
                row
                for row in bucket
                if metric_score_tuple(row, metric_id) is not None
                and coerce_float(row.get("primary_nc_depth_ratio")) is not None
            ]
            if len(metric_bucket) < 2:
                continue
            best_metric = min(metric_bucket, key=lambda row: blind_sort_key(row, metric_id))
            best_primary = min(
                metric_bucket,
                key=lambda row: (
                    coerce_float(row.get("primary_nc_depth_ratio")),
                    *blind_sort_key(row, metric_id),
                ),
            )
            primary_values = sorted(
                coerce_float(row.get("primary_nc_depth_ratio"))
                for row in metric_bucket
            )
            median_primary = primary_values[len(primary_values) // 2]
            best_metric_primary = coerce_float(
                best_metric.get("primary_nc_depth_ratio")
            )
            best_primary_value = coerce_float(
                best_primary.get("primary_nc_depth_ratio")
            )
            pairwise_stats = pairwise_metric_stats(metric_bucket, metric_id)
            local_costs = [
                metric_scalar(row, metric_id) for row in metric_bucket
            ]
            local_primary = [
                coerce_float(row.get("primary_nc_depth_ratio"))
                for row in metric_bucket
            ]
            summaries.append(
                {
                    "metric_id": metric_id,
                    "metric_fields": ";".join(METRIC_SPECS[metric_id]),
                    "partition_id": partition_id,
                    "partition_kind": best_metric.get("partition_kind"),
                    "circuit_id": circuit_id,
                    "tcount_after": tcount,
                    "num_candidates": len(metric_bucket),
                    "total_candidate_pairs": pairwise_stats["total_pairs"],
                    "best_metric_candidate_id": best_metric.get("candidate_id"),
                    "best_metric_key": metric_key_string(best_metric, metric_id),
                    "best_metric_primary_nc_depth_ratio": best_metric_primary,
                    "best_primary_candidate_id": best_primary.get("candidate_id"),
                    "best_primary_nc_depth_ratio": best_primary_value,
                    "best_metric_hits_best_primary": int(
                        best_metric.get("candidate_id")
                        == best_primary.get("candidate_id")
                    ),
                    "best_metric_at_or_below_median_primary": int(
                        best_metric_primary is not None
                        and best_metric_primary <= median_primary
                    ),
                    "within_tcount_pairwise_win_rate": pairwise_stats["win_rate"],
                    "within_tcount_pairwise_pairs": pairwise_stats["comparisons"],
                    "metric_tie_pairs": pairwise_stats["metric_ties"],
                    "primary_tie_pairs": pairwise_stats["primary_ties"],
                    "tie_rate": pairwise_stats["tie_rate"],
                    "coverage": pairwise_stats["coverage"],
                    "within_tcount_spearman": spearman(
                        [value for value in local_costs if value is not None],
                        [value for value in local_primary if value is not None],
                    ),
                    "topk_reranking_gain": (
                        None
                        if best_metric_primary is None
                        else median_primary - best_metric_primary
                    ),
                    "topk_regret": (
                        None
                        if best_metric_primary is None or best_primary_value is None
                        else best_metric_primary - best_primary_value
                    ),
                }
            )
    return summaries


def semantic_vs_random_rows(bucket_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    random_by_bucket: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    semantic_rows: list[dict[str, Any]] = []
    for row in bucket_summaries:
        key = (
            str(row.get("metric_id")),
            str(row.get("circuit_id")),
            int(float(row.get("tcount_after"))),
        )
        if row.get("partition_kind") == "random":
            random_by_bucket[key].append(row)
        elif str(row.get("partition_kind", "")).startswith("semantic"):
            semantic_rows.append(row)

    summaries: list[dict[str, Any]] = []
    for row in semantic_rows:
        key = (
            str(row.get("metric_id")),
            str(row.get("circuit_id")),
            int(float(row.get("tcount_after"))),
        )
        random_rows = random_by_bucket.get(key, [])
        win_rate = coerce_float(row.get("within_tcount_pairwise_win_rate"))
        gain = coerce_float(row.get("topk_reranking_gain"))
        regret = coerce_float(row.get("topk_regret"))
        tie_rate = coerce_float(row.get("tie_rate"))
        random_win_rates = [
            value
            for random_row in random_rows
            if (value := coerce_float(random_row.get("within_tcount_pairwise_win_rate")))
            is not None
        ]
        random_gains = [
            value
            for random_row in random_rows
            if (value := coerce_float(random_row.get("topk_reranking_gain"))) is not None
        ]
        random_regrets = [
            value
            for random_row in random_rows
            if (value := coerce_float(random_row.get("topk_regret"))) is not None
        ]
        random_tie_rates = [
            value
            for random_row in random_rows
            if (value := coerce_float(random_row.get("tie_rate"))) is not None
        ]
        summaries.append(
            {
                **row,
                "num_random_controls": len(random_rows),
                "random_partition_percentile": percentile_against_controls(
                    win_rate, random_win_rates
                ),
                "random_gain_percentile": percentile_against_controls(
                    gain, random_gains
                ),
                "random_regret_percentile": lower_is_better_percentile_against_controls(
                    regret, random_regrets
                ),
                "random_tie_rate_percentile": lower_is_better_percentile_against_controls(
                    tie_rate, random_tie_rates
                ),
                "random_mean_pairwise_win_rate": mean(random_win_rates),
                "random_mean_topk_reranking_gain": mean(random_gains),
                "random_mean_regret": mean(random_regrets),
                "random_mean_tie_rate": mean(random_tie_rates),
            }
        )
    return summaries


def classify_signal(semantic_random_summaries: list[dict[str, Any]]) -> str:
    v3_rows = [
        row
        for row in semantic_random_summaries
        if row.get("metric_id") == "score_v3_lex"
    ]
    selected_rows = v3_rows or semantic_random_summaries
    semantic_win_rate = mean(
        row.get("within_tcount_pairwise_win_rate") for row in selected_rows
    )
    semantic_percentile = mean(
        row.get("random_partition_percentile") for row in selected_rows
    )
    semantic_regret_percentile = mean(
        row.get("random_regret_percentile") for row in selected_rows
    )
    if (
        len(selected_rows) >= 3
        and semantic_win_rate is not None
        and semantic_percentile is not None
        and semantic_regret_percentile is not None
        and semantic_win_rate >= 0.60
        and semantic_percentile >= 0.75
        and semantic_regret_percentile >= 0.75
    ):
        return "supportive"
    if (
        selected_rows
        and semantic_win_rate is not None
        and (
            semantic_win_rate > 0.50
            or (
                semantic_regret_percentile is not None
                and semantic_regret_percentile > 0.50
            )
        )
    ):
        return "mixed"
    return "not-supported"


def pairwise_metric_stats(rows: list[dict[str, Any]], metric_id: str) -> dict[str, Any]:
    wins = 0
    comparisons = 0
    metric_ties = 0
    primary_ties = 0
    total_pairs = 0
    for left_index, right_index in itertools.combinations(range(len(rows)), 2):
        left = rows[left_index]
        right = rows[right_index]
        left_cost = metric_score_tuple(left, metric_id)
        right_cost = metric_score_tuple(right, metric_id)
        left_primary = coerce_float(left.get("primary_nc_depth_ratio"))
        right_primary = coerce_float(right.get("primary_nc_depth_ratio"))
        if (
            left_cost is None
            or right_cost is None
            or left_primary is None
            or right_primary is None
        ):
            continue
        total_pairs += 1
        if left_primary == right_primary:
            primary_ties += 1
            continue
        if left_cost == right_cost:
            metric_ties += 1
            continue
        comparisons += 1
        if (left_cost < right_cost and left_primary < right_primary) or (
            right_cost < left_cost and right_primary < left_primary
        ):
            wins += 1
    active_pairs = comparisons + metric_ties
    return {
        "wins": wins,
        "comparisons": comparisons,
        "metric_ties": metric_ties,
        "primary_ties": primary_ties,
        "total_pairs": total_pairs,
        "win_rate": None if comparisons == 0 else wins / comparisons,
        "tie_rate": None if active_pairs == 0 else metric_ties / active_pairs,
        "coverage": None if total_pairs == 0 else comparisons / total_pairs,
    }


def percentile_against_controls(
    value: float | None,
    controls: list[float],
) -> float | None:
    if value is None or not controls:
        return None
    return sum(control <= value for control in controls) / len(controls)


def lower_is_better_percentile_against_controls(
    value: float | None,
    controls: list[float],
) -> float | None:
    if value is None or not controls:
        return None
    return sum(control >= value for control in controls) / len(controls)


def validation_summary_rows(
    bucket_summaries: list[dict[str, Any]],
    semantic_random_summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in bucket_summaries:
        rows.append(
            {
                "metric_scope": "tcount_bucket",
                "tensor_split_status": "ok",
                **row,
            }
        )
    for row in semantic_random_summaries:
        rows.append(
            {
                "metric_scope": "semantic_vs_random",
                "tensor_split_status": "ok",
                **row,
            }
        )
    return rows


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    return pearson(rank_values(xs), rank_values(ys))


def rank_values(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        rank = (index + end - 1) / 2.0
        for ordered_index in order[index:end]:
            ranks[ordered_index] = rank
        index = end
    return ranks


def mean(values: Any) -> float | None:
    finite = [coerce_float(value) for value in values]
    finite = [value for value in finite if value is not None]
    return None if not finite else float(sum(finite) / len(finite))


def max_float(values: Any) -> float | None:
    finite = [coerce_float(value) for value in values]
    finite = [value for value in finite if value is not None]
    return None if not finite else float(max(finite))


def write_report(
    *,
    rows: list[dict[str, Any]],
    partition_summaries: list[dict[str, Any]],
    bucket_summaries: list[dict[str, Any]],
    semantic_random_summaries: list[dict[str, Any]],
    signal_status: str,
    report_path: Path,
    figure_path: Path,
) -> None:
    candidate_rows = [
        row for row in rows if row.get("metric_scope") == "candidate"
    ]
    ok_candidate_rows = [
        row for row in candidate_rows if row.get("tensor_split_status") == "ok"
    ]
    covered_formal = sorted(
        {
            row.get("circuit_id")
            for row in ok_candidate_rows
            if row.get("circuit_id") in FORMAL_CORE_CIRCUITS
        },
        key=natural_sort_key,
    )
    lines = [
        "# Tensor Split Evidence Report",
        "",
        f"- Signal status: `{signal_status}`",
        f"- Candidate-scope rows: {len(candidate_rows)}",
        f"- Candidate-scope ok rows: {len(ok_candidate_rows)}",
        f"- Formal core circuits covered: {', '.join(covered_formal) or 'none'}",
        f"- Figure: `{figure_path.relative_to(PROJECT_ROOT)}`",
        "- Selection diagnostics use blind tie-breaks: metric key, stable factor hash, candidate id.",
        "",
        "## Metric v3 Summary",
        "",
        "| metric | partition | kind | candidates | buckets | win | tie | coverage | hit | regret | max regret | Spearman |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    key_partitions = {
        "semantic_role_k2",
        "semantic_family_k2",
        "semantic_register_k2",
        "balanced_contiguous_k2",
        "tensor_graph_spectral_k2",
    }
    for row in partition_summaries:
        if row.get("metric_id") not in REPORT_METRICS:
            continue
        if row.get("partition_id") not in key_partitions:
            continue
        lines.append(
            "| {metric_id} | {partition_id} | {partition_kind} | {num_candidates} | {num_buckets} | {win_rate} | {tie_rate} | {coverage} | {hit} | {regret} | {max_regret} | {spearman} |".format(
                metric_id=row.get("metric_id"),
                partition_id=row.get("partition_id"),
                partition_kind=row.get("partition_kind"),
                num_candidates=row.get("num_candidates"),
                num_buckets=row.get("num_tcount_buckets"),
                win_rate=format_float(row.get("mean_pairwise_win_rate")),
                tie_rate=format_float(row.get("mean_tie_rate")),
                coverage=format_float(row.get("mean_coverage")),
                hit=format_float(row.get("topk_hit_rate")),
                regret=format_float(row.get("mean_regret")),
                max_regret=format_float(row.get("max_regret")),
                spearman=format_float(row.get("spearman_cost_vs_primary")),
            )
        )
    lines.extend(
        [
            "",
            "## Semantic Pi vs Random Under v3",
            "",
            "| metric | partition | circuit | T-count | win | random pct | tie | regret | regret pct | random controls |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in semantic_random_summaries:
        if row.get("metric_id") not in REPORT_METRICS:
            continue
        lines.append(
            "| {metric_id} | {partition_id} | {circuit_id} | {tcount_after} | {win_rate} | {percentile} | {tie_rate} | {regret} | {regret_percentile} | {controls} |".format(
                metric_id=row.get("metric_id"),
                partition_id=row.get("partition_id"),
                circuit_id=row.get("circuit_id"),
                tcount_after=row.get("tcount_after"),
                win_rate=format_float(row.get("within_tcount_pairwise_win_rate")),
                percentile=format_float(row.get("random_partition_percentile")),
                tie_rate=format_float(row.get("tie_rate")),
                regret=format_float(row.get("topk_regret")),
                regret_percentile=format_float(row.get("random_regret_percentile")),
                controls=row.get("num_random_controls"),
            )
        )
    lines.extend(
        [
            "",
            "## Failure Cases",
            "",
            "| metric | partition | circuit | T-count | n | best metric | best primary | regret | tie | coverage |",
            "|---|---|---|---:|---:|---|---|---:|---:|---:|",
        ]
    )
    failure_keys = {
        ("gf_2pow2_mult", 21),
        ("qft_4", 67),
        ("vbe_adder_3", 19),
        ("mod_5_4", 7),
    }
    for row in bucket_summaries:
        if row.get("metric_id") not in ("gadget_aware_mixed_cost", "score_v3_lex"):
            continue
        if row.get("partition_id") not in ("semantic_role_k2", "semantic_family_k2"):
            continue
        key = (str(row.get("circuit_id")), int(float(row.get("tcount_after"))))
        if key not in failure_keys:
            continue
        lines.append(
            "| {metric_id} | {partition_id} | {circuit_id} | {tcount_after} | {num_candidates} | {best_metric_candidate_id} | {best_primary_candidate_id} | {regret} | {tie_rate} | {coverage} |".format(
                metric_id=row.get("metric_id"),
                partition_id=row.get("partition_id"),
                circuit_id=row.get("circuit_id"),
                tcount_after=row.get("tcount_after"),
                num_candidates=row.get("num_candidates"),
                best_metric_candidate_id=row.get("best_metric_candidate_id"),
                best_primary_candidate_id=row.get("best_primary_candidate_id"),
                regret=format_float(row.get("topk_regret")),
                tie_rate=format_float(row.get("tie_rate")),
                coverage=format_float(row.get("coverage")),
            )
        )
    lines.extend(
        [
            "",
            "## Blind Tie-Break Evaluation",
            "",
            "| metric | partition | circuit | T-count | n | win | Spearman | gain | regret | hit |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in bucket_summaries:
        if row.get("metric_id") not in REPORT_METRICS:
            continue
        if row.get("partition_id") not in key_partitions:
            continue
        lines.append(
            "| {metric_id} | {partition_id} | {circuit_id} | {tcount_after} | {num_candidates} | {win_rate} | {spearman} | {gain} | {regret} | {hit} |".format(
                metric_id=row.get("metric_id"),
                partition_id=row.get("partition_id"),
                circuit_id=row.get("circuit_id"),
                tcount_after=row.get("tcount_after"),
                num_candidates=row.get("num_candidates"),
                win_rate=format_float(row.get("within_tcount_pairwise_win_rate")),
                spearman=format_float(row.get("within_tcount_spearman")),
                gain=format_float(row.get("topk_reranking_gain")),
                regret=format_float(row.get("topk_regret")),
                hit=row.get("best_metric_hits_best_primary"),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This analysis is AlphaQuantum-only up to the tensor metric. ZX/feynver-derived columns are used only as external outcome labels already present in prior candidate diagnostics. The signal status is based on score_v3_lex semantic partition performance against random controls, with blind tie-breaks and regret/tie-rate diagnostics.",
            "",
        ]
    )
    ensure_dir(report_path.parent)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def make_figure(rows: list[dict[str, Any]], figure_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    candidate_rows = [
        row
        for row in rows
        if row.get("metric_scope") == "candidate"
        and row.get("tensor_split_status") == "ok"
        and coerce_float(row.get("primary_nc_depth_ratio")) is not None
        and coerce_float(row.get("gadget_aware_mixed_cost")) is not None
        and coerce_float(row.get("mixed_excess_norm")) is not None
    ]
    if not candidate_rows:
        return
    available_partitions = {row.get("partition_id") for row in candidate_rows}
    preferred_partition = next(
        (
            partition_id
            for partition_id in (
                "semantic_role_k2",
                "semantic_family_k2",
                "semantic_register_k2",
                "tensor_graph_spectral_k2",
                "balanced_contiguous_k2",
            )
            if partition_id in available_partitions
        ),
        "balanced_contiguous_k2",
    )
    plot_rows = [
        row for row in candidate_rows if row.get("partition_id") == preferred_partition
    ]
    old_x = [coerce_float(row.get("gadget_aware_mixed_cost")) for row in plot_rows]
    new_x = [coerce_float(row.get("mixed_excess_norm")) for row in plot_rows]
    y = [coerce_float(row.get("primary_nc_depth_ratio")) for row in plot_rows]
    c = [coerce_float(row.get("tcount_after")) or 0.0 for row in plot_rows]
    labels = [row.get("circuit_id") for row in plot_rows]
    fig, axes = plt.subplots(
        1, 2, figsize=(11.5, 4.6), sharey=True, constrained_layout=True
    )
    scatter = axes[0].scatter(
        old_x, y, c=c, cmap="viridis", edgecolor="black", linewidth=0.4
    )
    axes[1].scatter(
        new_x, y, c=c, cmap="viridis", edgecolor="black", linewidth=0.4
    )
    for ax, xs, title, xlabel in (
        (
            axes[0],
            old_x,
            "v2 effective mixed cost",
            "gadget_aware_mixed_cost",
        ),
        (
            axes[1],
            new_x,
            "v3 normalized mixed excess",
            "mixed_excess_norm",
        ),
    ):
        for xi, yi, label in zip(xs, y, labels):
            ax.annotate(str(label), (xi, yi), fontsize=6, alpha=0.65)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
    axes[0].set_ylabel("primary_nc_depth_ratio")
    fig.suptitle(f"Tensor split metric ablation ({preferred_partition})")
    fig.colorbar(scatter, ax=axes.ravel().tolist(), label="T-count")
    ensure_dir(figure_path.parent)
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)


def format_float(value: Any) -> str:
    numeric = coerce_float(value)
    return "" if numeric is None else f"{numeric:.3f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-csv",
        action="append",
        type=Path,
        default=None,
        help="Candidate diagnostics CSV. Can be supplied multiple times.",
    )
    parser.add_argument("--inventory-csv", type=Path, default=DEFAULT_INVENTORY_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--figure-path", type=Path, default=DEFAULT_FIGURE_PATH)
    parser.add_argument(
        "--random-seeds",
        default=",".join(str(seed) for seed in DEFAULT_RANDOM_SEEDS),
    )
    return parser.parse_args()


def parse_random_seed_spec(spec: str) -> tuple[int, ...]:
    parts = [part.strip() for part in spec.split(",") if part.strip()]
    if "..." not in parts:
        return tuple(int(part) for part in parts)
    ellipsis_index = parts.index("...")
    if ellipsis_index < 2 or ellipsis_index != len(parts) - 2:
        raise ValueError(
            "Ellipsis seed specs must look like '0,1,...,49'."
        )
    start = int(parts[0])
    next_value = int(parts[1])
    end = int(parts[-1])
    step = next_value - start
    if step <= 0:
        raise ValueError("Ellipsis seed specs require a positive step.")
    return tuple(range(start, end + 1, step))


def main() -> None:
    args = parse_args()
    candidate_csvs = args.candidate_csv or list(DEFAULT_CANDIDATE_CSVS)
    random_seeds = parse_random_seed_spec(args.random_seeds)
    candidates = merged_candidate_rows(candidate_csvs)
    inventory_index = load_inventory_index(args.inventory_csv)
    decomposition_index = load_decomposition_index()
    rows = tensor_split_rows(
        candidates,
        inventory_index=inventory_index,
        decomposition_index=decomposition_index,
        random_seeds=random_seeds,
    )
    add_score_v3_rank(rows)
    (
        partition_summaries,
        bucket_summaries,
        semantic_random_summaries,
        signal_status,
    ) = evidence_summaries(rows)
    write_csv_rows(
        [
            *rows,
            *validation_summary_rows(bucket_summaries, semantic_random_summaries),
        ],
        args.output_csv,
    )
    make_figure(rows, args.figure_path)
    write_report(
        rows=rows,
        partition_summaries=partition_summaries,
        bucket_summaries=bucket_summaries,
        semantic_random_summaries=semantic_random_summaries,
        signal_status=signal_status,
        report_path=args.report_md,
        figure_path=args.figure_path,
    )
    append_command(
        {
            "command": "python scripts/analyze_tensor_split_evidence.py",
            "outputs": [
                str(args.output_csv.relative_to(PROJECT_ROOT)),
                str(args.report_md.relative_to(PROJECT_ROOT)),
                str(args.figure_path.relative_to(PROJECT_ROOT)),
            ],
            "signal_status": signal_status,
        }
    )


if __name__ == "__main__":
    main()
