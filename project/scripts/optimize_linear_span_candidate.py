from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._manifest import append_command
from scripts.analyze_action_dictionary_span import TARGETS
from scripts.analyze_action_dictionary_span import action_vector
from scripts.export_linear_span_candidate import factor_from_action
from scripts.export_linear_span_candidate import selected_actions
from scripts.export_linear_span_candidate import write_manifest
from scripts.materialize_split_reward_candidate import find_benchmark_dir
from scripts.materialize_split_reward_candidate import rank_one_tensor_sum
from scripts.tensor_split_core import balanced_contiguous_partition
from scripts.tensor_split_core import is_bridge_factor
from scripts.tensor_split_core import mixed_weight
from scripts.tensor_split_core import outer3

EXTERNAL_ROOT = PROJECT_ROOT / "external"
if str(EXTERNAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EXTERNAL_ROOT))

from alphatensor_quantum.src import tensors

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "alphaq_linear_span_optimized"


@dataclass(frozen=True)
class MilpSpanResult:
    coefficients: np.ndarray
    objective_value: float
    solver_status: int
    solver_message: str
    is_optimal: bool
    elapsed_sec: float


def solve_mod2_milp(
    columns: list[np.ndarray],
    target: np.ndarray,
    *,
    objective_weights: np.ndarray | None = None,
    max_factors: int | None = None,
    pair_incidence: np.ndarray | None = None,
    max_pair_overlap: int | None = None,
    time_limit_sec: float = 300.0,
    mip_rel_gap: float = 0.0,
) -> MilpSpanResult:
    try:
        from scipy.optimize import Bounds
        from scipy.optimize import LinearConstraint
        from scipy.optimize import milp
        from scipy.sparse import coo_matrix
        from scipy.sparse import eye
        from scipy.sparse import hstack
        from scipy.sparse import lil_matrix
    except Exception as exc:  # pragma: no cover - exercised only without scipy.
        raise RuntimeError("scipy.optimize.milp is required for MILP span optimization.") from exc

    num_rows = int(target.size)
    num_cols = len(columns)
    if num_cols == 0:
        raise ValueError("Cannot solve a span MILP with an empty action dictionary.")
    weights = (
        np.ones((num_cols,), dtype=float)
        if objective_weights is None
        else np.asarray(objective_weights, dtype=float)
    )
    if weights.shape != (num_cols,):
        raise ValueError(
            f"Expected objective_weights shape {(num_cols,)}, got {weights.shape}."
        )
    if max_pair_overlap is not None and max_pair_overlap < 0:
        raise ValueError("max_pair_overlap must be non-negative.")
    if max_pair_overlap is not None:
        if pair_incidence is None:
            raise ValueError("pair_incidence is required when max_pair_overlap is set.")
        pair_incidence = np.asarray(pair_incidence, dtype=float)
        if pair_incidence.ndim != 2 or pair_incidence.shape[1] != num_cols:
            raise ValueError(
                "Expected pair_incidence with shape "
                f"(num_pairs, {num_cols}), got {pair_incidence.shape}."
            )

    row_chunks = []
    col_chunks = []
    for col_index, column in enumerate(columns):
        nonzero_rows = np.flatnonzero(np.asarray(column, dtype=np.uint8).reshape(-1))
        row_chunks.append(nonzero_rows)
        col_chunks.append(np.full(nonzero_rows.size, col_index, dtype=np.int64))
    row_indices = np.concatenate(row_chunks)
    col_indices = np.concatenate(col_chunks)
    action_matrix = coo_matrix(
        (np.ones(row_indices.size, dtype=float), (row_indices, col_indices)),
        shape=(num_rows, num_cols),
    ).tocsr()

    parity_slack = eye(num_rows, dtype=float, format="csr") * -2.0
    parity_matrix = hstack(
        [action_matrix, parity_slack],
        format="csr",
    )
    rhs = np.asarray(target, dtype=np.uint8).reshape(-1).astype(float)
    objective = np.concatenate([weights, np.zeros((num_rows,), dtype=float)])
    integrality = np.ones((num_cols + num_rows,), dtype=float)
    lower = np.zeros((num_cols + num_rows,), dtype=float)
    upper = np.concatenate(
        [
            np.ones((num_cols,), dtype=float),
            np.full((num_rows,), float(num_cols), dtype=float),
        ]
    )

    start = time.time()
    constraints = [LinearConstraint(parity_matrix, rhs, rhs)]
    if max_factors is not None:
        if max_factors < 0:
            raise ValueError("max_factors must be non-negative.")
        factor_count_row = lil_matrix((1, num_cols + num_rows), dtype=float)
        factor_count_row[0, :num_cols] = 1.0
        constraints.append(
            LinearConstraint(
                factor_count_row.tocsr(),
                np.array([0.0]),
                np.array([float(max_factors)]),
            )
        )
    if max_pair_overlap is not None and pair_incidence is not None and pair_incidence.size:
        from scipy.sparse import csr_matrix

        zeros = lil_matrix((pair_incidence.shape[0], num_rows), dtype=float)
        pair_matrix = hstack(
            [csr_matrix(pair_incidence), zeros.tocsr()],
            format="csr",
        )
        constraints.append(
            LinearConstraint(
                pair_matrix,
                np.zeros((pair_incidence.shape[0],), dtype=float),
                np.full((pair_incidence.shape[0],), float(max_pair_overlap), dtype=float),
            )
        )

    result = milp(
        objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=constraints,
        options={
            "time_limit": float(time_limit_sec),
            "mip_rel_gap": float(mip_rel_gap),
        },
    )
    elapsed = time.time() - start
    if result.x is None:
        raise RuntimeError(
            f"MILP did not find a feasible decomposition: {result.message}"
        )
    coefficients = np.rint(result.x[:num_cols]).astype(np.uint8)
    return MilpSpanResult(
        coefficients=coefficients,
        objective_value=float(result.fun),
        solver_status=int(result.status),
        solver_message=str(result.message),
        is_optimal=bool(result.status == 0),
        elapsed_sec=float(elapsed),
    )


def objective_weights_for_actions(
    *,
    tensor: np.ndarray,
    actions: list[int],
    objective: str,
    mixed_weight_scale: float,
    support_weight_scale: float,
    pair_weight_scale: float,
    overlap_bonus_scale: float = 0.35,
) -> np.ndarray:
    if objective == "factor-count":
        return np.ones((len(actions),), dtype=float)
    partition = balanced_contiguous_partition(int(tensor.shape[0]))
    target = np.asarray(tensor, dtype=np.uint8)
    target_weight = max(int(np.count_nonzero(target)), 1)
    weights = []
    for action in actions:
        factor = factor_from_action(int(tensor.shape[0]), action)
        factor_weight = int(np.count_nonzero(factor))
        factor_support_cost = max(factor_weight - 1, 0)
        factor_pair_cost = max(factor_weight * (factor_weight - 1) // 2, 0)
        support_penalty = support_weight_scale * factor_support_cost
        pair_penalty = pair_weight_scale * factor_pair_cost
        if objective == "bridge-count":
            weights.append(
                1.0
                + support_penalty
                + pair_penalty
                + (mixed_weight_scale if is_bridge_factor(factor, partition) else 0.0)
            )
        elif objective == "mixed-weight":
            factor_mixed_weight = mixed_weight(outer3(factor), partition)
            denominator = max(int(np.count_nonzero(tensor)), 1)
            weights.append(1.0 + mixed_weight_scale * factor_mixed_weight / denominator)
        elif objective == "support-weight":
            weights.append(1.0 + support_penalty)
        elif objective == "pair-weight":
            weights.append(1.0 + pair_penalty)
        elif objective == "mixed-support":
            factor_mixed_weight = mixed_weight(outer3(factor), partition)
            denominator = max(int(np.count_nonzero(tensor)), 1)
            weights.append(
                1.0
                + support_penalty
                + pair_penalty
                + mixed_weight_scale * factor_mixed_weight / denominator
            )
        elif objective == "mixed-pair":
            factor_mixed_weight = mixed_weight(outer3(factor), partition)
            denominator = max(int(np.count_nonzero(tensor)), 1)
            weights.append(
                1.0
                + pair_penalty
                + mixed_weight_scale * factor_mixed_weight / denominator
            )
        elif objective == "depth-guarded-mixed-pair":
            factor_mixed_weight = mixed_weight(outer3(factor), partition)
            denominator = max(int(np.count_nonzero(tensor)), 1)
            weights.append(
                1.0
                + support_penalty
                + pair_penalty
                + mixed_weight_scale * factor_mixed_weight / denominator
            )
        elif objective == "frontier-pair":
            weights.append(
                frontier_pair_action_weight(
                    target=target,
                    factor=factor,
                    partition=partition.block_of,
                    target_weight=target_weight,
                    mixed_weight_scale=mixed_weight_scale,
                    support_weight_scale=support_weight_scale,
                    pair_weight_scale=pair_weight_scale,
                    overlap_bonus_scale=overlap_bonus_scale,
                )
            )
        elif objective == "t-preserving-frontier-pair":
            weights.append(
                t_preserving_frontier_pair_action_weight(
                    target=target,
                    factor=factor,
                    partition=partition.block_of,
                    target_weight=target_weight,
                    mixed_weight_scale=mixed_weight_scale,
                    support_weight_scale=support_weight_scale,
                    pair_weight_scale=pair_weight_scale,
                )
            )
        else:
            raise ValueError(
                "Unknown objective "
                f"{objective!r}; expected factor-count, bridge-count, mixed-weight, "
                "support-weight, pair-weight, mixed-support, mixed-pair, "
                "depth-guarded-mixed-pair, frontier-pair, "
                "or t-preserving-frontier-pair."
            )
    return np.asarray(weights, dtype=float)


def frontier_pair_action_weight(
    *,
    target: np.ndarray,
    factor: np.ndarray,
    partition: np.ndarray,
    target_weight: int,
    mixed_weight_scale: float,
    support_weight_scale: float,
    pair_weight_scale: float,
    overlap_bonus_scale: float,
) -> float:
    """Cost proxy for the article-style Clifford/non-Clifford frontier.

    The ZX detector in arXiv:2504.16004 shrinks the non-Clifford region by
    pushing Clifford structure away and closing only crossing gates.  In the
    tensor objective we cannot run ZX, so this proxy penalizes the action-level
    ingredients that later materialize as hard frontiers: cross-partition pairs,
    broad parity supports, and tensor mass outside the target.  It also gives a
    bounded bonus to factors that cover many target entries, which prevents the
    objective from becoming purely local and too fragmented.
    """

    vector = np.asarray(factor, dtype=np.uint8) % 2
    support = np.flatnonzero(vector)
    tensor_size = int(vector.shape[0])
    max_pairs = max(tensor_size * (tensor_size - 1) // 2, 1)
    support_pairs = max(len(support) * (len(support) - 1) // 2, 0)
    cross_pairs = 0
    for left_index, left in enumerate(support):
        for right in support[left_index + 1 :]:
            if int(partition[left]) != int(partition[right]):
                cross_pairs += 1

    action_tensor = outer3(vector)
    action_weight = max(int(np.count_nonzero(action_tensor)), 1)
    target_overlap = int(np.count_nonzero(action_tensor & target))
    off_target = action_weight - target_overlap

    frontier_penalty = cross_pairs / max_pairs
    support_penalty = support_pairs / max_pairs
    impurity_penalty = off_target / action_weight
    coverage_bonus = target_overlap / max(target_weight, 1)

    weight = (
        1.0
        + mixed_weight_scale * frontier_penalty
        + support_weight_scale * support_penalty
        + pair_weight_scale * impurity_penalty
        - overlap_bonus_scale * min(coverage_bonus, 1.0)
    )
    return max(0.05, float(weight))


def t_preserving_frontier_pair_action_weight(
    *,
    target: np.ndarray,
    factor: np.ndarray,
    partition: np.ndarray,
    target_weight: int,
    mixed_weight_scale: float,
    support_weight_scale: float,
    pair_weight_scale: float,
) -> float:
    """Frontier proxy with no negative action bonus.

    This objective is meant to be paired with a hard `max_factors` guard from
    the factor-count baseline.  The per-action weights are still useful as a
    tie-breaker inside that feasible set, but they never go below one, so the
    frontier term cannot buy extra factors by itself.
    """

    vector = np.asarray(factor, dtype=np.uint8) % 2
    support = np.flatnonzero(vector)
    tensor_size = int(vector.shape[0])
    max_pairs = max(tensor_size * (tensor_size - 1) // 2, 1)
    support_pairs = max(len(support) * (len(support) - 1) // 2, 0)
    cross_pairs = 0
    for left_index, left in enumerate(support):
        for right in support[left_index + 1 :]:
            if int(partition[left]) != int(partition[right]):
                cross_pairs += 1

    action_tensor = outer3(vector)
    action_weight = max(int(np.count_nonzero(action_tensor)), 1)
    target_overlap = int(np.count_nonzero(action_tensor & target))
    off_target = action_weight - target_overlap

    frontier_penalty = cross_pairs / max_pairs
    support_penalty = support_pairs / max_pairs
    impurity_penalty = off_target / action_weight
    coverage_penalty = 1.0 - min(target_overlap / max(target_weight, 1), 1.0)

    return float(
        1.0
        + mixed_weight_scale * frontier_penalty
        + support_weight_scale * support_penalty
        + pair_weight_scale * impurity_penalty
        + 0.05 * coverage_penalty
    )


def pair_incidence_for_actions(tensor_size: int, actions: list[int]) -> np.ndarray:
    pairs = [
        (left, right)
        for left in range(tensor_size)
        for right in range(left + 1, tensor_size)
    ]
    incidence = np.zeros((len(pairs), len(actions)), dtype=np.uint8)
    for action_index, action in enumerate(actions):
        factor = factor_from_action(tensor_size, action)
        active = set(int(index) for index in np.flatnonzero(factor))
        for pair_index, (left, right) in enumerate(pairs):
            if left in active and right in active:
                incidence[pair_index, action_index] = 1
    return incidence


def write_solution_manifest(
    path: Path,
    *,
    target: str,
    candidate_kind: str,
    factor_path: Path,
    change_of_basis_path: Path,
    num_moves: int,
    effective_t_cost: int,
    objective_value: float,
    solver_status: int,
    solver_message: str,
    is_optimal: bool,
) -> None:
    fieldnames = [
        "target",
        "candidate_kind",
        "status",
        "num_moves",
        "return",
        "effective_t_cost",
        "residual_weight",
        "factor_path",
        "change_of_basis_path",
        "is_canonical_basis",
        "span_objective_value",
        "span_solver_status",
        "span_solver_message",
        "span_is_optimal",
    ]
    row = {
        "target": target,
        "candidate_kind": candidate_kind,
        "status": "solved",
        "num_moves": num_moves,
        "return": "",
        "effective_t_cost": effective_t_cost,
        "residual_weight": 0.0,
        "factor_path": str(factor_path),
        "change_of_basis_path": str(change_of_basis_path),
        "is_canonical_basis": True,
        "span_objective_value": objective_value,
        "span_solver_status": solver_status,
        "span_solver_message": solver_message,
        "span_is_optimal": is_optimal,
    }
    rows: list[dict[str, Any]] = []
    if path.exists():
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows = [
            existing
            for existing in rows
            if not (
                existing.get("target") == target
                and existing.get("candidate_kind") == candidate_kind
            )
        ]
    rows.append(row)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export an exact AlphaQuantum-only decomposition by solving a "
            "minimum-cost GF(2) span MILP."
        )
    )
    parser.add_argument("--target", required=True)
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=None,
        help=(
            "Optional circuit-to-tensor benchmark directory. If omitted, known "
            "AlphaTensor targets use the local registry and other targets are "
            "resolved by name under external/circuit-to-tensor/benchmarks."
        ),
    )
    parser.add_argument(
        "--action-dictionary",
        choices=("low-weight", "tensor-overlap"),
        default="low-weight",
    )
    parser.add_argument("--max-action-weight", type=int, default=4)
    parser.add_argument("--tensor-overlap-max-weight", type=int, default=5)
    parser.add_argument("--tensor-overlap-max-actions-per-target", type=int, default=175)
    parser.add_argument(
        "--objective",
        choices=(
            "factor-count",
            "bridge-count",
            "mixed-weight",
            "support-weight",
            "pair-weight",
            "mixed-support",
            "mixed-pair",
            "depth-guarded-mixed-pair",
            "frontier-pair",
            "t-preserving-frontier-pair",
        ),
        default="factor-count",
    )
    parser.add_argument("--mixed-weight-scale", type=float, default=1.0)
    parser.add_argument("--support-weight-scale", type=float, default=0.25)
    parser.add_argument("--pair-weight-scale", type=float, default=0.0)
    parser.add_argument("--overlap-bonus-scale", type=float, default=0.35)
    parser.add_argument(
        "--max-factors",
        type=int,
        default=None,
        help="Optional hard cap on the number of selected factors.",
    )
    parser.add_argument(
        "--max-pair-overlap",
        type=int,
        default=None,
        help=(
            "Optional hard cap on how many selected factors may contain the "
            "same pair of tensor indices."
        ),
    )
    parser.add_argument("--time-limit-sec", type=float, default=300.0)
    parser.add_argument("--mip-rel-gap", type=float, default=0.0)
    parser.add_argument("--candidate-kind", default="milp_span")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def load_target_tensor(target: str, benchmark_dir: Path | None = None) -> np.ndarray:
    if target in TARGETS and benchmark_dir is None:
        return np.asarray(tensors.get_signature_tensor(TARGETS[target]), dtype=np.uint8)
    resolved_dir = benchmark_dir if benchmark_dir is not None else find_benchmark_dir(target)
    tensor_path = resolved_dir / f"{target}.tensor.npy"
    if not tensor_path.exists():
        raise FileNotFoundError(f"Missing target tensor: {tensor_path}")
    return np.load(tensor_path).astype(np.uint8)


def run(args: argparse.Namespace) -> int:
    tensor = load_target_tensor(
        args.target,
        getattr(args, "benchmark_dir", None),
    )
    actions = selected_actions(
        tensor,
        action_dictionary=args.action_dictionary,
        max_action_weight=args.max_action_weight,
        tensor_overlap_max_weight=args.tensor_overlap_max_weight,
        tensor_overlap_max_actions_per_target=args.tensor_overlap_max_actions_per_target,
    )
    columns = [action_vector(tensor.shape[0], action) for action in actions]
    weights = objective_weights_for_actions(
        tensor=tensor,
        actions=actions,
        objective=args.objective,
        mixed_weight_scale=args.mixed_weight_scale,
        support_weight_scale=args.support_weight_scale,
        pair_weight_scale=getattr(args, "pair_weight_scale", 0.0),
        overlap_bonus_scale=getattr(args, "overlap_bonus_scale", 0.35),
    )
    max_pair_overlap = getattr(args, "max_pair_overlap", None)
    solution = solve_mod2_milp(
        columns,
        tensor.reshape(-1),
        objective_weights=weights,
        max_factors=args.max_factors,
        pair_incidence=(
            pair_incidence_for_actions(tensor.shape[0], actions)
            if max_pair_overlap is not None
            else None
        ),
        max_pair_overlap=max_pair_overlap,
        time_limit_sec=args.time_limit_sec,
        mip_rel_gap=args.mip_rel_gap,
    )
    chosen_actions = [
        action for action, coefficient in zip(actions, solution.coefficients) if coefficient
    ]
    factors = np.array(
        [factor_from_action(tensor.shape[0], action) for action in chosen_actions],
        dtype=np.uint8,
    )
    reconstruction_ok = np.array_equal(rank_one_tensor_sum(factors), tensor.astype(bool))
    if not reconstruction_ok:
        raise RuntimeError("MILP-selected factors do not reconstruct target tensor.")

    output_dir = args.output_root / (
        f"{args.target}_{args.action_dictionary}_"
        f"w{args.max_action_weight}_k{args.tensor_overlap_max_actions_per_target}_"
        f"{args.objective}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    factor_path = output_dir / f"{args.target}.{args.candidate_kind}.npy"
    change_of_basis_path = output_dir / f"{args.target}.{args.candidate_kind}.change_of_basis.npy"
    manifest_path = output_dir / "candidate_factors_manifest.csv"
    np.save(factor_path, factors.astype(np.int32))
    np.save(change_of_basis_path, np.eye(tensor.shape[0], dtype=np.int32))
    write_solution_manifest(
        manifest_path,
        target=args.target,
        candidate_kind=args.candidate_kind,
        factor_path=factor_path,
        change_of_basis_path=change_of_basis_path,
        num_moves=factors.shape[0],
        effective_t_cost=factors.shape[0],
        objective_value=solution.objective_value,
        solver_status=solution.solver_status,
        solver_message=solution.solver_message,
        is_optimal=solution.is_optimal,
    )
    summary: dict[str, Any] = {
        "status": "ok",
        "target": args.target,
        "candidate_kind": args.candidate_kind,
        "benchmark_dir": str(getattr(args, "benchmark_dir", "") or ""),
        "action_dictionary": args.action_dictionary,
        "max_action_weight": args.max_action_weight,
        "tensor_overlap_max_weight": args.tensor_overlap_max_weight,
        "tensor_overlap_max_actions_per_target": args.tensor_overlap_max_actions_per_target,
        "objective": args.objective,
        "mixed_weight_scale": args.mixed_weight_scale,
        "support_weight_scale": args.support_weight_scale,
        "pair_weight_scale": getattr(args, "pair_weight_scale", 0.0),
        "overlap_bonus_scale": getattr(args, "overlap_bonus_scale", 0.35),
        "max_factors": args.max_factors,
        "max_pair_overlap": max_pair_overlap,
        "num_actions": len(actions),
        "num_factors": int(factors.shape[0]),
        "span_objective_value": solution.objective_value,
        "solver_status": solution.solver_status,
        "solver_message": solution.solver_message,
        "is_optimal": solution.is_optimal,
        "elapsed_sec": solution.elapsed_sec,
        "reconstruction_ok": bool(reconstruction_ok),
        "factor_path": str(factor_path),
        "change_of_basis_path": str(change_of_basis_path),
        "manifest_path": str(manifest_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    append_command(
        {
            "tool": "optimize_linear_span_candidate.py",
            "command": " ".join(sys.argv),
            "cwd": str(PROJECT_ROOT),
            "manifest_path": str(manifest_path),
            "exit_code": 0,
        }
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def main_from_namespace_for_test(**kwargs: object) -> int:
    return run(argparse.Namespace(**kwargs))


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
