from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._manifest import append_command
from scripts.analyze_action_dictionary_span import TARGETS
from scripts.analyze_action_dictionary_span import action_vector
from scripts.analyze_action_dictionary_span import low_weight_actions
from scripts.analyze_action_dictionary_span import tensor_overlap_actions
from scripts.materialize_split_reward_candidate import rank_one_tensor_sum

EXTERNAL_ROOT = PROJECT_ROOT / "external"
if str(EXTERNAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EXTERNAL_ROOT))

from alphatensor_quantum.src import tensors

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "alphaq_linear_span"


def solve_gf2(columns: list[np.ndarray], target: np.ndarray) -> tuple[np.ndarray, int, int]:
    num_rows = target.size
    num_cols = len(columns)
    rows = [0] * num_rows
    rhs = [int(value) for value in target.reshape(-1)]
    for col_index, column in enumerate(columns):
        bit = 1 << col_index
        for row_index, value in enumerate(column.reshape(-1)):
            if value:
                rows[row_index] ^= bit

    rank = 0
    pivots: list[int] = []
    for col_index in range(num_cols):
        mask = 1 << col_index
        pivot = None
        for row_index in range(rank, num_rows):
            if rows[row_index] & mask:
                pivot = row_index
                break
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        rhs[rank], rhs[pivot] = rhs[pivot], rhs[rank]
        for row_index in range(num_rows):
            if row_index != rank and (rows[row_index] & mask):
                rows[row_index] ^= rows[rank]
                rhs[row_index] ^= rhs[rank]
        pivots.append(col_index)
        rank += 1

    if any(rows[row] == 0 and rhs[row] for row in range(rank, num_rows)):
        raise ValueError("Target tensor is not in the selected action dictionary span.")

    solution = np.zeros((num_cols,), dtype=np.uint8)
    for row_index, col_index in enumerate(pivots):
        solution[col_index] = rhs[row_index]
    return solution, rank, num_cols - rank


def factor_from_action(size: int, action: int) -> np.ndarray:
    bits = action + 1
    return np.array([(bits >> index) & 1 for index in range(size)], dtype=np.uint8)


def selected_actions(
    tensor: np.ndarray,
    *,
    action_dictionary: str,
    max_action_weight: int,
    tensor_overlap_max_weight: int,
    tensor_overlap_max_actions_per_target: int,
) -> list[int]:
    if action_dictionary == "low-weight":
        return low_weight_actions(tensor.shape[0], max_action_weight)
    if action_dictionary == "tensor-overlap":
        return tensor_overlap_actions(
            tensor,
            base_max_weight=max_action_weight,
            overlap_max_weight=tensor_overlap_max_weight,
            per_target_limit=tensor_overlap_max_actions_per_target,
        )
    raise ValueError(
        f"Unsupported action dictionary {action_dictionary!r}; use low-weight or tensor-overlap."
    )


def write_manifest(
    path: Path,
    *,
    target: str,
    candidate_kind: str,
    factor_path: Path,
    change_of_basis_path: Path,
    num_moves: int,
    effective_t_cost: int,
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
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export an exact AlphaQuantum-only GF(2) span decomposition candidate."
    )
    parser.add_argument("--target", choices=tuple(TARGETS), required=True)
    parser.add_argument(
        "--action-dictionary",
        choices=("low-weight", "tensor-overlap"),
        default="low-weight",
    )
    parser.add_argument("--max-action-weight", type=int, default=3)
    parser.add_argument("--tensor-overlap-max-weight", type=int, default=5)
    parser.add_argument("--tensor-overlap-max-actions-per-target", type=int, default=175)
    parser.add_argument("--candidate-kind", default="linear_span")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    tensor = np.asarray(tensors.get_signature_tensor(TARGETS[args.target]), dtype=np.uint8)
    actions = selected_actions(
        tensor,
        action_dictionary=args.action_dictionary,
        max_action_weight=args.max_action_weight,
        tensor_overlap_max_weight=args.tensor_overlap_max_weight,
        tensor_overlap_max_actions_per_target=args.tensor_overlap_max_actions_per_target,
    )
    columns = [action_vector(tensor.shape[0], action) for action in actions]
    solution, rank, nullity = solve_gf2(columns, tensor.reshape(-1))
    chosen_actions = [
        action for action, coefficient in zip(actions, solution) if coefficient
    ]
    factors = np.array(
        [factor_from_action(tensor.shape[0], action) for action in chosen_actions],
        dtype=np.uint8,
    )
    reconstruction_ok = np.array_equal(rank_one_tensor_sum(factors), tensor.astype(bool))
    if not reconstruction_ok:
        raise RuntimeError("Exported linear-span factors do not reconstruct target tensor.")

    output_dir = args.output_root / (
        f"{args.target}_{args.action_dictionary}_"
        f"w{args.max_action_weight}_k{args.tensor_overlap_max_actions_per_target}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    factor_path = output_dir / f"{args.target}.{args.candidate_kind}.npy"
    change_of_basis_path = output_dir / f"{args.target}.{args.candidate_kind}.change_of_basis.npy"
    manifest_path = output_dir / "candidate_factors_manifest.csv"
    np.save(factor_path, factors.astype(np.int32))
    np.save(change_of_basis_path, np.eye(tensor.shape[0], dtype=np.int32))
    write_manifest(
        manifest_path,
        target=args.target,
        candidate_kind=args.candidate_kind,
        factor_path=factor_path,
        change_of_basis_path=change_of_basis_path,
        num_moves=factors.shape[0],
        effective_t_cost=factors.shape[0],
    )
    summary = {
        "status": "ok",
        "target": args.target,
        "candidate_kind": args.candidate_kind,
        "action_dictionary": args.action_dictionary,
        "num_actions": len(actions),
        "rank": rank,
        "nullity": nullity,
        "num_factors": int(factors.shape[0]),
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
            "tool": "export_linear_span_candidate.py",
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
