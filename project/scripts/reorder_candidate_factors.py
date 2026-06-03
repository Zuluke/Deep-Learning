from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_action_dictionary_span import TARGETS
from scripts.export_linear_span_candidate import write_manifest
from scripts.materialize_split_reward_candidate import rank_one_tensor_sum
from scripts.tensor_split_core import balanced_contiguous_partition
from scripts.tensor_split_core import canonicalize_factors
from scripts.tensor_split_core import mixed_weight
from scripts.tensor_split_core import outer3
from scripts.tensor_split_core import project_mixed_tensor

EXTERNAL_ROOT = PROJECT_ROOT / "external"
if str(EXTERNAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EXTERNAL_ROOT))

from alphatensor_quantum.src import tensors

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "alphaq_reordered_candidates"


def load_manifest_row(path: Path, *, target: str, candidate_kind: str) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("target") == target and row.get("candidate_kind") == candidate_kind
        ]
    if len(rows) != 1:
        raise ValueError(
            f"Expected exactly one row for target={target!r}, candidate_kind={candidate_kind!r}; got {len(rows)}."
        )
    return rows[0]


def order_indices(
    factors: np.ndarray,
    target_tensor: np.ndarray,
    *,
    strategy: str,
) -> list[int]:
    if strategy == "original":
        return list(range(len(factors)))
    if strategy == "support-ascending":
        return sorted(
            range(len(factors)),
            key=lambda index: (int(np.count_nonzero(factors[index])), tuple(factors[index].tolist()), index),
        )
    if strategy == "support-descending":
        return sorted(
            range(len(factors)),
            key=lambda index: (-int(np.count_nonzero(factors[index])), tuple(factors[index].tolist()), index),
        )
    if strategy in {"greedy-residual", "greedy-mixed"}:
        return greedy_order(factors, target_tensor, mixed=(strategy == "greedy-mixed"))
    raise ValueError(
        f"Unknown reorder strategy {strategy!r}; expected original, support-ascending, support-descending, greedy-residual, or greedy-mixed."
    )


def greedy_order(factors: np.ndarray, target_tensor: np.ndarray, *, mixed: bool) -> list[int]:
    remaining = set(range(len(factors)))
    residual = np.asarray(target_tensor, dtype=np.uint8).copy()
    factor_tensors = [outer3(factor) for factor in factors]
    partition = balanced_contiguous_partition(target_tensor.shape[0])
    ordered: list[int] = []
    while remaining:
        scored = []
        for index in remaining:
            next_residual = residual ^ factor_tensors[index]
            if mixed:
                score = mixed_weight(next_residual, partition)
                tie_score = int(np.count_nonzero(next_residual))
            else:
                score = int(np.count_nonzero(next_residual))
                tie_score = mixed_weight(next_residual, partition)
            scored.append(
                (
                    score,
                    tie_score,
                    int(np.count_nonzero(factors[index])),
                    tuple(factors[index].tolist()),
                    index,
                )
            )
        chosen = min(scored)[-1]
        residual ^= factor_tensors[chosen]
        ordered.append(chosen)
        remaining.remove(chosen)
    return ordered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reorder exact AlphaQ candidate factors.")
    parser.add_argument("--target", choices=tuple(TARGETS), required=True)
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--candidate-kind", required=True)
    parser.add_argument(
        "--strategy",
        choices=(
            "original",
            "support-ascending",
            "support-descending",
            "greedy-residual",
            "greedy-mixed",
        ),
        required=True,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    row = load_manifest_row(
        args.manifest_csv,
        target=args.target,
        candidate_kind=args.candidate_kind,
    )
    target_tensor = np.asarray(tensors.get_signature_tensor(TARGETS[args.target]), dtype=np.uint8)
    factors = canonicalize_factors(
        np.load(row["factor_path"], allow_pickle=True),
        tensor_size=target_tensor.shape[0],
    )
    indices = order_indices(factors, target_tensor, strategy=args.strategy)
    reordered = factors[indices]
    reconstruction_ok = np.array_equal(
        rank_one_tensor_sum(reordered),
        target_tensor.astype(bool),
    )
    if not reconstruction_ok:
        raise RuntimeError("Reordered factors do not reconstruct target tensor.")

    output_dir = args.output_root / f"{args.target}_{args.candidate_kind}_{args.strategy}"
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_kind = f"{args.candidate_kind}_{args.strategy.replace('-', '_')}"
    factor_path = output_dir / f"{args.target}.{candidate_kind}.npy"
    change_of_basis_path = output_dir / f"{args.target}.{candidate_kind}.change_of_basis.npy"
    manifest_path = output_dir / "candidate_factors_manifest.csv"
    np.save(factor_path, reordered.astype(np.int32))
    np.save(change_of_basis_path, np.eye(target_tensor.shape[0], dtype=np.int32))
    write_manifest(
        manifest_path,
        target=args.target,
        candidate_kind=candidate_kind,
        factor_path=factor_path,
        change_of_basis_path=change_of_basis_path,
        num_moves=reordered.shape[0],
        effective_t_cost=reordered.shape[0],
    )
    summary = {
        "status": "ok",
        "target": args.target,
        "source_manifest_csv": str(args.manifest_csv),
        "source_candidate_kind": args.candidate_kind,
        "candidate_kind": candidate_kind,
        "strategy": args.strategy,
        "num_factors": int(reordered.shape[0]),
        "initial_mixed_weight": mixed_weight(target_tensor, balanced_contiguous_partition(target_tensor.shape[0])),
        "final_mixed_weight": mixed_weight(
            target_tensor ^ rank_one_tensor_sum(reordered).astype(np.uint8),
            balanced_contiguous_partition(target_tensor.shape[0]),
        ),
        "reconstruction_ok": bool(reconstruction_ok),
        "factor_path": str(factor_path),
        "manifest_path": str(manifest_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def main_from_namespace_for_test(**kwargs: object) -> int:
    return run(argparse.Namespace(**kwargs))


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
