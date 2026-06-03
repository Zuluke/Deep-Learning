from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._analysis_common import dump_qasm_v2
from scripts._analysis_common import ensure_dir
from scripts._analysis_common import load_qasm_circuit
from scripts.assemble_resynth_circuit import assemble_circuit
from scripts.materialize_split_reward_candidate import canonicalize_factors
from scripts.materialize_split_reward_candidate import find_benchmark_dir
from scripts.materialize_split_reward_candidate import qasm_metrics
from scripts.materialize_split_reward_candidate import rank_one_tensor_sum
from scripts.materialize_split_reward_candidate import resolve_project_path
from scripts.materialize_split_reward_candidate import structural_metrics
from scripts.materialize_split_reward_candidate import write_json


def load_manifest_row(path: Path, *, target: str, candidate_kind: str) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("target") == target and row.get("candidate_kind") == candidate_kind
        ]
    if len(rows) != 1:
        raise ValueError(
            f"Expected exactly one manifest row for target={target!r}, "
            f"candidate_kind={candidate_kind!r}; got {len(rows)}."
        )
    return rows[0]


def load_mapping(path: Path, tensor_size: int) -> list[int]:
    mapping = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(mapping, list) or len(mapping) != tensor_size:
        raise ValueError(f"Expected mapping list of length {tensor_size} in {path}.")
    return [int(value) for value in mapping]


def gf2_inverse(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.uint8) % 2
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected square matrix, got {matrix.shape}.")
    size = matrix.shape[0]
    augmented = np.concatenate([matrix.copy(), np.eye(size, dtype=np.uint8)], axis=1)
    pivot_row = 0
    for col in range(size):
        candidates = np.flatnonzero(augmented[pivot_row:, col])
        if len(candidates) == 0:
            raise ValueError("Matrix is singular over GF(2).")
        pivot = pivot_row + int(candidates[0])
        if pivot != pivot_row:
            augmented[[pivot_row, pivot]] = augmented[[pivot, pivot_row]]
        for row in range(size):
            if row != pivot_row and augmented[row, col]:
                augmented[row] ^= augmented[pivot_row]
        pivot_row += 1
    return augmented[:, size:]


def factor_coefficients(rows: np.ndarray, parity: np.ndarray) -> np.ndarray:
    # rows[i] is the current linear form on tensor row i. We need coefficients c
    # with XOR_i c_i rows[i] = parity, i.e. rows.T @ c = parity over GF(2).
    inverse = gf2_inverse(rows.T)
    return (inverse @ (np.asarray(parity, dtype=np.uint8) % 2)) % 2


def ordered_factors(factors: np.ndarray, strategy: str) -> np.ndarray:
    factors = np.asarray(factors, dtype=np.uint8) % 2
    if strategy == "given":
        return factors.copy()
    if strategy == "support-ascending":
        keys = [(int(np.count_nonzero(factor)), tuple(int(v) for v in factor)) for factor in factors]
        return factors[sorted(range(len(factors)), key=lambda index: keys[index])]
    if strategy == "support-descending":
        keys = [(-int(np.count_nonzero(factor)), tuple(int(v) for v in factor)) for factor in factors]
        return factors[sorted(range(len(factors)), key=lambda index: keys[index])]
    if strategy == "lex":
        keys = [tuple(int(v) for v in factor) for factor in factors]
        return factors[sorted(range(len(factors)), key=lambda index: keys[index])]
    if strategy == "reverse":
        return factors[::-1].copy()
    raise ValueError(f"Unknown factor order strategy: {strategy}.")


def choose_target_index(
    active: list[int],
    *,
    rows: np.ndarray,
    parity: np.ndarray,
    mapping: list[int],
    strategy: str,
) -> int:
    if strategy == "min-index":
        return min(active)
    if strategy == "max-index":
        return max(active)
    if strategy == "min-mapping":
        return min(active, key=lambda index: (mapping[index], index))
    if strategy == "max-mapping":
        return max(active, key=lambda index: (mapping[index], -index))
    if strategy == "min-row-weight":
        return min(active, key=lambda index: (int(np.count_nonzero(rows[index])), index))
    if strategy == "max-row-weight":
        return max(active, key=lambda index: (-int(np.count_nonzero(rows[index])), index))
    if strategy == "min-change":
        return min(
            active,
            key=lambda index: (
                abs(int(np.count_nonzero(rows[index])) - int(np.count_nonzero(parity))),
                int(np.count_nonzero(rows[index] ^ parity)),
                index,
            ),
        )
    if strategy == "max-change":
        return max(
            active,
            key=lambda index: (
                abs(int(np.count_nonzero(rows[index])) - int(np.count_nonzero(parity))),
                int(np.count_nonzero(rows[index] ^ parity)),
                -index,
            ),
        )
    raise ValueError(f"Unknown target strategy: {strategy}.")


def greedy_cnot_order(factors: np.ndarray, *, target_strategy: str, mapping: list[int]) -> np.ndarray:
    remaining = [np.asarray(factor, dtype=np.uint8) % 2 for factor in factors]
    rows = np.eye(factors.shape[1], dtype=np.uint8)
    ordered: list[np.ndarray] = []
    while remaining:
        best_index = 0
        best_key: tuple[int, int, tuple[int, ...]] | None = None
        for index, factor in enumerate(remaining):
            if not np.any(factor):
                key = (0, 0, tuple(int(v) for v in factor))
            else:
                coeffs = factor_coefficients(rows, factor)
                active = [int(item) for item in np.flatnonzero(coeffs)]
                target = choose_target_index(
                    active,
                    rows=rows,
                    parity=factor,
                    mapping=mapping,
                    strategy=target_strategy,
                )
                key = (len(active) - 1, int(np.count_nonzero(rows[target] ^ factor)), tuple(int(v) for v in factor))
            if best_key is None or key < best_key:
                best_index = index
                best_key = key
        factor = remaining.pop(best_index)
        ordered.append(factor)
        if np.any(factor):
            coeffs = factor_coefficients(rows, factor)
            active = [int(item) for item in np.flatnonzero(coeffs)]
            target = choose_target_index(
                active,
                rows=rows,
                parity=factor,
                mapping=mapping,
                strategy=target_strategy,
            )
            for control in active:
                if control != target:
                    rows[target] ^= rows[control]
    return np.array(ordered, dtype=np.uint8)


def prepare_factors_for_synthesis(
    factors: np.ndarray,
    *,
    order_strategy: str,
    target_strategy: str,
    mapping: list[int],
) -> np.ndarray:
    if order_strategy == "greedy-cnot":
        return greedy_cnot_order(factors, target_strategy=target_strategy, mapping=mapping)
    return ordered_factors(factors, order_strategy)


def synthesize_shared_parity_circuit(
    factors: np.ndarray,
    *,
    num_qubits: int,
    mapping: list[int],
    target_strategy: str = "min-change",
) -> tuple[Any, list[tuple[int, int]]]:
    from qiskit import QuantumCircuit

    tensor_size = factors.shape[1]
    circuit = QuantumCircuit(num_qubits)
    rows = np.eye(tensor_size, dtype=np.uint8)
    cnots: list[tuple[int, int]] = []
    for factor in factors:
        parity = np.asarray(factor, dtype=np.uint8) % 2
        if not np.any(parity):
            continue
        coeffs = factor_coefficients(rows, parity)
        active = [index for index in np.flatnonzero(coeffs)]
        if not active:
            raise ValueError("Could not express nonzero parity in current basis.")
        target = choose_target_index(
            active,
            rows=rows,
            parity=parity,
            mapping=mapping,
            strategy=target_strategy,
        )
        for control in active:
            if control == target:
                continue
            circuit.cx(mapping[control], mapping[target])
            cnots.append((control, target))
            rows[target] ^= rows[control]
        if not np.array_equal(rows[target], parity):
            raise RuntimeError("Shared parity synthesis failed to realize requested parity.")
        circuit.t(mapping[target])
    for control, target in reversed(cnots):
        circuit.cx(mapping[control], mapping[target])
    return circuit, cnots


def phase_polynomial(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.uint8) % 2
    n_rows, n_cols = matrix.shape
    poly = np.zeros((n_rows, n_rows, n_rows), dtype=np.uint8)
    for col in range(n_cols):
        column = matrix[:, col]
        parity_weight = int(np.count_nonzero(column)) % 8
        for i in range(n_rows):
            for j in range(i):
                for k in range(j):
                    poly[i, j, k] = (
                        int(poly[i, j, k])
                        + int(column[i] and column[j] and column[k])
                    ) % 8
                poly[i, j, j] = (
                    int(poly[i, j, j])
                    + (11 - parity_weight) * int(column[i] and column[j])
                ) % 8
            poly[i, i, i] = (
                int(poly[i, i, i])
                + (((parity_weight + 6) * (parity_weight + 5)) // 2) * int(column[i])
            ) % 8

    for i in range(n_rows):
        for j in range(i):
            for k in range(j):
                value = int(poly[i, j, k])
                if value == 0:
                    continue
                poly[i, i, i] = (int(poly[i, i, i]) + 7 * value) % 8
                poly[j, j, j] = (int(poly[j, j, j]) + 7 * value) % 8
                poly[k, k, k] = (int(poly[k, k, k]) + 7 * value) % 8
                poly[i, j, j] = (int(poly[i, j, j]) + value) % 8
                poly[i, k, k] = (int(poly[i, k, k]) + value) % 8
                poly[j, k, k] = (int(poly[j, k, k]) + value) % 8
                poly[i, j, k] = (4 * value) % 8

    for i in range(n_rows):
        for j in range(i):
            value = int(poly[i, j, j])
            if value == 0:
                continue
            poly[i, i, i] = (int(poly[i, i, i]) + value) % 8
            poly[j, j, j] = (int(poly[j, j, j]) + value) % 8
            poly[i, j, j] = (6 * value) % 8
    return poly


def append_clifford_correction(
    circuit: Any,
    *,
    candidate_matrix: np.ndarray,
    original_matrix: np.ndarray,
    mapping: list[int],
) -> int:
    correction = phase_polynomial(original_matrix)
    correction = (correction + 8 - phase_polynomial(candidate_matrix)) % 8
    gate_count = 0
    tensor_size = candidate_matrix.shape[0]
    for i in range(tensor_size):
        for j in range(i):
            if int(correction[i, j, j]) == 4:
                circuit.cz(mapping[i], mapping[j])
                gate_count += 1
        phase = int(correction[i, i, i])
        if phase == 2:
            circuit.s(mapping[i])
            gate_count += 1
        elif phase == 4:
            circuit.z(mapping[i])
            gate_count += 1
        elif phase == 6:
            circuit.sdg(mapping[i])
            gate_count += 1
        elif phase != 0:
            raise ValueError(f"Unexpected non-Clifford correction phase {phase}.")
    return gate_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize exact factors with a shared parity-network synthesis."
    )
    parser.add_argument("--target", required=True)
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--candidate-kind", required=True)
    parser.add_argument(
        "--factor-order",
        choices=[
            "given",
            "support-ascending",
            "support-descending",
            "lex",
            "reverse",
            "greedy-cnot",
        ],
        default="given",
    )
    parser.add_argument(
        "--target-strategy",
        choices=[
            "min-change",
            "max-change",
            "min-index",
            "max-index",
            "min-mapping",
            "max-mapping",
            "min-row-weight",
            "max-row-weight",
        ],
        default="min-change",
    )
    parser.add_argument("--benchmark-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_csv = resolve_project_path(args.manifest_csv)
    output_root = resolve_project_path(args.output_root)
    benchmark_dir = (
        resolve_project_path(args.benchmark_dir)
        if args.benchmark_dir is not None
        else find_benchmark_dir(args.target)
    )
    ensure_dir(output_root)

    row = load_manifest_row(
        manifest_csv,
        target=args.target,
        candidate_kind=args.candidate_kind,
    )
    raw_factors = np.load(resolve_project_path(row["factor_path"]))
    change_of_basis_path = row.get("change_of_basis_path") or ""
    change_of_basis = (
        np.load(resolve_project_path(change_of_basis_path))
        if change_of_basis_path
        else None
    )
    factors = canonicalize_factors(raw_factors, change_of_basis).astype(np.uint8)
    target_tensor = np.load(benchmark_dir / f"{args.target}.tensor.npy").astype(bool)
    reconstruction_ok = np.array_equal(rank_one_tensor_sum(factors), target_tensor)
    if not reconstruction_ok:
        raise ValueError("Candidate factors do not reconstruct target tensor.")

    mapping = load_mapping(benchmark_dir / f"{args.target}.mapping.txt", factors.shape[1])
    synthesis_factors = prepare_factors_for_synthesis(
        factors,
        order_strategy=args.factor_order,
        target_strategy=args.target_strategy,
        mapping=mapping,
    )
    ordered_reconstruction_ok = np.array_equal(
        rank_one_tensor_sum(synthesis_factors),
        target_tensor,
    )
    if not ordered_reconstruction_ok:
        raise ValueError("Ordered candidate factors do not reconstruct target tensor.")
    original_qasm = benchmark_dir / f"{args.target}.qasm"
    num_qubits = max(load_qasm_circuit(original_qasm).num_qubits, max(mapping) + 1)
    block_circuit, shared_cnots = synthesize_shared_parity_circuit(
        synthesis_factors,
        num_qubits=num_qubits,
        mapping=mapping,
        target_strategy=args.target_strategy,
    )
    original_matrix = np.load(benchmark_dir / f"{args.target}.matrix.npy").astype(np.uint8)
    correction_gate_count = append_clifford_correction(
        block_circuit,
        candidate_matrix=factors.T,
        original_matrix=original_matrix,
        mapping=mapping,
    )

    block_qasm = output_root / f"{args.target}.qasm"
    dump_qasm_v2(block_circuit, block_qasm)
    assembled_qasm, assembled_summary = assemble_circuit(benchmark_dir, output_root)
    block_metrics = qasm_metrics(block_qasm)
    assembled_metrics = qasm_metrics(assembled_qasm)
    external_metrics = structural_metrics(assembled_qasm, original_qasm)
    summary: dict[str, Any] = {
        "status": "ok",
        "target": args.target,
        "candidate_kind": args.candidate_kind,
        "synthesis": "shared_parity_network",
        "factor_order": args.factor_order,
        "target_strategy": args.target_strategy,
        "manifest_csv": str(manifest_csv),
        "factor_path": str(resolve_project_path(row["factor_path"])),
        "num_factors": int(synthesis_factors.shape[0]),
        "num_shared_forward_cnots": len(shared_cnots),
        "num_shared_total_cnots": 2 * len(shared_cnots),
        "num_correction_gates": correction_gate_count,
        "reconstruction_ok": bool(reconstruction_ok),
        "ordered_reconstruction_ok": bool(ordered_reconstruction_ok),
        "benchmark_dir": str(benchmark_dir),
        "block_qasm": str(block_qasm),
        "assembled_qasm": str(assembled_qasm),
        "assembled_summary": assembled_summary,
        "block_metrics": block_metrics,
        "assembled_metrics": assembled_metrics,
        "external_structural_metrics": external_metrics,
    }
    write_json(output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
