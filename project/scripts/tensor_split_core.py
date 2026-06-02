from __future__ import annotations

from dataclasses import dataclass
import ast
import hashlib
import itertools
from pathlib import Path
import re
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class TensorPartition:
    partition_id: str
    block_of: np.ndarray
    kind: str
    semantic_partition_status: str | None = None
    semantic_partition_error: str | None = None

    @property
    def tensor_size(self) -> int:
        return int(self.block_of.shape[0])

    @property
    def block_sizes(self) -> tuple[int, ...]:
        return tuple(
            int(np.count_nonzero(self.block_of == block_id))
            for block_id in sorted(set(int(value) for value in self.block_of))
        )


@dataclass(frozen=True)
class FactorGroup:
    group_type: str
    factor_indices: tuple[int, ...]
    effective_cost: int

    @property
    def start(self) -> int:
        return min(self.factor_indices)

    @property
    def stop(self) -> int:
        return max(self.factor_indices) + 1


@dataclass(frozen=True)
class QubitRef:
    register: str
    index: int
    flat_index: int


def canonicalize_factors(
    factors: np.ndarray,
    *,
    tensor_size: int | None = None,
) -> np.ndarray:
    array = np.asarray(factors, dtype=np.uint8)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D factor array, got shape {array.shape}.")
    array = array % 2
    if tensor_size is None:
        return array
    if array.shape[1] == tensor_size:
        return array
    if array.shape[0] == tensor_size:
        return array.T.copy()
    raise ValueError(
        f"Cannot orient factor array with shape {array.shape} for tensor_size={tensor_size}."
    )


def load_factor_array(path: Path, *, tensor_size: int | None = None) -> np.ndarray:
    return canonicalize_factors(np.load(path, allow_pickle=True), tensor_size=tensor_size)


def outer3(factor: np.ndarray) -> np.ndarray:
    vector = np.asarray(factor, dtype=np.uint8) % 2
    return np.einsum("i,j,k->ijk", vector, vector, vector, optimize=True).astype(
        np.uint8
    )


def tensor_from_factors(factors: np.ndarray, *, tensor_size: int | None = None) -> np.ndarray:
    canonical = canonicalize_factors(factors, tensor_size=tensor_size)
    size = canonical.shape[1] if tensor_size is None else tensor_size
    tensor = np.zeros((size, size, size), dtype=np.uint8)
    for factor in canonical:
        tensor ^= outer3(factor)
    return tensor


def as_partition(partition: TensorPartition | Iterable[int]) -> TensorPartition:
    if isinstance(partition, TensorPartition):
        return partition
    block_of = np.asarray(list(partition), dtype=np.int32)
    return TensorPartition("custom", block_of, "custom")


def balanced_contiguous_partition(tensor_size: int) -> TensorPartition:
    if tensor_size < 2:
        raise ValueError("A split partition requires tensor_size >= 2.")
    split = tensor_size // 2
    block_of = np.ones((tensor_size,), dtype=np.int32)
    block_of[:split] = 0
    return TensorPartition("balanced_contiguous_k2", block_of, "contiguous")


def random_balanced_partition(tensor_size: int, *, seed: int) -> TensorPartition:
    if tensor_size < 2:
        raise ValueError("A split partition requires tensor_size >= 2.")
    rng = np.random.default_rng(seed)
    indices = np.arange(tensor_size)
    rng.shuffle(indices)
    split = tensor_size // 2
    block_of = np.ones((tensor_size,), dtype=np.int32)
    block_of[indices[:split]] = 0
    return TensorPartition(f"random_balanced_k2_seed{seed}", block_of, "random")


def semantic_partitions_from_qasm(
    *,
    circuit_id: str,
    qasm_path: Path,
    mapping_path: Path | None,
    tensor_size: int,
) -> list[TensorPartition]:
    qubits = parse_qasm_qubits(qasm_path)
    if not qubits:
        return []
    mapping = load_mapping_indices(mapping_path, tensor_size=tensor_size)
    partitions: list[TensorPartition] = []
    register_partition = _semantic_register_partition(
        circuit_id=circuit_id,
        qubits=qubits,
        mapping=mapping,
        tensor_size=tensor_size,
    )
    if register_partition is not None:
        partitions.append(register_partition)
    role_partition = _semantic_role_partition(
        circuit_id=circuit_id,
        qubits=qubits,
        mapping=mapping,
        tensor_size=tensor_size,
    )
    if role_partition is not None:
        partitions.append(role_partition)
    family_partition = _semantic_family_partition(
        circuit_id=circuit_id,
        qubits=qubits,
        mapping=mapping,
        tensor_size=tensor_size,
    )
    if family_partition is not None:
        partitions.append(family_partition)
    return partitions


def parse_qasm_qubits(path: Path) -> list[QubitRef]:
    text = path.read_text(encoding="utf-8")
    qubits: list[QubitRef] = []
    for match in re.finditer(r"\bqreg\s+([A-Za-z_]\w*)\[(\d+)\]\s*;", text):
        register = match.group(1)
        size = int(match.group(2))
        for index in range(size):
            qubits.append(QubitRef(register, index, len(qubits)))
    return qubits


def load_mapping_indices(
    mapping_path: Path | None,
    *,
    tensor_size: int,
) -> list[int]:
    if mapping_path is None or not mapping_path.exists():
        return list(range(tensor_size))
    try:
        loaded = ast.literal_eval(mapping_path.read_text(encoding="utf-8").strip())
    except (SyntaxError, ValueError):
        return list(range(tensor_size))
    if not isinstance(loaded, list):
        return list(range(tensor_size))
    mapping = [int(value) for value in loaded]
    if len(mapping) < tensor_size:
        mapping.extend(range(len(mapping), tensor_size))
    return mapping[:tensor_size]


def tensor_graph_spectral_partition(tensor: np.ndarray) -> TensorPartition:
    tensor = _as_binary_tensor(tensor)
    size = int(tensor.shape[0])
    if size < 2:
        raise ValueError("A split partition requires tensor_size >= 2.")
    adjacency = np.zeros((size, size), dtype=float)
    for i, j, k in np.argwhere(tensor):
        for a, b in ((i, j), (i, k), (j, k)):
            if a == b:
                continue
            adjacency[a, b] += 1.0
            adjacency[b, a] += 1.0
    if not np.any(adjacency):
        fallback = balanced_contiguous_partition(size)
        return TensorPartition(
            "tensor_graph_spectral_k2", fallback.block_of, "tensor-graph"
        )
    laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
    _, vectors = np.linalg.eigh(laplacian)
    fiedler = vectors[:, 1] if size > 1 else vectors[:, 0]
    ordered = sorted(range(size), key=lambda index: (float(fiedler[index]), index))
    split = size // 2
    block_of = np.ones((size,), dtype=np.int32)
    block_of[ordered[:split]] = 0
    if np.all(block_of == block_of[0]):
        block_of = balanced_contiguous_partition(size).block_of
    return TensorPartition("tensor_graph_spectral_k2", block_of, "tensor-graph")


def default_partitions(
    tensor: np.ndarray,
    *,
    random_seeds: tuple[int, ...] = (0, 1, 2),
) -> list[TensorPartition]:
    size = int(np.asarray(tensor).shape[0])
    return [
        balanced_contiguous_partition(size),
        tensor_graph_spectral_partition(tensor),
        *(random_balanced_partition(size, seed=seed) for seed in random_seeds),
    ]


def project_mixed_tensor(
    tensor: np.ndarray,
    partition: TensorPartition | Iterable[int],
) -> np.ndarray:
    tensor = _as_binary_tensor(tensor)
    split = as_partition(partition)
    if tensor.shape != (split.tensor_size, split.tensor_size, split.tensor_size):
        raise ValueError(
            f"Tensor shape {tensor.shape} does not match partition size {split.tensor_size}."
        )
    return np.where(_local_mask(split), 0, tensor).astype(np.uint8)


def project_mixed_factor(
    factor: np.ndarray,
    partition: TensorPartition | Iterable[int],
) -> np.ndarray:
    return project_mixed_tensor(outer3(factor), partition)


def is_bridge_factor(
    factor: np.ndarray,
    partition: TensorPartition | Iterable[int],
) -> bool:
    split = as_partition(partition)
    vector = np.asarray(factor, dtype=np.uint8) % 2
    if vector.shape != (split.tensor_size,):
        raise ValueError(
            f"Factor shape {vector.shape} does not match partition size {split.tensor_size}."
        )
    active_blocks = set(int(split.block_of[index]) for index in np.flatnonzero(vector))
    return len(active_blocks) > 1


def mixed_weight(
    tensor: np.ndarray,
    partition: TensorPartition | Iterable[int],
) -> int:
    return int(np.count_nonzero(project_mixed_tensor(tensor, partition)))


def mixed_auc(
    target_tensor: np.ndarray,
    factors: np.ndarray,
    partition: TensorPartition | Iterable[int],
) -> int:
    residual = _as_binary_tensor(target_tensor).copy()
    split = as_partition(partition)
    canonical = canonicalize_factors(factors, tensor_size=residual.shape[0])
    mixed_mask = ~_local_mask(split)
    area = int(np.count_nonzero(residual & mixed_mask))
    for factor in canonical:
        residual ^= outer3(factor)
        area += int(np.count_nonzero(residual & mixed_mask))
    return int(area)


def raw_bridge_count(
    factors: np.ndarray,
    partition: TensorPartition | Iterable[int],
) -> int:
    split = as_partition(partition)
    canonical = canonicalize_factors(factors, tensor_size=split.tensor_size)
    return sum(1 for factor in canonical if is_bridge_factor(factor, split))


def gadget_aware_mixed_cost(
    factors: np.ndarray,
    partition: TensorPartition | Iterable[int],
) -> int:
    return gadget_aware_mixed_stats(factors, partition)[
        "gadget_aware_effective_mixed_cost"
    ]


def gadget_aware_mixed_stats(
    factors: np.ndarray,
    partition: TensorPartition | Iterable[int],
    *,
    groups: list[FactorGroup] | None = None,
) -> dict[str, int]:
    split = as_partition(partition)
    canonical = canonicalize_factors(factors, tensor_size=split.tensor_size)
    effective_cost = 0
    total_mixed_weight = 0
    mixed_group_count = 0
    mixed_block_span = 0
    factor_groups = groups if groups is not None else group_multiset_gadgets(canonical)
    for group in factor_groups:
        group_factors = canonical[list(group.factor_indices)]
        group_mixed_weight = int(
            np.count_nonzero(project_mixed_tensor(tensor_from_factors(group_factors), split))
        )
        if group_mixed_weight == 0:
            continue
        effective_cost += group.effective_cost
        total_mixed_weight += group_mixed_weight
        mixed_group_count += 1
        mixed_block_span += group_block_span(group_factors, split)
    return {
        "gadget_aware_effective_mixed_cost": int(effective_cost),
        "gadget_mixed_weight": int(total_mixed_weight),
        "mixed_group_count": int(mixed_group_count),
        "mixed_block_span": int(mixed_block_span),
    }


def tensor_split_v3_stats(
    target_tensor: np.ndarray,
    factors: np.ndarray,
    partition: TensorPartition | Iterable[int],
    *,
    groups: list[FactorGroup] | None = None,
) -> dict[str, int | float | str]:
    split = as_partition(partition)
    target = _as_binary_tensor(target_tensor)
    canonical = canonicalize_factors(factors, tensor_size=split.tensor_size)
    factor_groups = groups if groups is not None else group_multiset_gadgets(canonical)
    target_mixed_weight = mixed_weight(target, split)
    denominator = max(target_mixed_weight, 1)
    mixed_group_tensors = [
        group_mixed_tensor(canonical, group, split) for group in factor_groups
    ]
    mixed_group_weights = [
        int(np.count_nonzero(mixed_tensor)) for mixed_tensor in mixed_group_tensors
    ]
    effective_cost = 0
    total_mixed_weight = 0
    mixed_group_count = 0
    mixed_block_span = 0
    for group, group_mixed_weight in zip(factor_groups, mixed_group_weights):
        if group_mixed_weight == 0:
            continue
        group_factors = canonical[list(group.factor_indices)]
        effective_cost += group.effective_cost
        total_mixed_weight += group_mixed_weight
        mixed_group_count += 1
        mixed_block_span += group_block_span(group_factors, split)
    target_mixed_tensor = project_mixed_tensor(target, split)
    mixed_auc_original = group_mixed_auc_from_mixed_tensors(
        target_mixed_tensor, mixed_group_tensors
    )
    mixed_auc_greedy_value = mixed_auc_greedy_from_mixed_tensors(
        target_mixed_tensor, mixed_group_tensors
    )
    group_denominator = max(len(factor_groups), 1)
    auc_denominator = (len(factor_groups) + 1) * denominator
    singleton_count = singleton_bridge_count(canonical, split, groups=factor_groups)
    union = np.zeros_like(target_mixed_tensor, dtype=bool)
    for mixed_tensor in mixed_group_tensors:
        union |= mixed_tensor.astype(bool)
    off_target_mixed = int(np.count_nonzero(union & ~target_mixed_tensor.astype(bool)))
    stable_hash = stable_factor_hash(canonical)
    mixed_excess_norm = (total_mixed_weight - target_mixed_weight) / denominator
    mixed_auc_greedy_norm = mixed_auc_greedy_value / auc_denominator
    singleton_bridge_count_norm = singleton_count / max(len(canonical), 1)
    return {
        "gadget_aware_effective_mixed_cost": int(effective_cost),
        "gadget_mixed_weight": int(total_mixed_weight),
        "mixed_group_count": int(mixed_group_count),
        "mixed_block_span": int(mixed_block_span),
        "target_mixed_weight": int(target_mixed_weight),
        "gadget_mixed_weight_norm": float(total_mixed_weight / denominator),
        "mixed_excess_norm": float(mixed_excess_norm),
        "mixed_auc_original": int(mixed_auc_original),
        "mixed_auc_original_norm": float(mixed_auc_original / auc_denominator),
        "mixed_auc_greedy": int(mixed_auc_greedy_value),
        "mixed_auc_greedy_norm": float(mixed_auc_greedy_norm),
        "off_target_mixed_weight": int(off_target_mixed),
        "off_target_mixed_weight_norm": float(off_target_mixed / denominator),
        "mixed_group_count_norm": float(mixed_group_count / group_denominator),
        "singleton_bridge_count": int(singleton_count),
        "singleton_bridge_count_norm": float(singleton_bridge_count_norm),
        "stable_factor_hash": stable_hash,
        "score_v3_lex": score_v3_lex_string(
            mixed_excess_norm=float(mixed_excess_norm),
            mixed_auc_greedy_norm=float(mixed_auc_greedy_norm),
            singleton_bridge_count_norm=float(singleton_bridge_count_norm),
            stable_hash=stable_hash,
        ),
    }


def group_mixed_auc(
    target_tensor: np.ndarray,
    factors: np.ndarray,
    partition: TensorPartition | Iterable[int],
    *,
    groups: list[FactorGroup] | None = None,
) -> int:
    split = as_partition(partition)
    residual = _as_binary_tensor(target_tensor).copy()
    canonical = canonicalize_factors(factors, tensor_size=split.tensor_size)
    factor_groups = groups if groups is not None else group_multiset_gadgets(canonical)
    residual_mixed = project_mixed_tensor(residual, split)
    return group_mixed_auc_from_mixed_tensors(
        residual_mixed,
        [
            group_mixed_tensor(canonical, group, split)
            for group in sorted(factor_groups, key=lambda item: item.start)
        ],
    )


def mixed_auc_greedy(
    target_tensor: np.ndarray,
    factors: np.ndarray,
    partition: TensorPartition | Iterable[int],
    *,
    groups: list[FactorGroup] | None = None,
) -> int:
    split = as_partition(partition)
    residual = _as_binary_tensor(target_tensor).copy()
    canonical = canonicalize_factors(factors, tensor_size=split.tensor_size)
    factor_groups = groups if groups is not None else group_multiset_gadgets(canonical)
    remaining = [
        (group, group_mixed_tensor(canonical, group, split))
        for group in sorted(factor_groups, key=lambda item: item.start)
    ]
    residual_mixed = project_mixed_tensor(residual, split)
    return mixed_auc_greedy_from_mixed_tensors(
        residual_mixed, [mixed_tensor for _, mixed_tensor in remaining]
    )


def group_mixed_auc_from_mixed_tensors(
    target_mixed_tensor: np.ndarray,
    mixed_tensors: list[np.ndarray],
) -> int:
    residual_mixed = np.asarray(target_mixed_tensor, dtype=np.uint8).copy()
    area = int(np.count_nonzero(residual_mixed))
    for mixed_tensor in mixed_tensors:
        residual_mixed ^= mixed_tensor
        area += int(np.count_nonzero(residual_mixed))
    return int(area)


def mixed_auc_greedy_from_mixed_tensors(
    target_mixed_tensor: np.ndarray,
    mixed_tensors: list[np.ndarray],
) -> int:
    residual_mixed = np.asarray(target_mixed_tensor, dtype=np.uint8).copy()
    remaining = list(mixed_tensors)
    area = int(np.count_nonzero(residual_mixed))
    while remaining:
        best_index, best_tensor, best_weight, _ = min(
            (
                (
                    index,
                    tensor,
                    int(np.count_nonzero(residual_mixed ^ tensor)),
                    stable_tensor_hash(tensor),
                )
                for index, tensor in enumerate(remaining)
            ),
            key=lambda item: (item[2], item[3], item[0]),
        )
        residual_mixed ^= best_tensor
        area += best_weight
        remaining.pop(best_index)
    return int(area)


def singleton_bridge_count(
    factors: np.ndarray,
    partition: TensorPartition | Iterable[int],
    *,
    groups: list[FactorGroup] | None = None,
) -> int:
    split = as_partition(partition)
    canonical = canonicalize_factors(factors, tensor_size=split.tensor_size)
    factor_groups = groups if groups is not None else group_multiset_gadgets(canonical)
    return sum(
        1
        for group in factor_groups
        if group.group_type == "factor"
        and is_bridge_factor(canonical[group.factor_indices[0]], split)
    )


def off_target_mixed_weight(
    target_tensor: np.ndarray,
    factors: np.ndarray,
    partition: TensorPartition | Iterable[int],
    *,
    groups: list[FactorGroup] | None = None,
) -> int:
    split = as_partition(partition)
    target_mixed = project_mixed_tensor(target_tensor, split).astype(bool)
    canonical = canonicalize_factors(factors, tensor_size=split.tensor_size)
    factor_groups = groups if groups is not None else group_multiset_gadgets(canonical)
    union = np.zeros_like(target_mixed, dtype=bool)
    for group in factor_groups:
        union |= group_mixed_tensor(canonical, group, split).astype(bool)
    return int(np.count_nonzero(union & ~target_mixed))


def group_tensor(factors: np.ndarray, group: FactorGroup) -> np.ndarray:
    return tensor_from_factors(factors[list(group.factor_indices)])


def group_mixed_tensor(
    factors: np.ndarray,
    group: FactorGroup,
    partition: TensorPartition | Iterable[int],
) -> np.ndarray:
    return project_mixed_tensor(group_tensor(factors, group), partition)


def stable_factor_hash(factors: np.ndarray) -> str:
    canonical = canonicalize_factors(factors)
    digest = hashlib.sha256()
    digest.update(str(tuple(canonical.shape)).encode("ascii"))
    digest.update(canonical.astype(np.uint8, copy=False).tobytes(order="C"))
    return digest.hexdigest()[:16]


def stable_tensor_hash(tensor: np.ndarray) -> str:
    array = np.asarray(tensor, dtype=np.uint8) % 2
    digest = hashlib.sha256()
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()[:16]


def score_v3_lex_string(
    *,
    mixed_excess_norm: float,
    mixed_auc_greedy_norm: float,
    singleton_bridge_count_norm: float,
    stable_hash: str,
) -> str:
    return (
        f"{mixed_excess_norm:.12g};"
        f"{mixed_auc_greedy_norm:.12g};"
        f"{singleton_bridge_count_norm:.12g};"
        f"{stable_hash}"
    )


def group_block_span(
    factors: np.ndarray,
    partition: TensorPartition | Iterable[int],
) -> int:
    split = as_partition(partition)
    canonical = canonicalize_factors(factors, tensor_size=split.tensor_size)
    active = set()
    for factor in canonical:
        active.update(int(split.block_of[index]) for index in np.flatnonzero(factor))
    return len(active)


def group_multiset_gadgets(factors: np.ndarray) -> list[FactorGroup]:
    canonical = canonicalize_factors(factors)
    used: set[int] = set()
    groups: list[FactorGroup] = []
    groups.extend(_find_toffoli_multisets(canonical, used))
    groups.extend(_find_cs_multisets(canonical, used))
    for index in range(len(canonical)):
        if index not in used:
            groups.append(FactorGroup("factor", (index,), 1))
    return sorted(groups, key=lambda group: group.start)


def group_consecutive_gadgets(factors: np.ndarray) -> list[FactorGroup]:
    canonical = canonicalize_factors(factors)
    groups: list[FactorGroup] = []
    index = 0
    while index < len(canonical):
        if index + 7 <= len(canonical) and factors_form_toffoli_gadget(
            canonical[index : index + 7]
        ):
            groups.append(FactorGroup("toffoli", tuple(range(index, index + 7)), 2))
            index += 7
            continue
        if index + 3 <= len(canonical) and factors_form_cs_gadget(
            canonical[index : index + 3]
        ):
            groups.append(FactorGroup("cs", tuple(range(index, index + 3)), 2))
            index += 3
            continue
        groups.append(FactorGroup("factor", (index,), 1))
        index += 1
    return groups


def factors_form_cs_gadget(factors: np.ndarray) -> bool:
    factors = canonicalize_factors(factors)
    if factors.shape[0] != 3:
        return False
    a, b, ab = factors
    return bool(np.any(a != b) and np.all(ab == ((a + b) % 2)))


def factors_form_toffoli_gadget(factors: np.ndarray) -> bool:
    factors = canonicalize_factors(factors)
    if factors.shape[0] != 7:
        return False
    a, b, c, ab, ac, abc, bc = factors
    if not _three_factors_linearly_independent(a, b, c):
        return False
    return bool(
        np.all(ab == ((a + b) % 2))
        and np.all(ac == ((a + c) % 2))
        and np.all(abc == ((a + b + c) % 2))
        and np.all(bc == ((b + c) % 2))
    )


def _semantic_register_partition(
    *,
    circuit_id: str,
    qubits: list[QubitRef],
    mapping: list[int],
    tensor_size: int,
) -> TensorPartition | None:
    registers = list(dict.fromkeys(qubit.register for qubit in qubits))
    if len(registers) < 2:
        return None
    output_register = registers[-1]
    return _mapped_semantic_partition(
        partition_id="semantic_register_k2",
        kind="semantic-register",
        circuit_id=circuit_id,
        qubits=qubits,
        mapping=mapping,
        tensor_size=tensor_size,
        block_for_qubit=lambda qubit: 1 if qubit.register == output_register else 0,
    )


def _semantic_role_partition(
    *,
    circuit_id: str,
    qubits: list[QubitRef],
    mapping: list[int],
    tensor_size: int,
) -> TensorPartition | None:
    def block_for_qubit(qubit: QubitRef) -> int:
        if circuit_id.startswith("gf_2pow"):
            return 1 if qubit.register == "c" else 0
        hamming_match = re.fullmatch(r"hamming_weight_n(\d+)", circuit_id)
        if hamming_match:
            return 0 if qubit.index < int(hamming_match.group(1)) else 1
        if circuit_id == "mod_5_4":
            return 0 if qubit.index < 4 else 1
        return 0

    return _mapped_semantic_partition(
        partition_id="semantic_role_k2",
        kind="semantic-role",
        circuit_id=circuit_id,
        qubits=qubits,
        mapping=mapping,
        tensor_size=tensor_size,
        block_for_qubit=block_for_qubit,
        extra_block=1,
    )


def _semantic_family_partition(
    *,
    circuit_id: str,
    qubits: list[QubitRef],
    mapping: list[int],
    tensor_size: int,
) -> TensorPartition | None:
    qft_match = re.fullmatch(r"qft_(\d+)", circuit_id)
    if qft_match:
        cutoff = max(1, int(qft_match.group(1)) // 2)
        block_for_qubit = lambda qubit: 0 if qubit.index < cutoff else 1
    elif circuit_id.startswith("gf_2pow"):
        block_for_qubit = lambda qubit: 1 if qubit.register == "c" else 0
    else:
        hamming_match = re.fullmatch(r"hamming_weight_n(\d+)", circuit_id)
        if hamming_match:
            cutoff = int(hamming_match.group(1))
            block_for_qubit = lambda qubit: 0 if qubit.index < cutoff else 1
        elif circuit_id == "mod_5_4":
            block_for_qubit = lambda qubit: 0 if qubit.index < 4 else 1
        else:
            cutoff = max(1, len(qubits) // 2)
            block_for_qubit = lambda qubit: 0 if qubit.flat_index < cutoff else 1

    return _mapped_semantic_partition(
        partition_id="semantic_family_k2",
        kind="semantic-family",
        circuit_id=circuit_id,
        qubits=qubits,
        mapping=mapping,
        tensor_size=tensor_size,
        block_for_qubit=block_for_qubit,
        extra_block=1,
    )


def _mapped_semantic_partition(
    *,
    partition_id: str,
    kind: str,
    circuit_id: str,
    qubits: list[QubitRef],
    mapping: list[int],
    tensor_size: int,
    block_for_qubit: object,
    extra_block: int = 1,
) -> TensorPartition | None:
    block_of = np.zeros((tensor_size,), dtype=np.int32)
    has_extra_mapping = False
    for tensor_index in range(tensor_size):
        mapped_index = mapping[tensor_index] if tensor_index < len(mapping) else tensor_index
        if 0 <= mapped_index < len(qubits):
            block_of[tensor_index] = int(block_for_qubit(qubits[mapped_index]))  # type: ignore[operator]
        else:
            block_of[tensor_index] = extra_block
            has_extra_mapping = True
    if len(set(int(value) for value in block_of)) < 2:
        return None
    status = "partial-mapping" if has_extra_mapping else "ok"
    return TensorPartition(
        partition_id=partition_id,
        block_of=block_of,
        kind=kind,
        semantic_partition_status=status,
        semantic_partition_error=None if status == "ok" else "extra tensor indices assigned to work/gadget block",
    )


def _find_toffoli_multisets(
    factors: np.ndarray,
    used: set[int],
) -> list[FactorGroup]:
    groups: list[FactorGroup] = []
    while True:
        found = _find_one_toffoli_multiset(factors, used)
        if found is None:
            return groups
        groups.append(FactorGroup("toffoli", found, 2))
        used.update(found)


def _find_one_toffoli_multiset(
    factors: np.ndarray,
    used: set[int],
) -> tuple[int, ...] | None:
    for start in range(max(0, len(factors) - 6)):
        candidate = tuple(range(start, start + 7))
        if any(index in used for index in candidate):
            continue
        if _factors_form_toffoli_multiset(factors[list(candidate)]):
            return candidate
    return None


def _find_cs_multisets(
    factors: np.ndarray,
    used: set[int],
) -> list[FactorGroup]:
    groups: list[FactorGroup] = []
    while True:
        found = _find_one_cs_multiset(factors, used)
        if found is None:
            return groups
        groups.append(FactorGroup("cs", found, 2))
        used.update(found)


def _find_one_cs_multiset(
    factors: np.ndarray,
    used: set[int],
) -> tuple[int, ...] | None:
    for start in range(max(0, len(factors) - 2)):
        candidate = tuple(range(start, start + 3))
        if any(index in used for index in candidate):
            continue
        if _factors_form_cs_multiset(factors[list(candidate)]):
            return candidate
    return None


def _factors_form_toffoli_multiset(factors: np.ndarray) -> bool:
    if factors.shape[0] != 7:
        return False
    vector_index = _vector_index(factors)
    for i, j, k in itertools.combinations(range(7), 3):
        a, b, c = factors[i], factors[j], factors[k]
        if not _three_factors_linearly_independent(a, b, c):
            continue
        forbidden = {i, j, k}
        chosen = []
        for vector in (
            (a + b) % 2,
            (a + c) % 2,
            (a + b + c) % 2,
            (b + c) % 2,
        ):
            found = _first_available_index(vector_index, vector, forbidden | set(chosen))
            if found is None:
                break
            chosen.append(found)
        if len(chosen) == 4:
            return True
    return False


def _factors_form_cs_multiset(factors: np.ndarray) -> bool:
    if factors.shape[0] != 3:
        return False
    vector_index = _vector_index(factors)
    for i, j in itertools.combinations(range(3), 2):
        if np.all(factors[i] == factors[j]):
            continue
        found = _first_available_index(vector_index, (factors[i] + factors[j]) % 2, {i, j})
        if found is not None:
            return True
    return False


def _vector_index(factors: np.ndarray) -> dict[tuple[int, ...], list[int]]:
    index: dict[tuple[int, ...], list[int]] = {}
    for factor_index, factor in enumerate(factors):
        index.setdefault(tuple(int(value) for value in factor), []).append(factor_index)
    return index


def _first_available_index(
    vector_index: dict[tuple[int, ...], list[int]],
    vector: np.ndarray,
    forbidden: set[int],
) -> int | None:
    key = tuple(int(value) for value in vector)
    for candidate in vector_index.get(key, []):
        if candidate not in forbidden:
            return candidate
    return None


def _three_factors_linearly_independent(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> bool:
    return bool(
        np.any(a != b)
        and np.any(a != c)
        and np.any(b != c)
        and np.any(c != ((a + b) % 2))
    )


def _as_binary_tensor(tensor: np.ndarray) -> np.ndarray:
    array = np.asarray(tensor, dtype=np.uint8) % 2
    if array.ndim != 3 or len(set(array.shape)) != 1:
        raise ValueError(f"Expected a cubic rank-3 tensor, got shape {array.shape}.")
    return array


def _local_mask(partition: TensorPartition) -> np.ndarray:
    blocks = partition.block_of
    return (
        (blocks[:, None, None] == blocks[None, :, None])
        & (blocks[:, None, None] == blocks[None, None, :])
    )
