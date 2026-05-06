from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Literal

import numpy as np

from atq_repro.tensor import gf2_rank


GadgetKind = Literal["toffoli", "cs", "t"]


@dataclass(frozen=True)
class GadgetGroup:
    kind: GadgetKind
    start: int
    length: int


@dataclass(frozen=True)
class GadgetSummary:
    num_factors: int
    num_toffoli: int
    num_cs: int
    num_t_remaining: int
    effective_tcount: int
    groups: tuple[GadgetGroup, ...]


def _same_vector(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(np.array_equal(left, right))


def is_cs_group(group: np.ndarray) -> bool:
    if len(group) != 3:
        return False
    first, second, third = group
    return gf2_rank(np.asarray([first, second])) == 2 and _same_vector(third, first ^ second)


def is_toffoli_group(group: np.ndarray) -> bool:
    if len(group) != 7 or gf2_rank(group) != 3:
        return False

    target = {tuple(row.tolist()) for row in group}
    for basis_indices in combinations(range(7), 3):
        basis = [group[index] for index in basis_indices]
        if gf2_rank(np.asarray(basis)) != 3:
            continue
        first, second, third = basis
        span = {
            tuple(row.tolist())
            for row in (
                first,
                second,
                third,
                first ^ second,
                first ^ third,
                second ^ third,
                first ^ second ^ third,
            )
        }
        if span == target:
            return True
    return False


def analyze_gadgetization(factors: np.ndarray, *, use_gadgets: bool) -> GadgetSummary:
    factor_array = np.asarray(factors, dtype=np.uint8)
    if factor_array.ndim != 2:
        raise ValueError(f"Expected factors with shape (rank, tensor_size), got {factor_array.shape}.")

    if not use_gadgets:
        return GadgetSummary(
            num_factors=int(len(factor_array)),
            num_toffoli=0,
            num_cs=0,
            num_t_remaining=int(len(factor_array)),
            effective_tcount=int(len(factor_array)),
            groups=tuple(GadgetGroup("t", index, 1) for index in range(len(factor_array))),
        )

    groups: list[GadgetGroup] = []
    num_toffoli = 0
    num_cs = 0
    num_t_remaining = 0
    index = 0
    while index < len(factor_array):
        if index + 7 <= len(factor_array) and is_toffoli_group(factor_array[index : index + 7]):
            groups.append(GadgetGroup("toffoli", index, 7))
            num_toffoli += 1
            index += 7
            continue
        if index + 3 <= len(factor_array) and is_cs_group(factor_array[index : index + 3]):
            groups.append(GadgetGroup("cs", index, 3))
            num_cs += 1
            index += 3
            continue
        groups.append(GadgetGroup("t", index, 1))
        num_t_remaining += 1
        index += 1

    return GadgetSummary(
        num_factors=int(len(factor_array)),
        num_toffoli=num_toffoli,
        num_cs=num_cs,
        num_t_remaining=num_t_remaining,
        effective_tcount=2 * num_toffoli + 2 * num_cs + num_t_remaining,
        groups=tuple(groups),
    )
