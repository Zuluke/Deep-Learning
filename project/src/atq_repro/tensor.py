from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TensorValidation:
    equal: bool
    mismatch_count: int | None
    target_shape: tuple[int, ...] | None
    reconstructed_shape: tuple[int, ...]
    error: str | None = None


def gf2_rank(matrix: np.ndarray) -> int:
    rows = np.asarray(matrix, dtype=np.uint8).copy()
    if rows.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape {rows.shape}.")
    if rows.size == 0:
        return 0

    n_rows, n_cols = rows.shape
    rank = 0
    for col in range(n_cols):
        pivot = None
        for row in range(rank, n_rows):
            if rows[row, col]:
                pivot = row
                break
        if pivot is None:
            continue
        if pivot != rank:
            rows[[rank, pivot]] = rows[[pivot, rank]]
        for row in range(n_rows):
            if row != rank and rows[row, col]:
                rows[row] ^= rows[rank]
        rank += 1
        if rank == n_rows:
            break
    return rank


def symmetric_tensor_from_factors(factors: np.ndarray) -> np.ndarray:
    factor_array = np.asarray(factors, dtype=np.uint8)
    if factor_array.ndim != 2:
        raise ValueError(f"Expected factors with shape (rank, tensor_size), got {factor_array.shape}.")

    tensor_size = factor_array.shape[1]
    tensor = np.zeros((tensor_size, tensor_size, tensor_size), dtype=np.uint8)
    for factor in factor_array:
        rank_one = (
            factor[:, None, None]
            & factor[None, :, None]
            & factor[None, None, :]
        )
        tensor ^= rank_one.astype(np.uint8)
    return tensor.astype(bool)


def validate_decomposition(factors: np.ndarray, target_tensor: np.ndarray | None) -> TensorValidation:
    reconstructed = symmetric_tensor_from_factors(factors)
    if target_tensor is None:
        return TensorValidation(
            equal=False,
            mismatch_count=None,
            target_shape=None,
            reconstructed_shape=tuple(reconstructed.shape),
            error="target tensor not found",
        )

    target = np.asarray(target_tensor, dtype=bool)
    if target.shape != reconstructed.shape:
        return TensorValidation(
            equal=False,
            mismatch_count=None,
            target_shape=tuple(target.shape),
            reconstructed_shape=tuple(reconstructed.shape),
            error="shape mismatch",
        )

    mismatch = np.logical_xor(target, reconstructed)
    mismatch_count = int(mismatch.sum())
    return TensorValidation(
        equal=mismatch_count == 0,
        mismatch_count=mismatch_count,
        target_shape=tuple(target.shape),
        reconstructed_shape=tuple(reconstructed.shape),
    )
