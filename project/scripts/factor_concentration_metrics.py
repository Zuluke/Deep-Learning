from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np


def factor_concentration_metrics(factors: np.ndarray) -> dict[str, Any]:
    factors = np.asarray(factors, dtype=np.uint8) % 2
    if factors.ndim != 2:
        raise ValueError(f"Expected factors with shape (num_factors, tensor_size), got {factors.shape}.")
    factor_count, tensor_size = factors.shape
    support_weights = factors.sum(axis=1).astype(float)
    unique_parities, multiplicities = unique_factor_multiplicities(factors)
    qubit_incidence = factors.sum(axis=0).astype(float)
    return {
        "factor_count": int(factor_count),
        "tensor_size": int(tensor_size),
        "unique_parity_count": int(len(unique_parities)),
        "parity_reuse_ratio": safe_ratio(factor_count - len(unique_parities), factor_count),
        "parity_concentration_index": concentration_index(multiplicities),
        "qubit_concentration_index": concentration_index(qubit_incidence),
        "support_weight_mean": float(support_weights.mean()) if factor_count else 0.0,
        "support_weight_std": float(support_weights.std()) if factor_count else 0.0,
        "support_weight_max": int(support_weights.max()) if factor_count else 0,
        "pairwise_support_overlap_mean": pairwise_support_overlap_mean(factors),
        "pairwise_jaccard_mean": pairwise_jaccard_mean(factors),
    }


def unique_factor_multiplicities(factors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if factors.size == 0:
        return np.empty((0, factors.shape[1]), dtype=np.uint8), np.empty((0,), dtype=float)
    unique, counts = np.unique(factors, axis=0, return_counts=True)
    return unique.astype(np.uint8), counts.astype(float)


def concentration_index(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    total = float(weights.sum())
    if total <= 0:
        return 0.0
    return float(np.sum(weights * weights) / (total * total))


def pairwise_support_overlap_mean(factors: np.ndarray) -> float:
    if len(factors) < 2:
        return 0.0
    overlaps = [
        int(np.count_nonzero(factors[i] & factors[j]))
        for i, j in combinations(range(len(factors)), 2)
    ]
    return float(np.mean(overlaps)) if overlaps else 0.0


def pairwise_jaccard_mean(factors: np.ndarray) -> float:
    if len(factors) < 2:
        return 0.0
    values: list[float] = []
    for i, j in combinations(range(len(factors)), 2):
        intersection = int(np.count_nonzero(factors[i] & factors[j]))
        union = int(np.count_nonzero(factors[i] | factors[j]))
        values.append(0.0 if union == 0 else intersection / union)
    return float(np.mean(values)) if values else 0.0


def safe_ratio(numerator: float | int, denominator: float | int) -> float | None:
    denominator = float(denominator)
    if denominator <= 0:
        return None
    return float(numerator) / denominator
