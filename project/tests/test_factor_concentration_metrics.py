from __future__ import annotations

import numpy as np

from scripts.factor_concentration_metrics import concentration_index
from scripts.factor_concentration_metrics import factor_concentration_metrics


def test_concentration_index_is_larger_for_concentrated_weights() -> None:
    assert concentration_index(np.array([3, 0, 0])) > concentration_index(np.array([1, 1, 1]))


def test_factor_concentration_metrics_detects_reused_parities() -> None:
    factors = np.array(
        [
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
        ],
        dtype=np.uint8,
    )

    metrics = factor_concentration_metrics(factors)

    assert metrics["factor_count"] == 3
    assert metrics["unique_parity_count"] == 2
    assert metrics["parity_reuse_ratio"] == 1 / 3
    assert metrics["support_weight_max"] == 2
    assert metrics["pairwise_support_overlap_mean"] > 0
