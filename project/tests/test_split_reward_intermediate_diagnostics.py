from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_split_reward_intermediates import initial_action_audit
from scripts.analyze_split_reward_intermediates import replay_candidate
from scripts.tensor_split_core import outer3


def test_replay_candidate_detects_zero_net_cancellation_cycle() -> None:
    factor = np.asarray([1, 0, 1], dtype=np.uint8)
    target = outer3(np.asarray([1, 1, 0], dtype=np.uint8))
    factors = np.stack([factor, factor], axis=0)

    diagnostics = replay_candidate(target_tensor=target, factors=factors)

    assert diagnostics["final_residual_weight"] == diagnostics["initial_residual_weight"]
    assert diagnostics["net_tensor_weight"] == 0
    assert diagnostics["odd_factor_count"] == 0
    assert diagnostics["cancellation_fraction"] == 1.0


def test_initial_action_audit_counts_improving_actions() -> None:
    factor = np.asarray([1, 0, 1], dtype=np.uint8)
    target = outer3(factor)

    audit = initial_action_audit(target, max_weight=2)

    weight_two = next(row for row in audit["rows"] if row["weight"] == 2)
    assert weight_two["best_delta"] == -8
    assert weight_two["improving_actions"] == 1
