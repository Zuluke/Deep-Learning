from __future__ import annotations

import csv

import numpy as np

from scripts.materialize_split_reward_candidate import rank_one_tensor_sum
from scripts.optimize_linear_span_candidate import solve_mod2_milp


def test_solve_mod2_milp_minimizes_exact_toy_solution():
    columns = [
        np.array([1, 0, 1], dtype=np.uint8),
        np.array([0, 1, 1], dtype=np.uint8),
        np.array([1, 1, 0], dtype=np.uint8),
    ]
    target = np.array([1, 1, 0], dtype=np.uint8)

    result = solve_mod2_milp(columns, target, time_limit_sec=5.0)

    assert result.is_optimal
    assert result.objective_value == 1.0
    assert result.coefficients.tolist() == [0, 0, 1]


def test_optimize_linear_span_candidate_reconstructs_hamming_n4_loww3(tmp_path):
    from scripts import optimize_linear_span_candidate

    output_root = tmp_path / "milp_span"
    exit_code = optimize_linear_span_candidate.main_from_namespace_for_test(
        target="hamming_weight_n4",
        action_dictionary="low-weight",
        max_action_weight=3,
        tensor_overlap_max_weight=5,
        tensor_overlap_max_actions_per_target=175,
        objective="factor-count",
        mixed_weight_scale=1.0,
        support_weight_scale=0.25,
        pair_weight_scale=0.0,
        max_factors=None,
        time_limit_sec=20.0,
        mip_rel_gap=0.0,
        candidate_kind="milp_span",
        output_root=output_root,
    )

    assert exit_code == 0
    manifest = next(output_root.glob("*/candidate_factors_manifest.csv"))
    with manifest.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    factors = np.load(row["factor_path"])
    target = np.load(
        "external/circuit-to-tensor/benchmarks/applications/"
        "hamming_weight_n4/hamming_weight_n4.tensor.npy"
    ).astype(bool)

    assert row["span_is_optimal"] == "True"
    assert factors.shape == (40, 9)
    assert np.array_equal(rank_one_tensor_sum(factors), target)
