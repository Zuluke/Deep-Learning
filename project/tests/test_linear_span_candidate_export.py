from __future__ import annotations

import csv

import numpy as np

from scripts.export_linear_span_candidate import solve_gf2
from scripts.materialize_split_reward_candidate import rank_one_tensor_sum


def test_solve_gf2_returns_exact_solution():
    columns = [
        np.array([1, 0, 1], dtype=np.uint8),
        np.array([0, 1, 1], dtype=np.uint8),
    ]
    target = np.array([1, 1, 0], dtype=np.uint8)

    solution, rank, nullity = solve_gf2(columns, target)

    assert rank == 2
    assert nullity == 0
    assert solution.tolist() == [1, 1]


def test_export_linear_span_candidate_reconstructs_hamming_n4(tmp_path):
    from scripts import export_linear_span_candidate

    output_root = tmp_path / "linear_span"
    exit_code = export_linear_span_candidate.main_from_namespace_for_test(
        target="hamming_weight_n4",
        action_dictionary="low-weight",
        max_action_weight=3,
        tensor_overlap_max_weight=5,
        tensor_overlap_max_actions_per_target=175,
        candidate_kind="linear_span",
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

    assert factors.shape == (40, 9)
    assert np.array_equal(rank_one_tensor_sum(factors), target)
