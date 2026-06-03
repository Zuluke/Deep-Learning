from __future__ import annotations

import csv

import numpy as np

from scripts.materialize_split_reward_candidate import rank_one_tensor_sum
from scripts.optimize_linear_span_candidate import pair_incidence_for_actions
from scripts.optimize_linear_span_candidate import solve_mod2_milp
from scripts.optimize_linear_span_candidate import write_solution_manifest


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


def test_solve_mod2_milp_respects_pair_overlap_cap():
    columns = [
        np.array([1, 1, 0], dtype=np.uint8),
        np.array([1, 0, 0], dtype=np.uint8),
        np.array([0, 1, 0], dtype=np.uint8),
    ]
    target = np.array([1, 1, 0], dtype=np.uint8)
    pair_incidence = np.array([[1, 0, 0]], dtype=np.uint8)

    result = solve_mod2_milp(
        columns,
        target,
        pair_incidence=pair_incidence,
        max_pair_overlap=0,
        time_limit_sec=5.0,
    )

    assert result.is_optimal
    assert result.objective_value == 2.0
    assert result.coefficients.tolist() == [0, 1, 1]


def test_pair_incidence_for_actions_marks_shared_support_pairs():
    # action + 1 bit patterns: 001, 011, 101.
    incidence = pair_incidence_for_actions(3, [0, 2, 4])

    assert incidence.shape == (3, 3)
    assert incidence[:, 0].tolist() == [0, 0, 0]
    assert incidence[:, 1].tolist() == [1, 0, 0]
    assert incidence[:, 2].tolist() == [0, 1, 0]


def test_write_solution_manifest_preserves_existing_candidate_rows(tmp_path):
    manifest = tmp_path / "candidate_factors_manifest.csv"
    factor_a = tmp_path / "a.npy"
    factor_b = tmp_path / "b.npy"
    cob = tmp_path / "basis.npy"

    write_solution_manifest(
        manifest,
        target="toy",
        candidate_kind="a",
        factor_path=factor_a,
        change_of_basis_path=cob,
        num_moves=1,
        effective_t_cost=1,
        objective_value=1.0,
        solver_status=0,
        solver_message="ok",
        is_optimal=True,
    )
    write_solution_manifest(
        manifest,
        target="toy",
        candidate_kind="b",
        factor_path=factor_b,
        change_of_basis_path=cob,
        num_moves=2,
        effective_t_cost=2,
        objective_value=2.0,
        solver_status=0,
        solver_message="ok",
        is_optimal=True,
    )

    with manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert [row["candidate_kind"] for row in rows] == ["a", "b"]


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
        max_pair_overlap=None,
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
