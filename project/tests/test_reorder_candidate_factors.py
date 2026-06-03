from __future__ import annotations

import csv

import numpy as np

from scripts.materialize_split_reward_candidate import rank_one_tensor_sum


def test_reorder_candidate_factors_preserves_reconstruction(tmp_path):
    from scripts import optimize_linear_span_candidate
    from scripts import reorder_candidate_factors

    output_root = tmp_path / "milp"
    optimize_linear_span_candidate.main_from_namespace_for_test(
        target="hamming_weight_n4",
        action_dictionary="low-weight",
        max_action_weight=3,
        tensor_overlap_max_weight=5,
        tensor_overlap_max_actions_per_target=175,
        objective="factor-count",
        mixed_weight_scale=1.0,
        support_weight_scale=0.25,
        max_factors=None,
        time_limit_sec=20.0,
        mip_rel_gap=0.0,
        candidate_kind="milp_span",
        output_root=output_root,
    )
    source_manifest = next(output_root.glob("*/candidate_factors_manifest.csv"))
    reorder_root = tmp_path / "reordered"

    exit_code = reorder_candidate_factors.main_from_namespace_for_test(
        target="hamming_weight_n4",
        manifest_csv=source_manifest,
        candidate_kind="milp_span",
        strategy="greedy-residual",
        output_root=reorder_root,
    )

    assert exit_code == 0
    manifest = next(reorder_root.glob("*/candidate_factors_manifest.csv"))
    with manifest.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    factors = np.load(row["factor_path"])
    target = np.load(
        "external/circuit-to-tensor/benchmarks/applications/"
        "hamming_weight_n4/hamming_weight_n4.tensor.npy"
    ).astype(bool)
    assert row["candidate_kind"] == "milp_span_greedy_residual"
    assert np.array_equal(rank_one_tensor_sum(factors), target)
