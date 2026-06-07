from __future__ import annotations

from scripts.run_objective_beam_policy_grid import best_policy_beams
from scripts.run_objective_beam_policy_grid import decomp_rows_for_materialization
from scripts.run_objective_beam_policy_grid import policy_summary_rows


def test_decomp_rows_for_materialization_skips_failed_objectives() -> None:
    rows = [
        {
            "target": "toy",
            "objective_variant": "factor_count",
            "execution_status": "ok",
            "tcount": "5",
        },
        {
            "target": "toy",
            "objective_variant": "mixed_pair",
            "execution_status": "failed",
            "error_message": "timeout",
        },
    ]

    materialized = decomp_rows_for_materialization(rows)

    assert len(materialized) == 1
    assert materialized[0]["objective_variant"] == "factor_count"


def test_best_policy_beams_selects_best_beam_for_one_objective() -> None:
    rows = [
        {
            "target": "toy",
            "objective_variant": "factor_count_pair_cap",
            "materializer": "selected-beam-shared-parity-w4",
            "qasm_depth_ratio": "2.0",
            "primary_nc_depth_ratio": "0.8",
            "num_total_cnots": "20",
        },
        {
            "target": "toy",
            "objective_variant": "factor_count_pair_cap",
            "materializer": "selected-beam-shared-parity-w16",
            "qasm_depth_ratio": "1.8",
            "primary_nc_depth_ratio": "0.9",
            "num_total_cnots": "18",
        },
        {
            "target": "toy",
            "objective_variant": "mixed_pair",
            "materializer": "selected-beam-shared-parity-w4",
            "qasm_depth_ratio": "1.0",
            "primary_nc_depth_ratio": "0.1",
            "num_total_cnots": "1",
        },
    ]

    best = best_policy_beams(rows, "factor_count_pair_cap")

    assert best["toy"]["materializer"] == "selected-beam-shared-parity-w16"


def test_policy_summary_rows_compares_fixed_policies_against_current_beam() -> None:
    grid_rows = [
        {
            "target": "toy",
            "objective_variant": "factor_count",
            "materializer": "selected-beam-shared-parity-w4",
            "tcount": "8",
            "primary_nc_depth_ratio": "0.8",
            "qasm_depth": "80",
            "tdepth": "2",
            "num_total_cnots": "20",
            "qasm_depth_ratio": "2.0",
        },
        {
            "target": "toy",
            "objective_variant": "factor_count_pair_cap",
            "materializer": "selected-beam-shared-parity-w4",
            "tcount": "10",
            "primary_nc_depth_ratio": "0.7",
            "qasm_depth": "70",
            "tdepth": "2",
            "num_total_cnots": "20",
            "qasm_depth_ratio": "1.8",
        },
        {
            "target": "toy",
            "objective_variant": "mixed_pair",
            "materializer": "selected-beam-shared-parity-w4",
            "tcount": "12",
            "primary_nc_depth_ratio": "1.1",
            "qasm_depth": "110",
            "tdepth": "3",
            "num_total_cnots": "30",
            "qasm_depth_ratio": "3.0",
        },
    ]
    current_rows = [
        {
            "target": "toy",
            "materializer": "beam-shared-parity-w4",
            "tcount": "10",
            "primary_nc_depth_ratio": "1.0",
            "qasm_depth": "100",
            "tdepth": "2",
            "num_total_cnots": "20",
            "qasm_depth_ratio": "2.5",
        }
    ]

    rows = {row["policy"]: row for row in policy_summary_rows(grid_rows=grid_rows, current_rows=current_rows)}

    assert rows["factor_count"]["tcount_nonworse"] == 1
    assert rows["factor_count"]["tcount_wins"] == 1
    assert rows["factor_count"]["primary_wins"] == 1
    assert rows["factor_count"]["qasm_wins"] == 1
    assert rows["factor_count_pair_cap"]["joint_nonworse"] == 1
    assert rows["mixed_pair"]["tcount_nonworse"] == 0
