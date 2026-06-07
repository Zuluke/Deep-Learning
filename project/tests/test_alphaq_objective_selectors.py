from __future__ import annotations

from scripts.analyze_alphaq_objective_selectors import beam_oracle_objective_by_target
from scripts.analyze_alphaq_objective_selectors import detail_rows
from scripts.analyze_alphaq_objective_selectors import selected_rows_by_policy
from scripts.analyze_alphaq_objective_selectors import selector_specs
from scripts.analyze_alphaq_objective_selectors import summary_rows


def test_beam_oracle_objective_uses_post_materialization_t_primary_qasm_order() -> None:
    grid_rows = [
        {
            "target": "toy",
            "objective_variant": "mixed_pair",
            "materializer": "selected-beam-shared-parity-w4",
            "tcount": "10",
            "primary_nc_depth_ratio": "0.2",
            "qasm_depth": "2",
        },
        {
            "target": "toy",
            "objective_variant": "factor_count_pair_cap",
            "materializer": "selected-beam-shared-parity-w4",
            "tcount": "9",
            "primary_nc_depth_ratio": "0.8",
            "qasm_depth": "8",
        },
    ]

    assert beam_oracle_objective_by_target(grid_rows)["toy"] == "factor_count_pair_cap"


def test_min_factor_count_then_overlap_selector_prefers_lower_overlap_on_tie() -> None:
    rows = [
        {
            "target": "toy",
            "objective_variant": "factor_count",
            "factor_count": "8",
            "factor_pairwise_support_overlap_mean": "2.0",
        },
        {
            "target": "toy",
            "objective_variant": "factor_count_pair_cap",
            "factor_count": "8",
            "factor_pairwise_support_overlap_mean": "1.0",
        },
    ]

    selected = selected_rows_by_policy(
        rows,
        "min_factor_count_then_overlap",
        selector_specs()["min_factor_count_then_overlap"],
    )

    assert selected[0]["objective_variant"] == "factor_count_pair_cap"


def test_detail_and_summary_rows_compare_selector_to_current_beam() -> None:
    decomp_rows = [
        {
            "target": "toy",
            "objective_variant": "factor_count",
            "factor_count": "8",
            "factor_pairwise_support_overlap_mean": "1.0",
            "factor_pairwise_jaccard_mean": "1.0",
            "factor_qubit_concentration_index": "1.0",
            "factor_support_weight_mean": "1.0",
            "tcount": "8",
            "primary_nc_depth_ratio": "0.7",
            "qasm_depth_ratio": "0.7",
        },
        {
            "target": "toy",
            "objective_variant": "mixed_pair",
            "factor_count": "10",
            "factor_pairwise_support_overlap_mean": "0.5",
            "factor_pairwise_jaccard_mean": "0.5",
            "factor_qubit_concentration_index": "0.5",
            "factor_support_weight_mean": "0.5",
            "tcount": "10",
            "primary_nc_depth_ratio": "0.2",
            "qasm_depth_ratio": "0.2",
        },
    ]
    grid_rows = [
        {
            "target": "toy",
            "objective_variant": "factor_count",
            "materializer": "selected-beam-shared-parity-w4",
            "tcount": "8",
            "primary_nc_depth_ratio": "0.8",
            "qasm_depth": "80",
            "qasm_depth_ratio": "2.0",
            "num_total_cnots": "10",
        },
        {
            "target": "toy",
            "objective_variant": "mixed_pair",
            "materializer": "selected-beam-shared-parity-w4",
            "tcount": "10",
            "primary_nc_depth_ratio": "1.1",
            "qasm_depth": "110",
            "qasm_depth_ratio": "3.0",
            "num_total_cnots": "12",
        },
    ]
    current_rows = [
        {
            "target": "toy",
            "materializer": "beam-shared-parity-w4",
            "tcount": "10",
            "primary_nc_depth_ratio": "1.0",
            "qasm_depth": "100",
            "qasm_depth_ratio": "2.5",
            "num_total_cnots": "10",
        }
    ]

    details = detail_rows(decomp_rows=decomp_rows, grid_rows=grid_rows, current_rows=current_rows)
    summaries = {row["selector"]: row for row in summary_rows(details)}

    assert summaries["min_factor_count"]["oracle_matches"] == 1
    assert summaries["min_factor_count"]["tcount_wins"] == 1
    assert summaries["min_overlap_then_factor_count"]["oracle_matches"] == 0


def test_detail_rows_use_beam_oracle_not_decomposition_oracle() -> None:
    decomp_rows = [
        {
            "target": "toy",
            "objective_variant": "factor_count",
            "factor_count": "8",
            "factor_pairwise_support_overlap_mean": "1.0",
            "factor_pairwise_jaccard_mean": "1.0",
            "factor_qubit_concentration_index": "1.0",
            "factor_support_weight_mean": "1.0",
            "tcount": "8",
            "primary_nc_depth_ratio": "0.7",
            "qasm_depth_ratio": "0.7",
        },
        {
            "target": "toy",
            "objective_variant": "mixed_pair",
            "factor_count": "9",
            "factor_pairwise_support_overlap_mean": "0.5",
            "factor_pairwise_jaccard_mean": "0.5",
            "factor_qubit_concentration_index": "0.5",
            "factor_support_weight_mean": "0.5",
            "tcount": "9",
            "primary_nc_depth_ratio": "0.9",
            "qasm_depth_ratio": "0.9",
        },
    ]
    grid_rows = [
        {
            "target": "toy",
            "objective_variant": "factor_count",
            "materializer": "selected-beam-shared-parity-w4",
            "tcount": "9",
            "primary_nc_depth_ratio": "0.9",
            "qasm_depth": "90",
            "qasm_depth_ratio": "9",
            "num_total_cnots": "9",
        },
        {
            "target": "toy",
            "objective_variant": "mixed_pair",
            "materializer": "selected-beam-shared-parity-w4",
            "tcount": "7",
            "primary_nc_depth_ratio": "1.1",
            "qasm_depth": "110",
            "qasm_depth_ratio": "11",
            "num_total_cnots": "11",
        },
    ]
    current_rows = [
        {
            "target": "toy",
            "materializer": "beam-shared-parity-w4",
            "tcount": "10",
            "primary_nc_depth_ratio": "1.0",
            "qasm_depth": "100",
            "qasm_depth_ratio": "10",
            "num_total_cnots": "10",
        }
    ]

    details = detail_rows(decomp_rows=decomp_rows, grid_rows=grid_rows, current_rows=current_rows)
    by_selector = {row["selector"]: row for row in details}

    assert by_selector["min_factor_count"]["oracle_objective"] == "mixed_pair"
    assert by_selector["min_factor_count"]["matches_oracle"] is False
