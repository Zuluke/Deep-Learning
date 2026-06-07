from __future__ import annotations

from scripts.run_best_objective_beam_ablation import comparison_rows
from scripts.run_best_objective_beam_ablation import objective_selection_key
from scripts.run_best_objective_beam_ablation import selected_objective_rows


def test_selected_objective_rows_prioritizes_tcount_then_primary_then_qasm() -> None:
    rows = [
        {
            "target": "toy",
            "objective_variant": "mixed_pair",
            "tcount": "10",
            "primary_nc_depth_ratio": "0.4",
            "qasm_depth_ratio": "2.0",
        },
        {
            "target": "toy",
            "objective_variant": "factor_count_pair_cap",
            "tcount": "9",
            "primary_nc_depth_ratio": "0.8",
            "qasm_depth_ratio": "1.0",
        },
        {
            "target": "toy",
            "objective_variant": "factor_count",
            "tcount": "9",
            "primary_nc_depth_ratio": "0.7",
            "qasm_depth_ratio": "3.0",
        },
    ]

    selected = selected_objective_rows(rows, targets=["toy"])

    assert objective_selection_key(selected[0]) == (9.0, 0.7, 3.0, "factor_count")
    assert selected[0]["objective_variant"] == "factor_count"


def test_comparison_rows_compares_selected_best_beam_against_current_beam() -> None:
    selected_rows = [
        {
            "target": "toy",
            "materializer": "selected-shared-parity",
            "objective_variant": "factor_count",
            "qasm_depth": "100",
            "primary_nc_depth_ratio": "0.8",
        },
        {
            "target": "toy",
            "materializer": "selected-beam-shared-parity-w4",
            "objective_variant": "factor_count",
            "tcount": "8",
            "tdepth": "2",
            "qasm_depth": "80",
            "qasm_depth_ratio": "2.0",
            "num_total_cnots": "20",
            "primary_nc_depth_ratio": "0.7",
        },
        {
            "target": "toy",
            "materializer": "selected-beam-shared-parity-w16",
            "objective_variant": "factor_count",
            "tcount": "8",
            "tdepth": "3",
            "qasm_depth": "90",
            "qasm_depth_ratio": "2.2",
            "num_total_cnots": "18",
            "primary_nc_depth_ratio": "0.6",
        },
    ]
    current_rows = [
        {
            "target": "toy",
            "materializer": "beam-shared-parity-w4",
            "tcount": "10",
            "tdepth": "4",
            "qasm_depth": "160",
            "num_total_cnots": "40",
            "primary_nc_depth_ratio": "1.4",
            "qasm_depth_ratio": "4.0",
        }
    ]

    rows = comparison_rows(selected_rows=selected_rows, current_rows=current_rows)

    assert len(rows) == 1
    row = rows[0]
    assert row["selected_objective"] == "factor_count"
    assert row["selected_vs_current_tcount_ratio"] == 0.8
    assert row["selected_vs_current_qasm_depth_ratio"] == 0.5
    assert row["selected_vs_current_primary_nc_ratio"] == 0.5
    assert row["selected_vs_selected_shared_qasm_depth_ratio"] == 0.8
