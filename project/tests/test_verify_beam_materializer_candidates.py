from __future__ import annotations

from scripts.verify_beam_materializer_candidates import best_beam_rows
from scripts.verify_beam_materializer_candidates import filter_rows


def test_best_beam_rows_selects_lowest_qasm_then_primary() -> None:
    rows = [
        {
            "target": "toy",
            "materializer": "shared-parity",
            "qasm_depth_ratio": "2.0",
            "primary_nc_depth_ratio": "1.0",
            "num_total_cnots": "20",
        },
        {
            "target": "toy",
            "materializer": "beam-shared-parity-w4",
            "qasm_depth_ratio": "1.8",
            "primary_nc_depth_ratio": "0.9",
            "num_total_cnots": "22",
        },
        {
            "target": "toy",
            "materializer": "beam-shared-parity-w16",
            "qasm_depth_ratio": "1.8",
            "primary_nc_depth_ratio": "0.8",
            "num_total_cnots": "24",
        },
    ]

    best = best_beam_rows(rows)

    assert len(best) == 1
    assert best[0]["materializer"] == "beam-shared-parity-w16"


def test_best_beam_rows_ignores_baseline_rows() -> None:
    rows = [
        {
            "target": "toy",
            "materializer": "shared-parity",
            "qasm_depth_ratio": "1.0",
            "primary_nc_depth_ratio": "1.0",
            "num_total_cnots": "20",
        }
    ]

    assert best_beam_rows(rows) == []


def test_best_beam_rows_accepts_custom_materializer_prefix() -> None:
    rows = [
        {
            "target": "toy",
            "materializer": "beam-shared-parity-w4",
            "qasm_depth_ratio": "0.5",
            "primary_nc_depth_ratio": "0.5",
            "num_total_cnots": "1",
        },
        {
            "target": "toy",
            "materializer": "selected-beam-shared-parity-w4",
            "qasm_depth_ratio": "1.0",
            "primary_nc_depth_ratio": "0.7",
            "num_total_cnots": "10",
        },
        {
            "target": "toy",
            "materializer": "selected-beam-shared-parity-w16",
            "qasm_depth_ratio": "0.9",
            "primary_nc_depth_ratio": "0.8",
            "num_total_cnots": "12",
        },
    ]

    best = best_beam_rows(rows, materializer_prefix="selected-beam-shared-parity")

    assert len(best) == 1
    assert best[0]["materializer"] == "selected-beam-shared-parity-w16"


def test_filter_rows_keeps_requested_objective_variant() -> None:
    rows = [
        {"target": "a", "objective_variant": "factor_count"},
        {"target": "a", "objective_variant": "factor_count_pair_cap"},
        {"target": "a", "objective_variant": "mixed_pair"},
    ]

    filtered = filter_rows(rows, "factor_count_pair_cap")

    assert filtered == [{"target": "a", "objective_variant": "factor_count_pair_cap"}]
    assert filter_rows(rows, None) == rows
