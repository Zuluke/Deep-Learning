from __future__ import annotations

from scripts.run_beam_materializer_ablation import best_pairs
from scripts.run_beam_materializer_ablation import materializers
from scripts.run_beam_materializer_ablation import paired_rows


def test_materializers_include_baseline_and_requested_beams() -> None:
    assert materializers([4, 16]) == [
        ("shared-parity", None),
        ("beam-shared-parity", 4),
        ("beam-shared-parity", 16),
    ]


def test_paired_rows_compare_each_beam_against_shared() -> None:
    rows = [
        {
            "target": "toy",
            "materializer": "shared-parity",
            "primary_nc_depth_ratio": 1.0,
            "qasm_depth_ratio": 2.0,
            "tdepth": 4,
            "num_total_cnots": 20,
        },
        {
            "target": "toy",
            "materializer": "beam-shared-parity-w4",
            "primary_nc_depth_ratio": 1.1,
            "qasm_depth_ratio": 1.8,
            "tdepth": 4,
            "num_total_cnots": 18,
        },
        {
            "target": "toy",
            "materializer": "beam-shared-parity-w16",
            "primary_nc_depth_ratio": 0.9,
            "qasm_depth_ratio": 2.1,
            "tdepth": 5,
            "num_total_cnots": 16,
        },
    ]

    pairs = paired_rows(rows)

    assert len(pairs) == 2
    assert pairs[0]["materializer"] == "beam-shared-parity-w16"
    assert pairs[1]["materializer"] == "beam-shared-parity-w4"
    assert pairs[1]["qasm_ratio"] == 0.9


def test_best_pairs_select_lowest_qasm_then_primary() -> None:
    rows = [
        {
            "target": "toy",
            "materializer": "shared-parity",
            "primary_nc_depth_ratio": 1.0,
            "qasm_depth_ratio": 2.0,
            "tdepth": 4,
            "num_total_cnots": 20,
        },
        {
            "target": "toy",
            "materializer": "beam-shared-parity-w4",
            "primary_nc_depth_ratio": 0.8,
            "qasm_depth_ratio": 1.9,
            "tdepth": 4,
            "num_total_cnots": 18,
        },
        {
            "target": "toy",
            "materializer": "beam-shared-parity-w16",
            "primary_nc_depth_ratio": 0.7,
            "qasm_depth_ratio": 1.9,
            "tdepth": 4,
            "num_total_cnots": 18,
        },
    ]

    best = best_pairs(rows)[0]

    assert best["materializer"] == "beam-shared-parity-w16"
    assert best["qasm_ratio"] == 0.95
