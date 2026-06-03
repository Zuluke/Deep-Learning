from __future__ import annotations

from pathlib import Path

from scripts.analyze_split_reward_sweep_results import collision_groups
from scripts.analyze_split_reward_sweep_results import enrich_rows


def test_enrich_rows_computes_frontier_drop_from_target_weight():
    rows = enrich_rows([
        {
            "target": "hamming_weight_n4",
            "target_tensor_weight": "120",
            "best_return_residual_weight": "140",
            "best_frontier_residual_weight": "111",
            "max_num_moves": "80",
        }
    ])

    assert rows[0]["terminal_residual_drop"] == -20.0
    assert rows[0]["frontier_residual_drop"] == 9.0
    assert rows[0]["normalized_frontier_residual_drop"] == 0.075
    assert rows[0]["has_positive_frontier"]


def test_enrich_rows_does_not_treat_plateau_as_positive_frontier():
    rows = enrich_rows([
        {
            "target": "hamming_weight_n4",
            "target_tensor_weight": "120",
            "best_frontier_residual_weight": "120",
        }
    ])

    assert rows[0]["frontier_residual_drop"] == 0.0
    assert not rows[0]["has_positive_frontier"]


def test_collision_groups_detect_same_candidate_dir_across_horizons():
    rows = enrich_rows([
        {
            "source_csv": str(Path("m40.csv")),
            "target": "hamming_weight_n4",
            "max_num_moves": "40",
            "candidate_output_dir": "/tmp/shared",
        },
        {
            "source_csv": str(Path("m80.csv")),
            "target": "hamming_weight_n4",
            "max_num_moves": "80",
            "candidate_output_dir": "/tmp/shared",
        },
    ])

    groups = collision_groups(rows)

    assert groups == [("/tmp/shared", ["m40.csv:m40", "m80.csv:m80"])]
