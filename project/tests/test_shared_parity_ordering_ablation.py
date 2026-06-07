from __future__ import annotations

from scripts.run_shared_parity_ordering_ablation import beats_fraction
from scripts.run_shared_parity_ordering_ablation import expand_orders
from scripts.run_shared_parity_ordering_ablation import materialization_grid
from scripts.run_shared_parity_ordering_ablation import percentile
from scripts.run_shared_parity_ordering_ablation import summarize_target_rows


def test_expand_orders_appends_deterministic_random_labels() -> None:
    assert expand_orders(
        ["given", "random-seed-3"],
        random_orders=3,
        random_seed_start=2,
    ) == ["given", "random-seed-3", "random-seed-2", "random-seed-4"]


def test_materialization_grid_crosses_targets_orders_and_strategies() -> None:
    grid = materialization_grid(
        targets=["a", "b"],
        orders=["given", "reverse"],
        target_strategies=["min-change"],
    )

    assert grid == [
        ("a", "given", "min-change"),
        ("a", "reverse", "min-change"),
        ("b", "given", "min-change"),
        ("b", "reverse", "min-change"),
    ]


def test_summarize_target_rows_reports_selected_rank_and_medians() -> None:
    rows = [
        {
            "target": "toy",
            "factor_order": "given",
            "target_strategy": "min-change",
            "checkpoint_selected": True,
            "qasm_depth_ratio": 3.0,
            "primary_nc_depth_ratio": 0.6,
        },
        {
            "target": "toy",
            "factor_order": "reverse",
            "target_strategy": "min-change",
            "checkpoint_selected": False,
            "qasm_depth_ratio": 2.0,
            "primary_nc_depth_ratio": 0.5,
        },
        {
            "target": "toy",
            "factor_order": "lex",
            "target_strategy": "min-change",
            "checkpoint_selected": False,
            "qasm_depth_ratio": 4.0,
            "primary_nc_depth_ratio": 0.7,
        },
    ]

    summary = summarize_target_rows(rows)[0]

    assert summary["median_qasm_depth_ratio"] == 3.0
    assert summary["best_qasm_depth_ratio"] == 2.0
    assert summary["selected_qasm_rank"] == 2
    assert summary["selected_primary_rank"] == 2


def test_percentile_and_beats_fraction_are_lower_is_better() -> None:
    assert percentile([1.0, 3.0, 5.0], 25) == 2.0
    assert percentile([1.0, 3.0, 5.0], 75) == 4.0
    assert beats_fraction(3.0, [2.0, 3.0, 4.0, 5.0]) == 0.75
