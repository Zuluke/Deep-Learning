from __future__ import annotations

from scripts.analyze_alphaq_budget_regret import summarize_details, tcount_regret
from scripts.analyze_alphaq_clustered_bootstrap import (
    bootstrap_cluster_metrics,
    geometric_mean,
    metric_values,
)


def test_tcount_regret_and_summary() -> None:
    details = [
        {
            "final_tcount": "8",
            "oracle_tcount": "7",
            "baseline_tcount": "10",
            "oracle_t_recovered": False,
        },
        {
            "final_tcount": "12",
            "oracle_tcount": "12",
            "baseline_tcount": "12",
            "oracle_t_recovered": True,
        },
    ]

    assert tcount_regret(details[0]) == 1
    summary = summarize_details("LOTO", "groups", "guarded_top2", 2, details)

    assert summary["groups"] == 2
    assert summary["oracle_t_recovered"] == 1
    assert summary["misses"] == 1
    assert summary["max_regret"] == 1
    assert summary["wins_vs_baseline"] == 1
    assert summary["losses_vs_baseline"] == 0


def test_clustered_bootstrap_over_targets_is_deterministic() -> None:
    rows = [
        {
            "target": "a",
            "tcount_ratio_vs_baseline": "0.8",
            "tcount_regression_vs_baseline": "-2",
        },
        {
            "target": "a",
            "tcount_ratio_vs_baseline": "1.0",
            "tcount_regression_vs_baseline": "0",
        },
        {
            "target": "b",
            "tcount_ratio_vs_baseline": "1.1",
            "tcount_regression_vs_baseline": "1",
        },
    ]

    point = metric_values(rows)
    intervals_a = bootstrap_cluster_metrics(rows, samples=200, seed=11)
    intervals_b = bootstrap_cluster_metrics(rows, samples=200, seed=11)

    assert abs(point["geomean_t_ratio"] - geometric_mean([0.8, 1.0, 1.1])) < 1e-12
    assert point["win_count"] == 1
    assert point["max_regression"] == 1
    assert intervals_a == intervals_b
    assert intervals_a["win_count"]["ci_low"] <= point["win_count"] <= intervals_a["win_count"]["ci_high"]
