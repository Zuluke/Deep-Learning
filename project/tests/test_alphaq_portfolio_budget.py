from __future__ import annotations

import csv
import random
import subprocess
import sys
from pathlib import Path

from scripts.analyze_alphaq_portfolio_budget import (
    BASELINE_OBJECTIVE,
    PORTFOLIO_OBJECTIVES,
    bootstrap_median_ci,
    dedupe_groups,
    evaluate_scope,
    materialized_set,
    sign_test_one_sided,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def synthetic_rows() -> list[dict[str, str]]:
    """Six targets where high jaccard means a worse final T-count.

    The oracle objective alternates so the selector has a learnable signal:
    candidates with the lowest pairwise jaccard always have the lowest
    materialized T-count.
    """
    rows = []
    base_t = {"a": 40, "b": 50, "c": 60, "d": 44, "e": 52, "f": 48}
    for index, target in enumerate(sorted(base_t)):
        for position, objective in enumerate(PORTFOLIO_OBJECTIVES):
            # Rotate which objective is best per target, but keep the feature
            # correlation identical: lower jaccard <-> lower T-count.
            offset = (position - index) % len(PORTFOLIO_OBJECTIVES)
            tcount = base_t[target] + 3 * offset
            rows.append(
                {
                    "source_split": "external_synthetic",
                    "target": target,
                    "objective_variant": objective,
                    "execution_status": "ok",
                    "has_beam_candidate": "True",
                    "train_ready": "True",
                    "factor_count": str(10 + offset),
                    "factor_qubit_concentration_index": "0.2",
                    "factor_support_weight_mean": "3.0",
                    "factor_pairwise_support_overlap_mean": "1.0",
                    "factor_pairwise_jaccard_mean": str(0.1 + 0.2 * offset),
                    "best_beam_tcount": str(tcount),
                    "best_beam_qasm_depth": str(100 + offset),
                    "best_beam_primary_nc_depth_ratio": "0.5",
                }
            )
    return rows


def test_sign_test_matches_exact_binomial() -> None:
    assert sign_test_one_sided(0, 0) is None
    assert sign_test_one_sided(1, 0) == 0.5
    assert abs(sign_test_one_sided(5, 0) - 0.03125) < 1e-12
    assert abs(sign_test_one_sided(3, 1) - 0.3125) < 1e-12


def test_bootstrap_median_ci_is_deterministic_and_bounded() -> None:
    rng = random.Random(7)
    ci = bootstrap_median_ci([0.8, 0.9, 1.0, 1.0, 1.0], rng, 500)

    assert ci is not None
    low, high = ci
    assert 0.8 <= low <= high <= 1.0


def test_guarded_budget_always_contains_baseline() -> None:
    ranking = [
        {"objective_variant": "mixed_pair"},
        {"objective_variant": "frontier_pair"},
        {"objective_variant": BASELINE_OBJECTIVE},
        {"objective_variant": "factor_count_pair_cap"},
    ]
    for budget in (1, 2, 3, 4):
        chosen = materialized_set(ranking, budget, guarded=True)
        assert len(chosen) == budget
        assert any(row["objective_variant"] == BASELINE_OBJECTIVE for row in chosen)


def test_full_budget_recovers_oracle_and_guard_never_loses() -> None:
    rows = synthetic_rows()
    rng = random.Random(3)
    details, summaries = evaluate_scope(
        "groups",
        rows,
        rng,
        bootstrap_samples=200,
        permutation_samples=50,
    )

    full = next(row for row in summaries if row["policy"] == f"top{len(PORTFOLIO_OBJECTIVES)}")
    assert full["oracle_t_recovered"] == full["groups"]

    for budget in (2, 3, 4):
        guarded = next(row for row in summaries if row["policy"] == f"guarded_top{budget}")
        assert guarded["tcount_losses_vs_baseline"] == 0

    guarded_detail = [row for row in details if row["policy"] == "guarded_top2"]
    assert all(
        BASELINE_OBJECTIVE in row["materialized_objectives"].split(",")
        for row in guarded_detail
    )


def test_dedupe_groups_prefers_complete_external_groups() -> None:
    rows = synthetic_rows()
    duplicate = [
        {**row, "source_split": "internal"}
        for row in rows
        if row["target"] == "a" and row["objective_variant"] != "frontier_pair"
    ]
    deduped = dedupe_groups(rows + duplicate)
    target_a = [row for row in deduped if row["target"] == "a"]

    assert len(target_a) == len(PORTFOLIO_OBJECTIVES)
    assert {row["source_split"] for row in target_a} == {"external_synthetic"}


def test_cli_runs_on_synthetic_dataset(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.csv"
    rows = synthetic_rows()
    with dataset.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = tmp_path / "summary.csv"
    detail = tmp_path / "detail.csv"
    report = tmp_path / "report.md"
    figure = tmp_path / "figure.png"

    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "analyze_alphaq_portfolio_budget.py"),
            "--dataset-csv",
            str(dataset),
            "--summary-csv",
            str(summary),
            "--detail-csv",
            str(detail),
            "--report-path",
            str(report),
            "--figure-path",
            str(figure),
            "--bootstrap-samples",
            "100",
            "--permutation-samples",
            "20",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=PROJECT_ROOT,
    )

    assert completed.returncode == 0, completed.stderr
    assert summary.exists()
    assert detail.exists()
    assert report.exists()
    assert figure.exists()
    with summary.open(encoding="utf-8", newline="") as handle:
        policies = {row["policy"] for row in csv.DictReader(handle)}
    assert "guarded_top2" in policies
    assert "oracle_full_portfolio" in policies
