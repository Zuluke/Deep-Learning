from __future__ import annotations

import pytest

from scripts.run_decomposition_objective_ablation import OBJECTIVE_VARIANTS
from scripts.run_decomposition_objective_ablation import parse_objective_variants


def test_parse_objective_variants_filters_requested_variants() -> None:
    variants = parse_objective_variants("factor_count_pair_cap,mixed_pair")

    assert [variant.name for variant in variants] == ["factor_count_pair_cap", "mixed_pair"]


def test_parse_objective_variants_rejects_unknown_variant() -> None:
    with pytest.raises(ValueError, match="Unknown objective"):
        parse_objective_variants("not_a_real_objective")


def test_parse_objective_variants_default_names_cover_all_variants() -> None:
    variants = parse_objective_variants(",".join(variant.name for variant in OBJECTIVE_VARIANTS))

    assert variants == OBJECTIVE_VARIANTS

import scripts.run_decomposition_objective_ablation as ablation
from scripts.run_decomposition_objective_ablation import compare_rows
from scripts.run_decomposition_objective_ablation import collect_rows
from scripts.run_decomposition_objective_ablation import pairwise_summary


def test_collect_rows_can_record_objective_failures_without_fallback(monkeypatch, tmp_path) -> None:
    def fail_optimization(**_kwargs):
        raise RuntimeError("MILP timeout")

    monkeypatch.setattr(ablation, "run_optimization", fail_optimization)

    rows = collect_rows(
        ["mod_5_4"],
        tmp_path,
        time_limit_sec=1.0,
        force=False,
        continue_on_error=True,
    )

    assert len(rows) == 3
    assert {row["execution_status"] for row in rows} == {"failed"}
    assert all(row["summary_path"] == "" for row in rows)
    assert all("timeout" in row["error_message"] for row in rows)


def test_collect_rows_writes_incremental_checkpoint(monkeypatch, tmp_path) -> None:
    def fail_optimization(**_kwargs):
        raise RuntimeError("MILP timeout")

    monkeypatch.setattr(ablation, "run_optimization", fail_optimization)
    checkpoint_csv = tmp_path / "checkpoint.csv"

    collect_rows(
        ["mod_5_4"],
        tmp_path,
        time_limit_sec=1.0,
        force=False,
        objective_variants=OBJECTIVE_VARIANTS[:1],
        continue_on_error=True,
        checkpoint_csv=checkpoint_csv,
    )

    text = checkpoint_csv.read_text(encoding="utf-8")
    assert "mod_5_4" in text
    assert "factor_count" in text
    assert "failed" in text
    assert "MILP timeout" in text


def test_collect_rows_raises_by_default_on_objective_failure(monkeypatch, tmp_path) -> None:
    def fail_optimization(**_kwargs):
        raise RuntimeError("MILP timeout")

    monkeypatch.setattr(ablation, "run_optimization", fail_optimization)

    with pytest.raises(RuntimeError, match="timeout"):
        collect_rows(["mod_5_4"], tmp_path, time_limit_sec=1.0, force=False)


def test_compare_rows_reports_mixed_vs_baseline_ratios() -> None:
    mixed = {
        "factor_count": 8,
        "factor_parity_reuse_ratio": 0.25,
        "factor_parity_concentration_index": 0.2,
        "primary_nc_depth_ratio": 0.4,
        "qasm_depth_ratio": 2.0,
        "tdepth": 3,
        "tcount": 8,
    }
    baseline = {
        "factor_count": 10,
        "factor_parity_reuse_ratio": 0.1,
        "factor_parity_concentration_index": 0.1,
        "primary_nc_depth_ratio": 0.8,
        "qasm_depth_ratio": 4.0,
        "tdepth": 6,
        "tcount": 10,
    }

    row = compare_rows("toy", "mixed_vs_baseline", mixed, baseline)

    assert row["factor_count_ratio"] == 0.8
    assert row["parity_reuse_delta"] == 0.15
    assert row["parity_ci_ratio"] == 2.0
    assert row["primary_ratio"] == 0.5
    assert row["qasm_depth_ratio"] == 0.5
    assert row["tdepth_ratio"] == 0.5


def test_pairwise_summary_includes_unmatched_and_matched_comparisons() -> None:
    rows = [
        {"target": "mod_5_4", "objective_variant": "factor_count", "factor_count": 10},
        {
            "target": "mod_5_4",
            "objective_variant": "factor_count_pair_cap",
            "factor_count": 9,
        },
        {"target": "mod_5_4", "objective_variant": "mixed_pair", "factor_count": 8},
    ]

    comparisons = [row["comparison"] for row in pairwise_summary(rows)]

    assert comparisons == [
        "mixed_pair_vs_factor_count",
        "mixed_pair_vs_factor_count_pair_cap",
    ]
