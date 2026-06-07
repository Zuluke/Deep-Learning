from __future__ import annotations

from scripts.analyze_alphaq_learned_objective_selector import detail_rows
from scripts.analyze_alphaq_learned_objective_selector import linear_score
from scripts.analyze_alphaq_learned_objective_selector import normalize_rows
from scripts.analyze_alphaq_learned_objective_selector import selected_objective
from scripts.analyze_alphaq_learned_objective_selector import train_weights


def test_normalize_rows_scales_features_within_target() -> None:
    rows = [
        make_decomp("toy", "a", factor_count=2, overlap=4),
        make_decomp("toy", "b", factor_count=6, overlap=8),
    ]

    normalized = normalize_rows(rows, ("factor_count", "factor_pairwise_support_overlap_mean"))

    assert normalized[0]["norm_factor_count"] == 0.0
    assert normalized[1]["norm_factor_count"] == 1.0
    assert normalized[0]["norm_factor_pairwise_support_overlap_mean"] == 0.0
    assert normalized[1]["norm_factor_pairwise_support_overlap_mean"] == 1.0


def test_selected_objective_uses_lowest_linear_score() -> None:
    rows = normalize_rows(
        [
            make_decomp("toy", "a", factor_count=2, overlap=8),
            make_decomp("toy", "b", factor_count=6, overlap=4),
        ],
        ("factor_count", "factor_pairwise_support_overlap_mean"),
    )

    assert selected_objective(rows, "toy", ("factor_count",), (1,)) == "a"
    assert selected_objective(rows, "toy", ("factor_pairwise_support_overlap_mean",), (1,)) == "b"
    assert linear_score(rows[0], ("factor_count",), (1,)) == 0.0


def test_train_weights_prefers_training_equivalence_then_sparsity() -> None:
    rows = normalize_rows(
        [
            make_decomp("train", "a", factor_count=1, overlap=5),
            make_decomp("train", "b", factor_count=2, overlap=1),
            make_decomp("holdout", "a", factor_count=1, overlap=5),
            make_decomp("holdout", "b", factor_count=2, overlap=1),
        ],
        ("factor_count", "factor_pairwise_support_overlap_mean"),
    )
    oracle = {
        "train": {"oracle_objective": "a", "equivalent_objectives": ["a"]},
        "holdout": {"oracle_objective": "b", "equivalent_objectives": ["b"]},
    }

    weights = train_weights(
        rows=rows,
        targets=["train"],
        features=("factor_count", "factor_pairwise_support_overlap_mean"),
        levels=(0, 1),
        oracle=oracle,
    )

    assert weights == (1, 0)


def test_detail_rows_report_loto_learned_selector_metrics() -> None:
    decomp_rows = [
        make_decomp("a", "factor_count", factor_count=1, overlap=4),
        make_decomp("a", "factor_count_pair_cap", factor_count=2, overlap=1),
        make_decomp("b", "factor_count", factor_count=1, overlap=4),
        make_decomp("b", "factor_count_pair_cap", factor_count=2, overlap=1),
    ]
    grid_rows = [
        make_grid("a", "factor_count", tcount=1, primary=1, qasm=10),
        make_grid("a", "factor_count_pair_cap", tcount=2, primary=2, qasm=20),
        make_grid("b", "factor_count", tcount=1, primary=1, qasm=10),
        make_grid("b", "factor_count_pair_cap", tcount=2, primary=2, qasm=20),
    ]
    current_rows = [make_current("a"), make_current("b")]

    rows = detail_rows(decomp_rows=decomp_rows, grid_rows=grid_rows, current_rows=current_rows)
    alphaq_rows = [row for row in rows if row["selector"] == "loto_linear_alphaq_nonnegative"]

    assert len(alphaq_rows) == 2
    assert all(row["selected_objective"] == "factor_count" for row in alphaq_rows)
    assert all(row["oracle_equivalent"] for row in alphaq_rows)


def make_decomp(target: str, objective: str, *, factor_count: int, overlap: int) -> dict[str, str]:
    return {
        "target": target,
        "objective_variant": objective,
        "factor_count": str(factor_count),
        "factor_qubit_concentration_index": str(overlap),
        "factor_support_weight_mean": str(overlap),
        "factor_pairwise_support_overlap_mean": str(overlap),
        "factor_pairwise_jaccard_mean": str(overlap),
        "tdepth": str(factor_count),
        "qasm_depth_ratio": str(factor_count),
    }


def make_grid(target: str, objective: str, *, tcount: int, primary: int, qasm: int) -> dict[str, str]:
    return {
        "target": target,
        "objective_variant": objective,
        "materializer": "selected-beam-shared-parity-w4",
        "tcount": str(tcount),
        "primary_nc_depth_ratio": str(primary),
        "qasm_depth": str(qasm),
        "num_total_cnots": "1",
    }


def make_current(target: str) -> dict[str, str]:
    return {
        "target": target,
        "materializer": "beam-shared-parity-w4",
        "tcount": "2",
        "primary_nc_depth_ratio": "2",
        "qasm_depth": "20",
        "num_total_cnots": "1",
    }
