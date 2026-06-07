from __future__ import annotations

from scripts.analyze_alphaq_learned_objective_selector import normalize_rows
from scripts.analyze_alphaq_pairwise_selector_separability import clear_pair_rows
from scripts.analyze_alphaq_pairwise_selector_separability import detail_rows
from scripts.analyze_alphaq_pairwise_selector_separability import pair_correct
from scripts.analyze_alphaq_pairwise_selector_separability import train_weights


def test_clear_pair_rows_excludes_metric_equivalent_objective_ties() -> None:
    normalized = normalize_rows(
        [
            make_decomp("tie", "factor_count", factor_count=1, overlap=2),
            make_decomp("tie", "factor_count_pair_cap", factor_count=1, overlap=1),
            make_decomp("clear", "factor_count", factor_count=1, overlap=2),
            make_decomp("clear", "factor_count_pair_cap", factor_count=2, overlap=1),
        ],
        ("factor_count", "factor_pairwise_support_overlap_mean"),
    )
    grid_rows = [
        make_grid("tie", "factor_count", tcount=1, primary=1, qasm=1),
        make_grid("tie", "factor_count_pair_cap", tcount=1, primary=1, qasm=1),
        make_grid("clear", "factor_count", tcount=1, primary=1, qasm=1),
        make_grid("clear", "factor_count_pair_cap", tcount=2, primary=2, qasm=2),
    ]

    pairs = clear_pair_rows(
        normalized_rows=normalized,
        grid_rows=grid_rows,
        features=("factor_count", "factor_pairwise_support_overlap_mean"),
    )

    assert len(pairs) == 1
    assert pairs[0]["target"] == "clear"
    assert pairs[0]["preferred_objective"] == "factor_count"
    assert pairs[0]["rejected_objective"] == "factor_count_pair_cap"


def test_pairwise_training_selects_feature_direction_from_training_pairs() -> None:
    pairs = [
        {
            "target": "train",
            "delta_factor_count": -1.0,
            "delta_factor_pairwise_support_overlap_mean": 1.0,
        },
        {
            "target": "holdout",
            "delta_factor_count": -1.0,
            "delta_factor_pairwise_support_overlap_mean": 1.0,
        },
    ]
    profile = {
        "features": ("factor_count", "factor_pairwise_support_overlap_mean"),
        "kind": "single",
        "levels": (-1, 1),
    }

    weights = train_weights(pairs=pairs, train_targets=["train"], profile=profile)

    assert pair_correct(pairs[1], profile["features"], weights)


def test_detail_rows_reports_loto_pairwise_results() -> None:
    decomp_rows = [
        make_decomp("a", "factor_count", factor_count=1, overlap=3),
        make_decomp("a", "factor_count_pair_cap", factor_count=2, overlap=1),
        make_decomp("b", "factor_count", factor_count=1, overlap=3),
        make_decomp("b", "factor_count_pair_cap", factor_count=2, overlap=1),
    ]
    grid_rows = [
        make_grid("a", "factor_count", tcount=1, primary=1, qasm=1),
        make_grid("a", "factor_count_pair_cap", tcount=2, primary=2, qasm=2),
        make_grid("b", "factor_count", tcount=1, primary=1, qasm=1),
        make_grid("b", "factor_count_pair_cap", tcount=2, primary=2, qasm=2),
    ]

    rows = detail_rows(decomp_rows, grid_rows)
    single_rows = [row for row in rows if row["selector"] == "loto_pairwise_single_alphaq"]

    assert len(single_rows) == 2
    assert all(row["pair_correct"] for row in single_rows)


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
