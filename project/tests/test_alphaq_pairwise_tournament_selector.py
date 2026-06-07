from __future__ import annotations

from scripts.analyze_alphaq_learned_objective_selector import normalize_rows
from scripts.analyze_alphaq_pairwise_tournament_selector import detail_rows
from scripts.analyze_alphaq_pairwise_tournament_selector import select_tournament_objective


def test_select_tournament_objective_uses_pairwise_wins() -> None:
    rows = normalize_rows(
        [
            make_decomp("toy", "factor_count", factor_count=1, overlap=3),
            make_decomp("toy", "factor_count_pair_cap", factor_count=2, overlap=2),
            make_decomp("toy", "mixed_pair", factor_count=3, overlap=1),
        ],
        ("factor_count", "factor_pairwise_support_overlap_mean"),
    )

    selected, wins, _ = select_tournament_objective(
        target_rows=rows,
        features=("factor_count",),
        weights=(1,),
    )

    assert selected == "factor_count"
    assert wins["factor_count"] == 2


def test_select_tournament_objective_uses_stable_factor_count_tie_break() -> None:
    rows = normalize_rows(
        [
            make_decomp("toy", "factor_count", factor_count=2, overlap=1),
            make_decomp("toy", "factor_count_pair_cap", factor_count=1, overlap=1),
        ],
        ("factor_count", "factor_pairwise_support_overlap_mean"),
    )

    selected, _, _ = select_tournament_objective(
        target_rows=rows,
        features=("factor_pairwise_support_overlap_mean",),
        weights=(1,),
    )

    assert selected == "factor_count_pair_cap"


def test_detail_rows_reports_pairwise_tournament_selection() -> None:
    decomp_rows = [
        make_decomp("a", "factor_count", factor_count=1, overlap=3),
        make_decomp("a", "factor_count_pair_cap", factor_count=2, overlap=2),
        make_decomp("b", "factor_count", factor_count=1, overlap=3),
        make_decomp("b", "factor_count_pair_cap", factor_count=2, overlap=2),
    ]
    grid_rows = [
        make_grid("a", "factor_count", tcount=1, primary=1, qasm=1),
        make_grid("a", "factor_count_pair_cap", tcount=2, primary=2, qasm=2),
        make_grid("b", "factor_count", tcount=1, primary=1, qasm=1),
        make_grid("b", "factor_count_pair_cap", tcount=2, primary=2, qasm=2),
    ]
    current_rows = [make_current("a"), make_current("b")]

    rows = detail_rows(decomp_rows=decomp_rows, grid_rows=grid_rows, current_rows=current_rows)
    tournament_rows = [row for row in rows if row["selector"] == "loto_tournament_single_alphaq"]

    assert len(tournament_rows) == 2
    assert all(row["selected_objective"] == "factor_count" for row in tournament_rows)
    assert all(row["oracle_equivalent"] for row in tournament_rows)


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
        "qasm_depth": "2",
        "num_total_cnots": "1",
    }
