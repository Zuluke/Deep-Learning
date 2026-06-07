from __future__ import annotations

from scripts.analyze_alphaq_oracle_equivalence import equivalence_detail_rows
from scripts.analyze_alphaq_oracle_equivalence import oracle_info
from scripts.analyze_alphaq_oracle_equivalence import summary_rows


def test_oracle_info_marks_metric_equivalent_objectives() -> None:
    grid_rows = [
        make_grid("toy", "factor_count", tcount=1, primary=2, qasm=3),
        make_grid("toy", "factor_count_pair_cap", tcount=1, primary=2, qasm=3),
        make_grid("toy", "mixed_pair", tcount=2, primary=1, qasm=1),
    ]

    info = oracle_info(grid_rows)

    assert info["toy"]["oracle_objective"] == "factor_count"
    assert info["toy"]["equivalent_objectives"] == ["factor_count", "factor_count_pair_cap"]


def test_equivalence_detail_recovers_apparent_oracle_miss() -> None:
    grid_rows = [
        make_grid("toy", "factor_count", tcount=1, primary=2, qasm=3),
        make_grid("toy", "factor_count_pair_cap", tcount=1, primary=2, qasm=3),
        make_grid("toy", "mixed_pair", tcount=2, primary=1, qasm=1),
    ]
    detail_rows = [
        {
            "selector": "min_factor_count",
            "target": "toy",
            "selected_objective": "factor_count_pair_cap",
            "tcount_ratio": "1",
            "primary_ratio": "1",
            "qasm_ratio": "1",
        }
    ]

    details = equivalence_detail_rows(grid_rows=grid_rows, selector_detail_rows=detail_rows)

    assert details[0]["exact_oracle_match"] is False
    assert details[0]["oracle_equivalent"] is True
    assert details[0]["recovered_by_equivalence"] is True


def test_summary_rows_counts_equivalent_matches_separately() -> None:
    rows = [
        {
            "selector": "s",
            "target": "a",
            "exact_oracle_match": False,
            "oracle_equivalent": True,
            "recovered_by_equivalence": True,
            "tcount_ratio": "1",
            "primary_ratio": "0.9",
            "qasm_ratio": "1.1",
            "selected_objective": "x",
            "equivalent_objectives": "x,y",
        },
        {
            "selector": "s",
            "target": "b",
            "exact_oracle_match": True,
            "oracle_equivalent": True,
            "recovered_by_equivalence": False,
            "tcount_ratio": "0.8",
            "primary_ratio": "0.9",
            "qasm_ratio": "0.9",
            "selected_objective": "x",
            "equivalent_objectives": "x",
        },
    ]

    summary = summary_rows(rows)[0]

    assert summary["exact_oracle_matches"] == 1
    assert summary["oracle_equivalent_matches"] == 2
    assert summary["recovered_by_equivalence"] == 1


def make_grid(target: str, objective: str, *, tcount: int, primary: int, qasm: int) -> dict[str, str]:
    return {
        "target": target,
        "objective_variant": objective,
        "materializer": "selected-beam-shared-parity-w4",
        "tcount": str(tcount),
        "primary_nc_depth_ratio": str(primary),
        "qasm_depth": str(qasm),
        "qasm_depth_ratio": str(qasm),
        "num_total_cnots": "1",
    }
