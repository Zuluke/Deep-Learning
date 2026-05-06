from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from atq_repro.gadgets import analyze_gadgetization
from atq_repro.gadgets import is_cs_group
from atq_repro.gadgets import is_toffoli_group
from atq_repro.io import load_target_tensor
from atq_repro.paper import aggregate_best_totals
from atq_repro.paper import benchmark_comparison_rows
from atq_repro.paper import build_long_results
import atq_repro.paper as paper_module
from atq_repro.tensor import gf2_rank
from atq_repro.tensor import symmetric_tensor_from_factors
from atq_repro.tensor import validate_decomposition


def test_symmetric_tensor_from_factors_matches_manual_xor() -> None:
    factors = np.asarray([[1, 0], [1, 1]], dtype=bool)
    tensor = symmetric_tensor_from_factors(factors)
    expected = np.zeros((2, 2, 2), dtype=bool)
    expected[0, 0, 0] ^= True
    expected ^= np.ones((2, 2, 2), dtype=bool)
    assert np.array_equal(tensor, expected)


def test_gf2_rank() -> None:
    matrix = np.asarray(
        [
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
        ],
        dtype=bool,
    )
    assert gf2_rank(matrix) == 3


def test_detects_cs_and_toffoli_patterns() -> None:
    a = np.asarray([1, 0, 0], dtype=bool)
    b = np.asarray([0, 1, 0], dtype=bool)
    c = np.asarray([0, 0, 1], dtype=bool)
    assert is_cs_group(np.asarray([a, b, a ^ b]))
    assert is_toffoli_group(np.asarray([a, b, c, a ^ b, a ^ c, b ^ c, a ^ b ^ c]))


def test_mod_5_4_public_decomposition_validates_and_costs_two() -> None:
    factors = np.load(
        PROJECT_ROOT
        / "external"
        / "alphatensor_quantum"
        / "decompositions"
        / "benchmarks_gadgets.npz"
    )["mod_5_4"][0]
    _, target = load_target_tensor("mod_5_4")

    validation = validate_decomposition(factors, target)
    gadget_summary = analyze_gadgetization(factors, use_gadgets=True)

    assert validation.equal
    assert gadget_summary.num_toffoli == 1
    assert gadget_summary.effective_tcount == 2


def test_paper_comparison_reproduces_known_values() -> None:
    rows = build_long_results()
    aggregate = aggregate_best_totals(rows)
    comparison = benchmark_comparison_rows(aggregate)
    indexed = {(row["source"], row["circuit_id"]): row for row in comparison}

    assert indexed[("paper_table_benchmark_gadgets", "mod_5_4")]["match"] == "true"
    assert indexed[("paper_fig4_gf_gadgets", "gf_2pow5_mult")]["reproduced_effective_tcount"] == 26
    assert indexed[("paper_fig4_binary_addition", "cuccaro_adder_n10")]["reproduced_effective_tcount"] == 18
    assert all(row["match"] == "true" for row in comparison)


def test_reproduce_paper_results_writes_main_csvs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(paper_module, "REPORTS_ROOT", tmp_path / "reports")
    output_dir = tmp_path / "paper"

    paths = paper_module.reproduce_paper_results(output_dir)

    for key in [
        "paper_results_long_csv",
        "paper_tensor_validation_csv",
        "paper_benchmark_comparison_csv",
        "paper_family_summary_csv",
        "paper_aggregate_best_csv",
        "paper_summary_json",
    ]:
        assert paths[key].exists()

    comparison = paths["paper_benchmark_comparison_csv"].read_text(encoding="utf-8")
    assert "match" in comparison
    assert "false" not in comparison
