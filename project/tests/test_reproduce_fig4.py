from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._fig4_reproduction import build_binary_addition_points
from scripts._fig4_reproduction import build_finite_field_points
from scripts._fig4_reproduction import generate_fig4_artifacts
from scripts._fig4_reproduction import generate_fig4b_artifacts
from scripts._fig4_reproduction import validate_binary_addition_points
from scripts._fig4_reproduction import validate_finite_field_points


def test_build_binary_addition_points_matches_expected_series() -> None:
    points = build_binary_addition_points()
    validate_binary_addition_points(points)

    assert [point.bits for point in points] == list(range(3, 11))
    assert [point.cuccaro_toffoli_count for point in points] == [4, 6, 8, 10, 12, 14, 16, 18]
    assert [point.vendored_qasm_ccx_count for point in points] == [6, 8, 10, 12, 14, 16, 18, 20]
    assert [point.alphatensor_toffoli_count for point in points] == [2, 3, 4, 5, 6, 7, 8, 9]
    assert [point.gidney_toffoli_count for point in points] == [2, 3, 4, 5, 6, 7, 8, 9]
    assert [point.effective_tcount for point in points] == [4, 6, 8, 10, 12, 14, 16, 18]
    assert all(point.num_cs_gadgets == 0 for point in points)


def test_build_finite_field_points_matches_expected_series() -> None:
    points = build_finite_field_points()
    validate_finite_field_points(points)

    assert [point.exponent_m for point in points] == list(range(2, 11))
    assert [point.circuit_before_optimization_toffoli_count for point in points] == [
        4,
        9,
        16,
        25,
        36,
        49,
        64,
        81,
        100,
    ]
    assert [point.alphatensor_toffoli_count for point in points] == [3, 6, 9, 13, 18, 22, 29, 35, 46]
    assert [point.best_classical_toffoli_upper_bound for point in points] == [
        3,
        6,
        9,
        13,
        17,
        22,
        26,
        31,
        35,
    ]
    assert [point.matches_best_known_upper_bound for point in points] == [
        True,
        True,
        True,
        True,
        False,
        True,
        False,
        False,
        False,
    ]
    assert all(point.num_cs_gadgets == 0 for point in points)


def test_generate_fig4b_artifacts_writes_expected_files(tmp_path: Path) -> None:
    artifact_paths = generate_fig4b_artifacts(tmp_path / "fig4b")

    for path in artifact_paths.values():
        assert path.exists()
        assert path.is_file()


def test_generate_fig4_artifacts_writes_expected_files(tmp_path: Path) -> None:
    artifact_paths = generate_fig4_artifacts(tmp_path / "fig4")

    for path in artifact_paths.values():
        assert path.exists()
        assert path.is_file()
