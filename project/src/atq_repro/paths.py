from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_ROOT = PROJECT_ROOT / "external"
BENCHMARKS_ROOT = EXTERNAL_ROOT / "circuit-to-tensor" / "benchmarks"
DECOMPOSITIONS_ROOT = EXTERNAL_ROOT / "alphatensor_quantum" / "decompositions"
RESULTS_ROOT = PROJECT_ROOT / "results"
PAPER_REPRO_ROOT = RESULTS_ROOT / "reproducibility" / "paper"
CSV_ROOT = RESULTS_ROOT / "csv"
FIGURES_ROOT = RESULTS_ROOT / "figures"
REPORTS_ROOT = RESULTS_ROOT / "reports"

PUBLIC_DECOMPOSITION_FILES = (
    "benchmarks_gadgets.npz",
    "benchmarks_no_gadgets.npz",
    "binary_addition.npz",
    "hamming_weight_phase_gradient.npz",
    "multiplication_finite_fields_gadgets.npz",
    "multiplication_finite_fields_no_gadgets.npz",
    "quantum_chemistry.npz",
    "unary_iteration_gadgets.npz",
    "unary_iteration_no_gadgets.npz",
)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def relative_to_project(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)
