from __future__ import annotations

import numpy as np

from scripts.verify_candidates_numeric import anf_degree
from scripts.verify_candidates_numeric import correction_analysis
from scripts.verify_candidates_numeric import proportionality


def random_unitary(dim: int, rng: np.random.Generator) -> np.ndarray:
    matrix = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(matrix)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def test_anf_degree_known_functions() -> None:
    # f(x) = 0
    assert anf_degree(np.zeros(8, dtype=np.uint8)) == 0
    # f(x) = x0
    x0 = np.array([x & 1 for x in range(8)], dtype=np.uint8)
    assert anf_degree(x0) == 1
    # f(x) = x0*x1 (degree 2)
    x0x1 = np.array([(x & 1) & ((x >> 1) & 1) for x in range(8)], dtype=np.uint8)
    assert anf_degree(x0x1) == 2
    # f(x) = x0*x1*x2 (degree 3)
    x012 = np.array([(x & 1) & ((x >> 1) & 1) & ((x >> 2) & 1) for x in range(8)], dtype=np.uint8)
    assert anf_degree(x012) == 3


def test_proportionality_detects_exact_scaling() -> None:
    rng = np.random.default_rng(11)
    unitary = random_unitary(8, rng)
    residual, constant = proportionality(0.25j * unitary, unitary)

    assert residual < 1e-12
    assert abs(abs(constant) - 0.25) < 1e-12


def test_clifford_correction_extracts_affine_relabel() -> None:
    rng = np.random.default_rng(7)
    n = 3
    dim = 2**n
    unitary = random_unitary(dim, rng)
    # K|x> = (-1)^{x0 x1} |sigma(x)>, sigma(x) = (x with bit2 ^= bit0) ^ 0b010
    correction = np.zeros((dim, dim), dtype=complex)
    for x in range(dim):
        image = x ^ ((x & 1) << 2) ^ 0b010
        sign = -1.0 if ((x & 1) and ((x >> 1) & 1)) else 1.0
        correction[image, x] = sign
    block = 0.5 * unitary @ correction

    result = correction_analysis(block, unitary)

    assert result is not None
    assert result["residual"] < 1e-10
    assert result["side"] == "input"
    assert result["x_frame_weight"] == 1
    assert result["parity_changes"] == 1
    assert result["phase_degree"] == 2


def test_correction_analysis_flags_non_clifford_phase() -> None:
    rng = np.random.default_rng(13)
    n = 3
    dim = 2**n
    unitary = random_unitary(dim, rng)
    correction = np.zeros((dim, dim), dtype=complex)
    for x in range(dim):
        # degree-3 phase: not Clifford
        sign = -1.0 if (x == 0b111) else 1.0
        correction[x, x] = sign
    block = unitary @ correction

    result = correction_analysis(block, unitary)

    assert result is not None
    assert result["phase_degree"] == 3
    assert result["defect_signature"]


def test_correction_analysis_handles_output_side() -> None:
    rng = np.random.default_rng(19)
    n = 3
    dim = 2**n
    unitary = random_unitary(dim, rng)
    correction = np.zeros((dim, dim), dtype=complex)
    for x in range(dim):
        sign = -1.0 if (x & 1) else 1.0
        correction[x ^ 0b100, x] = sign
    block = correction @ unitary

    result = correction_analysis(block, unitary)

    assert result is not None
    assert result["side"] == "output"
    assert result["phase_degree"] == 1
    assert result["x_frame_weight"] == 1


def test_correction_analysis_rejects_unrelated_unitaries() -> None:
    rng = np.random.default_rng(17)
    unitary = random_unitary(8, rng)
    other = random_unitary(8, rng)

    assert correction_analysis(other, unitary) is None
