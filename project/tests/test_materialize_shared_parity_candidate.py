from __future__ import annotations

import numpy as np

from scripts import materialize_shared_parity_candidate as shared


def test_gf2_inverse_and_factor_coefficients_for_changed_basis() -> None:
    rows = np.array(
        [
            [1, 0, 1],
            [0, 1, 0],
            [0, 0, 1],
        ],
        dtype=np.uint8,
    )
    parity = np.array([1, 1, 1], dtype=np.uint8)

    coeffs = shared.factor_coefficients(rows, parity)

    assert np.array_equal((coeffs @ rows) % 2, parity)


def test_shared_parity_synthesis_uses_one_t_per_nonzero_factor() -> None:
    factors = np.array(
        [
            [1, 0, 0],
            [1, 1, 0],
            [1, 1, 1],
        ],
        dtype=np.uint8,
    )

    circuit, cnots = shared.synthesize_shared_parity_circuit(
        factors,
        num_qubits=3,
        mapping=[0, 1, 2],
    )

    ops = circuit.count_ops()
    assert ops["t"] == 3
    assert ops["cx"] == 2 * len(cnots)
    assert len(cnots) > 0


def test_shared_parity_synthesis_skips_zero_factor() -> None:
    factors = np.array(
        [
            [0, 0, 0],
            [0, 1, 0],
        ],
        dtype=np.uint8,
    )

    circuit, cnots = shared.synthesize_shared_parity_circuit(
        factors,
        num_qubits=3,
        mapping=[0, 1, 2],
    )

    assert circuit.count_ops()["t"] == 1
    assert cnots == []


def test_clifford_correction_is_empty_for_identical_phase_matrix() -> None:
    from qiskit import QuantumCircuit

    matrix = np.array(
        [
            [1, 0, 1],
            [0, 1, 1],
            [0, 0, 1],
        ],
        dtype=np.uint8,
    )
    circuit = QuantumCircuit(3)

    gate_count = shared.append_clifford_correction(
        circuit,
        candidate_matrix=matrix,
        original_matrix=matrix,
        mapping=[0, 1, 2],
    )

    assert gate_count == 0
    assert circuit.count_ops() == {}
