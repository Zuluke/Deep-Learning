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


def test_factor_order_strategies_are_deterministic() -> None:
    factors = np.array(
        [
            [1, 1, 1],
            [0, 1, 0],
            [1, 0, 0],
        ],
        dtype=np.uint8,
    )

    ordered = shared.ordered_factors(factors, "support-ascending")

    assert [int(np.count_nonzero(factor)) for factor in ordered] == [1, 1, 3]
    assert np.array_equal(shared.ordered_factors(factors, "reverse"), factors[::-1])


def test_random_factor_order_is_seeded_and_deterministic() -> None:
    factors = np.eye(5, dtype=np.uint8)

    ordered_once = shared.ordered_factors(factors, "random-seed-7")
    ordered_twice = shared.ordered_factors(factors, "random-seed-7")
    different_seed = shared.ordered_factors(factors, "random-seed-8")

    assert np.array_equal(ordered_once, ordered_twice)
    assert sorted(map(tuple, ordered_once.tolist())) == sorted(map(tuple, factors.tolist()))
    assert not np.array_equal(ordered_once, different_seed)


def test_target_strategy_changes_shared_network_shape() -> None:
    factors = np.array(
        [
            [1, 1, 0],
            [1, 1, 1],
        ],
        dtype=np.uint8,
    )

    _, min_cnots = shared.synthesize_shared_parity_circuit(
        factors,
        num_qubits=3,
        mapping=[0, 1, 2],
        target_strategy="min-index",
    )
    _, max_cnots = shared.synthesize_shared_parity_circuit(
        factors,
        num_qubits=3,
        mapping=[0, 1, 2],
        target_strategy="max-index",
    )

    assert min_cnots != max_cnots


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


def test_naive_parity_synthesis_recomputes_each_factor() -> None:
    factors = np.array(
        [
            [1, 1, 0],
            [1, 1, 0],
        ],
        dtype=np.uint8,
    )

    circuit, cnots = shared.synthesize_naive_parity_circuit(
        factors,
        num_qubits=3,
        mapping=[0, 1, 2],
        target_strategy="min-index",
    )

    ops = circuit.count_ops()
    assert ops["t"] == 2
    assert ops["cx"] == 4
    assert len(cnots) == 4


def test_depth_aware_target_plan_avoids_busy_target_when_possible() -> None:
    rows = np.eye(3, dtype=np.uint8)
    parity = np.array([1, 1, 1], dtype=np.uint8)

    target, controls, trial_depths = shared.depth_aware_target_plan(
        [0, 1, 2],
        rows=rows,
        parity=parity,
        mapping=[0, 1, 2],
        qubit_depths=[10, 0, 0],
    )

    assert target == 1
    assert controls == [2, 0]
    assert trial_depths[1] == 12


def test_depth_aware_shared_parity_synthesis_preserves_t_count() -> None:
    factors = np.array(
        [
            [1, 1, 0],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=np.uint8,
    )

    circuit, cnots = shared.synthesize_depth_aware_shared_parity_circuit(
        factors,
        num_qubits=3,
        mapping=[0, 1, 2],
    )

    ops = circuit.count_ops()
    assert ops["t"] == 3
    assert ops["cx"] == 2 * len(cnots)
    assert len(cnots) > 0


def test_beam_plan_consumes_each_nonzero_factor_once() -> None:
    factors = np.array(
        [
            [1, 1, 0],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=np.uint8,
    )

    plan = shared.beam_plan_shared_parity(
        factors,
        num_qubits=3,
        mapping=[0, 1, 2],
        beam_width=4,
    )

    assert sorted(item[0] for item in plan.plan) == [0, 1, 2]
    assert plan.remaining == ()
    assert plan.cnot_count == sum(len(item[2]) for item in plan.plan)


def test_beam_shared_parity_synthesis_preserves_t_count() -> None:
    factors = np.array(
        [
            [1, 1, 0],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=np.uint8,
    )

    circuit, cnots, metadata = shared.synthesize_beam_shared_parity_circuit(
        factors,
        num_qubits=3,
        mapping=[0, 1, 2],
        beam_width=4,
    )

    ops = circuit.count_ops()
    assert ops["t"] == 3
    assert ops["cx"] == 2 * len(cnots)
    assert metadata["beam_width"] == 4
    assert len(metadata["beam_plan_factor_order"]) == 3


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


def test_phase_polynomial_matches_exact_parity_phase_function() -> None:
    """Each column models one T gate on the parity of its support.

    The lifted polynomial must therefore reproduce sum_cols parity(x) mod 8
    for every input x and for every column weight. A historical
    weight-dependent closed form broke down for columns of weight >= 8,
    producing Z-frame assembly defects on targets whose original matrices
    contain heavy columns (nc_tof_4, vbe_adder_3, nc_tof_5).
    """
    rng = np.random.default_rng(7)
    n = 10

    def eval_poly(poly: np.ndarray, x: int) -> int:
        bits = [(x >> q) & 1 for q in range(n)]
        total = 0
        for i in range(n):
            for j in range(i + 1):
                for k in range(j + 1):
                    if poly[i, j, k]:
                        total += int(poly[i, j, k]) * bits[i] * bits[j] * bits[k]
        return total % 8

    def true_phase(matrix: np.ndarray, x: int) -> int:
        total = 0
        for col in range(matrix.shape[1]):
            parity = 0
            for q in range(n):
                if matrix[q, col]:
                    parity ^= (x >> q) & 1
            total += parity
        return total % 8

    matrices = [
        np.eye(n, 1, dtype=np.uint8),
    ]
    for weight in range(1, n + 1):
        single = np.zeros((n, 1), dtype=np.uint8)
        single[:weight, 0] = 1
        matrices.append(single)
    for _ in range(5):
        random_matrix = (rng.random((n, 9)) < 0.45).astype(np.uint8)
        random_matrix = random_matrix[:, random_matrix.sum(axis=0) > 0]
        matrices.append(random_matrix)

    for matrix in matrices:
        poly = shared.phase_polynomial(matrix)
        for x in range(2**n):
            assert eval_poly(poly, x) == true_phase(matrix, x)
