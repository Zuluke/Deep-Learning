from __future__ import annotations

from pathlib import Path
import sys

from qiskit import QuantumCircuit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.diagnose_block_replacement_equivalence import diagnose_block_circuits
from scripts.diagnose_block_replacement_equivalence import diagnose_monomial_phase_difference
from scripts.diagnose_block_replacement_equivalence import format_mapped_terms


def test_block_replacement_diagnostic_finds_global_match() -> None:
    reference = QuantumCircuit(1)
    reference.h(0)
    candidate = QuantumCircuit(1)
    candidate.h(0)

    result = diagnose_block_circuits(reference, candidate, max_columns=2)

    assert result["block_diagnostic_status"] == "sampled-global-match"
    assert result["max_global_residual"] < 1e-9


def test_block_replacement_diagnostic_detects_input_dependent_phase() -> None:
    reference = QuantumCircuit(1)
    candidate = QuantumCircuit(1)
    candidate.z(0)

    result = diagnose_block_circuits(reference, candidate, max_columns=2)

    assert result["block_diagnostic_status"] == "sampled-columnwise-phase-mismatch"
    assert result["max_column_residual"] < 1e-9
    assert result["max_global_residual"] > 0.5
    assert result["monomial_phase_status"] == "ok"
    assert result["monomial_phase_delta_values"] == "0:1;4:1"
    assert result["monomial_phase_delta_sign_only"] == 1
    assert result["monomial_phase_delta_degree"] == 1
    assert result["monomial_phase_delta_terms"] == "q0"


def test_block_replacement_diagnostic_detects_mismatch() -> None:
    reference = QuantumCircuit(1)
    reference.x(0)
    candidate = QuantumCircuit(1)

    result = diagnose_block_circuits(reference, candidate, max_columns=2)

    assert result["block_diagnostic_status"] == "sampled-mismatch"
    assert result["max_column_residual"] > 0.5


def test_block_replacement_diagnostic_skips_too_many_qubits() -> None:
    reference = QuantumCircuit(3)
    candidate = QuantumCircuit(3)

    result = diagnose_block_circuits(reference, candidate, max_qubits=2)

    assert result["block_diagnostic_status"] == "too-many-qubits"
    assert "exceeds max_qubits" in result["block_diagnostic_error"]


def test_monomial_phase_diagnostic_reports_global_zero_delta() -> None:
    reference = QuantumCircuit(1)
    candidate = QuantumCircuit(1)

    result = diagnose_monomial_phase_difference(reference, candidate)

    assert result["monomial_phase_status"] == "ok"
    assert result["monomial_phase_delta_values"] == "0:2"
    assert result["monomial_phase_delta_global_only"] == 1
    assert result["monomial_phase_delta_degree"] == 0
    assert result["monomial_phase_delta_num_terms"] == 0


def test_monomial_phase_diagnostic_recovers_cz_anf_term() -> None:
    reference = QuantumCircuit(2)
    candidate = QuantumCircuit(2)
    candidate.cz(0, 1)

    result = diagnose_monomial_phase_difference(reference, candidate)

    assert result["monomial_phase_status"] == "ok"
    assert result["monomial_phase_delta_values"] == "0:3;4:1"
    assert result["monomial_phase_delta_sign_only"] == 1
    assert result["monomial_phase_delta_degree"] == 2
    assert result["monomial_phase_delta_terms"] == "q0*q1"


def test_phase_term_mapping_marks_work_indices() -> None:
    assert (
        format_mapped_terms(
            "q0;q2*q3",
            mapping=[0, 1, 10, 12],
            original_num_qubits=10,
        )
        == "orig[0];work[10]*work[12]"
    )
