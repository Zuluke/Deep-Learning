from __future__ import annotations

from pathlib import Path
import sys

from qiskit import QuantumCircuit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.diagnose_postselection_equivalence import diagnose_circuits
from scripts.diagnose_postselection_equivalence import project_extra_pattern


def test_project_extra_pattern_keeps_low_index_data_qubits() -> None:
    state = [0j] * 8
    state[0b100] = 2 + 0j
    state[0b101] = 3 + 0j
    state[0b110] = 5 + 0j
    state[0b111] = 7 + 0j

    projected = project_extra_pattern(
        state,
        data_qubits=2,
        total_qubits=3,
        pattern=1,
    )

    assert projected.tolist() == [2 + 0j, 3 + 0j, 5 + 0j, 7 + 0j]


def test_postselection_diagnostic_finds_global_match_with_idle_ancilla() -> None:
    original = QuantumCircuit(1)
    original.h(0)
    candidate = QuantumCircuit(2)
    candidate.h(0)

    result = diagnose_circuits(original, candidate, max_columns=2)

    assert result["diagnostic_status"] == "sampled-global-match"
    assert result["best_postselection_pattern"] == 0
    assert result["max_global_residual"] < 1e-9


def test_postselection_diagnostic_detects_columnwise_phase_mismatch() -> None:
    original = QuantumCircuit(1)
    candidate = QuantumCircuit(1)
    candidate.z(0)

    result = diagnose_circuits(original, candidate, max_columns=2)

    assert result["diagnostic_status"] == "sampled-columnwise-phase-mismatch"
    assert result["max_column_residual"] < 1e-9
    assert result["max_global_residual"] > 0.5


def test_postselection_diagnostic_detects_sampled_mismatch() -> None:
    original = QuantumCircuit(1)
    original.x(0)
    candidate = QuantumCircuit(1)

    result = diagnose_circuits(original, candidate, max_columns=2)

    assert result["diagnostic_status"] == "sampled-mismatch"
    assert result["max_column_residual"] > 0.5
