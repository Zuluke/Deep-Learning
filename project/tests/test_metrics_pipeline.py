from __future__ import annotations

from pathlib import Path
import sys

from qiskit import QuantumCircuit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._analysis_common import compute_structural_metrics
from scripts.assemble_resynth_circuit import assemble_circuit


def test_compute_structural_metrics_empty_circuit() -> None:
    circuit = QuantumCircuit(2)
    metrics = compute_structural_metrics(circuit)
    assert metrics["tcount"] == 0
    assert metrics["tdepth"] == 0
    assert metrics["rho_t"] == 0.0
    assert metrics["n_clifford_blocks"] == 0
    assert metrics["n_nonclifford_blocks"] == 0


def test_compute_structural_metrics_alternating_sequence() -> None:
    circuit = QuantumCircuit(1)
    circuit.h(0)
    circuit.t(0)
    circuit.h(0)
    circuit.tdg(0)
    metrics = compute_structural_metrics(circuit)
    assert metrics["tcount"] == 2
    assert metrics["tdepth"] == 2
    assert metrics["n_clifford_blocks"] == 2
    assert metrics["n_nonclifford_blocks"] == 2
    assert metrics["avg_nonclifford_block_len"] == 1.0
    assert metrics["hadamard_boundary_density"] > 0.0


def test_compute_structural_metrics_clifford_only() -> None:
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.s(1)
    metrics = compute_structural_metrics(circuit)
    assert metrics["tcount"] == 0
    assert metrics["n_clifford_blocks"] == 1
    assert metrics["n_nonclifford_blocks"] == 0
    assert metrics["rho_w"] == 0.0


def test_assemble_circuit_with_real_compile_layout(tmp_path: Path) -> None:
    compile_dir = Path(
        "/Users/caio/Deep-Learning/project/external/circuit-to-tensor/benchmarks/arithmetic/mod_5_4"
    )
    resynth_root = tmp_path / "resynth"
    resynth_root.mkdir(parents=True, exist_ok=True)
    source_block = compile_dir / "mod_5_4.qasm"
    replacement = resynth_root / "mod_5_4.qasm"
    replacement.write_text(source_block.read_text(encoding="utf-8"), encoding="utf-8")
    output_path, summary = assemble_circuit(compile_dir, resynth_root)
    assert output_path.exists()
    assert summary["num_qubits"] > 0
    assert "piece_paths" in summary
