from __future__ import annotations

from pathlib import Path
import sys

from qiskit import QuantumCircuit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._analysis_common import compute_structural_metrics
from scripts.assemble_resynth_circuit import assemble_circuit
from scripts.make_entrega1_outputs import compute_locality_metrics_from_circuit
from scripts.make_entrega1_outputs import t_locations_by_scheduled_layer
from scripts.run_formal_verification import classify_proof
from scripts.run_formal_verification import feynver_path


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


def test_entrega1_locality_metrics_clifford_only() -> None:
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    metrics = compute_locality_metrics_from_circuit(circuit)
    assert metrics["active_t_qubits"] == 0
    assert metrics["nonclifford_core_layers"] == 0
    assert metrics["nonclifford_core_area"] == 0
    assert metrics["clifford_prefix_ratio"] == 1.0
    assert metrics["clifford_suffix_ratio"] == 1.0
    assert metrics["heuristic_splitting_score"] == 1.0


def test_entrega1_locality_metrics_tracks_t_span() -> None:
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.t(0)
    circuit.cx(0, 1)
    circuit.tdg(1)

    locations, total_layers = t_locations_by_scheduled_layer(circuit)
    metrics = compute_locality_metrics_from_circuit(circuit)

    assert [(item.layer, item.qubit, item.gate_name) for item in locations] == [
        (1, 0, "t"),
        (3, 1, "tdg"),
    ]
    assert total_layers == 4
    assert metrics["active_t_qubits"] == 2
    assert metrics["t_locality_span_qubits"] == 2
    assert metrics["first_t_layer"] == 1
    assert metrics["last_t_layer"] == 3
    assert metrics["nonclifford_core_layers"] == 3
    assert metrics["nonclifford_core_area"] == 6


def test_formal_verification_classifies_feynver_outputs() -> None:
    assert classify_proof(0, "Equal (took 0.004s)\n", "") == ("equal", None)
    assert classify_proof(0, "Inconclusive (took 0.011s)\nReduced form:\n...", "")[
        0
    ] == "inconclusive"
    assert classify_proof(1, "", "Error: parser failed\n")[0] == "parse-error"
    assert classify_proof(1, "", "Not equal\n")[0] == "not-equal"


def test_formal_verification_finds_local_feynver() -> None:
    path = feynver_path()
    assert path is None or Path(path).name == "feynver"


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
