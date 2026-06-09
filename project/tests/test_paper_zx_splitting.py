from __future__ import annotations

from qiskit import QuantumCircuit

from scripts.zx_splitting import compute_paper_zx_splitting_metrics
from scripts.zx_splitting import compute_zx_splitting_metrics


def test_paper_zx_detector_preserves_clifford_only_split() -> None:
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.s(1)

    metrics = compute_paper_zx_splitting_metrics(circuit)

    assert metrics["paper_zx_split_status"] == "ok"
    assert metrics["paper_zx_rewrite_status"] == "ok"
    assert metrics["paper_zx_num_nonclifford_spiders"] == 0
    assert metrics["paper_zx_best_clifford_fraction"] == 1.0
    assert metrics["paper_zx_best_nonclifford_depth"] == 0


def test_paper_zx_detector_exposes_clifford_padding_around_t_gate() -> None:
    circuit = QuantumCircuit(1)
    circuit.h(0)
    circuit.t(0)
    circuit.h(0)

    baseline = compute_zx_splitting_metrics(circuit)
    metrics = compute_paper_zx_splitting_metrics(circuit)

    assert metrics["paper_zx_split_status"] == "ok"
    assert metrics["paper_zx_num_nonclifford_spiders"] == 1
    assert metrics["paper_zx_total_depth"] <= baseline["zx_total_depth"]
    assert metrics["paper_zx_best_nonclifford_depth"] > 0
    assert metrics["paper_zx_best_clifford_depth"] >= 0


def test_paper_zx_detector_closes_crossing_two_qubit_gate() -> None:
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.t(0)
    circuit.cx(0, 1)
    circuit.h(1)

    metrics = compute_paper_zx_splitting_metrics(circuit, rewrite_level="graphlike")

    assert metrics["paper_zx_split_status"] == "ok"
    assert metrics["paper_zx_num_nonclifford_spiders"] == 1
    assert metrics["paper_zx_left_closure_iters"] >= 1
    assert metrics["paper_zx_left_nonclifford_depth"] >= 2


def test_paper_zx_detector_reports_rewrite_effect() -> None:
    circuit = QuantumCircuit(1)
    circuit.h(0)
    circuit.t(0)
    circuit.h(0)

    metrics = compute_paper_zx_splitting_metrics(circuit)

    assert metrics["paper_zx_detector_variant"] == "2504.16004-clifford"
    assert metrics["paper_zx_vertices_before_rewrite"] > metrics["paper_zx_vertices_after_rewrite"]
    assert metrics["paper_zx_edges_before_rewrite"] >= metrics["paper_zx_edges_after_rewrite"]
