from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from qiskit import QuantumCircuit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.alphatensor_structural_cost import compute_selection_metrics
from scripts.alphatensor_structural_cost import structural_selection_key
from scripts.alphatensor_structural_cost import tcount_selection_key
from scripts.alphatensor_reranker import FeatureStats
from scripts.alphatensor_reranker import leave_one_circuit_out_eval
from scripts.alphatensor_reranker import MLPModel
from scripts.alphatensor_reranker import predictions_rows
from scripts.alphatensor_reranker import train_mlp
from scripts._analysis_common import compute_structural_metrics
from scripts.assemble_resynth_circuit import assemble_circuit
from scripts.make_entrega1_outputs import compute_locality_metrics_from_circuit
from scripts.make_entrega1_outputs import t_locations_by_scheduled_layer
from scripts.run_formal_verification import classify_proof
from scripts.run_formal_verification import feynver_path
from scripts.structural_target import attach_structural_target_metrics
from scripts.structural_target import compute_structural_target_metrics
from scripts.zx_splitting import compute_zx_splitting_metrics


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


def test_zx_splitting_detects_fully_clifford_circuit() -> None:
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)

    metrics = compute_zx_splitting_metrics(circuit)

    assert metrics["zx_split_status"] == "ok"
    assert metrics["zx_num_nonclifford_spiders"] == 0
    assert metrics["zx_best_clifford_fraction"] == 1.0
    assert metrics["zx_best_nonclifford_depth"] == 0


def test_zx_splitting_finds_prefix_and_suffix_around_t_gate() -> None:
    circuit = QuantumCircuit(1)
    circuit.h(0)
    circuit.t(0)
    circuit.h(0)

    metrics = compute_zx_splitting_metrics(circuit)

    assert metrics["zx_split_status"] == "ok"
    assert metrics["zx_total_depth"] == 3
    assert metrics["zx_num_nonclifford_spiders"] == 1
    assert metrics["zx_left_clifford_depth"] == 1
    assert metrics["zx_right_clifford_depth"] == 1
    assert metrics["zx_best_clifford_fraction"] == 1 / 3


def test_zx_splitting_tracks_artificial_clifford_padding() -> None:
    circuit = QuantumCircuit(1)
    circuit.h(0)
    circuit.h(0)
    circuit.t(0)
    circuit.h(0)
    circuit.h(0)

    metrics = compute_zx_splitting_metrics(circuit)

    assert metrics["zx_split_status"] == "ok"
    assert metrics["zx_total_depth"] >= 5
    assert metrics["zx_best_clifford_depth"] > 0
    assert metrics["zx_best_nonclifford_depth"] > 0


def test_zx_splitting_closes_two_qubit_crossings() -> None:
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.t(0)
    circuit.cx(0, 1)
    circuit.h(1)

    metrics = compute_zx_splitting_metrics(circuit)

    assert metrics["zx_split_status"] == "ok"
    assert metrics["zx_left_closure_iters"] == 1
    assert metrics["zx_left_clifford_depth"] == 1
    assert metrics["zx_left_nonclifford_depth"] == 3
    assert metrics["zx_right_clifford_depth"] == 2
    assert metrics["zx_best_side"] == "right"


def test_primary_structural_target_uses_original_zx_depth_denominator() -> None:
    original = {
        "circuit_id": "toy",
        "method": "original",
        "zx_split_status": "ok",
        "zx_total_depth": 100,
        "zx_best_nonclifford_depth": 40,
        "depth_after": 50,
        "tcount_after": 10,
        "zx_best_clifford_fraction": 0.60,
    }
    candidate = {
        "circuit_id": "toy",
        "method": "pyzx",
        "zx_split_status": "ok",
        "zx_total_depth": 200,
        "zx_best_nonclifford_depth": 30,
        "depth_after": 75,
        "tcount_after": 8,
        "zx_best_clifford_fraction": 0.85,
    }

    metrics = compute_structural_target_metrics(candidate, original)

    assert metrics["structural_target_status"] == "ok"
    assert metrics["primary_nc_depth_ratio"] == 0.30
    assert metrics["zx_total_depth_ratio"] == 2.0
    assert metrics["qasm_depth_ratio"] == 1.5


def test_primary_structural_target_resists_clifford_padding_proxy() -> None:
    original = {
        "circuit_id": "toy",
        "method": "original",
        "zx_split_status": "ok",
        "zx_total_depth": 3,
        "zx_best_nonclifford_depth": 2,
        "depth_after": 3,
        "tcount_after": 1,
        "zx_best_clifford_fraction": 1 / 3,
    }
    padded_candidate = {
        "circuit_id": "toy",
        "method": "pyzx",
        "zx_split_status": "ok",
        "zx_total_depth": 30,
        "zx_best_nonclifford_depth": 2,
        "depth_after": 30,
        "tcount_after": 1,
        "zx_best_clifford_fraction": 28 / 30,
    }

    metrics = compute_structural_target_metrics(padded_candidate, original)

    assert metrics["structural_target_status"] == "ok"
    assert metrics["primary_nc_depth_ratio"] == 2 / 3
    assert metrics["zx_total_depth_ratio"] == 10
    assert metrics["zx_depth_inflation"] is True


def test_primary_structural_target_reports_controlled_statuses() -> None:
    candidate = {
        "circuit_id": "toy",
        "method": "pyzx",
        "zx_split_status": "ok",
        "zx_total_depth": 1,
        "zx_best_nonclifford_depth": 1,
    }
    invalid_original = {
        "circuit_id": "toy",
        "method": "original",
        "zx_split_status": "ok",
        "zx_total_depth": 0,
        "zx_best_nonclifford_depth": 0,
    }
    missing_zx_original = {
        **invalid_original,
        "zx_split_status": "missing-qasm",
        "zx_total_depth": None,
    }

    assert (
        compute_structural_target_metrics(candidate, None)["structural_target_status"]
        == "missing-original"
    )
    assert (
        compute_structural_target_metrics(candidate, missing_zx_original)[
            "structural_target_status"
        ]
        == "missing-zx"
    )
    assert (
        compute_structural_target_metrics(candidate, invalid_original)[
            "structural_target_status"
        ]
        == "invalid-depth"
    )


def test_attach_structural_target_metrics_indexes_original_by_benchmark() -> None:
    rows = [
        {
            "circuit_id": "toy",
            "method": "original",
            "zx_split_status": "ok",
            "zx_total_depth": 8,
            "zx_best_nonclifford_depth": 4,
            "depth_after": 8,
            "tcount_after": 2,
        },
        {
            "circuit_id": "toy",
            "method": "alphatensor_public",
            "zx_split_status": "ok",
            "zx_total_depth": 6,
            "zx_best_nonclifford_depth": 3,
            "depth_after": 10,
            "tcount_after": 1,
        },
    ]

    enriched = attach_structural_target_metrics(rows)

    assert [row["structural_target_status"] for row in enriched] == ["ok", "ok"]
    assert enriched[1]["primary_nc_depth_ratio"] == 3 / 8
    assert enriched[1]["qasm_depth_ratio"] == 10 / 8


def test_alphatensor_structural_cost_prefers_primary_target_over_tcount() -> None:
    original = {
        "zx_split_status": "ok",
        "zx_total_depth": 10,
        "zx_best_nonclifford_depth": 8,
        "depth_after": 10,
        "tcount_after": 10,
    }
    structurally_good = compute_selection_metrics(
        {
            "zx_split_status": "ok",
            "zx_total_depth": 12,
            "zx_best_nonclifford_depth": 4,
            "depth_after": 12,
            "tcount_after": 9,
        },
        original,
    )
    tcount_good = compute_selection_metrics(
        {
            "zx_split_status": "ok",
            "zx_total_depth": 11,
            "zx_best_nonclifford_depth": 7,
            "depth_after": 11,
            "tcount_after": 5,
        },
        original,
    )

    assert min(
        [tcount_good, structurally_good],
        key=lambda item: structural_selection_key(item),
    ) is structurally_good


def test_alphatensor_structural_cost_tiebreaks_on_tcount() -> None:
    original = {
        "zx_split_status": "ok",
        "zx_total_depth": 10,
        "zx_best_nonclifford_depth": 8,
        "depth_after": 10,
        "tcount_after": 10,
    }
    low_tcount = compute_selection_metrics(
        {
            "zx_split_status": "ok",
            "zx_total_depth": 12,
            "zx_best_nonclifford_depth": 4,
            "depth_after": 12,
            "tcount_after": 5,
        },
        original,
    )
    high_tcount = compute_selection_metrics(
        {
            "zx_split_status": "ok",
            "zx_total_depth": 11,
            "zx_best_nonclifford_depth": 4,
            "depth_after": 11,
            "tcount_after": 9,
        },
        original,
    )

    assert min(
        [high_tcount, low_tcount],
        key=lambda item: structural_selection_key(item),
    ) is low_tcount


def test_alphatensor_structural_cost_reports_missing_zx() -> None:
    original = {
        "zx_split_status": "ok",
        "zx_total_depth": 10,
        "zx_best_nonclifford_depth": 8,
        "depth_after": 10,
        "tcount_after": 10,
    }
    candidate = {
        "zx_split_status": "failed",
        "zx_total_depth": None,
        "zx_best_nonclifford_depth": None,
        "depth_after": 8,
        "tcount_after": 4,
    }

    metrics = compute_selection_metrics(candidate, original)

    assert metrics["selection_status"] == "missing-zx"
    assert metrics["structural_cost"] is None


def test_tcount_selection_key_matches_legacy_ordering() -> None:
    candidates = [
        {"tcount": 7, "tdepth": 1},
        {"tcount": 5, "tdepth": 9},
        {"tcount": 5, "tdepth": 2},
    ]

    assert min(candidates, key=lambda item: tcount_selection_key(item)) == {
        "tcount": 5,
        "tdepth": 2,
    }


def test_alphatensor_reranker_learns_simple_structural_ordering() -> None:
    rows = [
        {
            "circuit_id": "a",
            "candidate_id": "a0",
            "selection_status": "ok",
            "primary_nc_depth_ratio": "0.1",
            "tcount_after": "1",
            "tdepth_after": "1",
            "depth_after": "1",
            "gate_count_after": "1",
            "tcount_ratio": "0.1",
            "qasm_depth_ratio": "1",
            "uses_any_gadget_source": "0",
        },
        {
            "circuit_id": "a",
            "candidate_id": "a1",
            "selection_status": "ok",
            "primary_nc_depth_ratio": "0.9",
            "tcount_after": "9",
            "tdepth_after": "9",
            "depth_after": "9",
            "gate_count_after": "9",
            "tcount_ratio": "0.9",
            "qasm_depth_ratio": "9",
            "uses_any_gadget_source": "1",
        },
        {
            "circuit_id": "b",
            "candidate_id": "b0",
            "selection_status": "ok",
            "primary_nc_depth_ratio": "0.2",
            "tcount_after": "2",
            "tdepth_after": "2",
            "depth_after": "2",
            "gate_count_after": "2",
            "tcount_ratio": "0.2",
            "qasm_depth_ratio": "2",
            "uses_any_gadget_source": "0",
        },
        {
            "circuit_id": "b",
            "candidate_id": "b1",
            "selection_status": "ok",
            "primary_nc_depth_ratio": "0.8",
            "tcount_after": "8",
            "tdepth_after": "8",
            "depth_after": "8",
            "gate_count_after": "8",
            "tcount_ratio": "0.8",
            "qasm_depth_ratio": "8",
            "uses_any_gadget_source": "1",
        },
    ]

    model = train_mlp(rows, hidden_size=4, epochs=300, learning_rate=0.03, seed=7)
    selected = [
        row["candidate_id"]
        for row in predictions_rows(rows, model)
        if row["reranker_selected"]
    ]

    assert selected == ["a0", "b0"]


def test_alphatensor_reranker_tolerance_prefers_lower_tcount() -> None:
    rows = [
        {
            "circuit_id": "a",
            "candidate_id": "a0",
            "selection_status": "ok",
            "primary_nc_depth_ratio": "0.10",
            "depth_after": "0.10",
            "tcount_after": "9",
            "qasm_depth_ratio": "1.0",
            "combo_index": "0",
        },
        {
            "circuit_id": "a",
            "candidate_id": "a1",
            "selection_status": "ok",
            "primary_nc_depth_ratio": "0.12",
            "depth_after": "0.12",
            "tcount_after": "1",
            "qasm_depth_ratio": "1.0",
            "combo_index": "1",
        },
    ]
    model = MLPModel(
        feature_columns=("depth_after",),
        stats=FeatureStats(
            means=np.array([0.0]),
            stds=np.array([1.0]),
            impute_values=np.array([0.0]),
        ),
        w1=np.array([[1.0]]),
        b1=np.array([0.0]),
        w2=np.array([[1.0]]),
        b2=np.array([0.0]),
    )

    selected = [
        row["candidate_id"]
        for row in predictions_rows(rows, model, prediction_tolerance=0.05)
        if row["reranker_selected"]
    ]

    assert selected == ["a1"]


def test_alphatensor_reranker_leave_one_circuit_out_reports_regret() -> None:
    rows = [
        {
            "circuit_id": "a",
            "candidate_id": "a0",
            "selection_status": "ok",
            "primary_nc_depth_ratio": "0.2",
            "tcount_after": "2",
            "tdepth_after": "2",
            "depth_after": "2",
            "gate_count_after": "2",
            "tcount_ratio": "0.2",
            "qasm_depth_ratio": "2",
        },
        {
            "circuit_id": "b",
            "candidate_id": "b0",
            "selection_status": "ok",
            "primary_nc_depth_ratio": "0.3",
            "tcount_after": "3",
            "tdepth_after": "3",
            "depth_after": "3",
            "gate_count_after": "3",
            "tcount_ratio": "0.3",
            "qasm_depth_ratio": "3",
        },
    ]

    eval_rows = leave_one_circuit_out_eval(
        rows,
        hidden_size=2,
        epochs=10,
        learning_rate=0.01,
        weight_decay=0.0,
        seed=3,
    )

    assert len(eval_rows) == 2
    assert {"primary_regret_vs_true_best", "primary_gain_vs_tcount_best"} <= set(
        eval_rows[0]
    )


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
