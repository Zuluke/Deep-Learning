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
from scripts.alphatensor_reranker import baseline_rows
from scripts.alphatensor_reranker import leave_one_circuit_out_eval
from scripts.alphatensor_reranker import MLPModel
from scripts.alphatensor_reranker import predictions_rows
from scripts.alphatensor_reranker import select_with_prediction_tolerance
from scripts.alphatensor_reranker import train_ensemble
from scripts.alphatensor_reranker import train_mlp
from scripts.alphatensor_reranker import train_pairwise_mlp_ranker
from scripts.alphatensor_reranker import train_pairwise_ranker
from scripts.alphaq_border_proxy import compute_alphaq_border_metrics
from scripts.alphaq_border_proxy import compute_alphaq_border_target_metrics
from scripts._analysis_common import compute_structural_metrics
from scripts._analysis_common import artifact_stem_candidates
from scripts.analyze_tensor_v3_phase_slack_sensitivity import build_sensitivity_rows
from scripts.analyze_tensor_v3_phase_slack_sensitivity import parse_float_list
from scripts.analyze_tensor_v3_phase_slack_sensitivity import summarize_by_slack
from scripts.analyze_tensor_v3_guard_surface import best_non_worse_guard_rows
from scripts.analyze_tensor_v3_guard_surface import build_guard_surface_rows
from scripts.analyze_tensor_v3_guard_surface import guarded_accepts
from scripts.analyze_tensor_v3_guard_surface import robust_qasm_guard_rows
from scripts.analyze_tensor_v3_tcount_tolerance_surface import best_non_worse_rows
from scripts.analyze_tensor_v3_tcount_tolerance_surface import build_surface_rows
from scripts.assemble_resynth_circuit import assemble_circuit
from scripts.compare_tensor_v3_profiles import build_profile_comparison_rows
from scripts.export_paper_tensor_v3_guard_table import build_manifest
from scripts.export_paper_tensor_v3_guard_table import check_outputs
from scripts.export_paper_tensor_v3_guard_table import manifest_json
from scripts.export_paper_tensor_v3_guard_table import render_guard_table
from scripts.export_paper_tensor_v3_guard_table import sha256_text
from scripts.export_paper_tensor_v3_guard_table import write_table
from scripts.make_entrega1_outputs import compute_locality_metrics_from_circuit
from scripts.make_entrega1_outputs import t_locations_by_scheduled_layer
from scripts.materialize_tensor_v3_guarded_profile import (
    materialize_guarded_selection_manifest_rows,
)
from scripts.run_formal_verification import classify_proof
from scripts.run_formal_verification import feynver_path
from scripts.replay_public_decompositions import select_tensor_v3_combo
from scripts.replay_public_decompositions import tensor_v3_method_name
from scripts.replay_public_decompositions import tensor_v3_profile_settings
from scripts.replay_public_decompositions import tensor_v3_selection_key
from scripts.replay_public_decompositions import tensor_v3_manifest_row
from scripts.structural_target import attach_structural_target_metrics
from scripts.structural_target import compute_structural_target_metrics
from scripts.zx_splitting import compute_zx_splitting_metrics
from scripts import compute_metrics as compute_metrics_module


def _alphaq_row(
    *,
    core_area: int,
    total_depth: int,
    total_width: int,
    tcount: int = 1,
    crossing: int = 0,
    depth_after: int | None = None,
) -> dict[str, int | str | bool | None]:
    core_depth = 0 if core_area == 0 else max(1, core_area // max(total_width, 1))
    core_width = 0 if core_area == 0 else max(1, min(total_width, core_area))
    return {
        "alphaq_border_status": "ok",
        "alphaq_border_error": None,
        "alphaq_total_depth": total_depth,
        "alphaq_total_width": total_width,
        "alphaq_total_area": total_depth * total_width,
        "alphaq_nc_core_depth": core_depth,
        "alphaq_nc_core_width": core_width,
        "alphaq_nc_core_area": core_area,
        "alphaq_core_tcount": tcount,
        "alphaq_crossing_closure_count": crossing,
        "alphaq_has_nonclifford": tcount > 0,
        "tcount_after": tcount,
        "depth_after": total_depth if depth_after is None else depth_after,
    }


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


def test_alphaq_border_proxy_reports_empty_core_for_clifford_circuit() -> None:
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.s(1)

    metrics = compute_alphaq_border_metrics(circuit)

    assert metrics["alphaq_border_status"] == "ok"
    assert metrics["alphaq_nc_core_area"] == 0
    assert metrics["alphaq_crossing_closure_count"] == 0


def test_alphaq_border_proxy_peels_clifford_padding_around_single_t() -> None:
    circuit = QuantumCircuit(1)
    circuit.h(0)
    circuit.t(0)
    circuit.h(0)

    metrics = compute_alphaq_border_metrics(circuit)

    assert metrics["alphaq_border_status"] == "ok"
    assert metrics["alphaq_prefix_clifford_depth"] == 1
    assert metrics["alphaq_suffix_clifford_depth"] == 1
    assert metrics["alphaq_nc_core_depth"] == 1
    assert metrics["alphaq_nc_core_width"] == 1
    assert metrics["alphaq_nc_core_area"] == 1


def test_alphaq_border_proxy_closes_two_qubit_crossing_on_left_border() -> None:
    circuit = QuantumCircuit(2)
    circuit.t(0)
    circuit.cx(0, 1)
    circuit.h(1)

    metrics = compute_alphaq_border_metrics(circuit)

    assert metrics["alphaq_border_status"] == "ok"
    assert metrics["alphaq_left_crossing_closure_count"] == 1
    assert metrics["alphaq_left_nc_core_width"] == 2
    assert metrics["alphaq_nc_core_area"] == 1


def test_alphaq_border_proxy_reports_dependency_closure_for_crossing_cnot() -> None:
    circuit = QuantumCircuit(2)
    circuit.t(0)
    circuit.cx(0, 1)
    circuit.h(1)

    metrics = compute_alphaq_border_metrics(circuit)

    assert metrics["alphaq_border_status"] == "ok"
    assert metrics["alphaq_dependency_core_size"] == 2
    assert metrics["alphaq_dependency_entangling_count"] == 1
    assert metrics["alphaq_dependency_boundary_edge_count"] >= 1


def test_alphaq_border_target_uses_original_total_area_denominator() -> None:
    original = _alphaq_row(core_area=1, total_depth=3, total_width=1)
    padded_candidate = _alphaq_row(core_area=1, total_depth=7, total_width=1)

    metrics = compute_alphaq_border_target_metrics(padded_candidate, original)

    assert metrics["alphaq_target_status"] == "ok"
    assert metrics["alphaq_nc_core_area_ratio"] == 1 / 3
    assert metrics["alphaq_total_depth_ratio"] == 7 / 3


def test_alphatensor_structural_cost_prefers_alphaq_core_over_tcount() -> None:
    original = _alphaq_row(
        core_area=8,
        total_depth=10,
        total_width=2,
        tcount=10,
        depth_after=10,
    )
    structurally_good = compute_selection_metrics(
        _alphaq_row(
            core_area=4,
            total_depth=12,
            total_width=2,
            tcount=9,
            depth_after=12,
        ),
        original,
    )
    tcount_good = compute_selection_metrics(
        _alphaq_row(
            core_area=7,
            total_depth=11,
            total_width=2,
            tcount=5,
            depth_after=11,
        ),
        original,
    )

    assert min(
        [tcount_good, structurally_good],
        key=lambda item: structural_selection_key(item),
    ) is structurally_good


def test_alphatensor_structural_cost_tiebreaks_on_crossing_then_tcount() -> None:
    original = _alphaq_row(core_area=4, total_depth=10, total_width=2, tcount=10)
    low_crossing = compute_selection_metrics(
        _alphaq_row(
            core_area=4,
            crossing=0,
            total_depth=12,
            total_width=2,
            tcount=9,
        ),
        original,
    )
    low_tcount_high_crossing = compute_selection_metrics(
        _alphaq_row(
            core_area=4,
            crossing=2,
            total_depth=11,
            total_width=2,
            tcount=5,
        ),
        original,
    )
    lower_tcount_same_crossing = compute_selection_metrics(
        _alphaq_row(
            core_area=4,
            crossing=0,
            total_depth=12,
            total_width=2,
            tcount=5,
        ),
        original,
    )

    assert min(
        [low_tcount_high_crossing, low_crossing],
        key=lambda item: structural_selection_key(item),
    ) is low_crossing
    assert min(
        [low_crossing, lower_tcount_same_crossing],
        key=lambda item: structural_selection_key(item),
    ) is lower_tcount_same_crossing


def test_alphatensor_structural_cost_reports_missing_alphaq_border() -> None:
    original = _alphaq_row(core_area=8, total_depth=10, total_width=2, tcount=10)
    candidate = {
        "alphaq_border_status": "normalization-failed",
        "depth_after": 8,
        "tcount_after": 4,
    }

    metrics = compute_selection_metrics(candidate, original)

    assert metrics["selection_status"] == "missing-alphaq-border"
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


def test_tensor_v3_selection_prefers_mixed_score_inside_tcount_tolerance() -> None:
    low_tcount_worse_tensor = {
        "selection_status": "ok",
        "tensor_v3_status": "ok",
        "tensor_v3_tcount_tolerance": 0.10,
        "tensor_v3_mixed_excess_norm": 0.50,
        "tensor_v3_mixed_auc_greedy_norm": 0.50,
        "tensor_v3_singleton_bridge_count_norm": 0.50,
        "tensor_v3_stable_factor_hash": "b",
        "tcount_after": 10,
        "qasm_depth_ratio": 1.0,
    }
    higher_tcount_better_tensor = {
        **low_tcount_worse_tensor,
        "tensor_v3_mixed_excess_norm": 0.10,
        "tensor_v3_mixed_auc_greedy_norm": 0.10,
        "tensor_v3_singleton_bridge_count_norm": 0.10,
        "tensor_v3_stable_factor_hash": "a",
        "tcount_after": 11,
    }

    selected = min(
        [low_tcount_worse_tensor, higher_tcount_better_tensor],
        key=lambda row: tensor_v3_selection_key(
            row,
            best_tcount=10,
            tie_breaker=0,
        ),
    )

    assert selected is higher_tcount_better_tensor


def test_tensor_v3_selection_rejects_tensor_gain_outside_tcount_tolerance() -> None:
    low_tcount = {
        "selection_status": "ok",
        "tensor_v3_status": "ok",
        "tensor_v3_tcount_tolerance": 0.10,
        "tensor_v3_mixed_excess_norm": 0.50,
        "tensor_v3_mixed_auc_greedy_norm": 0.50,
        "tensor_v3_singleton_bridge_count_norm": 0.50,
        "tensor_v3_stable_factor_hash": "b",
        "tcount_after": 10,
        "qasm_depth_ratio": 1.0,
    }
    outside_tolerance = {
        **low_tcount,
        "tensor_v3_mixed_excess_norm": 0.01,
        "tensor_v3_mixed_auc_greedy_norm": 0.01,
        "tensor_v3_singleton_bridge_count_norm": 0.01,
        "tensor_v3_stable_factor_hash": "a",
        "tcount_after": 12,
    }

    selected = min(
        [low_tcount, outside_tolerance],
        key=lambda row: tensor_v3_selection_key(
            row,
            best_tcount=10,
            tie_breaker=0,
        ),
    )

    assert selected is low_tcount


def test_tensor_v3_phase_slack_prefers_tdepth_only_within_tensor_near_tie() -> None:
    base = {
        "selection_status": "ok",
        "tensor_v3_selection_status": "ok",
        "tensor_v3_status": "ok",
        "tensor_v3_tcount_tolerance": 0.10,
        "tensor_v3_mixed_auc_greedy_norm": 0.5,
        "tensor_v3_singleton_bridge_count_norm": 0.5,
        "tensor_v3_stable_factor_hash": "m",
        "tcount_after": 10,
    }
    tensor_best_high_tdepth = {
        **base,
        "combo_index": 0,
        "tensor_v3_mixed_excess_norm": 1.00,
        "tdepth_after": 10,
    }
    near_tensor_low_tdepth = {
        **base,
        "combo_index": 1,
        "tensor_v3_mixed_excess_norm": 1.10,
        "tdepth_after": 5,
    }
    outside_tensor_best_tdepth = {
        **base,
        "combo_index": 2,
        "tensor_v3_mixed_excess_norm": 1.30,
        "tdepth_after": 1,
    }

    selected = select_tensor_v3_combo(
        [tensor_best_high_tdepth, near_tensor_low_tdepth, outside_tensor_best_tdepth],
        best_tcount=10,
        ranking_strategy="phase-slack-v1",
        mixed_slack=0.15,
    )

    assert selected is near_tensor_low_tdepth


def test_tensor_v3_profiles_resolve_named_modes() -> None:
    assert tensor_v3_profile_settings(
        profile=None,
        tcount_tolerance=0.12,
        mixed_slack=0.20,
    ) == (None, 0.12, 0.20)
    assert tensor_v3_profile_settings(
        profile="splitting-aggressive",
        tcount_tolerance=0.12,
        mixed_slack=0.20,
    ) == ("splitting-aggressive", 0.40, 0.20)
    assert (
        tensor_v3_method_name(
            selection_objective="tensor-v3",
            ranking_strategy="phase-slack-v1",
            profile="splitting-aggressive",
        )
        == "public_resynth_tensor_v3_phase_slack_aggressive"
    )


def test_tensor_v3_phase_slack_sensitivity_finds_threshold() -> None:
    base = {
        "circuit_id": "toy",
        "status": "ok",
        "selection_status": "ok",
        "tensor_v3_status": "ok",
        "tcount_after": "10",
        "tensor_v3_mixed_auc_greedy_norm": "0.5",
        "tensor_v3_singleton_bridge_count_norm": "0.5",
        "tensor_v3_stable_factor_hash": "h",
    }
    frontier_rows = [
        {
            **base,
            "candidate_id": "toy:combo0",
            "combo_index": "0",
            "tdepth_after": "10",
            "primary_nc_depth_ratio": "0.5",
            "tensor_v3_mixed_excess_norm": "1.0",
        },
        {
            **base,
            "candidate_id": "toy:combo1",
            "combo_index": "1",
            "tdepth_after": "5",
            "primary_nc_depth_ratio": "0.3",
            "tensor_v3_mixed_excess_norm": "1.1",
        },
        {
            **base,
            "candidate_id": "toy:combo2",
            "combo_index": "2",
            "tdepth_after": "1",
            "primary_nc_depth_ratio": "0.2",
            "tensor_v3_mixed_excess_norm": "1.3",
        },
    ]
    rows = build_sensitivity_rows(
        frontier_rows,
        entrega_rows=[
            {
                "circuit_id": "toy",
                "method": "alphatensor_public",
                "method_status": "ok",
                "primary_nc_depth_ratio": "0.6",
            }
        ],
        circuit_ids=("toy",),
        slack_values=(0.0, 0.15, 0.30),
        tcount_tolerance=0.0,
    )

    selected_by_slack = {
        float(row["slack"]): row["selected_candidate_id"] for row in rows
    }
    assert selected_by_slack[0.0] == "toy:combo0"
    assert selected_by_slack[0.15] == "toy:combo1"
    assert selected_by_slack[0.30] == "toy:combo2"
    assert {row["baseline_source"] for row in rows} == {
        "entrega_alphatensor_public"
    }

    summary_by_slack = {row["slack"]: row for row in summarize_by_slack(rows)}
    assert summary_by_slack[0.30]["classification"] == "oracle-plateau"
    assert summary_by_slack[0.30]["oracle_hits"] == 1


def test_tensor_v3_phase_slack_sensitivity_falls_back_to_frontier_tcount() -> None:
    base = {
        "circuit_id": "toy",
        "status": "ok",
        "selection_status": "ok",
        "tensor_v3_status": "ok",
        "tensor_v3_mixed_auc_greedy_norm": "0.5",
        "tensor_v3_singleton_bridge_count_norm": "0.5",
        "tensor_v3_stable_factor_hash": "h",
    }
    rows = build_sensitivity_rows(
        [
            {
                **base,
                "candidate_id": "toy:combo0",
                "combo_index": "0",
                "tcount_after": "10",
                "tdepth_after": "2",
                "primary_nc_depth_ratio": "0.5",
                "tensor_v3_mixed_excess_norm": "1.0",
            },
            {
                **base,
                "candidate_id": "toy:combo1",
                "combo_index": "1",
                "tcount_after": "11",
                "tdepth_after": "1",
                "primary_nc_depth_ratio": "0.4",
                "tensor_v3_mixed_excess_norm": "1.1",
            },
        ],
        entrega_rows=[],
        circuit_ids=("toy",),
        slack_values=(0.15,),
        tcount_tolerance=0.10,
    )

    assert rows[0]["baseline_source"] == "frontier_tcount_only"
    assert rows[0]["baseline_candidate_id"] == "toy:combo0"
    assert rows[0]["baseline_primary_nc_depth_ratio"] == 0.5


def test_tensor_v3_tcount_tolerance_surface_tracks_global_oracle() -> None:
    base = {
        "circuit_id": "toy",
        "status": "ok",
        "selection_status": "ok",
        "tensor_v3_status": "ok",
        "tensor_v3_mixed_auc_greedy_norm": "0.5",
        "tensor_v3_singleton_bridge_count_norm": "0.5",
        "tensor_v3_stable_factor_hash": "h",
    }
    frontier_rows = [
        {
            **base,
            "candidate_id": "toy:combo0",
            "combo_index": "0",
            "tcount_after": "10",
            "tdepth_after": "2",
            "primary_nc_depth_ratio": "0.5",
            "tensor_v3_mixed_excess_norm": "1.0",
        },
        {
            **base,
            "candidate_id": "toy:combo1",
            "combo_index": "1",
            "tcount_after": "12",
            "tdepth_after": "1",
            "primary_nc_depth_ratio": "0.3",
            "tensor_v3_mixed_excess_norm": "0.1",
        },
    ]
    detail_rows, summary_rows = build_surface_rows(
        frontier_rows,
        entrega_rows=[],
        circuit_ids=("toy",),
        slack_values=(0.20,),
        tcount_tolerances=(0.0, 0.20),
    )

    by_tolerance = {float(row["tcount_tolerance"]): row for row in detail_rows}
    assert by_tolerance[0.0]["oracle_candidate_id"] == "toy:combo0"
    assert by_tolerance[0.0]["global_oracle_candidate_id"] == "toy:combo1"
    assert by_tolerance[0.0]["global_regret_vs_oracle"] == 0.2
    assert by_tolerance[0.20]["selected_candidate_id"] == "toy:combo1"

    best_rows = best_non_worse_rows(summary_rows)
    assert [(row["tcount_tolerance"], row["slack"]) for row in best_rows] == [
        (0.2, 0.2)
    ]


def test_tensor_v3_profile_comparison_reports_tradeoff() -> None:
    rows = build_profile_comparison_rows(
        conservative_rows=[
            {
                "circuit_id": "toy",
                "status": "ok",
                "combo_index": "0",
                "tcount_after_selection": "10",
                "primary_nc_depth_ratio": "0.5",
                "qasm_depth_ratio": "1.0",
                "tensor_v3_mixed_excess_norm": "1.0",
                "tensor_v3_profile": "conservative",
                "tensor_v3_tcount_tolerance": "0.12",
                "tensor_v3_mixed_slack": "0.2",
            }
        ],
        aggressive_rows=[
            {
                "circuit_id": "toy",
                "status": "ok",
                "combo_index": "1",
                "tcount_after_selection": "12",
                "primary_nc_depth_ratio": "0.3",
                "qasm_depth_ratio": "0.7",
                "tensor_v3_mixed_excess_norm": "0.1",
                "tensor_v3_profile": "splitting-aggressive",
                "tensor_v3_tcount_tolerance": "0.4",
                "tensor_v3_mixed_slack": "0.2",
            }
        ],
        frontier_rows=[
            {
                "circuit_id": "toy",
                "candidate_id": "toy:combo0",
                "status": "ok",
                "selection_status": "ok",
                "tcount_after": "10",
                "tdepth_after": "2",
                "primary_nc_depth_ratio": "0.5",
            },
            {
                "circuit_id": "toy",
                "candidate_id": "toy:combo1",
                "status": "ok",
                "selection_status": "ok",
                "tcount_after": "12",
                "tdepth_after": "1",
                "primary_nc_depth_ratio": "0.3",
            },
        ],
        guarded_qasm_depth_gain=0.10,
    )

    assert rows[0]["aggressive_relation_vs_conservative"] == "better"
    assert rows[0]["delta_tcount"] == 2
    assert rows[0]["aggressive_is_global_oracle"] == 1
    assert rows[0]["guarded_uses_aggressive"] == 1
    assert rows[0]["guarded_relation_vs_conservative"] == "better"


def test_tensor_v3_guard_surface_blocks_low_qasm_gain_regression() -> None:
    conservative_rows = [
        {
            "circuit_id": "good",
            "status": "ok",
            "combo_index": "0",
            "tcount_after_selection": "10",
            "primary_nc_depth_ratio": "0.5",
            "qasm_depth_ratio": "1.0",
            "tensor_v3_mixed_excess_norm": "10.0",
            "tensor_v3_profile": "conservative",
        },
        {
            "circuit_id": "bad",
            "status": "ok",
            "combo_index": "0",
            "tcount_after_selection": "10",
            "primary_nc_depth_ratio": "0.4",
            "qasm_depth_ratio": "1.0",
            "tensor_v3_mixed_excess_norm": "10.0",
            "tensor_v3_profile": "conservative",
        },
    ]
    aggressive_rows = [
        {
            "circuit_id": "good",
            "status": "ok",
            "combo_index": "1",
            "tcount_after_selection": "12",
            "primary_nc_depth_ratio": "0.3",
            "qasm_depth_ratio": "0.8",
            "tensor_v3_mixed_excess_norm": "1.0",
            "tensor_v3_profile": "splitting-aggressive",
        },
        {
            "circuit_id": "bad",
            "status": "ok",
            "combo_index": "1",
            "tcount_after_selection": "12",
            "primary_nc_depth_ratio": "0.6",
            "qasm_depth_ratio": "0.95",
            "tensor_v3_mixed_excess_norm": "1.0",
            "tensor_v3_profile": "splitting-aggressive",
        },
    ]
    frontier_rows = [
        {
            "circuit_id": circuit_id,
            "candidate_id": f"{circuit_id}:combo{combo}",
            "status": "ok",
            "selection_status": "ok",
            "tcount_after": str(10 + 2 * combo),
            "tdepth_after": "1",
            "primary_nc_depth_ratio": primary,
        }
        for circuit_id, primary in (
            ("good", "0.5"),
            ("good", "0.3"),
            ("bad", "0.4"),
            ("bad", "0.6"),
        )
        for combo in ([0] if primary in {"0.5", "0.4"} else [1])
    ]

    detail_rows, summary_rows = build_guard_surface_rows(
        conservative_rows=conservative_rows,
        aggressive_rows=aggressive_rows,
        frontier_rows=frontier_rows,
        qasm_depth_gains=(0.0, 0.1),
        mixed_drop_fractions=(0.5,),
    )

    by_qasm = {float(row["qasm_depth_gain"]): row for row in summary_rows}
    assert by_qasm[0.0]["accepted_aggressive_count"] == 2
    assert by_qasm[0.0]["worse_vs_conservative"] == 1
    assert by_qasm[0.1]["accepted_aggressive_count"] == 1
    assert by_qasm[0.1]["better_vs_conservative"] == 1
    assert by_qasm[0.1]["worse_vs_conservative"] == 0

    selected_bad = next(
        row
        for row in detail_rows
        if row["circuit_id"] == "bad" and float(row["qasm_depth_gain"]) == 0.1
    )
    assert selected_bad["guarded_uses_aggressive"] == 0
    assert selected_bad["selected_candidate_id"] == "bad:combo0"


def test_tensor_v3_guard_surface_picks_best_non_worse_plateau() -> None:
    rows = [
        {
            "qasm_depth_gain": 0.0,
            "mixed_drop_fraction": 0.0,
            "worse_vs_conservative": 1,
            "better_vs_conservative": 2,
            "global_oracle_hits": 2,
            "max_global_oracle_regret": 0.0,
            "tcount_overhead_total": 3,
        },
        {
            "qasm_depth_gain": 0.1,
            "mixed_drop_fraction": 0.0,
            "worse_vs_conservative": 0,
            "better_vs_conservative": 2,
            "global_oracle_hits": 2,
            "max_global_oracle_regret": 0.1,
            "tcount_overhead_total": 3,
        },
        {
            "qasm_depth_gain": 0.2,
            "mixed_drop_fraction": 0.0,
            "worse_vs_conservative": 0,
            "better_vs_conservative": 1,
            "global_oracle_hits": 1,
            "max_global_oracle_regret": 0.0,
            "tcount_overhead_total": 1,
        },
    ]

    assert best_non_worse_guard_rows(rows) == [rows[1]]


def test_tensor_v3_guard_surface_finds_robust_qasm_threshold() -> None:
    rows = [
        {
            "qasm_depth_gain": 0.0,
            "mixed_drop_fraction": 0.0,
            "worse_vs_conservative": 1,
            "better_vs_conservative": 2,
            "global_oracle_hits": 2,
            "max_global_oracle_regret": 0.0,
            "tcount_overhead_total": 3,
        },
        {
            "qasm_depth_gain": 0.0,
            "mixed_drop_fraction": 0.99,
            "worse_vs_conservative": 0,
            "better_vs_conservative": 2,
            "global_oracle_hits": 2,
            "max_global_oracle_regret": 0.0,
            "tcount_overhead_total": 3,
        },
        {
            "qasm_depth_gain": 0.1,
            "mixed_drop_fraction": 0.0,
            "worse_vs_conservative": 0,
            "better_vs_conservative": 2,
            "global_oracle_hits": 2,
            "max_global_oracle_regret": 0.0,
            "tcount_overhead_total": 3,
        },
        {
            "qasm_depth_gain": 0.1,
            "mixed_drop_fraction": 0.99,
            "worse_vs_conservative": 0,
            "better_vs_conservative": 2,
            "global_oracle_hits": 2,
            "max_global_oracle_regret": 0.0,
            "tcount_overhead_total": 3,
        },
    ]

    robust_rows = robust_qasm_guard_rows(rows)

    assert len(robust_rows) == 1
    assert robust_rows[0]["robust_qasm_depth_gain"] == 0.1


def test_tensor_v3_guard_acceptance_requires_real_mixed_drop() -> None:
    assert not guarded_accepts(
        conservative_qasm_depth=1.0,
        aggressive_qasm_depth=0.9,
        conservative_mixed=0.0,
        aggressive_mixed=0.0,
        qasm_depth_gain=0.05,
        mixed_drop_fraction=0.0,
    )
    assert guarded_accepts(
        conservative_qasm_depth=1.0,
        aggressive_qasm_depth=0.9,
        conservative_mixed=10.0,
        aggressive_mixed=1.0,
        qasm_depth_gain=0.05,
        mixed_drop_fraction=0.5,
    )


def test_export_paper_tensor_v3_guard_table_uses_csv_values() -> None:
    rows = [
        {
            "circuit_id": "qft_4",
            "status": "ok",
            "conservative_tcount": "53",
            "aggressive_tcount": "67",
            "guarded_tcount": "53",
            "conservative_primary_nc_depth_ratio": "1.079268292682927",
            "aggressive_primary_nc_depth_ratio": "1.1768292682926829",
            "guarded_primary_nc_depth_ratio": "1.079268292682927",
            "aggressive_relation_vs_conservative": "worse",
            "guarded_relation_vs_conservative": "tie",
            "guarded_qasm_depth_gain_threshold": "0.1",
            "guarded_mixed_drop_fraction_threshold": "0.0",
        },
        {
            "circuit_id": "barenco_tof_4",
            "status": "ok",
            "conservative_tcount": "23",
            "aggressive_tcount": "28",
            "guarded_tcount": "28",
            "conservative_primary_nc_depth_ratio": "1.6504854368932038",
            "aggressive_primary_nc_depth_ratio": "1.4368932038834952",
            "guarded_primary_nc_depth_ratio": "1.4368932038834952",
            "aggressive_relation_vs_conservative": "better",
            "guarded_relation_vs_conservative": "better",
            "guarded_qasm_depth_gain_threshold": "0.1",
            "guarded_mixed_drop_fraction_threshold": "0.0",
        },
    ]

    table = render_guard_table(rows, focus_circuits=("barenco_tof_4", "qft_4"))

    assert r"\texttt{barenco\_tof\_4}" in table
    assert r"\texttt{qft\_4}" in table
    assert "23 & 28 & 28 & 1.650 & 1.437 & 1.437" in table
    assert "53 & 67 & 53 & 1.079 & 1.177 & 1.079" in table
    assert "aggressive B/T/W is 1/0/1" in table
    assert "guarded B/T/W is 1/1/0" in table
    assert r"$\tau_d=0.10$" in table


def test_export_paper_tensor_v3_guard_manifest_hashes_source(tmp_path) -> None:
    comparison_csv = tmp_path / "comparison.csv"
    output_tex = tmp_path / "table.tex"
    comparison_csv.write_text(
        "circuit_id,status,conservative_tcount,aggressive_tcount,guarded_tcount,"
        "conservative_primary_nc_depth_ratio,aggressive_primary_nc_depth_ratio,"
        "guarded_primary_nc_depth_ratio,aggressive_relation_vs_conservative,"
        "guarded_relation_vs_conservative,guarded_qasm_depth_gain_threshold,"
        "guarded_mixed_drop_fraction_threshold,guarded_uses_aggressive\n"
        "qft_4,ok,53,67,53,1.079,1.177,1.079,worse,tie,0.1,0.0,0\n",
        encoding="utf-8",
    )
    rows = [
        {
            "circuit_id": "qft_4",
            "status": "ok",
            "conservative_tcount": "53",
            "aggressive_tcount": "67",
            "guarded_tcount": "53",
            "conservative_primary_nc_depth_ratio": "1.079",
            "aggressive_primary_nc_depth_ratio": "1.177",
            "guarded_primary_nc_depth_ratio": "1.079",
            "aggressive_relation_vs_conservative": "worse",
            "guarded_relation_vs_conservative": "tie",
            "guarded_qasm_depth_gain_threshold": "0.1",
            "guarded_mixed_drop_fraction_threshold": "0.0",
            "guarded_uses_aggressive": "0",
        }
    ]
    write_table(render_guard_table(rows, focus_circuits=("qft_4",)), output_tex)

    manifest = build_manifest(
        rows,
        comparison_csv=comparison_csv,
        output_tex=output_tex,
        focus_circuits=("qft_4",),
    )

    assert len(manifest["comparison_csv_sha256"]) == 64
    assert len(manifest["output_tex_sha256"]) == 64
    assert manifest["guarded_relation_counts"] == {
        "better": 0,
        "tie": 1,
        "worse": 0,
    }
    assert manifest["selected_rows"][0]["circuit_id"] == "qft_4"
    assert manifest["selected_rows"][0]["guarded_uses_aggressive"] == 0


def test_export_paper_tensor_v3_guard_check_detects_stale_outputs(tmp_path) -> None:
    comparison_csv = tmp_path / "comparison.csv"
    output_tex = tmp_path / "table.tex"
    output_json = tmp_path / "table.json"
    comparison_csv.write_text(
        "circuit_id,status,conservative_tcount,aggressive_tcount,guarded_tcount,"
        "conservative_primary_nc_depth_ratio,aggressive_primary_nc_depth_ratio,"
        "guarded_primary_nc_depth_ratio,aggressive_relation_vs_conservative,"
        "guarded_relation_vs_conservative,guarded_qasm_depth_gain_threshold,"
        "guarded_mixed_drop_fraction_threshold,guarded_uses_aggressive\n"
        "qft_4,ok,53,67,53,1.079,1.177,1.079,worse,tie,0.1,0.0,0\n",
        encoding="utf-8",
    )
    rows = [
        {
            "circuit_id": "qft_4",
            "status": "ok",
            "conservative_tcount": "53",
            "aggressive_tcount": "67",
            "guarded_tcount": "53",
            "conservative_primary_nc_depth_ratio": "1.079",
            "aggressive_primary_nc_depth_ratio": "1.177",
            "guarded_primary_nc_depth_ratio": "1.079",
            "aggressive_relation_vs_conservative": "worse",
            "guarded_relation_vs_conservative": "tie",
            "guarded_qasm_depth_gain_threshold": "0.1",
            "guarded_mixed_drop_fraction_threshold": "0.0",
            "guarded_uses_aggressive": "0",
        }
    ]
    table_text = render_guard_table(rows, focus_circuits=("qft_4",))
    manifest = build_manifest(
        rows,
        comparison_csv=comparison_csv,
        output_tex=output_tex,
        focus_circuits=("qft_4",),
        output_tex_sha256=sha256_text(table_text),
    )
    output_tex.write_text(table_text, encoding="utf-8")
    output_json.write_text(manifest_json(manifest), encoding="utf-8")

    current = check_outputs(
        expected_table_text=table_text,
        expected_manifest=manifest,
        output_tex=output_tex,
        output_json=output_json,
    )
    assert current["ok"]

    output_tex.write_text(table_text + "% stale edit\n", encoding="utf-8")
    stale = check_outputs(
        expected_table_text=table_text,
        expected_manifest=manifest,
        output_tex=output_tex,
        output_json=output_json,
    )
    assert not stale["ok"]
    assert not stale["table_current"]
    assert stale["manifest_current"]


def test_tensor_v3_phase_slack_parse_values_sorts_and_dedupes() -> None:
    assert parse_float_list("0.15, 0.0,0.15,0.05") == (0.0, 0.05, 0.15)


def test_artifact_stem_candidates_adds_toff_tof_alias() -> None:
    assert "barenco_tof_3" in artifact_stem_candidates("barenco_toff_3")
    assert "nc_tof_4" in artifact_stem_candidates("nc_toff_4")


def test_tensor_v3_manifest_omits_external_audit_fields() -> None:
    manifest = tensor_v3_manifest_row(
        circuit_id="toy",
        selected=True,
        combo_summary={
            "status": "ok",
            "combo_index": 3,
            "candidate_qasm_path": "candidate.qasm",
            "block_choices": [
                {
                    "source_method": "public_resynth_gadgets",
                    "decomposition_key": "toy_decomp",
                    "candidate_index": 1,
                    "tcount": 5,
                    "tdepth": 2,
                }
            ],
            "tcount_after": 5,
            "tdepth_after": 2,
            "depth_after": 8,
            "gate_count_after": 13,
            "tensor_v3_selection_status": "ok",
            "tensor_v3_selection_error": None,
            "tensor_v3_ranking_strategy": "lex-v1",
            "tensor_v3_status": "ok",
            "tensor_v3_mixed_excess_norm": 0.25,
            "primary_nc_depth_ratio": 0.1,
            "zx_total_depth_ratio": 0.2,
            "alphaq_dependency_core_area_ratio": 0.3,
            "structural_cost": 0.4,
        },
    )

    assert manifest["tensor_v3_selected"] == 1
    assert manifest["tensor_v3_mixed_excess_norm"] == 0.25
    assert not any(
        key.startswith(("primary", "zx_", "alphaq_", "structural"))
        for key in manifest
    )


def test_materialize_guarded_selection_manifest_omits_external_targets() -> None:
    rows = materialize_guarded_selection_manifest_rows(
        conservative_manifest_rows=[
            {
                "circuit_id": "toy",
                "candidate_id": "toy:combo0",
                "combo_index": "0",
                "tensor_v3_selected": "1",
                "status": "ok",
                "tensor_v3_profile": "conservative",
                "tcount_after": "10",
                "tensor_v3_mixed_excess_norm": "0.50",
            }
        ],
        aggressive_manifest_rows=[
            {
                "circuit_id": "toy",
                "candidate_id": "toy:combo1",
                "combo_index": "1",
                "tensor_v3_selected": "1",
                "status": "ok",
                "tensor_v3_profile": "splitting-aggressive",
                "tcount_after": "11",
                "tensor_v3_mixed_excess_norm": "0.20",
            }
        ],
        comparison_rows=[
            {
                "circuit_id": "toy",
                "status": "ok",
                "guarded_uses_aggressive": "1",
                "guarded_qasm_depth_gain_threshold": "0.1",
                "guarded_mixed_drop_fraction_threshold": "0.0",
                "qasm_depth_gain_aggressive_vs_conservative": "0.2",
                "mixed_drop_aggressive_vs_conservative": "0.3",
                "mixed_drop_fraction_aggressive_vs_conservative": "0.6",
                "guarded_delta_tcount_vs_conservative": "1",
                "guarded_primary_nc_depth_ratio": "0.123",
            }
        ],
    )

    assert len(rows) == 1
    assert rows[0]["candidate_id"] == "toy:combo1"
    assert rows[0]["tensor_v3_profile"] == "guarded-aggressive"
    assert rows[0]["guarded_source_profile"] == "splitting-aggressive"
    assert rows[0]["guarded_uses_aggressive"] == 1
    assert rows[0]["qasm_depth_gain_aggressive_vs_conservative"] == "0.2"
    assert "primary_nc_depth_ratio" not in rows[0]
    assert "guarded_primary_nc_depth_ratio" not in rows[0]


def test_compute_metrics_includes_tensor_v3_public_method(monkeypatch, tmp_path) -> None:
    original_qasm = tmp_path / "toy.qasm"
    tensor_qasm = tmp_path / "tensor.qasm"
    compile_dir = tmp_path / "compile"
    compile_dir.mkdir()
    original_qasm.write_text("OPENQASM 2.0;\nqreg q[1];\nt q[0];\n", encoding="utf-8")
    tensor_qasm.write_text("OPENQASM 2.0;\nqreg q[1];\nt q[0];\n", encoding="utf-8")

    def fake_metrics(path: Path) -> dict[str, int | float | str | None]:
        is_original = path == original_qasm
        return {
            "comparability_status": "ok",
            "normalization_status": "ok",
            "normalization_error": None,
            "tcount": 10 if is_original else 9,
            "tdepth": 10 if is_original else 9,
            "rho_t": 0.0,
            "rho_w": 0.0,
            "rho_w_lambda_5": 0.0,
            "rho_w_lambda_10": 0.0,
            "rho_w_lambda_20": 0.0,
            "n_clifford_blocks": 0,
            "n_nonclifford_blocks": 1,
            "avg_nonclifford_block_len": 1.0,
            "hadamard_boundary_density": 0.0,
            "hadamard_boundary_density_w1": 0.0,
            "hadamard_boundary_density_w2": 0.0,
            "tdepth_over_tcount": 1.0,
            "normalized_qasm_num_qubits": 1,
            "normalized_qasm_depth": 10 if is_original else 9,
            "normalized_qasm_size": 1,
        }

    monkeypatch.setattr(compute_metrics_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        compute_metrics_module,
        "compute_metrics_from_qasm_path",
        fake_metrics,
    )
    monkeypatch.setattr(compute_metrics_module, "tensor_size_from_directory", lambda _: 3)
    monkeypatch.setattr(compute_metrics_module, "latest_demo_logs", lambda _: {})
    monkeypatch.setattr(compute_metrics_module, "to_relative", lambda path: str(path))

    rows = compute_metrics_module.build_rows(
        inventory_rows=[
            {
                "circuit_id": "toy",
                "source": "unit",
                "family": "unit",
                "n_qubits": "1",
                "qasm_path": "toy.qasm",
                "vendored_compile_dir": "compile",
            }
        ],
        pyzx_rows=[],
        compile_quizx_rows=[],
        public_resynth_rows=[],
        structural_resynth_rows=[],
        tensor_v3_resynth_rows=[
            {
                "circuit_id": "toy",
                "method": "public_resynth_tensor_v3",
                "status": "ok",
                "assembled_qasm_path": str(tensor_qasm),
                "tensor_v3_status": "ok",
                "tensor_v3_mixed_excess_norm": "0.125",
            }
        ],
        tensor_v3_phase_slack_resynth_rows=[
            {
                "circuit_id": "toy",
                "method": "public_resynth_tensor_v3_phase_slack",
                "status": "ok",
                "assembled_qasm_path": str(tensor_qasm),
                "tensor_v3_status": "ok",
                "tensor_v3_ranking_strategy": "phase-slack-v1",
                "tensor_v3_profile": "conservative",
                "tensor_v3_mixed_slack": "0.2",
                "tensor_v3_mixed_excess_norm": "0.100",
            }
        ],
        tensor_v3_extra_resynth_rows=[
            {
                "circuit_id": "toy",
                "method": "public_resynth_tensor_v3_phase_slack_guarded",
                "status": "ok",
                "assembled_qasm_path": str(tensor_qasm),
                "tensor_v3_status": "ok",
                "tensor_v3_profile": "guarded-aggressive",
                "tensor_v3_mixed_excess_norm": "0.090",
                "guarded_uses_aggressive": "1",
                "guarded_qasm_depth_gain_threshold": "0.10",
                "guarded_mixed_drop_fraction_threshold": "0.0",
                "mixed_drop_fraction_aggressive_vs_conservative": "0.90",
            }
        ],
        reranker_resynth_rows=[],
        demo_log_dir=tmp_path / "logs",
    )

    tensor_row = next(row for row in rows if row["method"] == "public_resynth_tensor_v3")
    phase_slack_row = next(
        row for row in rows if row["method"] == "public_resynth_tensor_v3_phase_slack"
    )
    guarded_row = next(
        row
        for row in rows
        if row["method"] == "public_resynth_tensor_v3_phase_slack_guarded"
    )

    assert tensor_row["method_status"] == "ok"
    assert tensor_row["tcount_after"] == 9
    assert tensor_row["tensor_v3_status"] == "ok"
    assert tensor_row["tensor_v3_mixed_excess_norm"] == "0.125"
    assert phase_slack_row["method_status"] == "ok"
    assert phase_slack_row["tensor_v3_ranking_strategy"] == "phase-slack-v1"
    assert phase_slack_row["tensor_v3_profile"] == "conservative"
    assert phase_slack_row["tensor_v3_mixed_slack"] == "0.2"
    assert phase_slack_row["tensor_v3_mixed_excess_norm"] == "0.100"
    assert guarded_row["method_status"] == "ok"
    assert guarded_row["tensor_v3_profile"] == "guarded-aggressive"
    assert guarded_row["tensor_v3_mixed_excess_norm"] == "0.090"
    assert guarded_row["guarded_uses_aggressive"] == "1"
    assert guarded_row["guarded_qasm_depth_gain_threshold"] == "0.10"
    assert guarded_row["guarded_mixed_drop_fraction_threshold"] == "0.0"
    assert guarded_row["mixed_drop_fraction_aggressive_vs_conservative"] == "0.90"


def test_alphatensor_reranker_learns_simple_structural_ordering() -> None:
    rows = [
        {
            "circuit_id": "a",
            "candidate_id": "a0",
            "selection_status": "ok",
            "structural_cost": "0.1",
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
            "structural_cost": "0.9",
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
            "structural_cost": "0.2",
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
            "structural_cost": "0.8",
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


def test_pairwise_ranker_learns_within_circuit_ordering() -> None:
    rows = [
        {
            "circuit_id": "a",
            "candidate_id": "a-good",
            "selection_status": "ok",
            "structural_cost": "0.1",
            "tcount_after": "4",
            "depth_after": "1",
            "qasm_depth_ratio": "1",
        },
        {
            "circuit_id": "a",
            "candidate_id": "a-bad",
            "selection_status": "ok",
            "structural_cost": "0.9",
            "tcount_after": "1",
            "depth_after": "9",
            "qasm_depth_ratio": "9",
        },
        {
            "circuit_id": "b",
            "candidate_id": "b-good",
            "selection_status": "ok",
            "structural_cost": "0.2",
            "tcount_after": "5",
            "depth_after": "2",
            "qasm_depth_ratio": "2",
        },
        {
            "circuit_id": "b",
            "candidate_id": "b-bad",
            "selection_status": "ok",
            "structural_cost": "0.8",
            "tcount_after": "1",
            "depth_after": "8",
            "qasm_depth_ratio": "8",
        },
    ]

    model = train_pairwise_ranker(
        rows,
        feature_columns=("depth_after", "qasm_depth_ratio"),
        epochs=600,
        learning_rate=0.05,
        weight_decay=1e-4,
        seed=5,
    )
    selected = [
        row["candidate_id"]
        for row in predictions_rows(rows, model, prediction_tolerance=0.0)
        if row["reranker_selected"]
    ]

    assert selected == ["a-good", "b-good"]


def test_pairwise_mlp_ranker_learns_within_circuit_ordering() -> None:
    rows = [
        {
            "circuit_id": "a",
            "candidate_id": "a-good",
            "selection_status": "ok",
            "structural_cost": "0.1",
            "alphaq_dependency_core_area_ratio": "0.1",
            "tcount_after": "4",
            "depth_after": "1",
            "qasm_depth_ratio": "1",
        },
        {
            "circuit_id": "a",
            "candidate_id": "a-bad",
            "selection_status": "ok",
            "structural_cost": "0.9",
            "alphaq_dependency_core_area_ratio": "0.9",
            "tcount_after": "1",
            "depth_after": "9",
            "qasm_depth_ratio": "9",
        },
        {
            "circuit_id": "b",
            "candidate_id": "b-good",
            "selection_status": "ok",
            "structural_cost": "0.2",
            "alphaq_dependency_core_area_ratio": "0.2",
            "tcount_after": "5",
            "depth_after": "2",
            "qasm_depth_ratio": "2",
        },
        {
            "circuit_id": "b",
            "candidate_id": "b-bad",
            "selection_status": "ok",
            "structural_cost": "0.8",
            "alphaq_dependency_core_area_ratio": "0.8",
            "tcount_after": "1",
            "depth_after": "8",
            "qasm_depth_ratio": "8",
        },
    ]

    model = train_pairwise_mlp_ranker(
        rows,
        feature_columns=("depth_after", "qasm_depth_ratio"),
        hidden_size=4,
        epochs=700,
        learning_rate=0.03,
        weight_decay=1e-4,
        seed=5,
    )
    selected = [
        row["candidate_id"]
        for row in predictions_rows(rows, model, prediction_tolerance=0.0)
        if row["reranker_selected"]
    ]

    assert selected == ["a-good", "b-good"]


def test_alphatensor_reranker_tolerance_prefers_lower_tcount() -> None:
    rows = [
        {
            "circuit_id": "a",
            "candidate_id": "a0",
            "selection_status": "ok",
            "structural_cost": "0.10",
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
            "structural_cost": "0.12",
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


def test_alphatensor_reranker_tolerance_uses_alphaq_total_tiebreak() -> None:
    rows = [
        {
            "candidate_id": "lower-qasm",
            "tcount_after": "5",
            "alphaq_crossing_closure_count": "0",
            "alphaq_total_depth_ratio": "2.0",
            "qasm_depth_ratio": "1.0",
            "combo_index": "0",
        },
        {
            "candidate_id": "lower-alphaq-total",
            "tcount_after": "5",
            "alphaq_crossing_closure_count": "0",
            "alphaq_total_depth_ratio": "1.0",
            "qasm_depth_ratio": "2.0",
            "combo_index": "1",
        },
    ]
    predictions = np.array([0.10, 0.12])

    selected_index = select_with_prediction_tolerance(
        rows,
        predictions,
        prediction_tolerance=0.05,
    )

    assert rows[selected_index]["candidate_id"] == "lower-alphaq-total"


def test_alphatensor_reranker_ensemble_reports_uncertainty() -> None:
    rows = [
        {
            "circuit_id": "a",
            "candidate_id": "a0",
            "selection_status": "ok",
            "structural_cost": "0.1",
            "primary_nc_depth_ratio": "0.1",
            "tcount_after": "1",
            "depth_after": "1",
            "qasm_depth_ratio": "1",
        },
        {
            "circuit_id": "a",
            "candidate_id": "a1",
            "selection_status": "ok",
            "structural_cost": "0.9",
            "primary_nc_depth_ratio": "0.9",
            "tcount_after": "9",
            "depth_after": "9",
            "qasm_depth_ratio": "9",
        },
        {
            "circuit_id": "b",
            "candidate_id": "b0",
            "selection_status": "ok",
            "structural_cost": "0.2",
            "primary_nc_depth_ratio": "0.2",
            "tcount_after": "2",
            "depth_after": "2",
            "qasm_depth_ratio": "2",
        },
    ]

    model = train_ensemble(rows, hidden_size=3, epochs=25, ensemble_size=3, seed=11)
    predictions = predictions_rows(rows, model, prediction_tolerance=0.01)

    assert all("prediction_std" in row for row in predictions)
    assert all(float(row["prediction_std"]) >= 0.0 for row in predictions)


def test_alphatensor_reranker_baselines_report_proxy_regret() -> None:
    rows = [
        {
            "circuit_id": "a",
            "candidate_id": "structural",
            "selection_status": "ok",
            "structural_cost": "0.1",
            "primary_nc_depth_ratio": "0.1",
            "tcount_after": "5",
            "qasm_depth_ratio": "1",
            "zx_total_depth_ratio": "1",
            "tdepth_after": "3",
        },
        {
            "circuit_id": "a",
            "candidate_id": "tcount",
            "selection_status": "ok",
            "structural_cost": "0.4",
            "primary_nc_depth_ratio": "0.4",
            "tcount_after": "1",
            "qasm_depth_ratio": "2",
            "zx_total_depth_ratio": "2",
            "tdepth_after": "1",
        },
    ]

    baselines = baseline_rows(rows)
    tcount_baseline = next(row for row in baselines if row["objective"] == "tcount")

    assert tcount_baseline["selected_candidate_id"] == "tcount"
    assert tcount_baseline["primary_regret_vs_structural_best"] == 0.30000000000000004


def test_alphatensor_reranker_leave_one_circuit_out_reports_regret() -> None:
    rows = [
        {
            "circuit_id": "a",
            "candidate_id": "a0",
            "selection_status": "ok",
            "structural_cost": "0.2",
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
            "structural_cost": "0.3",
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
    assert eval_rows[0]["prediction_tolerance"] == 0.10


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
