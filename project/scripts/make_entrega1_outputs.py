from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._analysis_common import DEFAULT_CSV_ROOT
from scripts._analysis_common import DEFAULT_FIGURES_ROOT
from scripts._analysis_common import DEFAULT_REPORTS_ROOT
from scripts._analysis_common import DEFAULT_RESULTS_ROOT
from scripts._analysis_common import NON_CLIFFORD_GATES
from scripts._analysis_common import ensure_dir
from scripts._analysis_common import load_qasm_circuit
from scripts._analysis_common import natural_sort_key
from scripts._analysis_common import normalize_circuit_to_basis
from scripts._analysis_common import write_csv_rows
from scripts._analysis_common import write_json
from scripts.structural_target import attach_structural_target_metrics
from scripts.zx_splitting import compute_zx_splitting_metrics_from_qasm


DEFAULT_CIRCUIT_IDS = (
    "mod_5_4",
    "gf_2pow2_mult",
    "qft_4",
    "hamming_weight_n4",
    "hamming_weight_n5",
)

HEATMAP_CIRCUIT_IDS = ("mod_5_4", "qft_4", "hamming_weight_n4")

METHOD_LABELS = {
    "original": "Original",
    "pyzx": "PyZX",
    "alphatensor_public": "AlphaTensor-public",
    "alphaq_tensor_v3": "AlphaQ tensor-v3",
    "alphaq_tensor_v3_phase_slack": "AlphaQ tensor-v3 phase-slack",
    "alphaq_final": "AlphaQ-final",
}

METHOD_COLORS = {
    "original": "#4D4D4D",
    "pyzx": "#0072B2",
    "alphatensor_public": "#009E73",
    "alphaq_tensor_v3": "#D55E00",
    "alphaq_tensor_v3_phase_slack": "#56B4E9",
    "alphaq_final": "#CC79A7",
}

ENTREGA_METHODS = (
    "original",
    "pyzx",
    "alphatensor_public",
    "alphaq_tensor_v3",
    "alphaq_tensor_v3_phase_slack",
    "alphaq_final",
)
CANDIDATE_METHODS = (
    "pyzx",
    "alphatensor_public",
    "alphaq_tensor_v3",
    "alphaq_tensor_v3_phase_slack",
    "alphaq_final",
)


@dataclass(frozen=True)
class TLocation:
    layer: int
    qubit: int
    gate_name: str


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def coerce_float(value: Any) -> float | None:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def coerce_int(value: Any) -> int | None:
    value_f = coerce_float(value)
    return None if value_f is None else int(value_f)


def project_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_normalized_qasm(path: Path) -> tuple[Any | None, str, str | None]:
    circuit = load_qasm_circuit(path)
    return normalize_circuit_to_basis(circuit)


def t_locations_by_scheduled_layer(circuit: Any) -> tuple[list[TLocation], int]:
    qubit_layers = [0 for _ in range(circuit.num_qubits)]
    locations: list[TLocation] = []
    for instruction in circuit.data:
        qubits = [circuit.find_bit(qubit).index for qubit in instruction.qubits]
        if not qubits:
            continue
        layer = max(qubit_layers[qubit] for qubit in qubits)
        if instruction.operation.name in NON_CLIFFORD_GATES:
            for qubit in qubits:
                locations.append(
                    TLocation(
                        layer=layer,
                        qubit=qubit,
                        gate_name=instruction.operation.name,
                    )
                )
        for qubit in qubits:
            qubit_layers[qubit] = layer + 1
    return locations, max(qubit_layers, default=0)


def compute_locality_metrics_from_circuit(circuit: Any) -> dict[str, Any]:
    locations, total_layers = t_locations_by_scheduled_layer(circuit)
    tcount = len(locations)
    if tcount == 0:
        return {
            "scheduled_layers": total_layers,
            "active_t_qubits": 0,
            "t_qubit_fraction": 0.0,
            "t_locality_span_qubits": 0,
            "t_locality_span_fraction": 0.0,
            "first_t_layer": None,
            "last_t_layer": None,
            "nonclifford_core_layers": 0,
            "nonclifford_core_area": 0,
            "nonclifford_core_density": 0.0,
            "clifford_prefix_ratio": 1.0,
            "clifford_suffix_ratio": 1.0,
            "heuristic_splitting_score": 1.0,
        }

    t_qubits = sorted({location.qubit for location in locations})
    t_layers = [location.layer for location in locations]
    first_layer = min(t_layers)
    last_layer = max(t_layers)
    core_layers = last_layer - first_layer + 1
    span_qubits = max(t_qubits) - min(t_qubits) + 1
    core_area = max(core_layers * span_qubits, 1)
    layer_denominator = max(total_layers, 1)

    prefix_ratio = first_layer / layer_denominator
    suffix_ratio = max(total_layers - last_layer - 1, 0) / layer_denominator
    t_qubit_fraction = len(t_qubits) / max(circuit.num_qubits, 1)
    span_fraction = span_qubits / max(circuit.num_qubits, 1)
    core_density = tcount / core_area
    splitting_score = (
        0.35 * (1.0 - t_qubit_fraction)
        + 0.25 * (1.0 - span_fraction)
        + 0.20 * min(core_density, 1.0)
        + 0.20 * max(prefix_ratio, suffix_ratio)
    )

    return {
        "scheduled_layers": total_layers,
        "active_t_qubits": len(t_qubits),
        "t_qubit_fraction": t_qubit_fraction,
        "t_locality_span_qubits": span_qubits,
        "t_locality_span_fraction": span_fraction,
        "first_t_layer": first_layer,
        "last_t_layer": last_layer,
        "nonclifford_core_layers": core_layers,
        "nonclifford_core_area": core_area,
        "nonclifford_core_density": core_density,
        "clifford_prefix_ratio": prefix_ratio,
        "clifford_suffix_ratio": suffix_ratio,
        "heuristic_splitting_score": splitting_score,
    }


def compute_locality_metrics_from_qasm(path: Path) -> dict[str, Any]:
    normalized, status, error = load_normalized_qasm(path)
    if normalized is None:
        return {
            "locality_status": status,
            "locality_error": error,
        }
    metrics = compute_locality_metrics_from_circuit(normalized)
    metrics["locality_status"] = status
    metrics["locality_error"] = error
    return metrics


def choose_alphatensor_public_row(rows_by_method: dict[str, dict[str, str]]) -> dict[str, str] | None:
    candidates = [
        rows_by_method.get("public_resynth_no_gadgets"),
        rows_by_method.get("public_resynth_gadgets"),
    ]
    ok_candidates = [
        row
        for row in candidates
        if row
        and row.get("method_status") == "ok"
        and coerce_int(row.get("tcount_after")) is not None
    ]
    if not ok_candidates:
        return None
    return min(
        ok_candidates,
        key=lambda row: (
            coerce_int(row.get("tcount_after")) or 10**12,
            0 if row["method"] == "public_resynth_no_gadgets" else 1,
        ),
    )


def build_entrega1_rows(
    final_rows: list[dict[str, str]],
    circuit_ids: tuple[str, ...],
) -> list[dict[str, Any]]:
    by_circuit: dict[str, dict[str, dict[str, str]]] = {}
    for row in final_rows:
        by_circuit.setdefault(row["circuit_id"], {})[row["method"]] = row

    output_rows: list[dict[str, Any]] = []
    for circuit_id in circuit_ids:
        rows_by_method = by_circuit.get(circuit_id, {})
        selected = [
            ("original", rows_by_method.get("original")),
            ("pyzx", rows_by_method.get("pyzx")),
            ("alphatensor_public", choose_alphatensor_public_row(rows_by_method)),
            ("alphaq_tensor_v3", rows_by_method.get("public_resynth_tensor_v3")),
            (
                "alphaq_tensor_v3_phase_slack",
                rows_by_method.get("public_resynth_tensor_v3_phase_slack"),
            ),
            ("alphaq_final", rows_by_method.get("public_resynth_alphaq_final")),
        ]
        for entrega1_method, source_row in selected:
            if source_row is None:
                output_rows.append(
                    {
                        "circuit_id": circuit_id,
                        "method": entrega1_method,
                        "method_label": METHOD_LABELS[entrega1_method],
                        "source_method": None,
                        "method_status": "not-available",
                    }
                )
                continue

            qasm_path = project_path(source_row.get("qasm_artifact_path"))
            locality_metrics = (
                compute_locality_metrics_from_qasm(qasm_path)
                if qasm_path and qasm_path.exists()
                else {"locality_status": "missing-qasm", "locality_error": "Missing QASM artifact."}
            )
            zx_metrics = (
                compute_zx_splitting_metrics_from_qasm(qasm_path)
                if qasm_path and qasm_path.exists()
                else {"zx_split_status": "missing-qasm", "zx_split_error": "Missing QASM artifact."}
            )
            row = {
                "circuit_id": circuit_id,
                "method": entrega1_method,
                "method_label": METHOD_LABELS[entrega1_method],
                "source_method": source_row["method"],
                "source": source_row.get("source"),
                "family": source_row.get("family"),
                "faixa": source_row.get("faixa"),
                "n_qubits": coerce_int(source_row.get("n_qubits")),
                "method_status": source_row.get("method_status"),
                "verify_status": source_row.get("verify_status"),
                "qasm_artifact_path": source_row.get("qasm_artifact_path"),
                "tcount_before": coerce_int(source_row.get("tcount_before")),
                "tcount_after": coerce_int(source_row.get("tcount_after")),
                "delta_t": coerce_int(source_row.get("delta_t")),
                "rel_gain_t": coerce_float(source_row.get("rel_gain_t")),
                "tdepth_before": coerce_int(source_row.get("tdepth_before")),
                "tdepth_after": coerce_int(source_row.get("tdepth_after")),
                "delta_tdepth": coerce_int(source_row.get("delta_tdepth")),
                "depth_after": coerce_int(source_row.get("normalized_qasm_depth")),
                "gate_count_after": coerce_int(source_row.get("normalized_qasm_size")),
                "rho_t": coerce_float(source_row.get("rho_t")),
                "rho_w": coerce_float(source_row.get("rho_w")),
                "n_clifford_blocks": coerce_int(source_row.get("n_clifford_blocks")),
                "n_nonclifford_blocks": coerce_int(source_row.get("n_nonclifford_blocks")),
                "avg_nonclifford_block_len": coerce_float(
                    source_row.get("avg_nonclifford_block_len")
                ),
                "hadamard_boundary_density": coerce_float(
                    source_row.get("hadamard_boundary_density")
                ),
                "selection_objective": source_row.get("selection_objective"),
                "tensor_v3_selection_status": source_row.get(
                    "tensor_v3_selection_status"
                ),
                "tensor_v3_selection_error": source_row.get(
                    "tensor_v3_selection_error"
                ),
                "tensor_v3_ranking_strategy": source_row.get(
                    "tensor_v3_ranking_strategy"
                ),
                "tensor_v3_profile": source_row.get("tensor_v3_profile"),
                "tensor_v3_mixed_slack": coerce_float(
                    source_row.get("tensor_v3_mixed_slack")
                ),
                "structural_cost": coerce_float(source_row.get("structural_cost")),
                "predicted_structural_cost": coerce_float(
                    source_row.get("predicted_structural_cost")
                ),
                "prediction_std": coerce_float(source_row.get("prediction_std")),
                "prediction_tolerance": coerce_float(
                    source_row.get("prediction_tolerance")
                ),
                "alphaq_target_status": source_row.get("alphaq_target_status"),
                "alphaq_nc_core_area_ratio": coerce_float(
                    source_row.get("alphaq_nc_core_area_ratio")
                ),
                "alphaq_dependency_core_area_ratio": coerce_float(
                    source_row.get("alphaq_dependency_core_area_ratio")
                ),
                "alphaq_nc_core_depth_ratio": coerce_float(
                    source_row.get("alphaq_nc_core_depth_ratio")
                ),
                "alphaq_nc_core_width_ratio": coerce_float(
                    source_row.get("alphaq_nc_core_width_ratio")
                ),
                "alphaq_dependency_core_depth_ratio": coerce_float(
                    source_row.get("alphaq_dependency_core_depth_ratio")
                ),
                "alphaq_dependency_core_width_ratio": coerce_float(
                    source_row.get("alphaq_dependency_core_width_ratio")
                ),
                "alphaq_total_depth_ratio": coerce_float(
                    source_row.get("alphaq_total_depth_ratio")
                ),
                "alphaq_total_area_ratio": coerce_float(
                    source_row.get("alphaq_total_area_ratio")
                ),
                "alphaq_crossing_closure_count": coerce_int(
                    source_row.get("alphaq_crossing_closure_count")
                ),
                "alphaq_nc_core_depth": coerce_int(
                    source_row.get("alphaq_nc_core_depth")
                ),
                "alphaq_nc_core_width": coerce_int(
                    source_row.get("alphaq_nc_core_width")
                ),
                "alphaq_nc_core_area": coerce_int(
                    source_row.get("alphaq_nc_core_area")
                ),
                "alphaq_dependency_core_depth": coerce_int(
                    source_row.get("alphaq_dependency_core_depth")
                ),
                "alphaq_dependency_core_width": coerce_int(
                    source_row.get("alphaq_dependency_core_width")
                ),
                "alphaq_dependency_core_area": coerce_int(
                    source_row.get("alphaq_dependency_core_area")
                ),
                "alphaq_dependency_boundary_edge_count": coerce_int(
                    source_row.get("alphaq_dependency_boundary_edge_count")
                ),
                "alphaq_dependency_component_count": coerce_int(
                    source_row.get("alphaq_dependency_component_count")
                ),
                "alphaq_dependency_chain_depth": coerce_int(
                    source_row.get("alphaq_dependency_chain_depth")
                ),
                "tensor_v3_status": source_row.get("tensor_v3_status"),
                "tensor_v3_partition_id": source_row.get("tensor_v3_partition_id"),
                "tensor_v3_partition_kind": source_row.get("tensor_v3_partition_kind"),
                "tensor_v3_tcount_tolerance": coerce_float(
                    source_row.get("tensor_v3_tcount_tolerance")
                ),
                "tensor_v3_mixed_excess_norm": coerce_float(
                    source_row.get("tensor_v3_mixed_excess_norm")
                ),
                "tensor_v3_mixed_auc_greedy_norm": coerce_float(
                    source_row.get("tensor_v3_mixed_auc_greedy_norm")
                ),
                "tensor_v3_singleton_bridge_count_norm": coerce_float(
                    source_row.get("tensor_v3_singleton_bridge_count_norm")
                ),
                "tensor_v3_score_lex": source_row.get("tensor_v3_score_lex"),
            }
            row.update(locality_metrics)
            row.update(zx_metrics)
            output_rows.append(row)
    return output_rows


def attach_formal_verification_status(
    rows: list[dict[str, Any]],
    verification_csv_path: Path | None,
) -> list[dict[str, Any]]:
    verification_index: dict[tuple[str, str], dict[str, str]] = {}
    if verification_csv_path is not None and verification_csv_path.exists():
        for verification_row in load_csv_rows(verification_csv_path):
            verification_index[
                (verification_row["circuit_id"], verification_row["method"])
            ] = verification_row

    enriched_rows = []
    for row in rows:
        source_method = row.get("source_method")
        if row["method"] == "original":
            enriched_rows.append(
                {
                    **row,
                    "formal_verification_status": "not-applicable",
                    "formal_verification_error": None,
                    "formal_proof_path": None,
                }
            )
            continue

        verification_row = verification_index.get((row["circuit_id"], source_method))
        if verification_row is None:
            enriched_rows.append(
                {
                    **row,
                    "formal_verification_status": "not-run",
                    "formal_verification_error": None,
                    "formal_proof_path": None,
                }
            )
            continue

        enriched_rows.append(
            {
                **row,
                "formal_verification_status": verification_row.get("verification_status"),
                "formal_verification_error": verification_row.get("verification_error"),
                "formal_proof_path": verification_row.get("proof_path"),
            }
        )
    return enriched_rows


def rows_for_formal_report(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidate_rows = [
        row
        for row in rows
        if row["method"] != "original"
        and row.get("formal_verification_status") == "equal"
    ]
    verified_circuit_ids = {row["circuit_id"] for row in candidate_rows}
    original_rows = [
        row
        for row in rows
        if row["method"] == "original" and row["circuit_id"] in verified_circuit_ids
    ]
    return sorted(
        [*original_rows, *candidate_rows],
        key=lambda row: (
            natural_sort_key(row["circuit_id"]),
            ENTREGA_METHODS.index(row["method"]),
        ),
    )


def rows_for_method(rows: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["method"] == method and row.get("method_status") == "ok"]


def grouped_values(
    rows: list[dict[str, Any]],
    circuit_ids: tuple[str, ...],
    methods: tuple[str, ...],
    key: str,
) -> dict[str, list[float]]:
    indexed = {(row["circuit_id"], row["method"]): row for row in rows}
    values: dict[str, list[float]] = {}
    for method in methods:
        method_values = []
        for circuit_id in circuit_ids:
            value = indexed.get((circuit_id, method), {}).get(key)
            method_values.append(float(value) if value not in (None, "") else np.nan)
        values[method] = method_values
    return values


def methods_with_finite_values(
    values: dict[str, list[float]],
    methods: tuple[str, ...],
) -> tuple[str, ...]:
    active = tuple(
        method
        for method in methods
        if any(np.isfinite(value) for value in values.get(method, []))
    )
    return active or methods


def save_tcount_plot(rows: list[dict[str, Any]], circuit_ids: tuple[str, ...], output_dir: Path) -> Path:
    methods = ENTREGA_METHODS
    values = grouped_values(rows, circuit_ids, methods, "tcount_after")
    active_methods = methods_with_finite_values(values, methods)
    x = np.arange(len(circuit_ids))
    width = 0.82 / len(active_methods)
    fig, ax = plt.subplots(figsize=(12, 5.8), constrained_layout=True)
    for offset, method in enumerate(active_methods):
        bar_positions = x + (offset - (len(active_methods) - 1) / 2) * width
        bars = ax.bar(
            bar_positions,
            values[method],
            width,
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
        )
        ax.bar_label(bars, fmt=lambda value: "" if np.isnan(value) else f"{value:.0f}", fontsize=8)
    ax.set_title("T-count: Original vs baselines vs AlphaQ tensor-v3")
    ax.set_ylabel("T-count normalizado")
    ax.set_xticks(x, circuit_ids, rotation=25, ha="right")
    ax.legend(frameon=False, ncols=min(len(active_methods), 4))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    png_path = output_dir / "tcount_comparison.png"
    fig.savefig(png_path, dpi=300)
    fig.savefig(output_dir / "tcount_comparison.pdf")
    plt.close(fig)
    return png_path


def save_core_plot(rows: list[dict[str, Any]], circuit_ids: tuple[str, ...], output_dir: Path) -> Path:
    methods = ENTREGA_METHODS
    active = grouped_values(rows, circuit_ids, methods, "active_t_qubits")
    core_layers = grouped_values(rows, circuit_ids, methods, "nonclifford_core_layers")
    active_methods = methods_with_finite_values(active, methods)
    x = np.arange(len(circuit_ids))
    width = 0.82 / len(active_methods)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    for axis, values, ylabel, title in [
        (axes[0], active, "qubits", "Qubits tocados por portas T/Tdg"),
        (axes[1], core_layers, "camadas", "Span temporal do nucleo nao-Clifford"),
    ]:
        for offset, method in enumerate(active_methods):
            axis.bar(
                x + (offset - (len(active_methods) - 1) / 2) * width,
                values[method],
                width,
                label=METHOD_LABELS[method],
                color=METHOD_COLORS[method],
            )
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, ncols=min(len(active_methods), 4))
    axes[1].set_xticks(x, circuit_ids, rotation=25, ha="right")
    png_path = output_dir / "nonclifford_core_comparison.png"
    fig.savefig(png_path, dpi=300)
    fig.savefig(output_dir / "nonclifford_core_comparison.pdf")
    plt.close(fig)
    return png_path


def save_reduction_vs_locality_plot(
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> Path:
    original_by_id = {
        row["circuit_id"]: row
        for row in rows
        if row["method"] == "original" and row.get("method_status") == "ok"
    }
    points = []
    for row in rows:
        if row["method"] not in CANDIDATE_METHODS or row.get("method_status") != "ok":
            continue
        original = original_by_id.get(row["circuit_id"])
        if not original:
            continue
        delta_t = coerce_float(row.get("delta_t"))
        before_area = coerce_float(original.get("nonclifford_core_area"))
        after_area = coerce_float(row.get("nonclifford_core_area"))
        if delta_t is None or before_area is None or after_area is None:
            continue
        points.append(
            {
                "circuit_id": row["circuit_id"],
                "method": row["method"],
                "delta_t": delta_t,
                "delta_core_area": before_area - after_area,
            }
        )

    fig, ax = plt.subplots(figsize=(7.5, 5.4), constrained_layout=True)
    for method in CANDIDATE_METHODS:
        subset = [point for point in points if point["method"] == method]
        ax.scatter(
            [point["delta_core_area"] for point in subset],
            [point["delta_t"] for point in subset],
            s=54,
            alpha=0.86,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
        for point in subset:
            if point["circuit_id"] in {"mod_5_4", "qcla_mod_7", "hamming_15_low"}:
                ax.annotate(
                    point["circuit_id"],
                    (point["delta_core_area"], point["delta_t"]),
                    fontsize=8,
                    alpha=0.85,
                )
    ax.axhline(0, color="#777777", linewidth=1, linestyle="--")
    ax.axvline(0, color="#777777", linewidth=1, linestyle="--")
    ax.set_title("Reducao de T-count vs mudanca estrutural")
    ax.set_xlabel("Reducao da area do nucleo nao-Clifford")
    ax.set_ylabel("Reducao de T-count")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    png_path = output_dir / "tcount_reduction_vs_core_area.png"
    fig.savefig(png_path, dpi=300)
    fig.savefig(output_dir / "tcount_reduction_vs_core_area.pdf")
    plt.close(fig)
    return png_path


def save_zx_splitting_plot(rows: list[dict[str, Any]], circuit_ids: tuple[str, ...], output_dir: Path) -> Path:
    methods = ENTREGA_METHODS
    values = grouped_values(rows, circuit_ids, methods, "zx_best_clifford_fraction")
    active_methods = methods_with_finite_values(values, methods)
    x = np.arange(len(circuit_ids))
    width = 0.82 / len(active_methods)

    fig, ax = plt.subplots(figsize=(12, 5.8), constrained_layout=True)
    for offset, method in enumerate(active_methods):
        heights = [
            100.0 * value if not np.isnan(value) else np.nan
            for value in values[method]
        ]
        bars = ax.bar(
            x + (offset - (len(active_methods) - 1) / 2) * width,
            heights,
            width,
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
        )
        ax.bar_label(
            bars,
            fmt=lambda value: "" if np.isnan(value) else f"{value:.0f}%",
            fontsize=8,
        )

    ax.set_title("Best ZX-detected Clifford section")
    ax.set_ylabel("fraction of circuit-like ZX depth")
    ax.set_xticks(x, circuit_ids, rotation=25, ha="right")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, ncols=min(len(active_methods), 4))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    png_path = output_dir / "zx_splitting_comparison.png"
    fig.savefig(png_path, dpi=300)
    fig.savefig(output_dir / "zx_splitting_comparison.pdf")
    plt.close(fig)
    return png_path


def t_heatmap_matrix(qasm_path: Path) -> tuple[np.ndarray, int, str | None]:
    normalized, status, error = load_normalized_qasm(qasm_path)
    if normalized is None:
        return np.zeros((1, 1), dtype=float), 1, error or status
    locations, total_layers = t_locations_by_scheduled_layer(normalized)
    matrix = np.zeros((normalized.num_qubits, max(total_layers, 1)), dtype=float)
    for location in locations:
        matrix[location.qubit, location.layer] = 1.0 if location.gate_name == "t" else 2.0
    return matrix, total_layers, None


def save_heatmap_for_circuit(
    rows: list[dict[str, Any]],
    circuit_id: str,
    output_dir: Path,
) -> Path | None:
    method_rows = [
        row
        for method in ENTREGA_METHODS
        for row in rows
        if row["circuit_id"] == circuit_id and row["method"] == method and row.get("method_status") == "ok"
    ]
    if not method_rows:
        return None

    fig, axes = plt.subplots(
        len(method_rows),
        1,
        figsize=(11, 2.6 * len(method_rows)),
        constrained_layout=True,
    )
    if len(method_rows) == 1:
        axes = [axes]

    cmap = plt.matplotlib.colors.ListedColormap(["#FFFFFF", "#D55E00", "#0072B2"])
    for axis, row in zip(axes, method_rows):
        qasm_path = project_path(row.get("qasm_artifact_path"))
        if qasm_path is None or not qasm_path.exists():
            matrix, _, error = np.zeros((1, 1), dtype=float), 1, "QASM ausente"
        else:
            matrix, _, error = t_heatmap_matrix(qasm_path)
        axis.imshow(matrix, cmap=cmap, vmin=0, vmax=2, aspect="auto", interpolation="nearest")
        axis.set_title(
            f"{METHOD_LABELS[row['method']]}: T-count={row.get('tcount_after')}, "
            f"span={row.get('nonclifford_core_layers')}"
        )
        axis.set_ylabel("qubit")
        axis.set_xlabel("camada escalonada")
        if error:
            axis.text(0.5, 0.5, error, transform=axis.transAxes, ha="center", va="center")
        elif np.count_nonzero(matrix) == 0:
            axis.text(
                0.5,
                0.5,
                "sem portas T/Tdg",
                transform=axis.transAxes,
                ha="center",
                va="center",
                color="#555555",
            )

    png_path = output_dir / f"t_gate_heatmap_{circuit_id}.png"
    fig.savefig(png_path, dpi=300)
    fig.savefig(output_dir / f"t_gate_heatmap_{circuit_id}.pdf")
    plt.close(fig)
    return png_path


def best_improvement_lines(rows: list[dict[str, Any]]) -> list[str]:
    candidates = [
        row
        for row in rows
        if row["method"] != "original"
        and row.get("method_status") == "ok"
        and coerce_float(row.get("delta_t")) is not None
    ]
    candidates.sort(
        key=lambda row: (
            -(coerce_float(row.get("delta_t")) or 0.0),
            natural_sort_key(row["circuit_id"]),
            row["method"],
        )
    )
    return [
        (
            f"- `{row['circuit_id']}` / {row['method_label']}: "
            f"T-count {row['tcount_before']} -> {row['tcount_after']} "
            f"(delta={row['delta_t']}, ganho relativo={row['rel_gain_t']:.3f})"
        )
        for row in candidates[:8]
        if row.get("rel_gain_t") is not None
    ]


def formal_verification_lines(verification_csv_path: Path | None) -> list[str]:
    if verification_csv_path is None or not verification_csv_path.exists():
        return ["- Verificacao formal ainda nao executada para os artefatos da Entrega 1."]

    rows = load_csv_rows(verification_csv_path)
    counts: dict[str, int] = {}
    for row in rows:
        status = row.get("verification_status", "unknown")
        counts[status] = counts.get(status, 0) + 1

    ordered = ["equal", "inconclusive", "timeout", "normalization-failed", "parse-error"]
    status_parts = [
        f"{status}={counts[status]}"
        for status in ordered
        if counts.get(status, 0)
    ]
    extra_parts = [
        f"{status}={count}"
        for status, count in sorted(counts.items())
        if status not in ordered
    ]
    return [
        f"- Pares candidatos verificados: {len(rows)}.",
        f"- Resultado: {', '.join(status_parts + extra_parts) if counts else 'sem pares'}.",
        f"- CSV de verificacao: `{verification_csv_path}`.",
        "- A tabela da Entrega 1 inclui `formal_verification_status` e `formal_proof_path` por metodo.",
    ]


def paper_reproduction_lines(paper_comparison_csv_path: Path | None) -> list[str]:
    if paper_comparison_csv_path is None or not paper_comparison_csv_path.exists():
        return ["- Reproducao tensorial do artigo ainda nao executada."]

    rows = load_csv_rows(paper_comparison_csv_path)
    matches = sum(row.get("match") == "true" for row in rows)
    tensor_ok = sum(row.get("all_tensor_equal") == "true" for row in rows)
    sources = sorted({row.get("source", "unknown") for row in rows})
    return [
        f"- Linhas comparadas com o artigo: {len(rows)}.",
        f"- Matches exatos de T-count efetivo: {matches}/{len(rows)}.",
        f"- Linhas com decomposicoes selecionadas validadas tensorialmente: {tensor_ok}/{len(rows)}.",
        f"- Fontes cobertas: {', '.join(f'`{source}`' for source in sources)}.",
        f"- CSV de reproducao: `{paper_comparison_csv_path}`.",
    ]


def write_report(
    rows: list[dict[str, Any]],
    output_path: Path,
    figure_paths: dict[str, Path | list[Path]],
    table_path: Path,
    verification_csv_path: Path | None,
    paper_comparison_csv_path: Path | None,
) -> Path:
    ok_alpha = len(rows_for_method(rows, "alphatensor_public"))
    ok_tensor_v3 = len(rows_for_method(rows, "alphaq_tensor_v3"))
    ok_tensor_v3_phase_slack = len(
        rows_for_method(rows, "alphaq_tensor_v3_phase_slack")
    )
    ok_alphaq_final = len(rows_for_method(rows, "alphaq_final"))
    ok_pyzx = len(rows_for_method(rows, "pyzx"))
    selected_ids = sorted({row["circuit_id"] for row in rows}, key=natural_sort_key)
    limitations = [
        "O metodo `AlphaTensor-public` usa replay/ressintese de decomposicoes publicas, nao treino completo do artigo.",
        "A verificacao formal usa QASM normalizado para a base Clifford+T local; candidatos `inconclusive` ou `timeout` ficam fora da tabela principal.",
        "A deteccao ZX implementada e conservadora: usa a fronteira em diagramas circuit-like e fecha recursivamente portas de dois qubits que cruzam a separacao.",
        "O T-depth reportado segue a implementacao local atual e deve ser tratado como estimativa simples para a Entrega 1.",
    ]

    heatmaps = figure_paths.get("heatmaps", [])
    heatmap_lines = [
        f"- ![Heatmap {path.stem}]({path})"
        for path in heatmaps
        if isinstance(path, Path)
    ]
    text = [
        "# Entrega 1 - Resultados experimentais preliminares",
        "",
        "## Escopo executado",
        "",
        f"- Circuitos selecionados: {', '.join(f'`{item}`' for item in selected_ids)}.",
        f"- PyZX disponivel para {ok_pyzx}/{len(selected_ids)} circuitos.",
        f"- AlphaTensor-public disponivel para {ok_alpha}/{len(selected_ids)} circuitos.",
        f"- AlphaQ tensor-v3 disponivel para {ok_tensor_v3}/{len(selected_ids)} circuitos.",
        (
            "- AlphaQ tensor-v3 phase-slack disponivel para "
            f"{ok_tensor_v3_phase_slack}/{len(selected_ids)} circuitos."
        ),
        f"- AlphaQ-final disponivel para {ok_alphaq_final}/{len(selected_ids)} circuitos.",
        f"- Tabela consolidada: `{table_path}`.",
        "",
        "## Figuras geradas",
        "",
        f"- ![T-count comparison]({figure_paths['tcount']})",
        f"- ![Non-Clifford core comparison]({figure_paths['core']})",
        f"- ![ZX splitting comparison]({figure_paths['zx']})",
        f"- ![Reduction vs core area]({figure_paths['scatter']})",
        "",
        "## Heatmaps de portas T/Tdg",
        "",
        *heatmap_lines,
        "",
        "## Principais resultados preliminares",
        "",
        *best_improvement_lines(rows),
        "",
        "## Reproducao fiel do artigo",
        "",
        *paper_reproduction_lines(paper_comparison_csv_path),
        "",
        "## Verificacao formal",
        "",
        *formal_verification_lines(verification_csv_path),
        "",
        "## Leitura para discussao",
        "",
        (
            "Os resultados ja permitem uma Entrega 1 experimental: PyZX reduz T-count em todos os "
            "casos selecionados em que ha ganho, enquanto o replay publico AlphaTensor frequentemente "
            "melhora PyZX nos circuitos com decomposicoes publicas compativeis. A variante AlphaQ "
            "tensor-v3 aparece como selecao AlphaQuantum-only orientada pela geometria tensorial, "
            "sem usar ZX/feynver no loop de escolha. As figuras estruturais e a fronteira detectada "
            "em ZX mostram que a reducao de T-count nem sempre coincide com uma melhora monotona da "
            "separacao Clifford/nao-Clifford, o que sustenta a motivacao de medir estrutura e nao "
            "apenas contagem."
        ),
        "",
        "## Limitacoes documentadas",
        "",
        *[f"- {item}" for item in limitations],
        "",
        "## Proxima acao sugerida",
        "",
        (
            "Para escrever o draft, usar esta tabela e estas figuras nas secoes de Experimentos, "
            "Resultados preliminares e Discussao; a secao Solucao Proposta deve explicar que a "
            "contribuicao da Entrega 1 e a camada de analise estrutural sobre baselines publicos."
        ),
        "",
    ]
    ensure_dir(output_path.parent)
    output_path.write_text("\n".join(text), encoding="utf-8")
    return output_path


def write_formal_report(
    rows: list[dict[str, Any]],
    output_path: Path,
    figure_paths: dict[str, Path | list[Path]],
    table_path: Path,
    audit_table_path: Path,
) -> Path:
    selected_ids = sorted({row["circuit_id"] for row in rows}, key=natural_sort_key)
    text = [
        "# Entrega 1 - Tabela principal formalmente verificada",
        "",
        "## Criterio de inclusao",
        "",
        (
            "Esta tabela remove candidatos com `timeout`, `inconclusive` ou qualquer outro status "
            "diferente de `equal` na verificacao formal. A tabela completa de auditoria continua "
            "preservada para documentar tudo que foi reproduzido."
        ),
        "",
        f"- Circuitos com pelo menos um candidato formalmente verificado: {', '.join(f'`{item}`' for item in selected_ids)}.",
        f"- Tabela principal sem timeouts: `{table_path}`.",
        f"- Tabela completa de auditoria: `{audit_table_path}`.",
        "",
        "## Figuras principais",
        "",
        f"- ![T-count comparison]({figure_paths['tcount']})",
        f"- ![Non-Clifford core comparison]({figure_paths['core']})",
        f"- ![ZX splitting comparison]({figure_paths['zx']})",
        f"- ![Reduction vs core area]({figure_paths['scatter']})",
        "",
        "## Observacao para o texto",
        "",
        (
            "A tabela principal se limita aos benchmarks com candidatos formalmente verificados. "
            "Candidatos AlphaTensor-public inconclusivos em circuitos como `cuccaro_adder_n3` e "
            "`vbe_adder_3` permanecem apenas na tabela completa de auditoria."
        ),
        "",
    ]
    ensure_dir(output_path.parent)
    output_path.write_text("\n".join(text), encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Entrega 1 tables, figures and report.")
    parser.add_argument("--final-metrics-csv", type=Path, default=DEFAULT_CSV_ROOT / "final_metrics.csv")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV_ROOT / "entrega1_metrics.csv")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_CSV_ROOT / "entrega1_metrics.json")
    parser.add_argument(
        "--formal-output-csv",
        type=Path,
        default=DEFAULT_CSV_ROOT / "entrega1_metrics_formally_verified.csv",
    )
    parser.add_argument(
        "--formal-output-json",
        type=Path,
        default=DEFAULT_CSV_ROOT / "entrega1_metrics_formally_verified.json",
    )
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURES_ROOT / "entrega1")
    parser.add_argument(
        "--formal-figure-dir",
        type=Path,
        default=DEFAULT_FIGURES_ROOT / "entrega1_formal",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORTS_ROOT / "entrega1_experimental_summary.md",
    )
    parser.add_argument(
        "--formal-report-path",
        type=Path,
        default=DEFAULT_REPORTS_ROOT / "entrega1_formally_verified_summary.md",
    )
    parser.add_argument(
        "--verification-csv",
        type=Path,
        default=DEFAULT_RESULTS_ROOT / "verification" / "entrega1" / "verification_summary.csv",
    )
    parser.add_argument(
        "--paper-comparison-csv",
        type=Path,
        default=DEFAULT_RESULTS_ROOT
        / "reproducibility"
        / "paper"
        / "paper_benchmark_comparison.csv",
    )
    parser.add_argument("--circuit-id", action="append", dest="circuit_ids", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    circuit_ids = tuple(args.circuit_ids) if args.circuit_ids else DEFAULT_CIRCUIT_IDS
    rows = attach_structural_target_metrics(
        build_entrega1_rows(load_csv_rows(args.final_metrics_csv), circuit_ids)
    )
    rows = attach_formal_verification_status(rows, args.verification_csv)
    formal_rows = rows_for_formal_report(rows)
    ensure_dir(args.figure_dir)
    ensure_dir(args.formal_figure_dir)
    write_csv_rows(rows, args.output_csv)
    write_csv_rows(formal_rows, args.formal_output_csv)
    write_json(
        {
            "final_metrics_csv": str(args.final_metrics_csv),
            "output_csv": str(args.output_csv),
            "circuit_ids": list(circuit_ids),
            "num_rows": len(rows),
            "verification_csv": str(args.verification_csv),
            "paper_comparison_csv": str(args.paper_comparison_csv),
        },
        args.output_json,
    )
    write_json(
        {
            "final_metrics_csv": str(args.final_metrics_csv),
            "output_csv": str(args.formal_output_csv),
            "circuit_ids": sorted(
                {row["circuit_id"] for row in formal_rows},
                key=natural_sort_key,
            ),
            "num_rows": len(formal_rows),
            "verification_csv": str(args.verification_csv),
            "criterion": "original rows plus candidate rows with formal_verification_status == equal",
        },
        args.formal_output_json,
    )

    tcount_path = save_tcount_plot(rows, circuit_ids, args.figure_dir)
    core_path = save_core_plot(rows, circuit_ids, args.figure_dir)
    zx_path = save_zx_splitting_plot(rows, circuit_ids, args.figure_dir)
    scatter_path = save_reduction_vs_locality_plot(rows, args.figure_dir)
    heatmap_paths = [
        path
        for circuit_id in HEATMAP_CIRCUIT_IDS
        if (path := save_heatmap_for_circuit(rows, circuit_id, args.figure_dir)) is not None
    ]
    report_path = write_report(
        rows,
        args.report_path,
        {
            "tcount": tcount_path,
            "core": core_path,
            "zx": zx_path,
            "scatter": scatter_path,
            "heatmaps": heatmap_paths,
        },
        args.output_csv,
        args.verification_csv,
        args.paper_comparison_csv,
    )
    formal_circuit_ids = tuple(
        sorted({row["circuit_id"] for row in formal_rows}, key=natural_sort_key)
    )
    formal_tcount_path = save_tcount_plot(
        formal_rows, formal_circuit_ids, args.formal_figure_dir
    )
    formal_core_path = save_core_plot(
        formal_rows, formal_circuit_ids, args.formal_figure_dir
    )
    formal_zx_path = save_zx_splitting_plot(
        formal_rows, formal_circuit_ids, args.formal_figure_dir
    )
    formal_scatter_path = save_reduction_vs_locality_plot(
        formal_rows, args.formal_figure_dir
    )
    formal_report_path = write_formal_report(
        formal_rows,
        args.formal_report_path,
        {
            "tcount": formal_tcount_path,
            "core": formal_core_path,
            "zx": formal_zx_path,
            "scatter": formal_scatter_path,
        },
        args.formal_output_csv,
        args.output_csv,
    )
    print(
        {
            "output_csv": str(args.output_csv),
            "output_json": str(args.output_json),
            "formal_output_csv": str(args.formal_output_csv),
            "formal_output_json": str(args.formal_output_json),
            "figure_dir": str(args.figure_dir),
            "formal_figure_dir": str(args.formal_figure_dir),
            "report_path": str(report_path),
            "formal_report_path": str(formal_report_path),
            "heatmaps": [str(path) for path in heatmap_paths],
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
