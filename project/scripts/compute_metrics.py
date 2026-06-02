from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._analysis_common import DEFAULT_COMPILE_ROOT
from scripts._analysis_common import DEFAULT_CSV_ROOT
from scripts._analysis_common import DEFAULT_PYZX_ROOT
from scripts._analysis_common import DEFAULT_RESYNTH_ROOT
from scripts._analysis_common import compute_metrics_from_qasm_path
from scripts._analysis_common import ensure_dir
from scripts._analysis_common import natural_sort_key
from scripts._analysis_common import tensor_size_from_directory
from scripts._analysis_common import to_relative
from scripts._analysis_common import write_csv_rows
from scripts._analysis_common import write_json
from scripts._manifest import append_command


DEFAULT_STRUCTURAL_RESYNTH_ROOT = PROJECT_ROOT / "results" / "public_resynth_structural"
DEFAULT_TENSOR_V3_RESYNTH_ROOT = PROJECT_ROOT / "results" / "public_resynth_tensor_v3"
DEFAULT_TENSOR_V3_PHASE_SLACK_RESYNTH_ROOT = (
    PROJECT_ROOT / "results" / "public_resynth_tensor_v3_phase_slack"
)
DEFAULT_TENSOR_V3_PHASE_SLACK_AGGRESSIVE_RESYNTH_ROOT = (
    PROJECT_ROOT / "results" / "public_resynth_tensor_v3_phase_slack_aggressive"
)
DEFAULT_TENSOR_V3_PHASE_SLACK_GUARDED_RESYNTH_ROOT = (
    PROJECT_ROOT / "results" / "public_resynth_tensor_v3_phase_slack_guarded"
)
DEFAULT_RERANKER_RESYNTH_ROOT = PROJECT_ROOT / "results" / "public_resynth_reranker"
DEFAULT_ALPHAQ_FINAL_RESYNTH_ROOT = PROJECT_ROOT / "results" / "public_resynth_alphaq_final"
STRUCTURAL_PUBLIC_METHODS = ("public_resynth_structural",)
TENSOR_V3_PUBLIC_METHODS = ("public_resynth_tensor_v3",)
TENSOR_V3_PHASE_SLACK_PUBLIC_METHODS = ("public_resynth_tensor_v3_phase_slack",)
TENSOR_V3_EXTRA_PUBLIC_METHODS = (
    "public_resynth_tensor_v3_phase_slack_aggressive",
    "public_resynth_tensor_v3_phase_slack_guarded",
)
RERANKER_PUBLIC_METHODS = ("public_resynth_reranker", "public_resynth_alphaq_final")
STRUCTURAL_SUMMARY_COLUMNS = (
    "selection_objective",
    "selection_status",
    "selection_error",
    "tensor_v3_selection_status",
    "tensor_v3_selection_error",
    "tensor_v3_ranking_strategy",
    "tensor_v3_profile",
    "tensor_v3_mixed_slack",
    "structural_cost",
    "alphaq_target_status",
    "alphaq_target_error",
    "alphaq_nc_core_area_ratio",
    "alphaq_dependency_core_area_ratio",
    "alphaq_nc_core_area_delta_vs_original",
    "alphaq_dependency_core_area_delta_vs_original",
    "alphaq_nc_core_depth_ratio",
    "alphaq_nc_core_width_ratio",
    "alphaq_dependency_core_depth_ratio",
    "alphaq_dependency_core_width_ratio",
    "alphaq_total_depth_ratio",
    "alphaq_total_area_ratio",
    "alphaq_border_status",
    "alphaq_border_error",
    "alphaq_total_depth",
    "alphaq_total_width",
    "alphaq_total_area",
    "alphaq_nc_core_depth",
    "alphaq_nc_core_width",
    "alphaq_nc_core_area",
    "alphaq_core_tcount",
    "alphaq_crossing_closure_count",
    "alphaq_dependency_core_depth",
    "alphaq_dependency_core_width",
    "alphaq_dependency_core_area",
    "alphaq_dependency_core_size",
    "alphaq_dependency_core_tcount",
    "alphaq_dependency_entangling_count",
    "alphaq_dependency_closure_rounds",
    "alphaq_dependency_internal_edge_count",
    "alphaq_dependency_boundary_edge_count",
    "alphaq_dependency_component_count",
    "alphaq_dependency_largest_component_size",
    "alphaq_dependency_largest_component_fraction",
    "alphaq_dependency_chain_depth",
    "alphaq_dependency_edge_density",
    "alphaq_left_nc_core_depth",
    "alphaq_left_nc_core_width",
    "alphaq_left_nc_core_area",
    "alphaq_left_crossing_closure_count",
    "alphaq_right_nc_core_depth",
    "alphaq_right_nc_core_width",
    "alphaq_right_nc_core_area",
    "alphaq_right_crossing_closure_count",
    "alphaq_prefix_clifford_depth",
    "alphaq_suffix_clifford_depth",
    "alphaq_clifford_shaved_depth_fraction",
    "primary_nc_depth_ratio",
    "zx_total_depth_ratio",
    "qasm_depth_ratio",
    "tcount_ratio",
    "tcount_after_selection",
    "tdepth_after_selection",
    "combo_index",
)
TENSOR_V3_SUMMARY_COLUMNS = (
    *STRUCTURAL_SUMMARY_COLUMNS,
    "tensor_v3_status",
    "tensor_v3_error",
    "tensor_v3_partition_id",
    "tensor_v3_partition_kind",
    "tensor_v3_tcount_tolerance",
    "tensor_v3_target_mixed_weight",
    "tensor_v3_gadget_mixed_weight",
    "tensor_v3_gadget_mixed_weight_norm",
    "tensor_v3_mixed_excess_norm",
    "tensor_v3_mixed_auc_original",
    "tensor_v3_mixed_auc_original_norm",
    "tensor_v3_mixed_auc_greedy",
    "tensor_v3_mixed_auc_greedy_norm",
    "tensor_v3_singleton_bridge_count",
    "tensor_v3_singleton_bridge_count_norm",
    "tensor_v3_mixed_group_count",
    "tensor_v3_mixed_group_count_norm",
    "tensor_v3_num_factor_groups",
    "tensor_v3_factor_count",
    "tensor_v3_score_lex",
    "tensor_v3_stable_factor_hash",
    "tensor_v3_partition_metrics_json",
    "guarded_source_method",
    "guarded_source_profile",
    "guarded_uses_aggressive",
    "guarded_qasm_depth_gain_threshold",
    "guarded_mixed_drop_fraction_threshold",
    "qasm_depth_gain_aggressive_vs_conservative",
    "mixed_drop_aggressive_vs_conservative",
    "mixed_drop_fraction_aggressive_vs_conservative",
    "guarded_delta_tcount_vs_conservative",
    "guarded_delta_primary_vs_conservative",
    "guarded_global_oracle_regret",
    "guarded_is_global_oracle",
)
RERANKER_SUMMARY_COLUMNS = (
    *STRUCTURAL_SUMMARY_COLUMNS,
    "source_candidate_id",
    "predicted_structural_cost",
    "predicted_primary_nc_depth_ratio",
    "prediction_std",
    "prediction_tolerance",
    "prediction_margin_from_best",
    "model_json",
)


def load_csv_rows(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
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
    if value in (None, "", "None"):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def project_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def classify_faixa(tensor_size: int | None) -> str:
    if tensor_size is None:
        return "unknown"
    if tensor_size <= 8:
        return "S"
    if tensor_size <= 20:
        return "M"
    return "L"


def qasm_metrics_or_stub(qasm_path: Path | None) -> dict[str, Any]:
    if qasm_path is None or not qasm_path.exists():
        return {
            "comparability_status": "none",
            "normalization_status": "not-run",
            "normalization_error": "Missing QASM artifact.",
        }
    return compute_metrics_from_qasm_path(qasm_path)


def row_from_metrics(
    *,
    circuit_row: dict[str, str],
    method: str,
    method_status: str,
    qasm_path: Path | None,
    runtime_sec: float | None,
    verify_status: str,
    tensor_size_no_quizx: int | None,
    tensor_size_quizx: int | None,
    tcount_before: int | None,
    tdepth_before: int | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = qasm_metrics_or_stub(qasm_path)
    tcount_after = coerce_int(metrics.get("tcount"))
    tdepth_after = coerce_int(metrics.get("tdepth"))
    delta_t = None
    rel_gain_t = None
    delta_tdepth = None
    if tcount_before is not None and tcount_after is not None:
        delta_t = tcount_before - tcount_after
        rel_gain_t = delta_t / max(tcount_before, 1)
    if tdepth_before is not None and tdepth_after is not None:
        delta_tdepth = tdepth_before - tdepth_after

    row = {
        "circuit_id": circuit_row["circuit_id"],
        "method": method,
        "source": circuit_row["source"],
        "family": circuit_row["family"],
        "faixa": classify_faixa(tensor_size_no_quizx),
        "n_qubits": coerce_int(circuit_row.get("n_qubits")),
        "tensor_size": tensor_size_no_quizx,
        "tensor_size_no_quizx": tensor_size_no_quizx,
        "tensor_size_quizx": tensor_size_quizx,
        "verify_status": verify_status,
        "runtime_sec": runtime_sec,
        "method_status": method_status,
        "comparability_status": metrics.get("comparability_status", "none"),
        "normalization_status": metrics.get("normalization_status", "not-run"),
        "normalization_error": metrics.get("normalization_error"),
        "qasm_artifact_path": (
            None if qasm_path is None or not qasm_path.exists() else to_relative(qasm_path)
        ),
        "tcount_before": tcount_before,
        "tcount_after": tcount_after,
        "delta_t": delta_t,
        "rel_gain_t": rel_gain_t,
        "tdepth_before": tdepth_before,
        "tdepth_after": tdepth_after,
        "delta_tdepth": delta_tdepth,
        "rho_t": metrics.get("rho_t"),
        "rho_w": metrics.get("rho_w"),
        "rho_w_lambda_5": metrics.get("rho_w_lambda_5"),
        "rho_w_lambda_10": metrics.get("rho_w_lambda_10"),
        "rho_w_lambda_20": metrics.get("rho_w_lambda_20"),
        "n_clifford_blocks": metrics.get("n_clifford_blocks"),
        "n_nonclifford_blocks": metrics.get("n_nonclifford_blocks"),
        "avg_nonclifford_block_len": metrics.get("avg_nonclifford_block_len"),
        "hadamard_boundary_density": metrics.get("hadamard_boundary_density"),
        "hadamard_boundary_density_w1": metrics.get("hadamard_boundary_density_w1"),
        "hadamard_boundary_density_w2": metrics.get("hadamard_boundary_density_w2"),
        "tdepth_over_tcount": metrics.get("tdepth_over_tcount"),
        "normalized_qasm_num_qubits": metrics.get("normalized_qasm_num_qubits"),
        "normalized_qasm_depth": metrics.get("normalized_qasm_depth"),
        "normalized_qasm_size": metrics.get("normalized_qasm_size"),
    }
    if extra:
        row.update(extra)
    return row


def latest_demo_logs(log_dir: Path) -> dict[tuple[str, bool], dict[str, Any]]:
    latest: dict[tuple[str, bool], tuple[Path, dict[str, Any]]] = {}
    for path in sorted(log_dir.glob("*.json"), key=lambda item: item.name):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        mode = payload.get("mode")
        targets = payload.get("target_circuits") or []
        if mode not in {"quick", "control"} or len(targets) != 1:
            continue
        key = (str(targets[0]).lower(), bool(payload.get("use_gadgets", False)))
        latest[key] = (path, payload)
    return {key: {"path": value[0], "payload": value[1]} for key, value in latest.items()}


def summary_extra(public_row: dict[str, str] | None, columns: tuple[str, ...]) -> dict[str, Any]:
    if public_row is None:
        return {column: None for column in columns}
    return {column: public_row.get(column) for column in columns}


def build_rows(
    *,
    inventory_rows: list[dict[str, str]],
    pyzx_rows: list[dict[str, str]],
    compile_quizx_rows: list[dict[str, str]],
    public_resynth_rows: list[dict[str, str]],
    structural_resynth_rows: list[dict[str, str]],
    tensor_v3_resynth_rows: list[dict[str, str]],
    tensor_v3_phase_slack_resynth_rows: list[dict[str, str]],
    tensor_v3_extra_resynth_rows: list[dict[str, str]],
    reranker_resynth_rows: list[dict[str, str]],
    demo_log_dir: Path,
) -> list[dict[str, Any]]:
    pyzx_by_id = {row["circuit_id"]: row for row in pyzx_rows}
    compile_quizx_by_id = {row["circuit_id"]: row for row in compile_quizx_rows}
    public_resynth_by_key = {
        (row["circuit_id"], row["method"]): row for row in public_resynth_rows
    }
    structural_resynth_by_key = {
        (row["circuit_id"], row["method"]): row for row in structural_resynth_rows
    }
    structural_circuit_ids = {row["circuit_id"] for row in structural_resynth_rows}
    tensor_v3_resynth_by_key = {
        (row["circuit_id"], row["method"]): row for row in tensor_v3_resynth_rows
    }
    tensor_v3_circuit_ids = {row["circuit_id"] for row in tensor_v3_resynth_rows}
    tensor_v3_phase_slack_resynth_by_key = {
        (row["circuit_id"], row["method"]): row
        for row in tensor_v3_phase_slack_resynth_rows
    }
    tensor_v3_phase_slack_circuit_ids = {
        row["circuit_id"] for row in tensor_v3_phase_slack_resynth_rows
    }
    tensor_v3_extra_resynth_by_key = {
        (row["circuit_id"], row["method"]): row for row in tensor_v3_extra_resynth_rows
    }
    tensor_v3_extra_methods_by_circuit: dict[str, set[str]] = {}
    for row in tensor_v3_extra_resynth_rows:
        tensor_v3_extra_methods_by_circuit.setdefault(row["circuit_id"], set()).add(
            row["method"]
        )
    reranker_resynth_by_key = {
        (row["circuit_id"], row["method"]): row for row in reranker_resynth_rows
    }
    reranker_circuit_ids = {row["circuit_id"] for row in reranker_resynth_rows}
    demo_by_key = latest_demo_logs(demo_log_dir)
    rows: list[dict[str, Any]] = []

    for circuit_row in sorted(inventory_rows, key=lambda row: natural_sort_key(row["circuit_id"])):
        compile_dir = (
            PROJECT_ROOT / circuit_row["vendored_compile_dir"]
            if circuit_row.get("vendored_compile_dir")
            else None
        )
        tensor_size_no_quizx = (
            tensor_size_from_directory(compile_dir) if compile_dir and compile_dir.exists() else None
        )
        vendored_hopt = (
            compile_dir / f"{circuit_row['circuit_id']}.hopt.qasm"
            if compile_dir is not None
            else None
        )

        original_qasm = PROJECT_ROOT / circuit_row["qasm_path"]
        original_metrics = compute_metrics_from_qasm_path(original_qasm)
        original_tcount = coerce_int(original_metrics.get("tcount"))
        original_tdepth = coerce_int(original_metrics.get("tdepth"))

        tensor_size_quizx = None
        compile_quizx_row = compile_quizx_by_id.get(circuit_row["circuit_id"])
        if compile_quizx_row:
            tensor_size_quizx = coerce_int(compile_quizx_row.get("tensor_size"))

        rows.append(
            row_from_metrics(
                circuit_row=circuit_row,
                method="original",
                method_status="ok",
                qasm_path=original_qasm,
                runtime_sec=None,
                verify_status="not-run",
                tensor_size_no_quizx=tensor_size_no_quizx,
                tensor_size_quizx=tensor_size_quizx,
                tcount_before=original_tcount,
                tdepth_before=original_tdepth,
            )
        )

        pyzx_row = pyzx_by_id.get(circuit_row["circuit_id"])
        pyzx_qasm = None
        pyzx_status = "not-run"
        pyzx_runtime = None
        if pyzx_row:
            pyzx_status = pyzx_row.get("status", "unknown")
            pyzx_runtime = coerce_float(pyzx_row.get("runtime_sec"))
            output_qasm_path = pyzx_row.get("output_qasm_path")
            if output_qasm_path:
                pyzx_qasm = PROJECT_ROOT / output_qasm_path
        rows.append(
            row_from_metrics(
                circuit_row=circuit_row,
                method="pyzx",
                method_status=pyzx_status,
                qasm_path=pyzx_qasm if pyzx_status == "ok" else None,
                runtime_sec=pyzx_runtime,
                verify_status="not-run",
                tensor_size_no_quizx=tensor_size_no_quizx,
                tensor_size_quizx=tensor_size_quizx,
                tcount_before=original_tcount,
                tdepth_before=original_tdepth,
            )
        )

        rows.append(
            row_from_metrics(
                circuit_row=circuit_row,
                method="compile_no_quizx",
                method_status="ok" if vendored_hopt and vendored_hopt.exists() else "not-available",
                qasm_path=vendored_hopt if vendored_hopt and vendored_hopt.exists() else None,
                runtime_sec=None,
                verify_status="not-run",
                tensor_size_no_quizx=tensor_size_no_quizx,
                tensor_size_quizx=tensor_size_quizx,
                tcount_before=original_tcount,
                tdepth_before=original_tdepth,
            )
        )

        compile_quizx_qasm = None
        compile_quizx_status = "not-run"
        compile_quizx_runtime = None
        if compile_quizx_row:
            compile_quizx_status = compile_quizx_row.get("status", "unknown")
            compile_quizx_runtime = coerce_float(compile_quizx_row.get("runtime_sec"))
            output_dir = compile_quizx_row.get("output_dir")
            if output_dir:
                compile_quizx_qasm = Path(output_dir) / f"{circuit_row['circuit_id']}.hopt.qasm"
        rows.append(
            row_from_metrics(
                circuit_row=circuit_row,
                method="compile_quizx",
                method_status=compile_quizx_status,
                qasm_path=compile_quizx_qasm if compile_quizx_status in {"ok", "skipped-existing"} else None,
                runtime_sec=compile_quizx_runtime,
                verify_status="not-run",
                tensor_size_no_quizx=tensor_size_no_quizx,
                tensor_size_quizx=tensor_size_quizx,
                tcount_before=original_tcount,
                tdepth_before=original_tdepth,
            )
        )

        for method in ("public_resynth_no_gadgets", "public_resynth_gadgets"):
            public_row = public_resynth_by_key.get((circuit_row["circuit_id"], method))
            public_qasm = None
            public_status = "not-run"
            if public_row:
                public_status = public_row.get("status", "unknown")
                assembled_qasm = public_row.get("assembled_qasm_path")
                if assembled_qasm:
                    public_qasm = project_path(assembled_qasm)
            rows.append(
                row_from_metrics(
                    circuit_row=circuit_row,
                    method=method,
                    method_status=public_status,
                    qasm_path=public_qasm if public_status == "ok" else None,
                    runtime_sec=None,
                    verify_status="not-run",
                    tensor_size_no_quizx=tensor_size_no_quizx,
                    tensor_size_quizx=tensor_size_quizx,
                    tcount_before=original_tcount,
                    tdepth_before=original_tdepth,
                )
            )

        if circuit_row["circuit_id"] in structural_circuit_ids:
            for method in STRUCTURAL_PUBLIC_METHODS:
                structural_row = structural_resynth_by_key.get(
                    (circuit_row["circuit_id"], method)
                )
                structural_qasm = None
                structural_status = "not-run"
                if structural_row:
                    structural_status = structural_row.get("status", "unknown")
                    assembled_qasm = structural_row.get("assembled_qasm_path")
                    if assembled_qasm:
                        structural_qasm = project_path(assembled_qasm)
                rows.append(
                    row_from_metrics(
                        circuit_row=circuit_row,
                        method=method,
                        method_status=structural_status,
                        qasm_path=structural_qasm if structural_status == "ok" else None,
                        runtime_sec=None,
                        verify_status="not-run",
                        tensor_size_no_quizx=tensor_size_no_quizx,
                        tensor_size_quizx=tensor_size_quizx,
                        tcount_before=original_tcount,
                        tdepth_before=original_tdepth,
                        extra=summary_extra(structural_row, STRUCTURAL_SUMMARY_COLUMNS),
                    )
                )

        if circuit_row["circuit_id"] in tensor_v3_circuit_ids:
            for method in TENSOR_V3_PUBLIC_METHODS:
                tensor_v3_row = tensor_v3_resynth_by_key.get(
                    (circuit_row["circuit_id"], method)
                )
                tensor_v3_qasm = None
                tensor_v3_status = "not-run"
                if tensor_v3_row:
                    tensor_v3_status = tensor_v3_row.get("status", "unknown")
                    assembled_qasm = tensor_v3_row.get("assembled_qasm_path")
                    if assembled_qasm:
                        tensor_v3_qasm = project_path(assembled_qasm)
                rows.append(
                    row_from_metrics(
                        circuit_row=circuit_row,
                        method=method,
                        method_status=tensor_v3_status,
                        qasm_path=tensor_v3_qasm if tensor_v3_status == "ok" else None,
                        runtime_sec=None,
                        verify_status="not-run",
                        tensor_size_no_quizx=tensor_size_no_quizx,
                        tensor_size_quizx=tensor_size_quizx,
                        tcount_before=original_tcount,
                        tdepth_before=original_tdepth,
                        extra=summary_extra(tensor_v3_row, TENSOR_V3_SUMMARY_COLUMNS),
                    )
                )

        if circuit_row["circuit_id"] in tensor_v3_phase_slack_circuit_ids:
            for method in TENSOR_V3_PHASE_SLACK_PUBLIC_METHODS:
                tensor_v3_phase_slack_row = tensor_v3_phase_slack_resynth_by_key.get(
                    (circuit_row["circuit_id"], method)
                )
                tensor_v3_phase_slack_qasm = None
                tensor_v3_phase_slack_status = "not-run"
                if tensor_v3_phase_slack_row:
                    tensor_v3_phase_slack_status = tensor_v3_phase_slack_row.get(
                        "status", "unknown"
                    )
                    assembled_qasm = tensor_v3_phase_slack_row.get("assembled_qasm_path")
                    if assembled_qasm:
                        tensor_v3_phase_slack_qasm = project_path(assembled_qasm)
                rows.append(
                    row_from_metrics(
                        circuit_row=circuit_row,
                        method=method,
                        method_status=tensor_v3_phase_slack_status,
                        qasm_path=(
                            tensor_v3_phase_slack_qasm
                            if tensor_v3_phase_slack_status == "ok"
                            else None
                        ),
                        runtime_sec=None,
                        verify_status="not-run",
                        tensor_size_no_quizx=tensor_size_no_quizx,
                        tensor_size_quizx=tensor_size_quizx,
                        tcount_before=original_tcount,
                        tdepth_before=original_tdepth,
                        extra=summary_extra(
                            tensor_v3_phase_slack_row,
                            TENSOR_V3_SUMMARY_COLUMNS,
                        ),
                    )
                )

        extra_methods = sorted(
            tensor_v3_extra_methods_by_circuit.get(circuit_row["circuit_id"], set())
        )
        for method in extra_methods:
            tensor_v3_extra_row = tensor_v3_extra_resynth_by_key.get(
                (circuit_row["circuit_id"], method)
            )
            tensor_v3_extra_qasm = None
            tensor_v3_extra_status = "not-run"
            if tensor_v3_extra_row:
                tensor_v3_extra_status = tensor_v3_extra_row.get("status", "unknown")
                assembled_qasm = tensor_v3_extra_row.get("assembled_qasm_path")
                if assembled_qasm:
                    tensor_v3_extra_qasm = project_path(assembled_qasm)
            rows.append(
                row_from_metrics(
                    circuit_row=circuit_row,
                    method=method,
                    method_status=tensor_v3_extra_status,
                    qasm_path=(
                        tensor_v3_extra_qasm
                        if tensor_v3_extra_status == "ok"
                        else None
                    ),
                    runtime_sec=None,
                    verify_status="not-run",
                    tensor_size_no_quizx=tensor_size_no_quizx,
                    tensor_size_quizx=tensor_size_quizx,
                    tcount_before=original_tcount,
                    tdepth_before=original_tdepth,
                    extra=summary_extra(tensor_v3_extra_row, TENSOR_V3_SUMMARY_COLUMNS),
                )
            )

        if circuit_row["circuit_id"] in reranker_circuit_ids:
            for method in RERANKER_PUBLIC_METHODS:
                reranker_row = reranker_resynth_by_key.get(
                    (circuit_row["circuit_id"], method)
                )
                reranker_qasm = None
                reranker_status = "not-run"
                if reranker_row:
                    reranker_status = reranker_row.get("status", "unknown")
                    assembled_qasm = reranker_row.get("assembled_qasm_path")
                    if assembled_qasm:
                        reranker_qasm = project_path(assembled_qasm)
                rows.append(
                    row_from_metrics(
                        circuit_row=circuit_row,
                        method=method,
                        method_status=reranker_status,
                        qasm_path=reranker_qasm if reranker_status == "ok" else None,
                        runtime_sec=None,
                        verify_status="not-run",
                        tensor_size_no_quizx=tensor_size_no_quizx,
                        tensor_size_quizx=tensor_size_quizx,
                        tcount_before=original_tcount,
                        tdepth_before=original_tdepth,
                        extra=summary_extra(reranker_row, RERANKER_SUMMARY_COLUMNS),
                    )
                )

        for use_gadgets, method in (
            (False, "demo_control_no_gadgets"),
            (True, "demo_control_gadgets"),
        ):
            key = (circuit_row["circuit_id"], use_gadgets)
            payload = demo_by_key.get(key)
            extra = {}
            if payload is None:
                rows.append(
                    row_from_metrics(
                        circuit_row=circuit_row,
                        method=method,
                        method_status="not-run",
                        qasm_path=None,
                        runtime_sec=None,
                        verify_status="not-run",
                        tensor_size_no_quizx=tensor_size_no_quizx,
                        tensor_size_quizx=tensor_size_quizx,
                        tcount_before=original_tcount,
                        tdepth_before=original_tdepth,
                        extra={"demo_log_path": None},
                    )
                )
                continue

            log_payload = payload["payload"]
            best_tcount = coerce_int(log_payload.get("best_tcount_observed"))
            delta_t = None
            rel_gain_t = None
            if original_tcount is not None and best_tcount is not None:
                delta_t = original_tcount - best_tcount
                rel_gain_t = delta_t / max(original_tcount, 1)
            extra = {
                "demo_log_path": to_relative(payload["path"]),
                "backend": log_payload.get("backend"),
                "devices": json.dumps(log_payload.get("devices", [])),
                "best_tcount_observed": best_tcount,
                "reference_best_tcount": coerce_int(log_payload.get("reference_best_tcount")),
                "matched_reference": log_payload.get("matched_reference"),
                "tcount_after": best_tcount,
                "delta_t": delta_t,
                "rel_gain_t": rel_gain_t,
            }
            demo_row = row_from_metrics(
                circuit_row=circuit_row,
                method=method,
                method_status="partial-log-only",
                qasm_path=None,
                runtime_sec=None,
                verify_status="not-run",
                tensor_size_no_quizx=tensor_size_no_quizx,
                tensor_size_quizx=tensor_size_quizx,
                tcount_before=original_tcount,
                tdepth_before=original_tdepth,
                extra=extra,
            )
            demo_row["tcount_after"] = best_tcount
            demo_row["delta_t"] = delta_t
            demo_row["rel_gain_t"] = rel_gain_t
            rows.append(demo_row)

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute canonical per-(circuit, method) metrics.")
    parser.add_argument("--inventory-csv", type=Path, required=True)
    parser.add_argument("--pyzx-summary-csv", type=Path, default=DEFAULT_PYZX_ROOT / "pyzx_summary.csv")
    parser.add_argument(
        "--compile-quizx-summary-csv",
        type=Path,
        default=DEFAULT_COMPILE_ROOT / "compile_on_summary.csv",
    )
    parser.add_argument(
        "--public-resynth-summary-csv",
        type=Path,
        default=DEFAULT_RESYNTH_ROOT / "public_resynth_summary.csv",
    )
    parser.add_argument(
        "--structural-resynth-summary-csv",
        type=Path,
        default=DEFAULT_STRUCTURAL_RESYNTH_ROOT / "public_resynth_summary.csv",
    )
    parser.add_argument(
        "--tensor-v3-resynth-summary-csv",
        type=Path,
        default=DEFAULT_TENSOR_V3_RESYNTH_ROOT / "public_resynth_summary.csv",
    )
    parser.add_argument(
        "--tensor-v3-phase-slack-resynth-summary-csv",
        type=Path,
        default=DEFAULT_TENSOR_V3_PHASE_SLACK_RESYNTH_ROOT
        / "public_resynth_summary.csv",
    )
    parser.add_argument(
        "--tensor-v3-extra-resynth-summary-csv",
        type=Path,
        action="append",
        default=[],
        help="Additional tensor-v3 profile summary CSVs to include as audit methods.",
    )
    parser.add_argument(
        "--reranker-resynth-summary-csv",
        type=Path,
        default=DEFAULT_RERANKER_RESYNTH_ROOT / "public_resynth_summary.csv",
    )
    parser.add_argument(
        "--alphaq-final-resynth-summary-csv",
        type=Path,
        default=DEFAULT_ALPHAQ_FINAL_RESYNTH_ROOT / "public_resynth_summary.csv",
    )
    parser.add_argument(
        "--demo-log-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "logs" / "demo",
    )
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV_ROOT / "final_metrics.csv")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_CSV_ROOT / "final_metrics.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_dir(args.output_csv.parent)
    inventory_rows = load_csv_rows(args.inventory_csv)
    pyzx_rows = load_csv_rows(args.pyzx_summary_csv)
    compile_quizx_rows = load_csv_rows(args.compile_quizx_summary_csv)
    public_resynth_rows = load_csv_rows(args.public_resynth_summary_csv)
    structural_resynth_rows = load_csv_rows(args.structural_resynth_summary_csv)
    tensor_v3_resynth_rows = load_csv_rows(args.tensor_v3_resynth_summary_csv)
    tensor_v3_phase_slack_resynth_rows = load_csv_rows(
        args.tensor_v3_phase_slack_resynth_summary_csv
    )
    tensor_v3_extra_resynth_rows = [
        row
        for path in args.tensor_v3_extra_resynth_summary_csv
        for row in load_csv_rows(path)
    ]
    reranker_resynth_rows = [
        *load_csv_rows(args.reranker_resynth_summary_csv),
        *load_csv_rows(args.alphaq_final_resynth_summary_csv),
    ]

    rows = build_rows(
        inventory_rows=inventory_rows,
        pyzx_rows=pyzx_rows,
        compile_quizx_rows=compile_quizx_rows,
        public_resynth_rows=public_resynth_rows,
        structural_resynth_rows=structural_resynth_rows,
        tensor_v3_resynth_rows=tensor_v3_resynth_rows,
        tensor_v3_phase_slack_resynth_rows=tensor_v3_phase_slack_resynth_rows,
        tensor_v3_extra_resynth_rows=tensor_v3_extra_resynth_rows,
        reranker_resynth_rows=reranker_resynth_rows,
        demo_log_dir=args.demo_log_dir,
    )
    write_csv_rows(rows, args.output_csv)
    write_json(
        {
            "inventory_csv": str(args.inventory_csv),
            "pyzx_summary_csv": str(args.pyzx_summary_csv),
            "compile_quizx_summary_csv": str(args.compile_quizx_summary_csv),
            "public_resynth_summary_csv": str(args.public_resynth_summary_csv),
            "structural_resynth_summary_csv": str(args.structural_resynth_summary_csv),
            "tensor_v3_resynth_summary_csv": str(args.tensor_v3_resynth_summary_csv),
            "tensor_v3_phase_slack_resynth_summary_csv": str(
                args.tensor_v3_phase_slack_resynth_summary_csv
            ),
            "tensor_v3_extra_resynth_summary_csv": [
                str(path) for path in args.tensor_v3_extra_resynth_summary_csv
            ],
            "reranker_resynth_summary_csv": str(args.reranker_resynth_summary_csv),
            "alphaq_final_resynth_summary_csv": str(args.alphaq_final_resynth_summary_csv),
            "demo_log_dir": str(args.demo_log_dir),
            "num_rows": len(rows),
        },
        args.output_json,
    )
    extra_summary_args = "".join(
        f"--tensor-v3-extra-resynth-summary-csv {path} "
        for path in args.tensor_v3_extra_resynth_summary_csv
    )
    append_command(
        {
            "tool": "compute_metrics.py",
            "command": (
                f"{sys.executable} scripts/compute_metrics.py --inventory-csv {args.inventory_csv} "
                f"--structural-resynth-summary-csv {args.structural_resynth_summary_csv} "
                f"--tensor-v3-resynth-summary-csv {args.tensor_v3_resynth_summary_csv} "
                f"--tensor-v3-phase-slack-resynth-summary-csv "
                f"{args.tensor_v3_phase_slack_resynth_summary_csv} "
                f"{extra_summary_args}"
                f"--reranker-resynth-summary-csv {args.reranker_resynth_summary_csv} "
                f"--alphaq-final-resynth-summary-csv {args.alphaq_final_resynth_summary_csv} "
                f"--output-csv {args.output_csv} --output-json {args.output_json}"
            ),
            "cwd": str(PROJECT_ROOT),
            "inventory_csv": str(args.inventory_csv),
            "output_csv": str(args.output_csv),
            "output_json": str(args.output_json),
            "exit_code": 0,
        }
    )
    print(json.dumps({"output_csv": str(args.output_csv), "output_json": str(args.output_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
