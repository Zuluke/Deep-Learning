from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from scripts._analysis_common import compute_metrics_from_qasm_path
from scripts.alphaq_border_proxy import ALPHAQ_BORDER_COST_KEY
from scripts.alphaq_border_proxy import ALPHAQ_DEPENDENCY_COST_KEY
from scripts.alphaq_border_proxy import ALPHAQ_BORDER_STATUS_KEY
from scripts.alphaq_border_proxy import compute_alphaq_border_metrics_from_qasm
from scripts.alphaq_border_proxy import compute_alphaq_border_target_metrics
from scripts.structural_target import coerce_float
from scripts.structural_target import compute_structural_target_metrics


INF = float("inf")


def circuit_selection_row_from_qasm(path: Path) -> dict[str, Any]:
    qasm_metrics = compute_metrics_from_qasm_path(path)
    zx_metrics = _safe_zx_splitting_metrics_from_qasm(path)
    alphaq_metrics = compute_alphaq_border_metrics_from_qasm(path)
    return {
        "qasm_path": str(path),
        "tcount_after": qasm_metrics.get("tcount"),
        "tdepth_after": qasm_metrics.get("tdepth"),
        "depth_after": qasm_metrics.get("normalized_qasm_depth"),
        "gate_count_after": qasm_metrics.get("normalized_qasm_size"),
        "rho_t": qasm_metrics.get("rho_t"),
        "rho_w": qasm_metrics.get("rho_w"),
        "n_clifford_blocks": qasm_metrics.get("n_clifford_blocks"),
        "n_nonclifford_blocks": qasm_metrics.get("n_nonclifford_blocks"),
        "avg_nonclifford_block_len": qasm_metrics.get("avg_nonclifford_block_len"),
        "hadamard_boundary_density": qasm_metrics.get("hadamard_boundary_density"),
        "tdepth_over_tcount": qasm_metrics.get("tdepth_over_tcount"),
        **zx_metrics,
        **alphaq_metrics,
    }


def compute_selection_metrics(
    candidate_row: Mapping[str, Any],
    original_row: Mapping[str, Any],
) -> dict[str, Any]:
    target_metrics = compute_structural_target_metrics(candidate_row, original_row)
    alphaq_target_metrics = compute_alphaq_border_target_metrics(
        candidate_row, original_row
    )
    status = alphaq_target_metrics.get("alphaq_target_status")
    tcount_ratio = _ratio(
        candidate_row.get("tcount_after"),
        original_row.get("tcount_after"),
    )
    tdepth_ratio = _ratio(
        candidate_row.get("tdepth_after"),
        original_row.get("tdepth_after"),
    )
    gate_count_ratio = _ratio(
        candidate_row.get("gate_count_after"),
        original_row.get("gate_count_after"),
    )
    qasm_depth_ratio = _ratio(
        candidate_row.get("depth_after"),
        original_row.get("depth_after"),
    )
    structural_cost = alphaq_target_metrics.get(ALPHAQ_DEPENDENCY_COST_KEY)
    if structural_cost is None:
        structural_cost = alphaq_target_metrics.get(ALPHAQ_BORDER_COST_KEY)
    return {
        "selection_status": "ok" if status == "ok" else status,
        "selection_error": alphaq_target_metrics.get("alphaq_target_error"),
        "structural_cost": structural_cost if status == "ok" else None,
        "tcount_ratio": tcount_ratio,
        "tdepth_ratio": tdepth_ratio,
        "gate_count_ratio": gate_count_ratio,
        "tcount_after": candidate_row.get("tcount_after"),
        "tdepth_after": candidate_row.get("tdepth_after"),
        "depth_after": candidate_row.get("depth_after"),
        "gate_count_after": candidate_row.get("gate_count_after"),
        "rho_t": candidate_row.get("rho_t"),
        "rho_w": candidate_row.get("rho_w"),
        "n_clifford_blocks": candidate_row.get("n_clifford_blocks"),
        "n_nonclifford_blocks": candidate_row.get("n_nonclifford_blocks"),
        "avg_nonclifford_block_len": candidate_row.get("avg_nonclifford_block_len"),
        "hadamard_boundary_density": candidate_row.get("hadamard_boundary_density"),
        "tdepth_over_tcount": candidate_row.get("tdepth_over_tcount"),
        **target_metrics,
        "qasm_depth_ratio": qasm_depth_ratio,
        **_copy_alphaq_border_fields(candidate_row),
        **alphaq_target_metrics,
    }


def compute_selection_metrics_from_qasm(
    *,
    candidate_qasm: Path,
    original_row: Mapping[str, Any],
) -> dict[str, Any]:
    candidate_row = circuit_selection_row_from_qasm(candidate_qasm)
    return compute_selection_metrics(candidate_row, original_row)


def structural_selection_key(
    metrics: Mapping[str, Any], tie_breaker: int = 0
) -> tuple[float, float, float, float, float, float, int]:
    if metrics.get("selection_status") != "ok":
        return (INF, INF, INF, INF, INF, INF, tie_breaker)
    return (
        _finite(metrics.get("structural_cost")),
        _finite(metrics.get("alphaq_dependency_boundary_edge_count")),
        _finite(metrics.get("alphaq_crossing_closure_count")),
        _finite(metrics.get("tcount_after")),
        _finite(metrics.get("alphaq_total_depth_ratio")),
        _finite(metrics.get("qasm_depth_ratio")),
        tie_breaker,
    )


def tcount_selection_key(metrics: Mapping[str, Any], tie_breaker: int = 0) -> tuple[float, float, int]:
    return (
        _finite(metrics.get("tcount")),
        _finite(metrics.get("tdepth")),
        tie_breaker,
    )


def _ratio(numerator: Any, denominator: Any) -> float | None:
    numerator_f = coerce_float(numerator)
    denominator_f = coerce_float(denominator)
    if numerator_f is None or denominator_f is None or numerator_f < 0 or denominator_f <= 0:
        return None
    return numerator_f / max(denominator_f, 1.0)


def _finite(value: Any) -> float:
    numeric = coerce_float(value)
    return INF if numeric is None else numeric


def _copy_alphaq_border_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key.startswith("alphaq_") or key == ALPHAQ_BORDER_STATUS_KEY
    }


def _safe_zx_splitting_metrics_from_qasm(path: Path) -> dict[str, Any]:
    try:
        from scripts.zx_splitting import compute_zx_splitting_metrics_from_qasm
    except Exception as exc:  # pragma: no cover - optional benchmark dependency
        return {
            "zx_split_status": "unavailable",
            "zx_split_error": str(exc),
        }
    return compute_zx_splitting_metrics_from_qasm(path)
