from __future__ import annotations

from typing import Any, Mapping


PRIMARY_RATIO_KEY = "primary_nc_depth_ratio"
STRUCTURAL_STATUS_KEY = "structural_target_status"


def coerce_float(value: Any) -> float | None:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compute_structural_target_metrics(
    row: Mapping[str, Any],
    original_row: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if original_row is None:
        return _status_metrics("missing-original", "Missing original row for circuit.")

    candidate_zx_depth = coerce_float(row.get("zx_total_depth"))
    candidate_nc_depth = coerce_float(row.get("zx_best_nonclifford_depth"))
    original_zx_depth = coerce_float(original_row.get("zx_total_depth"))
    original_nc_depth = coerce_float(original_row.get("zx_best_nonclifford_depth"))

    if row.get("zx_split_status") != "ok" or original_row.get("zx_split_status") != "ok":
        return _status_metrics("missing-zx", "ZX splitting metrics are not available.")
    if (
        candidate_zx_depth is None
        or candidate_nc_depth is None
        or original_zx_depth is None
        or original_nc_depth is None
    ):
        return _status_metrics("missing-zx", "Required ZX depth fields are missing.")
    if original_zx_depth <= 0 or candidate_zx_depth < 0 or candidate_nc_depth < 0:
        return _status_metrics("invalid-depth", "Invalid ZX depth values.")

    denominator = max(original_zx_depth, 1.0)
    original_primary = original_nc_depth / denominator
    primary_ratio = candidate_nc_depth / denominator
    zx_total_depth_ratio = candidate_zx_depth / denominator
    qasm_depth_ratio = _ratio(row.get("depth_after"), original_row.get("depth_after"))
    primary_delta = primary_ratio - original_primary

    tcount_after = coerce_float(row.get("tcount_after"))
    original_tcount_after = coerce_float(original_row.get("tcount_after"))
    clifford_fraction = coerce_float(row.get("zx_best_clifford_fraction"))
    original_clifford_fraction = coerce_float(
        original_row.get("zx_best_clifford_fraction")
    )

    return {
        STRUCTURAL_STATUS_KEY: "ok",
        "structural_target_error": None,
        PRIMARY_RATIO_KEY: primary_ratio,
        "primary_nc_depth_delta_vs_original": primary_delta,
        "zx_total_depth_ratio": zx_total_depth_ratio,
        "qasm_depth_ratio": qasm_depth_ratio,
        "tcount_improves_primary_worsens": _is_tcount_improves_primary_worsens(
            tcount_after,
            original_tcount_after,
            primary_delta,
        ),
        "clifford_fraction_misleading": _is_clifford_fraction_misleading(
            clifford_fraction,
            original_clifford_fraction,
            primary_delta,
        ),
        "zx_depth_inflation": zx_total_depth_ratio > 1.25,
        "qasm_depth_inflation": (
            False if qasm_depth_ratio is None else qasm_depth_ratio > 1.25
        ),
    }


def attach_structural_target_metrics(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    original_by_circuit = {
        row["circuit_id"]: row
        for row in rows
        if row.get("method") == "original"
    }
    enriched_rows = []
    for row in rows:
        original = original_by_circuit.get(row["circuit_id"])
        enriched_rows.append(
            {
                **row,
                **compute_structural_target_metrics(row, original),
            }
        )
    return enriched_rows


def _status_metrics(status: str, error: str) -> dict[str, Any]:
    return {
        STRUCTURAL_STATUS_KEY: status,
        "structural_target_error": error,
        PRIMARY_RATIO_KEY: None,
        "primary_nc_depth_delta_vs_original": None,
        "zx_total_depth_ratio": None,
        "qasm_depth_ratio": None,
        "tcount_improves_primary_worsens": False,
        "clifford_fraction_misleading": False,
        "zx_depth_inflation": False,
        "qasm_depth_inflation": False,
    }


def _ratio(numerator: Any, denominator: Any) -> float | None:
    numerator_f = coerce_float(numerator)
    denominator_f = coerce_float(denominator)
    if numerator_f is None or denominator_f is None:
        return None
    if denominator_f <= 0 or numerator_f < 0:
        return None
    return numerator_f / max(denominator_f, 1.0)


def _is_tcount_improves_primary_worsens(
    tcount_after: float | None,
    original_tcount_after: float | None,
    primary_delta: float,
) -> bool:
    return (
        tcount_after is not None
        and original_tcount_after is not None
        and tcount_after < original_tcount_after
        and primary_delta > 0
    )


def _is_clifford_fraction_misleading(
    clifford_fraction: float | None,
    original_clifford_fraction: float | None,
    primary_delta: float,
) -> bool:
    return (
        clifford_fraction is not None
        and original_clifford_fraction is not None
        and clifford_fraction > original_clifford_fraction
        and primary_delta > 0
    )
