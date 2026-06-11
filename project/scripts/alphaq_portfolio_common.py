from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.structural_target import coerce_float

BASELINE_OBJECTIVE = "factor_count"
DEFAULT_TENSOR_OVERLAP_MAX_ACTIONS_PER_TARGET = 175


def inf_if_missing(value: Any) -> float:
    numeric = coerce_float(value)
    return float("inf") if numeric is None else numeric


def canonical_objective_key(
    row: dict[str, Any],
    *,
    tcount_field: str = "best_beam_tcount",
    qasm_depth_field: str = "best_beam_qasm_depth",
    primary_nc_field: str = "best_beam_primary_nc_depth_ratio",
    objective_field: str = "objective_variant",
) -> tuple[float, float, float, str]:
    return (
        inf_if_missing(row.get(tcount_field)),
        inf_if_missing(row.get(qasm_depth_field)),
        inf_if_missing(row.get(primary_nc_field)),
        str(row.get(objective_field, "")),
    )


def linear_span_dir_name(
    target: str,
    *,
    action_dictionary: str,
    max_action_weight: int,
    tensor_overlap_max_actions_per_target: int = DEFAULT_TENSOR_OVERLAP_MAX_ACTIONS_PER_TARGET,
    objective: str,
) -> str:
    return (
        f"{target}_{action_dictionary}_w{max_action_weight}_"
        f"k{tensor_overlap_max_actions_per_target}_{objective}"
    )


def linear_span_run_dir(
    linear_root: Path,
    target: str,
    *,
    objective_variant: str | None = None,
    action_dictionary: str = "low-weight",
    max_action_weight: int,
    tensor_overlap_max_actions_per_target: int = DEFAULT_TENSOR_OVERLAP_MAX_ACTIONS_PER_TARGET,
    objective: str,
) -> Path:
    root = linear_root / objective_variant if objective_variant else linear_root
    return root / linear_span_dir_name(
        target,
        action_dictionary=action_dictionary,
        max_action_weight=max_action_weight,
        tensor_overlap_max_actions_per_target=tensor_overlap_max_actions_per_target,
        objective=objective,
    )


def linear_span_manifest_path(
    linear_root: Path,
    target: str,
    *,
    objective_variant: str | None = None,
    action_dictionary: str = "low-weight",
    max_action_weight: int,
    tensor_overlap_max_actions_per_target: int = DEFAULT_TENSOR_OVERLAP_MAX_ACTIONS_PER_TARGET,
    objective: str,
) -> Path:
    return (
        linear_span_run_dir(
            linear_root,
            target,
            objective_variant=objective_variant,
            action_dictionary=action_dictionary,
            max_action_weight=max_action_weight,
            tensor_overlap_max_actions_per_target=tensor_overlap_max_actions_per_target,
            objective=objective,
        )
        / "candidate_factors_manifest.csv"
    )
