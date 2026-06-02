from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ABLATION_CSV = PROJECT_ROOT / "results" / "csv" / "split_reward_ablation.csv"
DEFAULT_BENCHMARK_CSV = (
    PROJECT_ROOT / "results" / "csv" / "entrega1_metrics_formally_verified.csv"
)
DEFAULT_LOWWEIGHT_SWEEP_CSV = (
    PROJECT_ROOT / "results" / "csv" / "split_reward_target_sweep_lowweight_probe.csv"
)
DEFAULT_CANONICAL_SWEEP_CSV = (
    PROJECT_ROOT / "results" / "csv" / "split_reward_target_sweep_canonical_probe.csv"
)
DEFAULT_GADGET_CLOSURE_SWEEP_CSVS = (
    PROJECT_ROOT
    / "results"
    / "csv"
    / "split_reward_target_sweep_gadgetclosure_b2_canonical_probe.csv",
    PROJECT_ROOT
    / "results"
    / "csv"
    / "split_reward_target_sweep_gadgetclosure_hamming_canonical_probe.csv",
)
DEFAULT_EXTERNAL_SPLIT_REWARD_METRICS = (
    PROJECT_ROOT
    / "results"
    / "alphaq_split_reward_external"
    / "mod_5_4_v1_tiebreak_materialized"
    / "structural_metrics_from_qasm_original.json"
)
DEFAULT_LOG_DIR = PROJECT_ROOT / "results" / "logs" / "demo"
DEFAULT_OUTPUT_CSV = (
    PROJECT_ROOT / "results" / "csv" / "split_reward_core_validation.csv"
)
DEFAULT_REPORT = (
    PROJECT_ROOT / "results" / "reports" / "split_reward_core_validation.md"
)
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "results" / "figures" / "split_reward_core"

CONTROL_RUNS = [
    {
        "label": "Baseline none, basis mix",
        "run_key": "mod_5_4_none_1000_b32_m16",
        "scope": "mod_5_4_budget_real",
        "basis_regime": "basis-mix",
        "budget_source": "none",
    },
    {
        "label": "v1_tiebreak, basis mix",
        "run_key": "mod_5_4_v1_tiebreak_budget2_lam0005_basis_mix_1000_b32_m16_with_cob",
        "scope": "mod_5_4_budget_real",
        "basis_regime": "basis-mix",
        "budget_source": "explicit:mod_5_4=2",
    },
    {
        "label": "Baseline none, canonical",
        "run_key": "mod_5_4_none_forcecanonical_1000_b32_m16",
        "scope": "mod_5_4_canonical_control",
        "basis_regime": "canonical-only",
        "budget_source": "none",
    },
    {
        "label": "v1_tiebreak, canonical",
        "run_key": "mod_5_4_v1_tiebreak_budget2_lam0005_1000_b32_m16",
        "scope": "mod_5_4_canonical_control",
        "basis_regime": "canonical-only",
        "budget_source": "explicit:mod_5_4=2",
    },
]

INDIVIDUAL_RUNS = [
    {
        "label": "gf_2pow2_mult none, 1000 steps",
        "run_key": "gf_2pow2_mult_none_1000_b32_m16",
        "target": "gf_2pow2_mult",
        "mode": "none",
    },
    {
        "label": "gf_2pow2_mult v1, 1000 steps",
        "run_key": "gf_2pow2_mult_v1_1000_b32_m16",
        "target": "gf_2pow2_mult",
        "mode": "v1",
    },
    {
        "label": "gf_2pow2_mult v2_progress, low-weight<=2, 1000 steps",
        "run_key": "gf_2pow2_mult_v2_progress_loww2_1000_b16_m8",
        "target": "gf_2pow2_mult",
        "mode": "v2_progress",
    },
    {
        "label": "hamming_weight_n4 v2_progress, low-weight<=2, 1000 steps",
        "run_key": "hamming_weight_n4_v2_progress_loww2_1000_b16_m8",
        "target": "hamming_weight_n4",
        "mode": "v2_progress",
    },
]

ACTION_DICTIONARY_RUNS = [
    {
        "label": "hamming_weight_n5 v1, full action space, 100 steps",
        "run_key": "hamming_weight_n5_v1_full_smoke",
        "target": "hamming_weight_n5",
        "mode": "v1",
        "scope": "action_dictionary_probe",
    },
    {
        "label": "hamming_weight_n5 none, low-weight<=2, 100 steps",
        "run_key": "hamming_weight_n5_none_loww2_smoke",
        "target": "hamming_weight_n5",
        "mode": "none",
        "scope": "action_dictionary_probe",
    },
    {
        "label": "hamming_weight_n5 v1, low-weight<=2, 100 steps",
        "run_key": "hamming_weight_n5_v1_loww2_smoke",
        "target": "hamming_weight_n5",
        "mode": "v1",
        "scope": "action_dictionary_probe",
    },
    {
        "label": "hamming_weight_n5 v1, low-weight<=3, 100 steps",
        "run_key": "hamming_weight_n5_v1_loww3_smoke",
        "target": "hamming_weight_n5",
        "mode": "v1",
        "scope": "action_dictionary_probe",
    },
    {
        "label": "hamming_weight_n5 v1, low-weight<=2, 1000 steps",
        "run_key": "hamming_weight_n5_v1_loww2_1000_b16_m8",
        "target": "hamming_weight_n5",
        "mode": "v1",
        "scope": "action_dictionary_probe",
    },
]

MASKED_SPLIT4_RUNS = [
    {
        "label": "split4 none, low-weight<=2, masked, 1000 steps",
        "run_key": "split4_none_loww2_mask_1000_b16_m8",
        "mode": "none",
    },
    {
        "label": "split4 v1, low-weight<=2, masked, 1000 steps",
        "run_key": "split4_v1_loww2_mask_1000_b16_m8",
        "mode": "v1",
    },
    {
        "label": "split4 v2_progress, low-weight<=2, masked, 1000 steps",
        "run_key": "split4_v2_progress_loww2_mask_1000_b16_m8",
        "mode": "v2_progress",
    },
]

TENSOR_OVERLAP_RUNS = [
    {
        "label": "split4 none, tensor-overlap64/w5, masked, 1000 steps",
        "run_key": "split4_none_tensoroverlap64_w5_mask_1000_b16_m8",
        "mode": "none",
    },
    {
        "label": "split4 v2_progress, tensor-overlap64/w5, masked, 1000 steps",
        "run_key": "split4_v2_progress_tensoroverlap64_w5_mask_1000_b16_m8",
        "mode": "v2_progress",
    },
]

BENCHMARK_TARGETS = [
    "mod_5_4",
    "gf_2pow2_mult",
    "hamming_weight_n4",
    "hamming_weight_n5",
    "qft_4",
]
BENCHMARK_METHODS = [
    "original",
    "pyzx",
    "alphatensor_public",
    "alphaq_tensor_v3",
    "alphaq_tensor_v3_phase_slack",
    "alphaq_final",
    "alphaq_split_reward_v1_tiebreak",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment_scope",
        "label",
        "mode",
        "target",
        "basis_regime",
        "budget_source",
        "best_effective_t_cost",
        "best_return",
        "best_return_effective_t_cost",
        "best_return_residual_weight",
        "best_return_num_moves",
        "avg_return_final",
        "avg_split_sum_rewards_final",
        "avg_split_mixed_auc_sum_final",
        "avg_split_mixed_mass_sum_final",
        "matched_reference",
        "status",
        "action_dictionary",
        "max_action_weight",
        "tensor_overlap_max_weight",
        "tensor_overlap_max_actions_per_target",
        "gadget_closure_max_weight",
        "mask_padded_actions",
        "canonical_only",
        "force_canonical_basis",
        "num_actions",
        "training_steps",
        "batch_size",
        "num_mcts_simulations",
        "summary_json",
        "candidate_manifest_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _to_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _to_scalar(values: object, index: int = 0) -> object:
    if isinstance(values, list) and index < len(values):
        return values[index]
    return None


def _basis_regime_from_summary(summary: dict) -> str:
    return "canonical-only" if summary.get("force_canonical_basis") else "basis-mix"


def _history_final(summary: dict, key: str, index: int = 0) -> object:
    history = summary.get("history") or []
    if not history:
        return None
    values = history[-1].get(key)
    return _to_scalar(values, index)


def _find_summary(log_dir: Path, run_key: str) -> tuple[Path | None, dict | None]:
    for path in sorted(log_dir.glob("quick_cpu_*.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        output_dir = summary.get("candidate_output_dir") or ""
        if run_key in output_dir:
            return path, summary
    return None, None


def _control_rows(log_dir: Path) -> tuple[list[dict[str, object]], list[dict]]:
    rows: list[dict[str, object]] = []
    control_summaries = []
    for run in CONTROL_RUNS:
        summary_path, summary = _find_summary(log_dir, run["run_key"])
        if summary is None:
            rows.append(
                {
                    "experiment_scope": run["scope"],
                    "label": run["label"],
                    "basis_regime": run["basis_regime"],
                    "budget_source": run["budget_source"],
                    "summary_json": "",
                }
            )
            continue
        row = {
            "experiment_scope": run["scope"],
            "label": run["label"],
            "mode": summary.get("split_reward_mode"),
            "target": _to_scalar(summary.get("target_circuits")),
            "basis_regime": run["basis_regime"],
            "budget_source": run["budget_source"],
            "best_effective_t_cost": _to_scalar(
                summary.get("best_effective_t_cost")
            ),
            "best_return": _to_scalar(summary.get("best_return")),
            "best_return_effective_t_cost": _to_scalar(
                summary.get("best_return_effective_t_cost")
            ),
            "best_return_residual_weight": _to_scalar(
                summary.get("best_return_residual_weight")
            ),
            "best_return_num_moves": _to_scalar(
                summary.get("best_return_num_moves")
            ),
            "avg_return_final": _history_final(summary, "avg_return"),
            "avg_split_sum_rewards_final": _history_final(
                summary, "avg_split_sum_rewards"
            ),
            "avg_split_mixed_auc_sum_final": _history_final(
                summary, "avg_split_mixed_auc_sum"
            ),
            "avg_split_mixed_mass_sum_final": _history_final(
                summary, "avg_split_mixed_mass_sum"
            ),
            "matched_reference": _to_scalar(summary.get("matched_reference")),
            "summary_json": str(summary_path),
            "candidate_manifest_path": summary.get("candidate_manifest_path"),
        }
        rows.append(row)
        control_summaries.append({"run": run, "summary": summary})
    return rows, control_summaries


def _ablation_rows(path: Path) -> list[dict[str, object]]:
    rows = []
    for row in _read_csv(path):
        rows.append(
            {
                "experiment_scope": "split4_short_unbudgeted",
                "label": row.get("mode"),
                "mode": row.get("mode"),
                "target": row.get("target"),
                "basis_regime": "basis-mix",
                "budget_source": "not-applicable",
                "best_effective_t_cost": row.get("best_effective_t_cost"),
                "best_return": row.get("best_return"),
                "best_return_effective_t_cost": row.get(
                    "best_return_effective_t_cost"
                ),
                "best_return_residual_weight": row.get(
                    "best_return_residual_weight"
                ),
                "best_return_num_moves": row.get("best_return_num_moves"),
                "avg_return_final": row.get("avg_return_final"),
                "avg_split_sum_rewards_final": row.get(
                    "avg_split_sum_rewards_final"
                ),
                "avg_split_mixed_auc_sum_final": row.get(
                    "avg_split_mixed_auc_sum_final"
                ),
                "avg_split_mixed_mass_sum_final": row.get(
                    "avg_split_mixed_mass_sum_final"
                ),
                "matched_reference": "",
                "summary_json": row.get("summary_json"),
                "candidate_manifest_path": row.get("candidate_manifest_path"),
            }
        )
    return rows


def _individual_rows(log_dir: Path) -> tuple[list[dict[str, object]], list[dict]]:
    rows: list[dict[str, object]] = []
    summaries = []
    for run in INDIVIDUAL_RUNS:
        summary_path, summary = _find_summary(log_dir, run["run_key"])
        if summary is None:
            rows.append(
                {
                    "experiment_scope": "individual_probe",
                    "label": run["label"],
                    "mode": run["mode"],
                    "target": run["target"],
                    "basis_regime": "basis-mix",
                    "budget_source": "not-applicable",
                    "summary_json": "",
                }
            )
            continue
        row = {
            "experiment_scope": "individual_probe",
            "label": run["label"],
            "mode": summary.get("split_reward_mode"),
            "target": _to_scalar(summary.get("target_circuits")),
            "basis_regime": _basis_regime_from_summary(summary),
            "budget_source": "not-applicable",
            "best_effective_t_cost": _to_scalar(summary.get("best_effective_t_cost")),
            "best_return": _to_scalar(summary.get("best_return")),
            "best_return_effective_t_cost": _to_scalar(
                summary.get("best_return_effective_t_cost")
            ),
            "best_return_residual_weight": _to_scalar(
                summary.get("best_return_residual_weight")
            ),
            "best_return_num_moves": _to_scalar(summary.get("best_return_num_moves")),
            "avg_return_final": _history_final(summary, "avg_return"),
            "avg_split_sum_rewards_final": _history_final(
                summary, "avg_split_sum_rewards"
            ),
            "avg_split_mixed_auc_sum_final": _history_final(
                summary, "avg_split_mixed_auc_sum"
            ),
            "avg_split_mixed_mass_sum_final": _history_final(
                summary, "avg_split_mixed_mass_sum"
            ),
            "matched_reference": _to_scalar(summary.get("matched_reference")),
            "canonical_only": summary.get("canonical_only"),
            "force_canonical_basis": summary.get("force_canonical_basis"),
            "summary_json": str(summary_path),
            "candidate_manifest_path": summary.get("candidate_manifest_path"),
        }
        rows.append(row)
        summaries.append({"run": run, "summary": summary})
    return rows, summaries


def _row_from_training_summary(
    *,
    summary_path: Path | None,
    summary: dict | None,
    label: str,
    scope: str,
    fallback_target: str = "",
    fallback_mode: str = "",
    budget_source: str = "not-applicable",
) -> dict[str, object]:
    if summary is None:
        return {
            "experiment_scope": scope,
            "label": label,
            "target": fallback_target,
            "mode": fallback_mode,
            "budget_source": budget_source,
            "summary_json": "",
        }
    return {
        "experiment_scope": scope,
        "label": label,
        "mode": summary.get("split_reward_mode"),
        "target": _to_scalar(summary.get("target_circuits")),
        "basis_regime": _basis_regime_from_summary(summary),
        "budget_source": budget_source,
        "best_effective_t_cost": _to_scalar(summary.get("best_effective_t_cost")),
        "best_return": _to_scalar(summary.get("best_return")),
        "best_return_effective_t_cost": _to_scalar(
            summary.get("best_return_effective_t_cost")
        ),
        "best_return_residual_weight": _to_scalar(
            summary.get("best_return_residual_weight")
        ),
        "best_return_num_moves": _to_scalar(summary.get("best_return_num_moves")),
        "avg_return_final": _history_final(summary, "avg_return"),
        "avg_split_sum_rewards_final": _history_final(
            summary, "avg_split_sum_rewards"
        ),
        "avg_split_mixed_auc_sum_final": _history_final(
            summary, "avg_split_mixed_auc_sum"
        ),
        "avg_split_mixed_mass_sum_final": _history_final(
            summary, "avg_split_mixed_mass_sum"
        ),
        "matched_reference": _to_scalar(summary.get("matched_reference")),
        "action_dictionary": summary.get("action_dictionary"),
        "max_action_weight": summary.get("max_action_weight"),
        "mask_padded_actions": summary.get("mask_padded_actions"),
        "canonical_only": summary.get("canonical_only"),
        "force_canonical_basis": summary.get("force_canonical_basis"),
        "num_actions": summary.get("num_actions"),
        "training_steps": summary.get("training_steps"),
        "batch_size": summary.get("batch_size"),
        "num_mcts_simulations": summary.get("num_mcts_simulations"),
        "summary_json": "" if summary_path is None else str(summary_path),
        "candidate_manifest_path": summary.get("candidate_manifest_path"),
    }


def _action_dictionary_rows(log_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run in ACTION_DICTIONARY_RUNS:
        summary_path, summary = _find_summary(log_dir, run["run_key"])
        rows.append(
            _row_from_training_summary(
                summary_path=summary_path,
                summary=summary,
                label=run["label"],
                scope=run["scope"],
                fallback_target=run["target"],
                fallback_mode=run["mode"],
            )
        )
    return rows


def _lowweight_sweep_rows(path: Path) -> list[dict[str, object]]:
    rows = []
    for row in _read_csv(path):
        rows.append(
            {
                "experiment_scope": "lowweight_reward_probe",
                "label": (
                    f"{row.get('target')} {row.get('mode')} "
                    f"low-weight<={row.get('max_action_weight')}"
                ),
                "mode": row.get("mode"),
                "target": row.get("target"),
                "basis_regime": "basis-mix",
                "budget_source": "not-applicable",
                "best_effective_t_cost": row.get("best_effective_t_cost"),
                "best_return": row.get("best_return"),
                "best_return_effective_t_cost": row.get(
                    "best_return_effective_t_cost"
                ),
                "best_return_residual_weight": row.get(
                    "best_return_residual_weight"
                ),
                "best_return_num_moves": row.get("best_return_num_moves"),
                "avg_return_final": row.get("avg_return_final"),
                "avg_split_sum_rewards_final": row.get(
                    "avg_split_sum_rewards_final"
                ),
                "avg_split_mixed_auc_sum_final": row.get(
                    "avg_split_mixed_auc_sum_final"
                ),
                "avg_split_mixed_mass_sum_final": row.get(
                    "avg_split_mixed_mass_sum_final"
                ),
                "status": row.get("status"),
                "action_dictionary": row.get("action_dictionary"),
                "max_action_weight": row.get("max_action_weight"),
                "mask_padded_actions": row.get("mask_padded_actions"),
                "training_steps": row.get("training_steps"),
                "batch_size": row.get("batch_size"),
                "num_mcts_simulations": row.get("num_mcts_simulations"),
                "summary_json": row.get("summary_json"),
                "candidate_manifest_path": row.get("candidate_manifest_path"),
            }
        )
    return rows


def _canonical_probe_label(row: dict[str, str]) -> str:
    dictionary = row.get("action_dictionary") or "low-weight"
    if dictionary == "gadget-closure":
        return (
            f"{row.get('target')} {row.get('mode')} canonical "
            f"gadget-closure base<={row.get('max_action_weight')} "
            f"closure<={row.get('gadget_closure_max_weight')}"
        )
    return (
        f"{row.get('target')} {row.get('mode')} canonical "
        f"{dictionary}<={row.get('max_action_weight')}"
    )


def _canonical_probe_rows(path: Path, scope: str) -> list[dict[str, object]]:
    rows = []
    for row in _read_csv(path):
        rows.append(
            {
                "experiment_scope": scope,
                "label": _canonical_probe_label(row),
                "mode": row.get("mode"),
                "target": row.get("target"),
                "basis_regime": "canonical-only",
                "budget_source": "not-applicable",
                "best_effective_t_cost": row.get("best_effective_t_cost"),
                "best_return": row.get("best_return"),
                "best_return_effective_t_cost": row.get(
                    "best_return_effective_t_cost"
                ),
                "best_return_residual_weight": row.get(
                    "best_return_residual_weight"
                ),
                "best_return_num_moves": row.get("best_return_num_moves"),
                "avg_return_final": row.get("avg_return_final"),
                "avg_split_sum_rewards_final": row.get(
                    "avg_split_sum_rewards_final"
                ),
                "avg_split_mixed_auc_sum_final": row.get(
                    "avg_split_mixed_auc_sum_final"
                ),
                "avg_split_mixed_mass_sum_final": row.get(
                    "avg_split_mixed_mass_sum_final"
                ),
                "status": row.get("status"),
                "action_dictionary": row.get("action_dictionary"),
                "max_action_weight": row.get("max_action_weight"),
                "gadget_closure_max_weight": row.get(
                    "gadget_closure_max_weight"
                ),
                "mask_padded_actions": row.get("mask_padded_actions"),
                "canonical_only": row.get("canonical_only"),
                "force_canonical_basis": row.get("force_canonical_basis"),
                "training_steps": row.get("training_steps"),
                "batch_size": row.get("batch_size"),
                "num_mcts_simulations": row.get("num_mcts_simulations"),
                "summary_json": row.get("summary_json"),
                "candidate_manifest_path": row.get("candidate_manifest_path"),
            }
        )
    return rows


def _masked_split4_rows(log_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run in MASKED_SPLIT4_RUNS:
        summary_path, summary = _find_summary(log_dir, run["run_key"])
        if summary is None:
            rows.append(
                {
                    "experiment_scope": "masked_split4_probe",
                    "label": run["label"],
                    "mode": run["mode"],
                    "summary_json": "",
                }
            )
            continue
        targets = summary.get("target_circuits") or []
        for index, target in enumerate(targets):
            rows.append(
                {
                    "experiment_scope": "masked_split4_probe",
                    "label": run["label"],
                    "mode": summary.get("split_reward_mode"),
                    "target": target,
                    "basis_regime": "basis-mix",
                    "budget_source": "not-applicable",
                    "best_effective_t_cost": _to_scalar(
                        summary.get("best_effective_t_cost"), index
                    ),
                    "best_return": _to_scalar(summary.get("best_return"), index),
                    "best_return_residual_weight": _to_scalar(
                        summary.get("best_return_residual_weight"), index
                    ),
                    "best_return_num_moves": _to_scalar(
                        summary.get("best_return_num_moves"), index
                    ),
                    "avg_return_final": _history_final(summary, "avg_return", index),
                    "avg_split_sum_rewards_final": _history_final(
                        summary, "avg_split_sum_rewards", index
                    ),
                    "avg_split_mixed_auc_sum_final": _history_final(
                        summary, "avg_split_mixed_auc_sum", index
                    ),
                    "avg_split_mixed_mass_sum_final": _history_final(
                        summary, "avg_split_mixed_mass_sum", index
                    ),
                    "matched_reference": _to_scalar(
                        summary.get("matched_reference"), index
                    ),
                    "status": "ok",
                    "action_dictionary": summary.get("action_dictionary"),
                    "max_action_weight": summary.get("max_action_weight"),
                    "mask_padded_actions": summary.get("mask_padded_actions"),
                    "num_actions": summary.get("num_actions"),
                    "training_steps": summary.get("training_steps"),
                    "batch_size": summary.get("batch_size"),
                    "num_mcts_simulations": summary.get("num_mcts_simulations"),
                    "summary_json": str(summary_path),
                    "candidate_manifest_path": summary.get("candidate_manifest_path"),
                }
            )
    return rows


def _tensor_overlap_rows(log_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run in TENSOR_OVERLAP_RUNS:
        summary_path, summary = _find_summary(log_dir, run["run_key"])
        if summary is None:
            rows.append(
                {
                    "experiment_scope": "tensor_overlap_probe",
                    "label": run["label"],
                    "mode": run["mode"],
                    "summary_json": "",
                }
            )
            continue
        targets = summary.get("target_circuits") or []
        for index, target in enumerate(targets):
            rows.append(
                {
                    "experiment_scope": "tensor_overlap_probe",
                    "label": run["label"],
                    "mode": summary.get("split_reward_mode"),
                    "target": target,
                    "basis_regime": "basis-mix",
                    "budget_source": "not-applicable",
                    "best_effective_t_cost": _to_scalar(
                        summary.get("best_effective_t_cost"), index
                    ),
                    "best_return": _to_scalar(summary.get("best_return"), index),
                    "best_return_residual_weight": _to_scalar(
                        summary.get("best_return_residual_weight"), index
                    ),
                    "best_return_num_moves": _to_scalar(
                        summary.get("best_return_num_moves"), index
                    ),
                    "avg_return_final": _history_final(summary, "avg_return", index),
                    "avg_split_sum_rewards_final": _history_final(
                        summary, "avg_split_sum_rewards", index
                    ),
                    "avg_split_mixed_auc_sum_final": _history_final(
                        summary, "avg_split_mixed_auc_sum", index
                    ),
                    "avg_split_mixed_mass_sum_final": _history_final(
                        summary, "avg_split_mixed_mass_sum", index
                    ),
                    "matched_reference": _to_scalar(
                        summary.get("matched_reference"), index
                    ),
                    "status": "ok",
                    "action_dictionary": summary.get("action_dictionary"),
                    "max_action_weight": summary.get("max_action_weight"),
                    "tensor_overlap_max_weight": summary.get(
                        "tensor_overlap_max_weight"
                    ),
                    "tensor_overlap_max_actions_per_target": summary.get(
                        "tensor_overlap_max_actions_per_target"
                    ),
                    "mask_padded_actions": summary.get("mask_padded_actions"),
                    "num_actions": summary.get("num_actions"),
                    "training_steps": summary.get("training_steps"),
                    "batch_size": summary.get("batch_size"),
                    "num_mcts_simulations": summary.get("num_mcts_simulations"),
                    "summary_json": str(summary_path),
                    "candidate_manifest_path": summary.get("candidate_manifest_path"),
                }
            )
    return rows


def _benchmark_rows(
    path: Path,
    external_split_reward_metrics: Path,
) -> list[dict[str, str]]:
    rows = []
    for row in _read_csv(path):
        if row.get("circuit_id") in BENCHMARK_TARGETS and row.get(
            "method"
        ) in BENCHMARK_METHODS:
            rows.append(row)
    if external_split_reward_metrics.exists():
        metrics = json.loads(
            external_split_reward_metrics.read_text(encoding="utf-8")
        )
        rows.append(
            {
                "circuit_id": "mod_5_4",
                "method": "alphaq_split_reward_v1_tiebreak",
                "tcount_after": metrics.get("tcount_after"),
                "primary_nc_depth_ratio": metrics.get("primary_nc_depth_ratio"),
                "zx_best_clifford_fraction": metrics.get(
                    "zx_best_clifford_fraction"
                ),
                "qasm_depth_ratio": metrics.get("qasm_depth_ratio"),
                "zx_total_depth_ratio": metrics.get("zx_total_depth_ratio"),
                "selection_status": metrics.get("selection_status"),
            }
        )
    return rows


def _plot_control_curves(control_summaries: list[dict], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "mod_5_4_training_curves.png"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    groups = [("basis-mix", axes[0]), ("canonical-only", axes[1])]
    for basis, ax in groups:
        group_items = [
            item for item in control_summaries
            if item["run"]["basis_regime"] == basis
        ]
        for item_index, item in enumerate(group_items):
            run = item["run"]
            history = item["summary"].get("history") or []
            x_offset = (item_index - (len(group_items) - 1) / 2) * 4.0
            steps = [entry["step"] + x_offset for entry in history]
            values = [
                _to_float(_to_scalar(entry.get("best_effective_t_cost")))
                for entry in history
            ]
            ax.plot(
                steps,
                values,
                marker="o" if item_index == 0 else "x",
                linestyle="-" if item_index == 0 else "--",
                label=run["label"],
            )
        ax.axhline(2, color="0.4", linestyle="--", linewidth=1, label="budget 2")
        ax.set_title(basis)
        ax.set_xlabel("training step")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("best solved effective T-cost")
    fig.suptitle("mod_5_4 controlled training")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _plot_split4_residual(ablation_rows: list[dict[str, object]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "split4_short_residual_weight.png"
    targets = sorted({str(row["target"]) for row in ablation_rows})
    modes = [mode for mode in ("none", "mixed_drop", "mixed_auc", "v1")
             if any(row["mode"] == mode for row in ablation_rows)]
    width = 0.8 / max(len(modes), 1)
    x_positions = list(range(len(targets)))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for mode_index, mode in enumerate(modes):
        values = []
        for target in targets:
            match = next(
                (
                    row for row in ablation_rows
                    if row["mode"] == mode and row["target"] == target
                ),
                {},
            )
            values.append(_to_float(match.get("best_return_residual_weight")) or 0.0)
        offsets = [x + (mode_index - (len(modes) - 1) / 2) * width for x in x_positions]
        ax.bar(offsets, values, width=width, label=mode)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(targets, rotation=25, ha="right")
    ax.set_ylabel("best-return residual tensor weight")
    ax.set_title("split4 short ablation: unresolved residual after 200 steps")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _plot_individual_residual(
    individual_rows: list[dict[str, object]],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "individual_probe_residual_weight.png"
    rows = [
        row for row in individual_rows
        if row.get("target") == "gf_2pow2_mult" and row.get("summary_json")
    ]
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    labels = [str(row.get("mode")) for row in rows]
    values = [
        _to_float(row.get("best_return_residual_weight")) or 0.0
        for row in rows
    ]
    bars = ax.bar(labels, values, color=["0.35", "tab:red"][: len(labels)])
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel("best-return residual tensor weight")
    ax.set_title("gf_2pow2_mult individual 1000-step probe")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _plot_action_dictionary_probe(
    action_rows: list[dict[str, object]],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "action_dictionary_probe_residual_weight.png"
    rows = [row for row in action_rows if row.get("summary_json")]
    labels = []
    values = []
    colors = []
    for row in rows:
        mode = row.get("mode")
        dictionary = row.get("action_dictionary")
        steps = row.get("training_steps")
        max_weight = row.get("max_action_weight")
        action_count = row.get("num_actions")
        if dictionary == "full":
            dictionary_label = "full"
            colors.append("0.35")
        else:
            dictionary_label = f"low-w<={max_weight}"
            colors.append("tab:green" if mode == "none" else "tab:red")
        labels.append(f"{mode}\n{dictionary_label}\n{action_count} actions\n{steps} steps")
        values.append(_to_float(row.get("best_return_residual_weight")) or 0.0)
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    bars = ax.bar(range(len(labels)), values, color=colors)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("best-return residual tensor weight")
    ax.set_title("hamming_weight_n5: action dictionary and reward probe")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _plot_lowweight_reward_probe(
    lowweight_rows: list[dict[str, object]],
    action_rows: list[dict[str, object]],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "lowweight_reward_probe_residual_weight.png"
    rows = [
        row for row in lowweight_rows
        if row.get("status") == "ok"
        and row.get("best_return_residual_weight") not in (None, "")
    ]
    # Add the directly run hamming_weight_n5 100-step smoke pair. The 1000-step
    # n5 sweep timed out in the harness, so this keeps the comparison paired.
    for row in action_rows:
        if (
            row.get("target") == "hamming_weight_n5"
            and row.get("action_dictionary") == "low-weight"
            and str(row.get("training_steps")) == "100"
            and row.get("summary_json")
        ):
            rows.append(row)
    targets = ["gf_2pow2_mult", "hamming_weight_n4", "hamming_weight_n5"]
    modes = ["none", "v1"]
    target_steps = {}
    for target in targets:
        steps = {
            str(row.get("training_steps"))
            for row in rows
            if row.get("target") == target and row.get("training_steps") not in (None, "")
        }
        target_steps[target] = "/".join(sorted(steps)) if steps else "NA"
    width = 0.36
    x_positions = list(range(len(targets)))
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for mode_index, mode in enumerate(modes):
        values = []
        for target in targets:
            match = next(
                (
                    row for row in rows
                    if row.get("target") == target and row.get("mode") == mode
                ),
                {},
            )
            values.append(_to_float(match.get("best_return_residual_weight")) or 0.0)
        offsets = [
            x + (mode_index - (len(modes) - 1) / 2) * width
            for x in x_positions
        ]
        bars = ax.bar(offsets, values, width=width, label=mode)
        for bar, value in zip(bars, values, strict=True):
            if value:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value,
                    f"{value:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [f"{target}\n{target_steps[target]} steps" for target in targets],
        rotation=18,
        ha="right",
    )
    ax.set_ylabel("best-return residual tensor weight")
    ax.set_title("low-weight<=2 dictionary: none vs split reward v1")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _plot_canonical_reward_probe(
    canonical_rows: list[dict[str, object]],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "canonical_reward_probe_residual_weight.png"
    rows = [row for row in canonical_rows if row.get("status") == "ok"]
    targets = ["gf_2pow2_mult", "hamming_weight_n4"]
    modes = ["none", "v2_progress"]
    width = 0.36
    x_positions = list(range(len(targets)))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for mode_index, mode in enumerate(modes):
        values = []
        for target in targets:
            match = next(
                (
                    row for row in rows
                    if row.get("target") == target and row.get("mode") == mode
                ),
                {},
            )
            values.append(_to_float(match.get("best_return_residual_weight")) or 0.0)
        offsets = [
            x + (mode_index - (len(modes) - 1) / 2) * width
            for x in x_positions
        ]
        bars = ax.bar(offsets, values, width=width, label=mode)
        for bar, value in zip(bars, values, strict=True):
            if value:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value,
                    f"{value:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(targets, rotation=16, ha="right")
    ax.set_ylabel("best-return residual tensor weight")
    ax.set_title("canonical low-weight<=2 probe")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _plot_gadget_closure_probe(
    canonical_rows: list[dict[str, object]],
    gadget_rows: list[dict[str, object]],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "gadget_closure_canonical_probe_residual_weight.png"
    rows = [
        row for row in [*canonical_rows, *gadget_rows]
        if row.get("status") == "ok"
    ]
    targets = ["gf_2pow2_mult", "hamming_weight_n4"]
    series = [
        ("low-weight none", "low-weight", "none"),
        ("low-weight v2", "low-weight", "v2_progress"),
        ("gadget-closure none", "gadget-closure", "none"),
        ("gadget-closure v2", "gadget-closure", "v2_progress"),
    ]
    width = 0.19
    x_positions = list(range(len(targets)))
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    for series_index, (label, dictionary, mode) in enumerate(series):
        values = []
        for target in targets:
            match = next(
                (
                    row for row in rows
                    if row.get("target") == target
                    and row.get("action_dictionary") == dictionary
                    and row.get("mode") == mode
                ),
                {},
            )
            values.append(_to_float(match.get("best_return_residual_weight")) or 0.0)
        offsets = [
            x + (series_index - (len(series) - 1) / 2) * width
            for x in x_positions
        ]
        bars = ax.bar(offsets, values, width=width, label=label)
        for bar, value in zip(bars, values, strict=True):
            if value:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value,
                    f"{value:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(targets, rotation=16, ha="right")
    ax.set_ylabel("best-return residual tensor weight")
    ax.set_title("canonical action dictionary probe")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _plot_masked_split4_probe(
    masked_rows: list[dict[str, object]],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "masked_split4_probe_residual_weight.png"
    rows = [row for row in masked_rows if row.get("summary_json")]
    targets = [
        "mod_5_4",
        "gf_2pow2_mult",
        "hamming_weight_n4",
        "hamming_weight_n5",
    ]
    modes = ["none", "v1", "v2_progress"]
    width = 0.26
    x_positions = list(range(len(targets)))
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    for mode_index, mode in enumerate(modes):
        values = []
        for target in targets:
            match = next(
                (
                    row for row in rows
                    if row.get("target") == target and row.get("mode") == mode
                ),
                {},
            )
            values.append(_to_float(match.get("best_return_residual_weight")) or 0.0)
        offsets = [
            x + (mode_index - (len(modes) - 1) / 2) * width
            for x in x_positions
        ]
        bars = ax.bar(offsets, values, width=width, label=mode)
        for bar, value in zip(bars, values, strict=True):
            if value:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value,
                    f"{value:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(targets, rotation=18, ha="right")
    ax.set_ylabel("best-return residual tensor weight")
    ax.set_title("split4 masked low-weight<=2 probe")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _plot_tensor_overlap_probe(
    masked_rows: list[dict[str, object]],
    tensor_overlap_rows: list[dict[str, object]],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "tensor_overlap_probe_residual_weight.png"
    targets = [
        "mod_5_4",
        "gf_2pow2_mult",
        "hamming_weight_n4",
        "hamming_weight_n5",
    ]
    series = [
        ("low-weight v2", masked_rows, "v2_progress"),
        ("tensor-overlap none", tensor_overlap_rows, "none"),
        ("tensor-overlap v2", tensor_overlap_rows, "v2_progress"),
    ]
    width = 0.26
    x_positions = list(range(len(targets)))
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    for series_index, (label, rows, mode) in enumerate(series):
        values = []
        for target in targets:
            match = next(
                (
                    row for row in rows
                    if row.get("target") == target and row.get("mode") == mode
                ),
                {},
            )
            values.append(_to_float(match.get("best_return_residual_weight")) or 0.0)
        offsets = [
            x + (series_index - (len(series) - 1) / 2) * width
            for x in x_positions
        ]
        bars = ax.bar(offsets, values, width=width, label=label)
        for bar, value in zip(bars, values, strict=True):
            if value:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value,
                    f"{value:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(targets, rotation=18, ha="right")
    ax.set_ylabel("best-return residual tensor weight")
    ax.set_title("tensor-overlap dictionary probe")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _plot_external_benchmarks(benchmark_rows: list[dict[str, str]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "external_benchmark_primary_ratio.png"
    targets = [target for target in BENCHMARK_TARGETS if any(
        row.get("circuit_id") == target for row in benchmark_rows
    )]
    methods = [method for method in BENCHMARK_METHODS if any(
        row.get("method") == method for row in benchmark_rows
    )]
    width = 0.8 / max(len(methods), 1)
    x_positions = list(range(len(targets)))
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for method_index, method in enumerate(methods):
        values = []
        for target in targets:
            match = next(
                (
                    row for row in benchmark_rows
                    if row.get("method") == method
                    and row.get("circuit_id") == target
                ),
                {},
            )
            values.append(_to_float(match.get("primary_nc_depth_ratio")) or 0.0)
        offsets = [x + (method_index - (len(methods) - 1) / 2) * width for x in x_positions]
        ax.bar(offsets, values, width=width, label=method)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(targets, rotation=25, ha="right")
    ax.set_ylabel("external primary_nc_depth_ratio")
    ax.set_title("External benchmark context")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _fmt(value: object) -> str:
    if isinstance(value, bool):
        return str(value)
    number = _to_float(value)
    if number is None:
        return "NA" if value in (None, "") else str(value)
    return f"{number:.4g}"


def _markdown_table(rows: list[list[object]], headers: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(value) for value in row) + " |")
    return lines


def _write_report(
    path: Path,
    output_csv: Path,
    control_rows: list[dict[str, object]],
    individual_rows: list[dict[str, object]],
    action_rows: list[dict[str, object]],
    lowweight_rows: list[dict[str, object]],
    canonical_rows: list[dict[str, object]],
    gadget_closure_rows: list[dict[str, object]],
    masked_rows: list[dict[str, object]],
    tensor_overlap_rows: list[dict[str, object]],
    ablation_rows: list[dict[str, object]],
    benchmark_rows: list[dict[str, str]],
    figure_paths: list[Path],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    basis_mix = [
        row for row in control_rows
        if row.get("experiment_scope") == "mod_5_4_budget_real"
    ]
    canonical = [
        row for row in control_rows
        if row.get("experiment_scope") == "mod_5_4_canonical_control"
    ]
    lines = [
        "# Split reward core validation",
        "",
        "## Decision",
        "",
        "The polished internal variant is `v1_tiebreak`: it keeps effective "
        "T-cost as the hard primary signal and only applies a clipped terminal "
        "splitting penalty to solved candidates that are inside a real budget. "
        "Budgeted modes now require either a solved `none` baseline or explicit "
        "budgets, so short unsolved runs are not assigned synthetic budgets.",
        "",
        "The current evidence is supportive on `mod_5_4`, operationally "
        "supportive for using a restricted AlphaQuantum-only action dictionary, "
        "and negative for the present canonical reward-only design on the two "
        "larger local probes. A new `gadget-closure` action dictionary gives "
        "a small canonical improvement on `gf_2pow2_mult`, but it does not "
        "make the split reward itself win and it does not improve "
        "`hamming_weight_n4` in the local budget. On the controlled `mod_5_4` run with the normal basis "
        "mixture, `v1_tiebreak` preserves the baseline effective T-cost 2. "
        "After exporting the change-of-basis matrix, canonicalizing the solved "
        "factorization with `materialize_split_reward_candidate.py`, and "
        "resynthesizing it to QASM, the external "
        "`primary_nc_depth_ratio` improves to 0.322 at T-count 7. On "
        "`gf_2pow2_mult`, the 1000-step individual probe gives a small "
        "residual improvement for `v1` versus `none`, but neither run solves "
        "the tensor. The new `low-weight` action dictionary makes local probes "
        "of larger targets much more interpretable, but the split reward itself "
        "has only a small or tied residual advantage in those paired local "
        "runs. With the multi-target padded-action mask, `v1` and the new "
        "positive-only `v2_progress` improve the hamming residuals in the "
        "split4 setting, but still do not solve a large target locally. Dense "
        "`tensor-overlap` action expansion was also tested and did not improve "
        "over `low-weight<=2` in the local budget. Dense "
        "reward variants remain exploratory: in short "
        "`split4` runs they change residual profiles but do not solve the "
        "targets within 200 steps. The new canonical-only probe shows that "
        "`v2_progress` does not reduce residual tensor weight versus `none` "
        "for `gf_2pow2_mult` or `hamming_weight_n4` under the same "
        "`low-weight<=2` search budget.",
        "",
        f"Machine-readable summary: `{output_csv}`.",
        "",
        "## Figures",
        "",
    ]
    for figure in figure_paths:
        lines.append(f"- `{figure}`")
    lines.extend(["", "## mod_5_4 Controlled Runs", ""])
    lines.extend(
        _markdown_table(
            [
                [
                    row.get("label"),
                    row.get("basis_regime"),
                    row.get("budget_source"),
                    row.get("best_effective_t_cost"),
                    row.get("best_return_num_moves"),
                    row.get("matched_reference"),
                ]
                for row in basis_mix + canonical
            ],
            [
                "run",
                "basis",
                "budget",
                "best effective T-cost",
                "moves",
                "matched reference",
            ],
        )
    )
    lines.extend(
        [
            "",
            "Interpretation: the apparent earlier degradation of tiebreak mode "
            "was largely a basis-distribution confound. Canonical-only training "
            "is harder here: both `none` and `v1_tiebreak` stop at effective "
            "T-cost 8 in the 1000-step control, while the normal basis mixture "
            "reaches the known cost 2.",
            "",
            "The basis-mix `v1_tiebreak` candidate is not canonical in the raw "
            "tensor-game coordinates. The exported change-of-basis matrix is "
            "therefore required before QASM resynthesis. Canonicalizing with "
            "the inverse basis produces a valid 7-factor decomposition of the "
            "original tensor.",
            "",
            "## Individual Target Probe",
            "",
            "This probe isolates `gf_2pow2_mult` instead of mixing it with the "
            "other targets. It is still a short, unresolved run; the purpose is "
            "to test whether the split reward changes the search direction "
            "before we invest in long jobs.",
            "",
        ]
    )
    individual_table = []
    for row in individual_rows:
        individual_table.append(
            [
                row.get("target"),
                row.get("mode"),
                row.get("best_return"),
                row.get("best_return_residual_weight"),
                row.get("avg_split_sum_rewards_final"),
            ]
        )
    lines.extend(
        _markdown_table(
            individual_table,
            [
                "target",
                "mode",
                "best return",
                "best residual",
                "avg split reward",
            ],
        )
    )
    lines.extend(
        [
            "",
            "Interpretation: `v1` reduces the best residual from 18 to 16 in "
            "this 1000-step probe, which is a weak positive signal. Because "
            "neither run reaches residual zero, there is no real budget for "
            "`v1_tiebreak` on this target yet.",
            "",
            "## Action Dictionary Probe",
            "",
            "The main operational bottleneck is the full AlphaQuantum action "
            "space. For a tensor of size `n`, the full dictionary has "
            "`2^n - 1` actions. The experimental `low-weight` dictionary keeps "
            "only factors with Hamming weight at most two. This is still "
            "AlphaQuantum-only: it does not use ZX, PyZX, or feynver inside the "
            "loop. After these probes, the agent also gained an optional "
            "`mask_padded_actions` switch that masks factors touching padded "
            "coordinates in multi-target runs; the historical rows below did "
            "not use that mask, but the masked path is covered by unit and "
            "quick-loop smoke tests.",
            "",
        ]
    )
    action_table = []
    for row in action_rows:
        action_table.append(
            [
                row.get("target"),
                row.get("mode"),
                row.get("action_dictionary"),
                row.get("max_action_weight"),
                row.get("mask_padded_actions"),
                row.get("num_actions"),
                row.get("training_steps"),
                row.get("best_return_residual_weight"),
            ]
        )
    lines.extend(
        _markdown_table(
            action_table,
            [
                "target",
                "mode",
                "dictionary",
                "max weight",
                "mask padded",
                "actions",
                "steps",
                "best residual",
            ],
        )
    )
    lines.extend(
        [
            "",
            "Interpretation: on `hamming_weight_n5`, switching from the full "
            "dictionary to `low-weight<=2` reduces the 100-step residual from "
            "264 to 124. That gain is due to a better search space, not yet to "
            "the split reward: `none` and `v1` tie at residual 124 in the "
            "paired 100-step `low-weight<=2` smoke. The broader "
            "`low-weight<=3` dictionary has 175 actions and performs worse "
            "in this 100-step probe, reaching residual 200. Extending `v1` "
            "with `low-weight<=2` to 1000 steps reaches residual 108, but the "
            "corresponding long `none` run timed out in the sweep harness, so "
            "this is operational evidence rather than a paired reward win.",
            "",
            "## Low-weight Paired Reward Probe",
            "",
            "This table compares `none` and `v1` under the same "
            "`low-weight<=2` action dictionary. These are search-direction "
            "probes, not external benchmark wins, because no row solves the "
            "tensor exactly.",
            "",
        ]
    )
    lowweight_table = []
    for row in lowweight_rows:
        lowweight_table.append(
            [
                row.get("target"),
                row.get("mode"),
                row.get("status"),
                row.get("training_steps"),
                row.get("best_return_residual_weight"),
                row.get("avg_split_sum_rewards_final"),
            ]
        )
    lines.extend(
        _markdown_table(
            lowweight_table,
            [
                "target",
                "mode",
                "status",
                "steps",
                "best residual",
                "avg split reward",
            ],
        )
    )
    lines.extend(
        [
            "",
            "Interpretation: with the restricted dictionary, `v1` gives a small "
            "residual improvement on `gf_2pow2_mult` (44 to 42), ties on "
            "`hamming_weight_n4` (42 to 42), and both long `hamming_weight_n5` "
            "rows hit the local timeout in the sweep harness. The right claim "
            "is therefore cautious: the action dictionary is clearly useful; "
            "the current dense split reward is not yet a broad local win.",
            "",
            "## Canonical Reward Probe",
            "",
            "This is the cleanest local test of the internal splitting "
            "hypothesis so far. It forces the environment to start in the "
            "canonical basis and compares `none` against the positive-only "
            "`v2_progress` reward under the same `low-weight<=2`, 1000-step, "
            "batch-16, MCTS-8 configuration. All exported best-return rows are "
            "marked `is_canonical_basis=True` in their manifests.",
            "",
        ]
    )
    canonical_table = []
    for row in canonical_rows:
        canonical_table.append(
            [
                row.get("target"),
                row.get("mode"),
                row.get("status"),
                row.get("training_steps"),
                row.get("best_return_residual_weight"),
                row.get("best_return_effective_t_cost"),
                row.get("avg_split_sum_rewards_final"),
            ]
        )
    lines.extend(
        _markdown_table(
            canonical_table,
            [
                "target",
                "mode",
                "status",
                "steps",
                "best residual",
                "best-return effective cost",
                "avg split reward",
            ],
        )
    )
    lines.extend(
        [
            "",
            "Interpretation: this probe does not support a standalone "
            "canonical reward win. `gf_2pow2_mult` ties at residual 66 "
            "(`none` vs `v2_progress`), and `hamming_weight_n4` ties at "
            "residual 120. The split reward changes the scalar return and "
            "slightly changes the effective-cost profile on `gf_2pow2_mult`, "
            "but it does not close more of the target tensor. This means that "
            "the better basis-mix probes were relying heavily on the original "
            "AlphaQuantum change-of-basis machinery; the splitting reward "
            "alone is not yet competitive in canonical coordinates.",
            "",
            "## Gadget-closure Action Probe",
            "",
            "`gadget-closure` keeps the low-weight base actions available, "
            "adds higher-weight linear combinations up to a fixed closure "
            "weight, and masks those added actions unless the current factor "
            "history is a valid CS/Toffoli prefix. This is an internal "
            "AlphaQuantum action-space change: it does not use ZX, PyZX, "
            "feynver, or external circuit metrics.",
            "",
        ]
    )
    gadget_table = []
    for row in gadget_closure_rows:
        gadget_table.append(
            [
                row.get("target"),
                row.get("mode"),
                row.get("action_dictionary"),
                row.get("max_action_weight"),
                row.get("gadget_closure_max_weight"),
                row.get("training_steps"),
                row.get("best_return_residual_weight"),
                row.get("best_return_effective_t_cost"),
                row.get("avg_split_sum_rewards_final"),
            ]
        )
    lines.extend(
        _markdown_table(
            gadget_table,
            [
                "target",
                "mode",
                "dictionary",
                "base weight",
                "closure weight",
                "steps",
                "best residual",
                "best-return effective cost",
                "avg split reward",
            ],
        )
    )
    lines.extend(
        [
            "",
            "Interpretation: `gadget-closure` is a useful action-space idea but "
            "not yet a reward win. On `gf_2pow2_mult`, base<=2/closure<=4 "
            "improves the canonical residual from 66 to 64 and lowers the "
            "best-return effective cost from 30/29 to 27. However, "
            "`v2_progress` ties `none` at residual 64. On `hamming_weight_n4`, "
            "`gadget-closure` ties the low-weight canonical residual at 120. "
            "So the best current reading is that structured action access is "
            "helping slightly, while the splitting reward still needs a "
            "stronger or more state-aware objective.",
            "",
            "## Masked split4 Probe",
            "",
            "This probe uses the multi-target `split4` environment with "
            "`low-weight<=2` and `mask_padded_actions=true`. The mask prevents "
            "MCTS from spending actions on coordinates that are padding for the "
            "currently sampled target. This is the closest local proxy to the "
            "recommended cluster configuration.",
            "",
        ]
    )
    masked_table = []
    for row in masked_rows:
        masked_table.append(
            [
                row.get("target"),
                row.get("mode"),
                row.get("training_steps"),
                row.get("best_return_residual_weight"),
                row.get("avg_split_sum_rewards_final"),
                row.get("mask_padded_actions"),
            ]
        )
    lines.extend(
        _markdown_table(
            masked_table,
            [
                "target",
                "mode",
                "steps",
                "best residual",
                "avg split reward",
                "mask padded",
            ],
        )
    )
    lines.extend(
        [
            "",
            "Interpretation: the masked multi-target setup is a better local "
            "search harness than the original full-action split4 ablation. "
            "`v1` improves the hamming residuals (`75 -> 66` on "
            "`hamming_weight_n4`, `153 -> 144` on `hamming_weight_n5`) and ties "
            "`mod_5_4`/`gf_2pow2_mult`. `v2_progress` reaches the same residuals "
            "as `v1` in this probe, but its split reward is positive-only "
            "rather than dominated by dense penalties. This makes it the cleaner "
            "search-shaping candidate, while also showing that reward shaping "
            "alone is not yet enough for broad competitiveness.",
            "",
            "## Tensor-overlap Dictionary Probe",
            "",
            "The `tensor-overlap` dictionary keeps the `low-weight` base and "
            "adds target-guided higher-weight factors ranked only by overlap "
            "with the AlphaQuantum target tensor. This tests whether the "
            "`low-weight<=2` dictionary was too restrictive.",
            "",
        ]
    )
    tensor_overlap_table = []
    for row in tensor_overlap_rows:
        tensor_overlap_table.append(
            [
                row.get("target"),
                row.get("mode"),
                row.get("num_actions"),
                row.get("tensor_overlap_max_weight"),
                row.get("tensor_overlap_max_actions_per_target"),
                row.get("best_return_residual_weight"),
                row.get("avg_split_sum_rewards_final"),
            ]
        )
    lines.extend(
        _markdown_table(
            tensor_overlap_table,
            [
                "target",
                "mode",
                "actions",
                "max weight",
                "per-target extras",
                "best residual",
                "avg split reward",
            ],
        )
    )
    lines.extend(
        [
            "",
            "Interpretation: with 194 actions, `tensor-overlap` does not beat "
            "the smaller `low-weight<=2` dictionary in the 1000-step local "
            "probe. It worsens `mod_5_4`, `gf_2pow2_mult`, and "
            "`hamming_weight_n4`, and only ties the `hamming_weight_n5` residual "
            "at 144. This suggests the current overlap scoring admits too many "
            "distracting high-weight factors; `low-weight<=2` remains the "
            "recommended local/cluster default.",
            "",
            "## split4 Short Ablation",
            "",
            "This run intentionally excludes budgeted modes because the 200-step "
            "`none` baseline did not solve the four targets.",
            "",
        ]
    )
    split4_table = []
    for row in ablation_rows:
        split4_table.append(
            [
                row.get("mode"),
                row.get("target"),
                row.get("best_effective_t_cost"),
                row.get("best_return_residual_weight"),
                row.get("avg_split_sum_rewards_final"),
            ]
        )
    lines.extend(
        _markdown_table(
            split4_table,
            [
                "mode",
                "target",
                "solved T-cost",
                "best residual",
                "avg split reward",
            ],
        )
    )
    lines.extend(
        [
            "",
            "## External Benchmark Context",
            "",
            "The `alphaq_split_reward_v1_tiebreak` row is the newly materialized "
            "`mod_5_4` candidate from this sprint. Other rows are context from "
            "the existing verified benchmark pipeline.",
            "",
        ]
    )
    benchmark_table = []
    for row in benchmark_rows:
        benchmark_table.append(
            [
                row.get("circuit_id"),
                row.get("method"),
                row.get("tcount_after"),
                row.get("primary_nc_depth_ratio"),
                row.get("zx_best_clifford_fraction"),
            ]
        )
    lines.extend(
        _markdown_table(
            benchmark_table,
            [
                "target",
                "method",
                "T-count",
                "primary ratio",
                "Clifford fraction",
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Long-Run Harness",
            "",
            "Longer runs should use `scripts/run_split_reward_target_sweep.py` "
            "or the wrapper `scripts/run_split_reward_cluster_sweep.sh`. The "
            "SLURM template is `scripts/slurm_split_reward_target_sweep.sbatch`. "
            "These wrappers keep budgets explicit, write per-run logs, stop on "
            "timeouts, and materialize candidates only after exact tensor "
            "solutions. A timeout smoke test confirmed that unfinished runs are "
            "recorded as `timeout` rather than silently treated as failed "
            "scientific evidence.",
            "",
            "Recommended cluster command shape:",
            "",
            "```bash",
            "TARGETS=gf_2pow2_mult,hamming_weight_n4,hamming_weight_n5 \\",
            "MODES=none,v1,v2_progress \\",
            "TRAINING_STEPS=5000 EVAL_FREQUENCY=100 \\",
            "BATCH_SIZE=32 NUM_MCTS_SIMULATIONS=16 \\",
            "ACTION_DICTIONARY=low-weight MAX_ACTION_WEIGHT=2 \\",
            "MASK_PADDED_ACTIONS=1 FORCE_CANONICAL_BASIS=1 \\",
            "RUN_LABEL=split_reward_long scripts/run_split_reward_cluster_sweep.sh",
            "```",
            "",
            "Set `FORCE_CANONICAL_BASIS=0` only for a separate candidate-generation "
            "study that intentionally uses AlphaQuantum's original basis-mix "
            "machinery; it should not be mixed with the canonical reward test.",
            "",
            "A focused follow-up ablation can replace the dictionary line with "
            "`ACTION_DICTIONARY=gadget-closure MAX_ACTION_WEIGHT=2 "
            "GADGET_CLOSURE_MAX_WEIGHT=4`. This variant is not the default "
            "because it only improved `gf_2pow2_mult` slightly and tied "
            "`hamming_weight_n4` in the local canonical probe.",
            "",
            "For a true multi-target run where `mask_padded_actions` changes the "
            "MCTS policy, run `scripts/run_demo_train.py --target-preset "
            "split4 --action-dictionary low-weight --max-action-weight 2 "
            "--mask-padded-actions`. In per-target sweeps, each environment "
            "already uses the target tensor size, so the mask is mostly an "
            "audit flag rather than a search-space reduction.",
            "",
            "## Next Step",
            "",
            "The next scientifically meaningful step is to run the new "
            "`run_split_reward_target_sweep.py` harness on a larger machine "
            "or with a more efficient action dictionary. It now has explicit "
            "timeouts, no inferred budgets, and automatic materialization only "
            "for solved candidates. The strongest current claim is one-circuit "
            "external support plus weak search-direction evidence on "
            "`gf_2pow2_mult`, not a broad benchmark win.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation-csv", type=Path, default=DEFAULT_ABLATION_CSV)
    parser.add_argument("--benchmark-csv", type=Path, default=DEFAULT_BENCHMARK_CSV)
    parser.add_argument(
        "--lowweight-sweep-csv",
        type=Path,
        default=DEFAULT_LOWWEIGHT_SWEEP_CSV,
    )
    parser.add_argument(
        "--canonical-sweep-csv",
        type=Path,
        default=DEFAULT_CANONICAL_SWEEP_CSV,
    )
    parser.add_argument(
        "--gadget-closure-sweep-csv",
        type=Path,
        action="append",
        default=list(DEFAULT_GADGET_CLOSURE_SWEEP_CSVS),
    )
    parser.add_argument(
        "--external-split-reward-metrics",
        type=Path,
        default=DEFAULT_EXTERNAL_SPLIT_REWARD_METRICS,
    )
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    args = parser.parse_args()

    control_rows, control_summaries = _control_rows(args.log_dir)
    individual_rows, _ = _individual_rows(args.log_dir)
    action_rows = _action_dictionary_rows(args.log_dir)
    lowweight_rows = _lowweight_sweep_rows(args.lowweight_sweep_csv)
    canonical_rows = _canonical_probe_rows(
        args.canonical_sweep_csv,
        "canonical_reward_probe",
    )
    gadget_closure_rows = [
        row
        for path in args.gadget_closure_sweep_csv
        for row in _canonical_probe_rows(path, "gadget_closure_probe")
    ]
    masked_rows = _masked_split4_rows(args.log_dir)
    tensor_overlap_rows = _tensor_overlap_rows(args.log_dir)
    ablation_rows = _ablation_rows(args.ablation_csv)
    benchmark_rows = _benchmark_rows(
        args.benchmark_csv,
        args.external_split_reward_metrics,
    )
    all_rows = [
        *control_rows,
        *individual_rows,
        *action_rows,
        *lowweight_rows,
        *canonical_rows,
        *gadget_closure_rows,
        *masked_rows,
        *tensor_overlap_rows,
        *ablation_rows,
    ]
    _write_csv(args.output_csv, all_rows)

    figures = [
        _plot_control_curves(control_summaries, args.figure_dir),
        _plot_individual_residual(individual_rows, args.figure_dir),
        _plot_action_dictionary_probe(action_rows, args.figure_dir),
        _plot_lowweight_reward_probe(lowweight_rows, action_rows, args.figure_dir),
        _plot_canonical_reward_probe(canonical_rows, args.figure_dir),
        _plot_gadget_closure_probe(
            canonical_rows, gadget_closure_rows, args.figure_dir
        ),
        _plot_masked_split4_probe(masked_rows, args.figure_dir),
        _plot_tensor_overlap_probe(
            masked_rows, tensor_overlap_rows, args.figure_dir
        ),
        _plot_split4_residual(ablation_rows, args.figure_dir),
        _plot_external_benchmarks(benchmark_rows, args.figure_dir),
    ]
    _write_report(
        args.report_path,
        args.output_csv,
        control_rows,
        individual_rows,
        action_rows,
        lowweight_rows,
        canonical_rows,
        gadget_closure_rows,
        masked_rows,
        tensor_overlap_rows,
        ablation_rows,
        benchmark_rows,
        figures,
    )
    print(json.dumps({
        "output_csv": str(args.output_csv),
        "report": str(args.report_path),
        "figures": [str(path) for path in figures],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
