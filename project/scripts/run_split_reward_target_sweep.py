from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._manifest import append_command

DEFAULT_OUTPUT_CSV = (
    PROJECT_ROOT / "results" / "csv" / "split_reward_target_sweep.csv"
)
DEFAULT_REPORT = (
    PROJECT_ROOT / "results" / "reports" / "split_reward_target_sweep.md"
)
DEFAULT_LOG_ROOT = PROJECT_ROOT / "results" / "logs" / "split_reward_target_sweep"
DEFAULT_CANDIDATE_ROOT = PROJECT_ROOT / "results" / "alphaq_split_reward"
DEFAULT_MATERIALIZED_ROOT = (
    PROJECT_ROOT / "results" / "alphaq_split_reward_external"
)
TARGETS = (
    "mod_5_4",
    "gf_2pow2_mult",
    "hamming_weight_n4",
    "hamming_weight_n5",
)
MODES = (
    "none",
    "v1",
    "v2_progress",
    "v3_frontier",
    "v4_sticky_frontier",
    "v5_barrier_frontier",
    "v1_guarded",
    "v1_tiebreak",
)
BUDGETED_MODES = {"v1_guarded", "v1_tiebreak"}


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_csv_list(raw: str, allowed: tuple[str, ...], label: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError(f"At least one {label} must be selected.")
    unknown = sorted(set(values) - set(allowed))
    if unknown:
        raise ValueError(f"Unknown {label}: {unknown}. Expected one of {allowed}.")
    return values


def parse_budget_map(raw: str | None) -> dict[str, float]:
    if not raw:
        return {}
    budgets: dict[str, float] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        name, value = item.split("=", maxsplit=1)
        budgets[name.strip()] = float(value)
    return budgets


def _action_tag(args: argparse.Namespace) -> str:
    if args.action_dictionary == "low-weight":
        return f"loww{args.max_action_weight}"
    if args.action_dictionary == "tensor-overlap":
        return (
            f"tensoroverlap{args.tensor_overlap_max_actions_per_target}_"
            f"w{args.tensor_overlap_max_weight}_base{args.max_action_weight}"
        )
    if args.action_dictionary == "gadget-closure":
        return (
            f"gadgetclosure{args.gadget_closure_max_weight}_"
            f"base{args.max_action_weight}"
        )
    return "full"


def _basis_tag(args: argparse.Namespace) -> str:
    return "canonical" if args.force_canonical_basis else "basismix"


def _prior_tag(args: argparse.Namespace) -> str:
    if args.action_prior == "none":
        return "noprior"
    top_k_tag = (
        f"_topk{args.action_prior_top_k:g}"
        if getattr(args, "action_prior_top_k", 0)
        else ""
    )
    return (
        f"{args.action_prior}prior"
        f"_beta{args.action_prior_beta:g}"
        f"_r{args.action_prior_residual_weight:g}"
        f"_md{args.action_prior_mixed_drop_weight:g}"
        f"_mm{args.action_prior_mixed_mass_weight:g}"
        f"{top_k_tag}"
    )


def _replay_tag(args: argparse.Namespace) -> str:
    fraction = getattr(args, "frontier_replay_fraction", 0.0)
    if not fraction:
        return "noreplay"
    min_moves = getattr(args, "frontier_replay_min_moves", 0)
    min_drop = getattr(args, "frontier_replay_min_residual_drop", 0.0)
    return (
        f"replay{fraction:g}_minm{min_moves:g}_mindrop{min_drop:g}"
    ).replace(".", "p")


def _partition_tag(args: argparse.Namespace) -> str:
    return f"pi-{args.partition_preset.replace('_', '-').replace(' ', '-')}"


def _horizon_tag(args: argparse.Namespace) -> str:
    max_moves = getattr(args, "max_num_moves", 0)
    observed = getattr(args, "num_past_factors_to_observe", 0)
    return f"h{max_moves:g}_obs{observed:g}"


def _mask_tag(args: argparse.Namespace) -> str:
    padded = "padmask" if getattr(args, "mask_padded_actions", True) else "nopadmask"
    repeated = (
        "norepeat"
        if getattr(args, "mask_repeated_actions", False)
        else "repeatok"
    )
    return f"{padded}_{repeated}"


def _summary_value(summary: dict[str, Any] | None, key: str, index: int = 0) -> Any:
    if summary is None:
        return None
    values = summary.get(key)
    if isinstance(values, list) and index < len(values):
        return values[index]
    return values


def _final_history_value(
    summary: dict[str, Any] | None,
    key: str,
    index: int = 0,
) -> Any:
    if summary is None:
        return None
    history = summary.get("history") or []
    if not history:
        return None
    values = history[-1].get(key)
    if isinstance(values, list) and index < len(values):
        return values[index]
    return values


def _run_command(
    cmd: list[str],
    *,
    cwd: Path,
    log_path: Path,
    timeout_sec: int | None,
) -> tuple[str, int | None, str | None]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        try:
            completed = subprocess.run(
                cmd,
                cwd=cwd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_sec,
                check=False,
            )
        except subprocess.TimeoutExpired:
            log_file.write(
                f"\nTIMEOUT after {timeout_sec} seconds: {' '.join(cmd)}\n"
            )
            return "timeout", None, f"timeout after {timeout_sec}s"
    if completed.returncode == 0:
        return "ok", completed.returncode, None
    return "failed", completed.returncode, f"exit code {completed.returncode}"


def _run_training(
    *,
    target: str,
    mode: str,
    args: argparse.Namespace,
    run_root: Path,
    budgets: dict[str, float],
) -> dict[str, Any]:
    run_name = (
        f"{target}_{mode}_{args.training_steps}_eval{args.eval_frequency}_"
        f"b{args.batch_size}_m{args.num_mcts_simulations}_"
        f"{_action_tag(args)}_{_prior_tag(args)}_{_partition_tag(args)}_"
        f"{_basis_tag(args)}_{_horizon_tag(args)}_{_mask_tag(args)}_"
        f"{_replay_tag(args)}_seed{args.seed}"
    )
    summary_path = run_root / "summaries" / f"{run_name}.json"
    log_path = run_root / "logs" / f"{run_name}.log"
    candidate_dir = args.candidate_root / run_name
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_demo_train.py"),
        "--mode",
        "quick",
        "--profile",
        args.profile,
        "--target-preset",
        target,
        "--split-reward-mode",
        mode,
        "--use-gadgets",
        args.use_gadgets,
        "--training-steps",
        str(args.training_steps),
        "--eval-frequency",
        str(args.eval_frequency),
        "--batch-size",
        str(args.batch_size),
        "--num-mcts-simulations",
        str(args.num_mcts_simulations),
        "--action-dictionary",
        args.action_dictionary,
        "--max-action-weight",
        str(args.max_action_weight),
        "--tensor-overlap-max-weight",
        str(args.tensor_overlap_max_weight),
        "--tensor-overlap-max-actions-per-target",
        str(args.tensor_overlap_max_actions_per_target),
        "--gadget-closure-max-weight",
        str(args.gadget_closure_max_weight),
        "--max-num-moves",
        str(args.max_num_moves),
        "--num-past-factors-to-observe",
        str(args.num_past_factors_to_observe),
        "--lambda-drop",
        str(args.lambda_drop),
        "--lambda-auc",
        str(args.lambda_auc),
        "--lambda-mass",
        str(args.lambda_mass),
        "--lambda-residual",
        str(args.lambda_residual),
        "--lambda-frontier",
        str(args.lambda_frontier),
        "--lambda-budget",
        str(args.lambda_budget),
        "--drop-clip",
        str(args.drop_clip),
        "--action-prior",
        args.action_prior,
        "--action-prior-beta",
        str(args.action_prior_beta),
        "--action-prior-residual-weight",
        str(args.action_prior_residual_weight),
        "--action-prior-mixed-drop-weight",
        str(args.action_prior_mixed_drop_weight),
        "--action-prior-mixed-mass-weight",
        str(args.action_prior_mixed_mass_weight),
        "--action-prior-hamming-weight",
        str(args.action_prior_hamming_weight),
        "--action-prior-gadget-bonus",
        str(args.action_prior_gadget_bonus),
        "--action-prior-top-k",
        str(args.action_prior_top_k),
        "--frontier-replay-fraction",
        str(args.frontier_replay_fraction),
        "--frontier-replay-min-moves",
        str(args.frontier_replay_min_moves),
        "--frontier-replay-min-residual-drop",
        str(args.frontier_replay_min_residual_drop),
        "--partition-preset",
        args.partition_preset,
        "--seed",
        str(args.seed),
        "--candidate-output-dir",
        str(candidate_dir),
        "--summary-json",
        str(summary_path),
    ]
    if args.mask_padded_actions:
        cmd.append("--mask-padded-actions")
    else:
        cmd.append("--no-mask-padded-actions")
    if args.mask_repeated_actions:
        cmd.append("--mask-repeated-actions")
    else:
        cmd.append("--no-mask-repeated-actions")
    if not args.canonical_only:
        cmd.append("--no-canonical-only")
    if args.force_canonical_basis:
        cmd.append("--force-canonical-basis")
    if not args.action_prior_standardize:
        cmd.append("--no-action-prior-standardize")
    if not args.action_prior_canonical_only:
        cmd.append("--no-action-prior-canonical-only")
    if mode in BUDGETED_MODES:
        budget = budgets.get(target)
        if budget is None:
            return {
                "target": target,
                "mode": mode,
                "status": "skipped-budget-missing",
                "error": "budgeted mode requires explicit target budget",
                "summary_json": str(summary_path),
                "stdout_log": str(log_path),
            }
        cmd.extend(["--baseline-t-costs", f"{target}={budget}"])
    status, return_code, error = _run_command(
        cmd,
        cwd=PROJECT_ROOT,
        log_path=log_path,
        timeout_sec=args.timeout_sec,
    )
    summary = None
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    row = {
        "target": target,
        "mode": mode,
        "status": status,
        "error": error,
        "return_code": return_code,
        "summary_json": str(summary_path) if summary_path.exists() else "",
        "stdout_log": str(log_path),
        "candidate_output_dir": str(candidate_dir),
        "candidate_manifest_path": "" if summary is None else summary.get("candidate_manifest_path"),
        "best_effective_t_cost": _summary_value(summary, "best_effective_t_cost"),
        "best_return": _summary_value(summary, "best_return"),
        "best_return_effective_t_cost": _summary_value(
            summary, "best_return_effective_t_cost"
        ),
        "best_return_num_moves": _summary_value(summary, "best_return_num_moves"),
        "best_return_residual_weight": _summary_value(
            summary, "best_return_residual_weight"
        ),
        "target_tensor_weight": _summary_value(summary, "target_tensor_weight"),
        "best_frontier_residual_weight": _summary_value(
            summary, "best_frontier_residual_weight"
        ),
        "best_frontier_residual_drop": _summary_value(
            summary, "best_frontier_residual_drop"
        ),
        "best_frontier_effective_t_cost": _summary_value(
            summary, "best_frontier_effective_t_cost"
        ),
        "best_frontier_num_moves": _summary_value(
            summary, "best_frontier_num_moves"
        ),
        "avg_return_final": _final_history_value(summary, "avg_return"),
        "avg_split_sum_rewards_final": _final_history_value(
            summary, "avg_split_sum_rewards"
        ),
        "avg_split_mixed_auc_sum_final": _final_history_value(
            summary, "avg_split_mixed_auc_sum"
        ),
        "avg_split_mixed_mass_sum_final": _final_history_value(
            summary, "avg_split_mixed_mass_sum"
        ),
        "training_steps": args.training_steps,
        "eval_frequency": args.eval_frequency,
        "batch_size": args.batch_size,
        "num_mcts_simulations": args.num_mcts_simulations,
        "action_dictionary": args.action_dictionary,
        "max_action_weight": args.max_action_weight,
        "tensor_overlap_max_weight": args.tensor_overlap_max_weight,
        "tensor_overlap_max_actions_per_target": (
            args.tensor_overlap_max_actions_per_target
        ),
        "gadget_closure_max_weight": args.gadget_closure_max_weight,
        "max_num_moves": args.max_num_moves,
        "num_past_factors_to_observe": args.num_past_factors_to_observe,
        "lambda_drop": args.lambda_drop,
        "lambda_auc": args.lambda_auc,
        "lambda_mass": args.lambda_mass,
        "lambda_residual": args.lambda_residual,
        "lambda_frontier": args.lambda_frontier,
        "lambda_budget": args.lambda_budget,
        "drop_clip": args.drop_clip,
        "action_prior": args.action_prior,
        "action_prior_beta": args.action_prior_beta,
        "action_prior_residual_weight": args.action_prior_residual_weight,
        "action_prior_mixed_drop_weight": args.action_prior_mixed_drop_weight,
        "action_prior_mixed_mass_weight": args.action_prior_mixed_mass_weight,
        "action_prior_hamming_weight": args.action_prior_hamming_weight,
        "action_prior_gadget_bonus": args.action_prior_gadget_bonus,
        "action_prior_standardize": args.action_prior_standardize,
        "action_prior_top_k": args.action_prior_top_k,
        "action_prior_canonical_only": args.action_prior_canonical_only,
        "frontier_replay_fraction": args.frontier_replay_fraction,
        "frontier_replay_min_moves": args.frontier_replay_min_moves,
        "frontier_replay_min_residual_drop": (
            args.frontier_replay_min_residual_drop
        ),
        "partition_preset": args.partition_preset,
        "mask_padded_actions": args.mask_padded_actions,
        "mask_repeated_actions": args.mask_repeated_actions,
        "canonical_only": args.canonical_only,
        "force_canonical_basis": args.force_canonical_basis,
        "seed": args.seed,
    }
    return row


def _materialize_if_solved(
    row: dict[str, Any],
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if row.get("status") != "ok" or row.get("best_effective_t_cost") in (None, "", "None"):
        row.update(
            {
                "materialization_status": "not-solved",
                "materialization_error": "",
                "materialized_summary_json": "",
                "materialized_metrics_json": "",
                "primary_nc_depth_ratio": "",
                "tcount_after": "",
                "qasm_depth_ratio": "",
            }
        )
        return row
    manifest = row.get("candidate_manifest_path")
    if not manifest:
        row.update(
            {
                "materialization_status": "missing-manifest",
                "materialization_error": "missing candidate manifest",
                "materialized_summary_json": "",
                "materialized_metrics_json": "",
                "primary_nc_depth_ratio": "",
                "tcount_after": "",
                "qasm_depth_ratio": "",
            }
        )
        return row
    output_root = args.materialized_root / f"{row['target']}_{row['mode']}_sweep"
    log_path = output_root / "materialize.log"
    summary_path = output_root / "summary.json"
    metrics_path = output_root / "structural_metrics_from_qasm_original.json"
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "materialize_split_reward_candidate.py"),
        "--manifest-csv",
        str(manifest),
        "--target",
        str(row["target"]),
        "--candidate-kind",
        "best_solved",
        "--split-reward-mode",
        str(row["mode"]),
        "--output-root",
        str(output_root),
    ]
    status, _, error = _run_command(
        cmd,
        cwd=PROJECT_ROOT,
        log_path=log_path,
        timeout_sec=args.materialize_timeout_sec,
    )
    metrics = {}
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    row.update(
        {
            "materialization_status": status,
            "materialization_error": error or "",
            "materialized_summary_json": str(summary_path) if summary_path.exists() else "",
            "materialized_metrics_json": str(metrics_path) if metrics_path.exists() else "",
            "primary_nc_depth_ratio": metrics.get("primary_nc_depth_ratio", ""),
            "tcount_after": metrics.get("tcount_after", ""),
            "qasm_depth_ratio": metrics.get("qasm_depth_ratio", ""),
        }
    )
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "target",
        "mode",
        "status",
        "error",
        "return_code",
        "best_effective_t_cost",
        "best_return",
        "best_return_effective_t_cost",
        "best_return_num_moves",
        "best_return_residual_weight",
        "target_tensor_weight",
        "best_frontier_residual_weight",
        "best_frontier_residual_drop",
        "best_frontier_effective_t_cost",
        "best_frontier_num_moves",
        "avg_return_final",
        "avg_split_sum_rewards_final",
        "avg_split_mixed_auc_sum_final",
        "avg_split_mixed_mass_sum_final",
        "materialization_status",
        "materialization_error",
        "primary_nc_depth_ratio",
        "tcount_after",
        "qasm_depth_ratio",
        "training_steps",
        "eval_frequency",
        "batch_size",
        "num_mcts_simulations",
        "action_dictionary",
        "max_action_weight",
        "tensor_overlap_max_weight",
        "tensor_overlap_max_actions_per_target",
        "gadget_closure_max_weight",
        "max_num_moves",
        "num_past_factors_to_observe",
        "lambda_drop",
        "lambda_auc",
        "lambda_mass",
        "lambda_residual",
        "lambda_frontier",
        "lambda_budget",
        "drop_clip",
        "action_prior",
        "action_prior_beta",
        "action_prior_residual_weight",
        "action_prior_mixed_drop_weight",
        "action_prior_mixed_mass_weight",
        "action_prior_hamming_weight",
        "action_prior_gadget_bonus",
        "action_prior_standardize",
        "action_prior_top_k",
        "action_prior_canonical_only",
        "frontier_replay_fraction",
        "frontier_replay_min_moves",
        "frontier_replay_min_residual_drop",
        "partition_preset",
        "mask_padded_actions",
        "mask_repeated_actions",
        "canonical_only",
        "force_canonical_basis",
        "seed",
        "summary_json",
        "stdout_log",
        "candidate_output_dir",
        "candidate_manifest_path",
        "materialized_summary_json",
        "materialized_metrics_json",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict[str, Any]], output_csv: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Split reward target sweep",
        "",
        f"CSV: `{output_csv}`.",
        "",
        "Budgeted modes are only run when an explicit budget is provided. "
        "Unsolved rows are not materialized.",
        "",
        "| target | mode | basis | status | solved cost | terminal residual | frontier residual | frontier drop | primary ratio | materialization |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {target} | {mode} | {basis} | {status} | {cost} | {residual} | {frontier} | {drop} | {primary} | {mat} |".format(
                target=row.get("target", ""),
                mode=row.get("mode", ""),
                basis="canonical" if row.get("force_canonical_basis") else "basis-mix",
                status=row.get("status", ""),
                cost=row.get("best_effective_t_cost") or "",
                residual=row.get("best_return_residual_weight") or "",
                frontier=row.get("best_frontier_residual_weight") or "",
                drop=row.get("best_frontier_residual_drop") or "",
                primary=row.get("primary_nc_depth_ratio") or "",
                mat=row.get("materialization_status") or "",
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run per-target AlphaQuantum split-reward sweeps with timeout and audit outputs."
    )
    parser.add_argument("--targets", default=",".join(TARGETS))
    parser.add_argument("--modes", default="none,v1")
    parser.add_argument("--profile", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--use-gadgets", choices=("on", "off"), default="on")
    parser.add_argument("--training-steps", type=int, default=1000)
    parser.add_argument("--eval-frequency", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-mcts-simulations", type=int, default=16)
    parser.add_argument(
        "--action-dictionary",
        choices=("full", "low-weight", "tensor-overlap", "gadget-closure"),
        default="full",
    )
    parser.add_argument("--max-action-weight", type=int, default=3)
    parser.add_argument("--tensor-overlap-max-weight", type=int, default=5)
    parser.add_argument(
        "--tensor-overlap-max-actions-per-target",
        type=int,
        default=128,
    )
    parser.add_argument("--gadget-closure-max-weight", type=int, default=4)
    parser.add_argument("--max-num-moves", type=int, default=0)
    parser.add_argument("--num-past-factors-to-observe", type=int, default=0)
    parser.add_argument("--lambda-drop", type=float, default=0.05)
    parser.add_argument("--lambda-auc", type=float, default=0.05)
    parser.add_argument("--lambda-mass", type=float, default=0.05)
    parser.add_argument("--lambda-residual", type=float, default=0.10)
    parser.add_argument("--lambda-frontier", type=float, default=0.10)
    parser.add_argument("--lambda-budget", type=float, default=1.0)
    parser.add_argument("--drop-clip", type=float, default=1.0)
    parser.add_argument(
        "--action-prior",
        choices=("none", "residual", "split"),
        default="none",
    )
    parser.add_argument("--action-prior-beta", type=float, default=1.0)
    parser.add_argument("--action-prior-residual-weight", type=float, default=1.0)
    parser.add_argument("--action-prior-mixed-drop-weight", type=float, default=1.0)
    parser.add_argument("--action-prior-mixed-mass-weight", type=float, default=0.25)
    parser.add_argument("--action-prior-hamming-weight", type=float, default=0.05)
    parser.add_argument("--action-prior-gadget-bonus", type=float, default=0.25)
    parser.add_argument("--action-prior-top-k", type=int, default=0)
    parser.add_argument("--frontier-replay-fraction", type=float, default=0.0)
    parser.add_argument("--frontier-replay-min-moves", type=int, default=0)
    parser.add_argument("--frontier-replay-min-residual-drop", type=float, default=0.0)
    parser.add_argument(
        "--action-prior-standardize",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--action-prior-canonical-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--partition-preset",
        choices=("balanced", "shifted-contiguous", "tensor-spectral", "ensemble"),
        default="balanced",
    )
    parser.add_argument(
        "--mask-padded-actions",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--mask-repeated-actions",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--timeout-sec", type=int, default=900)
    parser.add_argument("--materialize-timeout-sec", type=int, default=300)
    parser.add_argument("--baseline-t-costs", default=None)
    parser.add_argument("--canonical-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-canonical-basis", action="store_true")
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE_ROOT)
    parser.add_argument("--materialized-root", type=Path, default=DEFAULT_MATERIALIZED_ROOT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    targets = parse_csv_list(args.targets, TARGETS, "targets")
    modes = parse_csv_list(args.modes, MODES, "modes")
    budgets = parse_budget_map(args.baseline_t_costs)
    run_root = args.run_root or DEFAULT_LOG_ROOT / _timestamp()
    run_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for target in targets:
        for mode in modes:
            row = _run_training(
                target=target,
                mode=mode,
                args=args,
                run_root=run_root,
                budgets=budgets,
            )
            rows.append(_materialize_if_solved(row, args=args))
            write_csv(args.output_csv, rows)
            write_report(args.report_path, rows, args.output_csv)
    append_command(
        {
            "tool": "run_split_reward_target_sweep.py",
            "command": " ".join(sys.argv),
            "cwd": str(PROJECT_ROOT),
            "output_csv": str(args.output_csv),
            "report_path": str(args.report_path),
            "run_root": str(run_root),
            "exit_code": 0,
        }
    )
    print(json.dumps({"output_csv": str(args.output_csv), "report": str(args.report_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
