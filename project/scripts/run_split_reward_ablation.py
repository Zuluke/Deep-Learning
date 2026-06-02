from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._manifest import append_command

DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "results" / "csv" / "split_reward_ablation.csv"
DEFAULT_REPORT = PROJECT_ROOT / "results" / "reports" / "split_reward_ablation.md"
DEFAULT_LOG_DIR = PROJECT_ROOT / "results" / "logs" / "demo_split_reward_ablation"
MODES = (
    "none",
    "mixed_drop",
    "mixed_auc",
    "v1",
    "v2_progress",
    "v3_frontier",
    "v4_sticky_frontier",
    "v1_guarded",
    "v1_tiebreak",
)
BUDGETED_MODES = {"v1_guarded", "v1_tiebreak"}


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _run_demo_train(args: argparse.Namespace, mode: str, summary_path: Path) -> dict:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_demo_train.py"),
        "--mode",
        "quick",
        "--profile",
        args.profile,
        "--target-preset",
        args.target_preset,
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
        "--seed",
        str(args.seed),
        "--log-dir",
        str(args.log_dir),
        "--summary-json",
        str(summary_path),
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
        "--t-guard-delta",
        str(args.t_guard_delta),
        "--interim-budget-slack",
        str(args.interim_budget_slack),
        "--terminal-tiebreak-clip",
        str(args.terminal_tiebreak_clip),
        "--frontier-replay-fraction",
        str(args.frontier_replay_fraction),
        "--frontier-replay-min-moves",
        str(args.frontier_replay_min_moves),
        "--frontier-replay-min-residual-drop",
        str(args.frontier_replay_min_residual_drop),
    ]
    if not args.canonical_only:
        cmd.append("--no-canonical-only")
    if args.force_canonical_basis:
        cmd.append("--force-canonical-basis")
    if mode in BUDGETED_MODES:
        if not args.baseline_t_costs:
            raise ValueError(f"{mode} requires baseline_t_costs.")
        cmd.extend(["--baseline-t-costs", args.baseline_t_costs])

    process = subprocess.run(cmd, cwd=PROJECT_ROOT, check=False)
    if process.returncode != 0:
        raise RuntimeError(
            f"run_demo_train failed for split reward mode {mode!r} "
            f"with exit code {process.returncode}."
        )
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _guard_budgets_from_none(summary: dict) -> str:
    budgets = []
    missing_targets = []
    for target, cost in zip(
        summary["target_circuits"], summary["best_effective_t_cost"], strict=True
    ):
        if cost is None:
            missing_targets.append(target)
        else:
            budgets.append(f"{target}={cost}")
    if missing_targets:
        names = ", ".join(missing_targets)
        raise RuntimeError(
            "Cannot run budgeted split-reward modes because the baseline "
            f"`none` run did not solve: {names}. Increase the baseline budget, "
            "pass explicit --baseline-t-costs, or select only unbudgeted modes "
            "with --modes."
        )
    return ",".join(budgets)


def _parse_modes(raw_modes: str) -> tuple[str, ...]:
    modes = tuple(mode.strip() for mode in raw_modes.split(",") if mode.strip())
    unknown = sorted(set(modes) - set(MODES))
    if unknown:
        raise ValueError(f"Unknown modes {unknown}. Expected one of {MODES}.")
    if not modes:
        raise ValueError("At least one mode must be selected.")
    return modes


def _final_history_value(summary: dict, key: str, index: int) -> float | None:
    history = summary.get("history") or []
    if not history:
        return None
    values = history[-1].get(key)
    if values is None:
        return None
    return values[index]


def _summary_value(summary: dict, key: str, index: int) -> object:
    values = summary.get(key) or []
    return values[index] if index < len(values) else None


def _rows_for_summary(summary: dict, summary_path: Path) -> list[dict[str, object]]:
    rows = []
    targets = summary["target_circuits"]
    for index, target in enumerate(targets):
        rows.append(
            {
                "mode": summary["split_reward_mode"],
                "target": target,
                "best_return": summary["best_return"][index],
                "best_tcount_from_return": summary["best_tcount_from_return"][index],
                "best_effective_t_cost": summary["best_effective_t_cost"][index],
                "best_return_effective_t_cost": _summary_value(
                    summary, "best_return_effective_t_cost", index
                ),
                "best_return_num_moves": _summary_value(
                    summary, "best_return_num_moves", index
                ),
                "best_return_residual_weight": _summary_value(
                    summary, "best_return_residual_weight", index
                ),
                "best_frontier_residual_weight": _summary_value(
                    summary, "best_frontier_residual_weight", index
                ),
                "best_frontier_effective_t_cost": _summary_value(
                    summary, "best_frontier_effective_t_cost", index
                ),
                "best_frontier_num_moves": _summary_value(
                    summary, "best_frontier_num_moves", index
                ),
                "avg_return_final": _final_history_value(
                    summary, "avg_return", index
                ),
                "avg_split_sum_rewards_final": _final_history_value(
                    summary, "avg_split_sum_rewards", index
                ),
                "avg_split_mixed_auc_sum_final": _final_history_value(
                    summary, "avg_split_mixed_auc_sum", index
                ),
                "avg_split_mixed_mass_sum_final": _final_history_value(
                    summary, "avg_split_mixed_mass_sum", index
                ),
                "frontier_replay_fraction": summary.get(
                    "frontier_replay_fraction", 0.0
                ),
                "frontier_replay_min_moves": summary.get(
                    "frontier_replay_min_moves", 0
                ),
                "frontier_replay_min_residual_drop": summary.get(
                    "frontier_replay_min_residual_drop", 0.0
                ),
                "summary_json": str(summary_path),
                "candidate_manifest_path": summary.get("candidate_manifest_path"),
            }
        )
    return rows


def _write_csv(rows: list[dict[str, object]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "mode",
        "target",
        "best_return",
        "best_tcount_from_return",
        "best_effective_t_cost",
        "best_return_effective_t_cost",
        "best_return_num_moves",
        "best_return_residual_weight",
        "best_frontier_residual_weight",
        "best_frontier_effective_t_cost",
        "best_frontier_num_moves",
        "avg_return_final",
        "avg_split_sum_rewards_final",
        "avg_split_mixed_auc_sum_final",
        "avg_split_mixed_mass_sum_final",
        "frontier_replay_fraction",
        "frontier_replay_min_moves",
        "frontier_replay_min_residual_drop",
        "summary_json",
        "candidate_manifest_path",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(rows: list[dict[str, object]], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Split reward ablation",
        "",
        "Short AlphaQuantum-only ablation over reward modes. The guarded mode "
        "and tiebreak mode use either explicit budgets or the solved effective "
        "T-costs from the `none` run. If the baseline does not solve a target, "
        "the budgeted modes are skipped unless a budget is provided explicitly.",
        "",
        "| mode | target | best return | best effective T-cost | frontier residual | avg split reward |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {mode} | {target} | {best_return} | {best_effective_t_cost} | "
            "{best_frontier_residual_weight} | "
            "{avg_split_sum_rewards_final} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Interpretation note: when `split_reward_mode != none`, `best_return` "
            "is no longer the negative T-count; use `best_effective_t_cost` for "
            "the gadget-aware T-cost proxy.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a short AlphaQuantum split-reward ablation."
    )
    parser.add_argument("--profile", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--target-preset", default="split4")
    parser.add_argument("--training-steps", type=int, default=200)
    parser.add_argument("--eval-frequency", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-mcts-simulations", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--use-gadgets", choices=("on", "off"), default="on")
    parser.add_argument("--lambda-drop", type=float, default=0.05)
    parser.add_argument("--lambda-auc", type=float, default=0.05)
    parser.add_argument("--lambda-mass", type=float, default=0.05)
    parser.add_argument("--lambda-residual", type=float, default=0.10)
    parser.add_argument("--lambda-frontier", type=float, default=0.10)
    parser.add_argument("--lambda-budget", type=float, default=1.0)
    parser.add_argument("--drop-clip", type=float, default=1.0)
    parser.add_argument("--t-guard-delta", type=float, default=0.0)
    parser.add_argument("--interim-budget-slack", type=float, default=5.0)
    parser.add_argument("--terminal-tiebreak-clip", type=float, default=0.25)
    parser.add_argument("--frontier-replay-fraction", type=float, default=0.0)
    parser.add_argument("--frontier-replay-min-moves", type=int, default=0)
    parser.add_argument("--frontier-replay-min-residual-drop", type=float, default=0.0)
    parser.add_argument(
        "--modes",
        default=",".join(MODES),
        help="Comma-separated subset of reward modes to run.",
    )
    parser.add_argument(
        "--baseline-t-costs",
        default=None,
        help=(
            "Explicit budgets for v1_guarded/v1_tiebreak, as name=value pairs "
            "or colon-separated values in target order."
        ),
    )
    parser.add_argument(
        "--canonical-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--force-canonical-basis", action="store_true")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_id = _timestamp()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir = args.log_dir / f"summaries_{run_id}"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, dict] = {}
    rows: list[dict[str, object]] = []
    modes = _parse_modes(args.modes)
    explicit_baseline_t_costs = args.baseline_t_costs
    for mode in modes:
        if mode in BUDGETED_MODES:
            if explicit_baseline_t_costs is not None:
                args.baseline_t_costs = explicit_baseline_t_costs
            else:
                if "none" not in summaries:
                    raise RuntimeError(
                        f"{mode} requires budgets. Run `none` first in --modes "
                        "or pass explicit --baseline-t-costs."
                    )
                args.baseline_t_costs = _guard_budgets_from_none(
                    summaries["none"]
                )
        else:
            args.baseline_t_costs = None
        summary_path = summaries_dir / f"{mode}.json"
        summaries[mode] = _run_demo_train(args, mode, summary_path)
        rows.extend(_rows_for_summary(summaries[mode], summary_path))

    _write_csv(rows, args.output_csv)
    _write_report(rows, args.report_path)
    append_command(
        {
            "tool": "run_split_reward_ablation.py",
            "command": " ".join(sys.argv),
            "cwd": str(PROJECT_ROOT),
            "output_csv": str(args.output_csv),
            "report_path": str(args.report_path),
            "exit_code": 0,
        }
    )
    print(json.dumps({"output_csv": str(args.output_csv), "report": str(args.report_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
