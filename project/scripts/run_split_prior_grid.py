from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._manifest import append_command

DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "results" / "csv" / "split_prior_grid.csv"
DEFAULT_REPORT = PROJECT_ROOT / "results" / "reports" / "split_prior_grid.md"
DEFAULT_RUN_ROOT = PROJECT_ROOT / "results" / "logs" / "split_prior_grid"
DEFAULT_CANDIDATE_ROOT = PROJECT_ROOT / "results" / "alphaq_split_prior_grid"
PARTITION_PRESETS = (
    "balanced",
    "shifted-contiguous",
    "tensor-spectral",
    "ensemble",
)


@dataclass(frozen=True)
class PriorGridConfig:
    partition_preset: str
    beta: float
    residual_weight: float
    mixed_drop_weight: float
    mixed_mass_weight: float
    top_k: int
    frontier_replay_fraction: float
    frontier_replay_min_moves: int
    frontier_replay_min_residual_drop: float

    @property
    def grid_id(self) -> str:
        top_k_tag = f"_topk{self.top_k}" if self.top_k else ""
        replay_tag = (
            f"_replay{self.frontier_replay_fraction:g}"
            f"_minm{self.frontier_replay_min_moves}"
            f"_mindrop{self.frontier_replay_min_residual_drop:g}"
            if self.frontier_replay_fraction
            else ""
        )
        return (
            f"pi-{self.partition_preset}"
            f"_beta{self.beta:g}"
            f"_r{self.residual_weight:g}"
            f"_md{self.mixed_drop_weight:g}"
            f"_mm{self.mixed_mass_weight:g}"
            f"{top_k_tag}"
            f"{replay_tag}"
        ).replace(".", "p")


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_csv_floats(raw: str, label: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError(f"At least one {label} value must be selected.")
    return values


def parse_csv_ints(raw: str, label: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError(f"At least one {label} value must be selected.")
    if any(value < 0 for value in values):
        raise ValueError(f"{label} values must be non-negative.")
    return values


def parse_csv_strings(
    raw: str,
    *,
    allowed: tuple[str, ...],
    label: str,
) -> tuple[str, ...]:
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError(f"At least one {label} must be selected.")
    unknown = sorted(set(values) - set(allowed))
    if unknown:
        raise ValueError(f"Unknown {label}: {unknown}. Expected one of {allowed}.")
    return values


def build_grid_configs(args: argparse.Namespace) -> list[PriorGridConfig]:
    partitions = parse_csv_strings(
        args.partition_presets,
        allowed=PARTITION_PRESETS,
        label="partition presets",
    )
    betas = parse_csv_floats(args.action_prior_betas, "action prior beta")
    residual_weights = parse_csv_floats(
        args.action_prior_residual_weights,
        "residual weight",
    )
    mixed_drop_weights = parse_csv_floats(
        args.action_prior_mixed_drop_weights,
        "mixed drop weight",
    )
    mixed_mass_weights = parse_csv_floats(
        args.action_prior_mixed_mass_weights,
        "mixed mass weight",
    )
    top_ks = parse_csv_ints(
        getattr(args, "action_prior_top_ks", None)
        or str(getattr(args, "action_prior_top_k", 0)),
        "action prior top-k",
    )
    replay_fractions = parse_csv_floats(
        getattr(args, "frontier_replay_fractions", None)
        or str(getattr(args, "frontier_replay_fraction", 0.0)),
        "frontier replay fraction",
    )
    if any(value < 0.0 or value > 1.0 for value in replay_fractions):
        raise ValueError("frontier replay fractions must be between 0 and 1.")
    replay_min_moves = parse_csv_ints(
        getattr(args, "frontier_replay_min_moves_values", None)
        or str(getattr(args, "frontier_replay_min_moves", 0)),
        "frontier replay min moves",
    )
    replay_min_drops = parse_csv_floats(
        getattr(args, "frontier_replay_min_residual_drops", None)
        or str(getattr(args, "frontier_replay_min_residual_drop", 0.0)),
        "frontier replay min residual drop",
    )
    if any(value < 0.0 for value in replay_min_drops):
        raise ValueError("frontier replay min residual drops must be non-negative.")
    return [
        PriorGridConfig(
            partition_preset=partition,
            beta=beta,
            residual_weight=residual_weight,
            mixed_drop_weight=mixed_drop_weight,
            mixed_mass_weight=mixed_mass_weight,
            top_k=top_k,
            frontier_replay_fraction=replay_fraction,
            frontier_replay_min_moves=min_moves,
            frontier_replay_min_residual_drop=min_drop,
        )
        for partition in partitions
        for beta in betas
        for residual_weight in residual_weights
        for mixed_drop_weight in mixed_drop_weights
        for mixed_mass_weight in mixed_mass_weights
        for top_k in top_ks
        for replay_fraction in replay_fractions
        for min_moves in replay_min_moves
        for min_drop in replay_min_drops
    ]


def _run_grid_config(
    config: PriorGridConfig,
    *,
    args: argparse.Namespace,
    run_root: Path,
) -> list[dict[str, Any]]:
    grid_root = run_root / config.grid_id
    output_csv = grid_root / "rows.csv"
    report_path = grid_root / "report.md"
    if args.resume and output_csv.exists():
        return _read_rows(output_csv, config)

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_split_reward_target_sweep.py"),
        "--targets",
        args.targets,
        "--modes",
        args.modes,
        "--profile",
        args.profile,
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
        "--lambda-residual",
        str(args.lambda_residual),
        "--lambda-frontier",
        str(args.lambda_frontier),
        "--action-prior",
        "split",
        "--action-prior-beta",
        str(config.beta),
        "--action-prior-residual-weight",
        str(config.residual_weight),
        "--action-prior-mixed-drop-weight",
        str(config.mixed_drop_weight),
        "--action-prior-mixed-mass-weight",
        str(config.mixed_mass_weight),
        "--action-prior-hamming-weight",
        str(args.action_prior_hamming_weight),
        "--action-prior-gadget-bonus",
        str(args.action_prior_gadget_bonus),
        "--action-prior-top-k",
        str(config.top_k),
        "--frontier-replay-fraction",
        str(config.frontier_replay_fraction),
        "--frontier-replay-min-moves",
        str(config.frontier_replay_min_moves),
        "--frontier-replay-min-residual-drop",
        str(config.frontier_replay_min_residual_drop),
        "--partition-preset",
        config.partition_preset,
        "--seed",
        str(args.seed),
        "--timeout-sec",
        str(args.timeout_sec),
        "--materialize-timeout-sec",
        str(args.materialize_timeout_sec),
        "--run-root",
        str(grid_root / "runs"),
        "--candidate-root",
        str(args.candidate_root / config.grid_id),
        "--output-csv",
        str(output_csv),
        "--report-path",
        str(report_path),
    ]
    if args.mask_padded_actions:
        cmd.append("--mask-padded-actions")
    else:
        cmd.append("--no-mask-padded-actions")
    if not args.canonical_only:
        cmd.append("--no-canonical-only")
    if args.force_canonical_basis:
        cmd.append("--force-canonical-basis")
    if not args.action_prior_standardize:
        cmd.append("--no-action-prior-standardize")
    if not args.action_prior_canonical_only:
        cmd.append("--no-action-prior-canonical-only")

    grid_root.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(cmd, cwd=PROJECT_ROOT, check=False)
    if completed.returncode != 0:
        return [
            {
                "grid_id": config.grid_id,
                "partition_preset": config.partition_preset,
                "action_prior_beta": config.beta,
                "action_prior_residual_weight": config.residual_weight,
                "action_prior_mixed_drop_weight": config.mixed_drop_weight,
                "action_prior_mixed_mass_weight": config.mixed_mass_weight,
                "action_prior_top_k": config.top_k,
                "frontier_replay_fraction": config.frontier_replay_fraction,
                "frontier_replay_min_moves": config.frontier_replay_min_moves,
                "frontier_replay_min_residual_drop": (
                    config.frontier_replay_min_residual_drop
                ),
                "max_num_moves": args.max_num_moves,
                "num_past_factors_to_observe": args.num_past_factors_to_observe,
                "lambda_residual": args.lambda_residual,
                "lambda_frontier": args.lambda_frontier,
                "target": "",
                "mode": "",
                "status": "failed-grid-command",
                "error": f"exit code {completed.returncode}",
                "intermediate_csv": str(output_csv),
                "intermediate_report": str(report_path),
            }
        ]
    return _read_rows(output_csv, config)


def _read_rows(path: Path, config: PriorGridConfig) -> list[dict[str, Any]]:
    if not path.exists():
        return [
            {
                "grid_id": config.grid_id,
                "partition_preset": config.partition_preset,
                "action_prior_beta": config.beta,
                "action_prior_residual_weight": config.residual_weight,
                "action_prior_mixed_drop_weight": config.mixed_drop_weight,
                "action_prior_mixed_mass_weight": config.mixed_mass_weight,
                "action_prior_top_k": config.top_k,
                "frontier_replay_fraction": config.frontier_replay_fraction,
                "frontier_replay_min_moves": config.frontier_replay_min_moves,
                "frontier_replay_min_residual_drop": (
                    config.frontier_replay_min_residual_drop
                ),
                "max_num_moves": "",
                "num_past_factors_to_observe": "",
                "lambda_residual": "",
                "lambda_frontier": "",
                "target": "",
                "mode": "",
                "status": "missing-output",
                "error": "intermediate CSV missing",
                "intermediate_csv": str(path),
            }
        ]
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            row.update(
                {
                    "grid_id": config.grid_id,
                    "partition_preset": config.partition_preset,
                    "action_prior_beta": config.beta,
                    "action_prior_residual_weight": config.residual_weight,
                    "action_prior_mixed_drop_weight": config.mixed_drop_weight,
                    "action_prior_mixed_mass_weight": config.mixed_mass_weight,
                    "action_prior_top_k": config.top_k,
                    "frontier_replay_fraction": config.frontier_replay_fraction,
                    "frontier_replay_min_moves": config.frontier_replay_min_moves,
                    "frontier_replay_min_residual_drop": (
                        config.frontier_replay_min_residual_drop
                    ),
                    "intermediate_csv": str(path),
                    "intermediate_report": str(path.with_name("report.md")),
                }
            )
            rows.append(row)
    return rows


def _float_or_inf(value: Any) -> float:
    try:
        if value in (None, ""):
            return float("inf")
        return float(value)
    except (TypeError, ValueError):
        return float("inf")


def _best_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = (row.get("target"), row.get("mode"))
        frontier_residual = row.get("best_frontier_residual_weight")
        frontier_cost = row.get("best_frontier_effective_t_cost")
        if frontier_residual in (None, ""):
            continue
        score = (
            _float_or_inf(frontier_residual),
            _float_or_inf(frontier_cost),
            _float_or_inf(row.get("best_return_residual_weight")),
            _float_or_inf(row.get("action_prior_beta")),
            row.get("grid_id", ""),
        )
        if key not in best or score < best[key][0]:
            best[key] = (score, row)
    return [row for _score, row in sorted(best.values(), key=lambda item: item[0])]


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "grid_id",
        "target",
        "mode",
        "status",
        "error",
        "partition_preset",
        "action_prior_beta",
        "action_prior_residual_weight",
        "action_prior_mixed_drop_weight",
        "action_prior_mixed_mass_weight",
        "action_prior_hamming_weight",
        "action_prior_gadget_bonus",
        "action_prior_top_k",
        "frontier_replay_fraction",
        "frontier_replay_min_moves",
        "frontier_replay_min_residual_drop",
        "best_return_residual_weight",
        "best_return_effective_t_cost",
        "best_frontier_residual_weight",
        "best_frontier_effective_t_cost",
        "best_frontier_num_moves",
        "best_effective_t_cost",
        "best_return",
        "avg_return_final",
        "training_steps",
        "batch_size",
        "num_mcts_simulations",
        "max_num_moves",
        "num_past_factors_to_observe",
        "lambda_residual",
        "lambda_frontier",
        "summary_json",
        "stdout_log",
        "candidate_manifest_path",
        "intermediate_csv",
        "intermediate_report",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_report(rows: list[dict[str, Any]], path: Path, output_csv: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    best_rows = _best_rows(rows)
    lines = [
        "# Split prior grid",
        "",
        f"CSV: `{output_csv}`.",
        "",
        "Lower residual tensor weight is better. Rows are unresolved unless "
        "`best_effective_t_cost` is finite.",
        "The Best Rows section uses explicit frontier metrics only; rows "
        "without `best_frontier_residual_weight` are omitted instead of being "
        "ranked by terminal residual as a substitute.",
        "",
        "## Best Rows",
        "",
        "| target | mode | grid | frontier residual | frontier cost | terminal residual | status |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in best_rows:
        lines.append(
            "| {target} | {mode} | {grid} | {frontier} | {frontier_cost} | {residual} | {status} |".format(
                target=row.get("target", ""),
                mode=row.get("mode", ""),
                grid=row.get("grid_id", ""),
                frontier=row.get("best_frontier_residual_weight", ""),
                frontier_cost=row.get("best_frontier_effective_t_cost", ""),
                residual=row.get("best_return_residual_weight", ""),
                status=row.get("status", ""),
            )
        )
    lines.extend(
        [
            "",
            "## All Rows",
            "",
            "| target | mode | grid | frontier residual | frontier cost | terminal residual | status |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {target} | {mode} | {grid} | {frontier} | {frontier_cost} | {residual} | {status} |".format(
                target=row.get("target", ""),
                mode=row.get("mode", ""),
                grid=row.get("grid_id", ""),
                frontier=row.get("best_frontier_residual_weight", ""),
                frontier_cost=row.get("best_frontier_effective_t_cost", ""),
                residual=row.get("best_return_residual_weight", ""),
                status=row.get("status", ""),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a compact grid over state-aware split-prior settings."
    )
    parser.add_argument("--targets", default="gf_2pow2_mult,hamming_weight_n4")
    parser.add_argument("--modes", default="none")
    parser.add_argument("--profile", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--use-gadgets", choices=("on", "off"), default="on")
    parser.add_argument("--training-steps", type=int, default=1000)
    parser.add_argument("--eval-frequency", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-mcts-simulations", type=int, default=8)
    parser.add_argument(
        "--action-dictionary",
        choices=("full", "low-weight", "tensor-overlap", "gadget-closure"),
        default="gadget-closure",
    )
    parser.add_argument("--max-action-weight", type=int, default=2)
    parser.add_argument("--tensor-overlap-max-weight", type=int, default=5)
    parser.add_argument("--tensor-overlap-max-actions-per-target", type=int, default=128)
    parser.add_argument("--gadget-closure-max-weight", type=int, default=4)
    parser.add_argument("--max-num-moves", type=int, default=0)
    parser.add_argument("--num-past-factors-to-observe", type=int, default=0)
    parser.add_argument("--lambda-residual", type=float, default=0.10)
    parser.add_argument("--lambda-frontier", type=float, default=0.10)
    parser.add_argument("--partition-presets", default="balanced,ensemble")
    parser.add_argument("--action-prior-betas", default="0.5,1.0,2.0")
    parser.add_argument("--action-prior-residual-weights", default="1.0")
    parser.add_argument("--action-prior-mixed-drop-weights", default="1.0")
    parser.add_argument("--action-prior-mixed-mass-weights", default="0.1,0.25")
    parser.add_argument("--action-prior-hamming-weight", type=float, default=0.05)
    parser.add_argument("--action-prior-gadget-bonus", type=float, default=0.25)
    parser.add_argument("--action-prior-top-k", type=int, default=0)
    parser.add_argument("--frontier-replay-fraction", type=float, default=0.0)
    parser.add_argument("--frontier-replay-min-moves", type=int, default=0)
    parser.add_argument("--frontier-replay-min-residual-drop", type=float, default=0.0)
    parser.add_argument(
        "--frontier-replay-fractions",
        default=None,
        help=(
            "Comma-separated replay fractions to include in the grid. "
            "Overrides --frontier-replay-fraction when provided."
        ),
    )
    parser.add_argument(
        "--frontier-replay-min-moves-values",
        default=None,
        help=(
            "Comma-separated minimum frontier depths to include in the grid. "
            "Overrides --frontier-replay-min-moves when provided."
        ),
    )
    parser.add_argument(
        "--frontier-replay-min-residual-drops",
        default=None,
        help=(
            "Comma-separated normalized residual-drop thresholds for frontier "
            "replay. Overrides --frontier-replay-min-residual-drop when provided."
        ),
    )
    parser.add_argument(
        "--action-prior-top-ks",
        default=None,
        help=(
            "Comma-separated top-k values to include in the grid. Overrides "
            "--action-prior-top-k when provided."
        ),
    )
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
        "--mask-padded-actions",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--canonical-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-canonical-basis", action="store_true")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--timeout-sec", type=int, default=900)
    parser.add_argument("--materialize-timeout-sec", type=int, default=120)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE_ROOT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_root = args.run_root or DEFAULT_RUN_ROOT / _timestamp()
    rows: list[dict[str, Any]] = []
    for config in build_grid_configs(args):
        rows.extend(_run_grid_config(config, args=args, run_root=run_root))
        _write_csv(rows, args.output_csv)
        _write_report(rows, args.report_path, args.output_csv)
    append_command(
        {
            "tool": "run_split_prior_grid.py",
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
