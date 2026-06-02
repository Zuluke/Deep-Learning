from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tensor_split_core import (  # noqa: E402
    balanced_contiguous_partition,
    gadget_aware_mixed_stats,
    outer3,
    raw_bridge_count,
    tensor_from_factors,
)

DEFAULT_SWEEP_CSV = (
    PROJECT_ROOT / "results" / "csv" / "split_reward_target_sweep_split4_prior.csv"
)
DEFAULT_OUTPUT_CSV = (
    PROJECT_ROOT / "results" / "csv" / "split_reward_intermediate_diagnostics.csv"
)
DEFAULT_REPORT = (
    PROJECT_ROOT / "results" / "reports" / "split_reward_intermediate_diagnostics.md"
)
REMOTE_PROJECT_PREFIXES = (
    "/home/CIN/cacl2/Deep-Learning/project",
    "/home/CIN/cacl2/Deep-Learning",
)


@dataclass(frozen=True)
class CandidateDiagnostics:
    target: str
    mode: str
    candidate_kind: str
    status: str
    num_factors: int
    initial_residual_weight: int
    final_residual_weight: int
    min_residual_weight: int
    min_residual_step: int
    residual_drop_from_initial: int
    net_tensor_weight: int
    unique_factor_count: int
    odd_factor_count: int
    even_factor_count: int
    cancellation_fraction: float
    action_weight_histogram: str
    residual_delta_histogram: str
    nonzero_overlap_steps: int
    reported_residual_weight: str
    reported_effective_t_cost: str
    raw_bridge_count: int
    gadget_aware_effective_mixed_cost: int
    gadget_mixed_weight: int
    mixed_group_count: int
    mixed_block_span: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose unsolved split-reward candidates by replaying exported "
            "factor prefixes against the target tensor."
        )
    )
    parser.add_argument("--sweep-csv", type=Path, default=DEFAULT_SWEEP_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--candidate-kinds",
        default="best_return,best_frontier",
        help="Comma-separated manifest candidate kinds to inspect.",
    )
    parser.add_argument(
        "--single-action-max-weight",
        type=int,
        default=4,
        help="Maximum factor Hamming weight for initial action viability audit.",
    )
    return parser.parse_args()


def resolve_project_path(raw_path: str | Path | None) -> Path | None:
    if raw_path is None:
        return None
    text = str(raw_path).strip()
    if not text:
        return None
    for prefix in REMOTE_PROJECT_PREFIXES:
        if text.startswith(prefix):
            suffix = text[len(prefix) :].lstrip("/")
            if prefix.endswith("/project"):
                return PROJECT_ROOT / suffix
            return PROJECT_ROOT.parent / suffix
    path = Path(text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def benchmark_tensor_path(target: str) -> Path:
    benchmark_root = PROJECT_ROOT / "external" / "circuit-to-tensor" / "benchmarks"
    matches = sorted(benchmark_root.glob(f"**/{target}/{target}.tensor.npy"))
    if not matches:
        raise FileNotFoundError(f"Could not find tensor for target {target!r}.")
    if len(matches) > 1:
        raise ValueError(f"Ambiguous tensor paths for target {target!r}: {matches}")
    return matches[0]


def load_target_tensor(target: str) -> np.ndarray:
    return np.load(benchmark_tensor_path(target)).astype(np.uint8) % 2


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def histogram(values: list[int]) -> str:
    counts = Counter(values)
    return ";".join(f"{key}:{counts[key]}" for key in sorted(counts))


def factor_key(factor: np.ndarray) -> tuple[int, ...]:
    return tuple(int(value) for value in np.asarray(factor, dtype=np.uint8) % 2)


def replay_candidate(
    *,
    target_tensor: np.ndarray,
    factors: np.ndarray,
) -> dict[str, Any]:
    factors = np.asarray(factors, dtype=np.uint8) % 2
    residual = target_tensor.copy()
    weights = [int(np.count_nonzero(residual))]
    deltas: list[int] = []
    overlaps: list[int] = []
    action_weights: list[int] = []
    for factor in factors:
        update = outer3(factor)
        overlap = int(np.count_nonzero(residual & update))
        previous = weights[-1]
        residual ^= update
        current = int(np.count_nonzero(residual))
        weights.append(current)
        deltas.append(current - previous)
        overlaps.append(overlap)
        action_weights.append(int(np.count_nonzero(factor)))

    net_tensor = tensor_from_factors(factors, tensor_size=target_tensor.shape[0])
    factor_counts = Counter(factor_key(factor) for factor in factors)
    odd_factor_count = sum(1 for count in factor_counts.values() if count % 2)
    even_factor_count = sum(1 for count in factor_counts.values() if count % 2 == 0)
    unique_factor_count = len(factor_counts)
    min_residual = min(weights)
    min_step = weights.index(min_residual)
    cancellation_fraction = (
        1.0 - odd_factor_count / unique_factor_count if unique_factor_count else 0.0
    )
    return {
        "initial_residual_weight": weights[0],
        "final_residual_weight": weights[-1],
        "min_residual_weight": min_residual,
        "min_residual_step": min_step,
        "residual_drop_from_initial": weights[0] - min_residual,
        "net_tensor_weight": int(np.count_nonzero(net_tensor)),
        "unique_factor_count": unique_factor_count,
        "odd_factor_count": odd_factor_count,
        "even_factor_count": even_factor_count,
        "cancellation_fraction": cancellation_fraction,
        "action_weight_histogram": histogram(action_weights),
        "residual_delta_histogram": histogram(deltas),
        "nonzero_overlap_steps": sum(1 for value in overlaps if value > 0),
    }


def initial_action_audit(
    target_tensor: np.ndarray,
    *,
    max_weight: int,
) -> dict[str, Any]:
    size = target_tensor.shape[0]
    initial_weight = int(np.count_nonzero(target_tensor))
    rows = []
    for weight in range(1, max_weight + 1):
        deltas = []
        for indices in combinations(range(size), weight):
            factor = np.zeros(size, dtype=np.uint8)
            factor[list(indices)] = 1
            updated = target_tensor ^ outer3(factor)
            deltas.append(int(np.count_nonzero(updated)) - initial_weight)
        rows.append(
            {
                "weight": weight,
                "best_delta": min(deltas) if deltas else "",
                "improving_actions": sum(1 for delta in deltas if delta < 0),
                "neutral_actions": sum(1 for delta in deltas if delta == 0),
                "total_actions": len(deltas),
            }
        )
    return {"initial_weight": initial_weight, "rows": rows}


def analyze_sweep(
    sweep_csv: Path,
    *,
    candidate_kinds: tuple[str, ...],
    single_action_max_weight: int,
) -> tuple[list[CandidateDiagnostics], dict[str, dict[str, Any]]]:
    with sweep_csv.open(encoding="utf-8", newline="") as handle:
        sweep_rows = list(csv.DictReader(handle))
    target_tensors = {
        target: load_target_tensor(target)
        for target in sorted({row["target"] for row in sweep_rows})
    }
    audits = {
        target: initial_action_audit(tensor, max_weight=single_action_max_weight)
        for target, tensor in target_tensors.items()
    }
    diagnostics: list[CandidateDiagnostics] = []
    for row in sweep_rows:
        target = row["target"]
        manifest_path = resolve_project_path(row.get("candidate_manifest_path"))
        if manifest_path is None or not manifest_path.exists():
            continue
        manifest_rows = load_manifest(manifest_path)
        target_tensor = target_tensors[target]
        partition = balanced_contiguous_partition(target_tensor.shape[0])
        for candidate_kind in candidate_kinds:
            matches = [
                item for item in manifest_rows if item.get("candidate_kind") == candidate_kind
            ]
            if not matches:
                continue
            item = matches[0]
            factor_path = resolve_project_path(item.get("factor_path"))
            if factor_path is None or not factor_path.exists():
                continue
            factors = np.load(factor_path).astype(np.uint8) % 2
            replay = replay_candidate(target_tensor=target_tensor, factors=factors)
            mixed_stats = gadget_aware_mixed_stats(factors, partition)
            diagnostics.append(
                CandidateDiagnostics(
                    target=target,
                    mode=row["mode"],
                    candidate_kind=candidate_kind,
                    status=item.get("status", ""),
                    num_factors=int(factors.shape[0]),
                    reported_residual_weight=item.get("residual_weight", ""),
                    reported_effective_t_cost=item.get("effective_t_cost", ""),
                    raw_bridge_count=raw_bridge_count(factors, partition),
                    **replay,
                    **mixed_stats,
                )
            )
    return diagnostics, audits


def write_csv(path: Path, rows: list[CandidateDiagnostics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(CandidateDiagnostics.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def candidate_summary(rows: list[CandidateDiagnostics]) -> list[str]:
    lines = [
        "| target | mode | kind | factors | min residual | final residual | net tensor | unique factors | odd unique | cancellation | mixed cost |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {target} | {mode} | {kind} | {factors} | {minr} | {finalr} | {net} | {unique} | {odd} | {cancel:.2f} | {mixed} |".format(
                target=row.target,
                mode=row.mode,
                kind=row.candidate_kind,
                factors=row.num_factors,
                minr=row.min_residual_weight,
                finalr=row.final_residual_weight,
                net=row.net_tensor_weight,
                unique=row.unique_factor_count,
                odd=row.odd_factor_count,
                cancel=row.cancellation_fraction,
                mixed=row.gadget_aware_effective_mixed_cost,
            )
        )
    return lines


def audit_summary(audits: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        "| target | initial residual | factor weight | best immediate delta | improving actions | total actions |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for target, audit in sorted(audits.items()):
        for row in audit["rows"]:
            lines.append(
                "| {target} | {initial} | {weight} | {delta} | {improving} | {total} |".format(
                    target=target,
                    initial=audit["initial_weight"],
                    weight=row["weight"],
                    delta=row["best_delta"],
                    improving=row["improving_actions"],
                    total=row["total_actions"],
                )
            )
    return lines


def write_report(
    path: Path,
    *,
    rows: list[CandidateDiagnostics],
    audits: dict[str, dict[str, Any]],
    output_csv: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    all_net_zero = all(row.net_tensor_weight == 0 for row in rows)
    any_residual_drop = any(row.residual_drop_from_initial > 0 for row in rows)
    all_odd_zero = all(row.odd_factor_count == 0 for row in rows)
    no_initial_improvers = all(
        item["improving_actions"] == 0
        for audit in audits.values()
        for item in audit["rows"]
    )
    conclusion = (
        "The intermediate sweep is dominated by cancellation cycles: every "
        "exported candidate has zero net tensor contribution, no candidate "
        "prefix improves the residual below the initial tensor, and all unique "
        "factors appear with even parity."
        if all_net_zero and not any_residual_drop and all_odd_zero
        else "The intermediate sweep contains non-trivial partial tensor progress."
    )
    code_implication = (
        "The next code change should not be circuit materialization for "
        "not-solved rows. The useful change is to make cancellation/no-progress "
        "diagnostics first-class and then add an anti-cycle or progress-aware "
        "training variant so the loop cannot spend the horizon on zero-net "
        "factor pairs."
        if no_initial_improvers
        else "Some immediate residual-improving actions exist; inspect whether "
        "the policy/action prior is ranking them too low."
    )
    lines = [
        "# Split Reward Intermediate Diagnostics",
        "",
        f"CSV: `{output_csv}`.",
        "",
        "## Main Finding",
        "",
        conclusion,
        "",
        "## Code Implication",
        "",
        code_implication,
        "",
        "## Candidate Replay",
        "",
        *candidate_summary(rows),
        "",
        "## Initial Single-Action Audit",
        "",
        *audit_summary(audits),
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    candidate_kinds = tuple(
        item.strip() for item in args.candidate_kinds.split(",") if item.strip()
    )
    rows, audits = analyze_sweep(
        args.sweep_csv,
        candidate_kinds=candidate_kinds,
        single_action_max_weight=args.single_action_max_weight,
    )
    write_csv(args.output_csv, rows)
    write_report(args.report, rows=rows, audits=audits, output_csv=args.output_csv)
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
