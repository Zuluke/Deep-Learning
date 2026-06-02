from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._analysis_common import DEFAULT_CSV_ROOT
from scripts._analysis_common import DEFAULT_REPORTS_ROOT
from scripts._analysis_common import DEFAULT_RESULTS_ROOT
from scripts._analysis_common import ensure_dir
from scripts._analysis_common import natural_sort_key
from scripts._analysis_common import write_csv_rows


DEFAULT_CIRCUIT_IDS = (
    "mod_5_4",
    "gf_2pow2_mult",
    "qft_4",
    "hamming_weight_n4",
    "hamming_weight_n5",
)

ObjectiveKey = Callable[[dict[str, str]], tuple[Any, ...]]


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
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
    value_float = coerce_float(value)
    return None if value_float is None else int(value_float)


def finite(value: Any) -> float:
    value_float = coerce_float(value)
    return float("inf") if value_float is None else value_float


def combo_index(row: dict[str, str]) -> int:
    return coerce_int(row.get("combo_index")) or 10**12


def stable_hash(row: dict[str, str]) -> str:
    return row.get("tensor_v3_stable_factor_hash") or ""


def valid_rows_for_circuit(
    rows: list[dict[str, str]],
    *,
    circuit_id: str,
    tcount_tolerance: float,
) -> list[dict[str, str]]:
    candidates = [
        row
        for row in rows
        if row.get("circuit_id") == circuit_id
        and row.get("status") == "ok"
        and row.get("selection_status") == "ok"
        and row.get("tensor_v3_status") == "ok"
        and coerce_float(row.get("primary_nc_depth_ratio")) is not None
        and coerce_float(row.get("tcount_after")) is not None
    ]
    if not candidates:
        return []
    best_tcount = min(finite(row.get("tcount_after")) for row in candidates)
    allowed = best_tcount * (1.0 + tcount_tolerance)
    return [row for row in candidates if finite(row.get("tcount_after")) <= allowed]


def objective_keys() -> dict[str, ObjectiveKey]:
    return {
        "current_lex_v1": lambda row: (
            finite(row.get("tensor_v3_mixed_excess_norm")),
            finite(row.get("tensor_v3_mixed_auc_greedy_norm")),
            finite(row.get("tensor_v3_singleton_bridge_count_norm")),
            stable_hash(row),
            finite(row.get("tcount_after")),
            combo_index(row),
        ),
        "no_greedy_auc": lambda row: (
            finite(row.get("tensor_v3_mixed_excess_norm")),
            finite(row.get("tensor_v3_singleton_bridge_count_norm")),
            stable_hash(row),
            finite(row.get("tcount_after")),
            combo_index(row),
        ),
        "auc_original": lambda row: (
            finite(row.get("tensor_v3_mixed_excess_norm")),
            finite(row.get("tensor_v3_mixed_auc_original_norm")),
            finite(row.get("tensor_v3_singleton_bridge_count_norm")),
            stable_hash(row),
            finite(row.get("tcount_after")),
            combo_index(row),
        ),
        "gadget_first": lambda row: (
            finite(row.get("tensor_v3_gadget_mixed_weight_norm")),
            finite(row.get("tensor_v3_mixed_excess_norm")),
            finite(row.get("tensor_v3_singleton_bridge_count_norm")),
            stable_hash(row),
            finite(row.get("tcount_after")),
            combo_index(row),
        ),
        "singleton_first": lambda row: (
            finite(row.get("tensor_v3_singleton_bridge_count_norm")),
            finite(row.get("tensor_v3_mixed_excess_norm")),
            finite(row.get("tensor_v3_mixed_auc_greedy_norm")),
            stable_hash(row),
            finite(row.get("tcount_after")),
            combo_index(row),
        ),
        "tcount_only": lambda row: (
            finite(row.get("tcount_after")),
            finite(row.get("tdepth_after")),
            combo_index(row),
        ),
        "primary_oracle": lambda row: (
            finite(row.get("primary_nc_depth_ratio")),
            finite(row.get("tcount_after")),
            finite(row.get("qasm_depth_ratio")),
            combo_index(row),
        ),
    }


def partition_metric_rows(row: dict[str, str]) -> list[dict[str, Any]]:
    raw_json = row.get("tensor_v3_partition_metrics_json")
    if not raw_json:
        return []
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def partition_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        finite(row.get("mixed_excess_norm")),
        finite(row.get("mixed_auc_greedy_norm")),
        finite(row.get("singleton_bridge_count_norm")),
        row.get("stable_factor_hash") or "",
    )


def ensemble_median_rank_selection(
    candidates: list[dict[str, str]],
) -> tuple[dict[str, str], dict[str, tuple[float, float]]]:
    partition_ids = sorted(
        {
            str(metric.get("partition_id"))
            for candidate in candidates
            for metric in partition_metric_rows(candidate)
            if metric.get("partition_id")
        }
    )
    if not partition_ids:
        return min(candidates, key=objective_keys()["current_lex_v1"]), {}

    rank_map: dict[str, list[int]] = {
        str(candidate.get("candidate_id")): [] for candidate in candidates
    }
    for partition_id in partition_ids:
        partition_candidates = []
        for candidate in candidates:
            metric = next(
                (
                    row
                    for row in partition_metric_rows(candidate)
                    if row.get("partition_id") == partition_id
                ),
                None,
            )
            if metric is None:
                continue
            partition_candidates.append((candidate, metric))
        partition_candidates.sort(
            key=lambda item: (
                *partition_key(item[1]),
                finite(item[0].get("tcount_after")),
                combo_index(item[0]),
            )
        )
        missing_rank = len(candidates) + 1
        for candidate in candidates:
            rank_map[str(candidate.get("candidate_id"))].append(missing_rank)
        for rank, (candidate, _metric) in enumerate(partition_candidates, start=1):
            rank_map[str(candidate.get("candidate_id"))][-1] = rank

    scores = {}
    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id"))
        ranks = sorted(rank_map[candidate_id])
        midpoint = len(ranks) // 2
        median = (
            ranks[midpoint]
            if len(ranks) % 2
            else 0.5 * (ranks[midpoint - 1] + ranks[midpoint])
        )
        scores[candidate_id] = (float(median), float(max(ranks)))

    selected = min(
        candidates,
        key=lambda candidate: (
            *scores[str(candidate.get("candidate_id"))],
            *objective_keys()["current_lex_v1"](candidate),
        ),
    )
    return selected, scores


def phase_slack_selection(
    candidates: list[dict[str, str]],
    *,
    mixed_slack: float,
) -> dict[str, str]:
    best_mixed = min(finite(row.get("tensor_v3_mixed_excess_norm")) for row in candidates)
    allowed_mixed = best_mixed + max(abs(best_mixed) * mixed_slack, 1e-12)
    near_tensor_candidates = [
        row
        for row in candidates
        if finite(row.get("tensor_v3_mixed_excess_norm")) <= allowed_mixed
    ]
    return min(
        near_tensor_candidates or candidates,
        key=lambda row: (
            finite(row.get("tdepth_after")),
            finite(row.get("tensor_v3_mixed_excess_norm")),
            finite(row.get("tensor_v3_mixed_auc_greedy_norm")),
            finite(row.get("tensor_v3_singleton_bridge_count_norm")),
            stable_hash(row),
            finite(row.get("tcount_after")),
            combo_index(row),
        ),
    )


def build_ablation_rows(
    frontier_rows: list[dict[str, str]],
    *,
    circuit_ids: tuple[str, ...],
    tcount_tolerance: float,
    mixed_slack: float,
) -> list[dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    keys = objective_keys()
    for circuit_id in circuit_ids:
        candidates = valid_rows_for_circuit(
            frontier_rows,
            circuit_id=circuit_id,
            tcount_tolerance=tcount_tolerance,
        )
        if not candidates:
            continue
        oracle = min(candidates, key=keys["primary_oracle"])
        current = min(candidates, key=keys["current_lex_v1"])
        ensemble_selected, ensemble_scores = ensemble_median_rank_selection(candidates)
        oracle_primary = finite(oracle.get("primary_nc_depth_ratio"))
        current_primary = finite(current.get("primary_nc_depth_ratio"))
        selected_by_objective = {
            objective: min(candidates, key=key) for objective, key in keys.items()
        }
        selected_by_objective["ensemble_median_rank"] = ensemble_selected
        selected_by_objective["phase_slack_v1"] = phase_slack_selection(
            candidates,
            mixed_slack=mixed_slack,
        )
        for objective, selected in selected_by_objective.items():
            selected_primary = finite(selected.get("primary_nc_depth_ratio"))
            ensemble_score = ensemble_scores.get(
                str(selected.get("candidate_id")), (None, None)
            )
            output_rows.append(
                {
                    "circuit_id": circuit_id,
                    "objective": objective,
                    "selected_candidate_id": selected.get("candidate_id"),
                    "selected_combo_index": selected.get("combo_index"),
                    "selected_tcount": coerce_int(selected.get("tcount_after")),
                    "selected_primary_nc_depth_ratio": selected_primary,
                    "oracle_candidate_id": oracle.get("candidate_id"),
                    "oracle_primary_nc_depth_ratio": oracle_primary,
                    "current_candidate_id": current.get("candidate_id"),
                    "current_primary_nc_depth_ratio": current_primary,
                    "primary_regret_vs_oracle": selected_primary - oracle_primary,
                    "primary_delta_vs_current": selected_primary - current_primary,
                    "tensor_v3_mixed_excess_norm": coerce_float(
                        selected.get("tensor_v3_mixed_excess_norm")
                    ),
                    "tensor_v3_mixed_auc_greedy_norm": coerce_float(
                        selected.get("tensor_v3_mixed_auc_greedy_norm")
                    ),
                    "tensor_v3_mixed_auc_original_norm": coerce_float(
                        selected.get("tensor_v3_mixed_auc_original_norm")
                    ),
                    "tensor_v3_singleton_bridge_count_norm": coerce_float(
                        selected.get("tensor_v3_singleton_bridge_count_norm")
                    ),
                    "tensor_v3_gadget_mixed_weight_norm": coerce_float(
                        selected.get("tensor_v3_gadget_mixed_weight_norm")
                    ),
                    "tensor_v3_ensemble_median_rank": ensemble_score[0],
                    "tensor_v3_ensemble_worst_rank": ensemble_score[1],
                    "num_candidates_within_tolerance": len(candidates),
                }
            )
    return output_rows


def relation(delta: float, *, eps: float = 1e-9) -> str:
    if delta < -eps:
        return "better"
    if delta > eps:
        return "worse"
    return "tie"


def write_report(rows: list[dict[str, Any]], output_path: Path, csv_path: Path) -> Path:
    objective_names = sorted({str(row["objective"]) for row in rows})
    summary_lines = [
        "| objective | better than current | tie current | worse than current | oracle hits | mean oracle regret | max oracle regret |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for objective in objective_names:
        subset = [row for row in rows if row["objective"] == objective]
        relations = [
            relation(float(row["primary_delta_vs_current"])) for row in subset
        ]
        oracle_hits = sum(
            relation(float(row["primary_regret_vs_oracle"])) == "tie"
            for row in subset
        )
        regrets = [float(row["primary_regret_vs_oracle"]) for row in subset]
        mean_regret = sum(regrets) / max(len(regrets), 1)
        max_regret = max(regrets) if regrets else 0.0
        summary_lines.append(
            "| "
            + " | ".join(
                [
                    f"`{objective}`",
                    str(relations.count("better")),
                    str(relations.count("tie")),
                    str(relations.count("worse")),
                    str(oracle_hits),
                    f"{mean_regret:.3f}",
                    f"{max_regret:.3f}",
                ]
            )
            + " |"
        )

    qft_rows = [
        row for row in rows if row["circuit_id"] == "qft_4"
    ]
    qft_lines = [
        "| objective | selected | primary | regret vs oracle | delta vs current |",
        "|---|---|---:|---:|---:|",
    ]
    for row in sorted(qft_rows, key=lambda item: str(item["objective"])):
        qft_lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['objective']}`",
                    f"`{row['selected_candidate_id']}`",
                    f"{float(row['selected_primary_nc_depth_ratio']):.3f}",
                    f"{float(row['primary_regret_vs_oracle']):.3f}",
                    f"{float(row['primary_delta_vs_current']):.3f}",
                ]
            )
            + " |"
        )

    text = [
        "# Tensor-v3 Selection Ablation",
        "",
        f"- Ablation CSV: `{csv_path}`.",
        "- Objectives are evaluated after the candidate frontier has been generated; `primary_oracle` is an external upper bound, not a valid selector.",
        "",
        "## Objective Summary",
        "",
        *summary_lines,
        "",
        "## qft_4 Diagnostic",
        "",
        *qft_lines,
        "",
        "## Interpretation",
        "",
        (
            "Use this report to decide whether the qft_4 failure comes from the greedy AUC "
            "component, the lexicographic order, or a broader mismatch between tensor partitions "
            "and global transform circuits. Any objective that improves qft_4 without hurting the "
            "other circuits is a candidate for the next frozen tensor-v3 selector. "
            "`phase_slack_v1` keeps the tensor score as a gate, but allows lower T-depth to decide "
            "among candidates whose mixed-excess score is within the configured slack."
        ),
        "",
    ]
    ensure_dir(output_path.parent)
    output_path.write_text("\n".join(text), encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablate tensor-v3 selection keys on an audited frontier.")
    parser.add_argument(
        "--candidate-frontier-csv",
        type=Path,
        default=DEFAULT_RESULTS_ROOT
        / "public_resynth_tensor_v3"
        / "candidate_frontier.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_CSV_ROOT / "tensor_v3_selection_ablation.csv",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_CSV_ROOT / "tensor_v3_selection_ablation.json",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORTS_ROOT / "tensor_v3_selection_ablation.md",
    )
    parser.add_argument("--tcount-tolerance", type=float, default=0.12)
    parser.add_argument("--mixed-slack", type=float, default=0.15)
    parser.add_argument("--circuit-id", action="append", dest="circuit_ids", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    circuit_ids = tuple(args.circuit_ids) if args.circuit_ids else DEFAULT_CIRCUIT_IDS
    frontier_rows = load_csv_rows(args.candidate_frontier_csv)
    rows = build_ablation_rows(
        frontier_rows,
        circuit_ids=circuit_ids,
        tcount_tolerance=args.tcount_tolerance,
        mixed_slack=args.mixed_slack,
    )
    write_csv_rows(rows, args.output_csv)
    args.output_json.write_text(
        json.dumps(
            {
                "candidate_frontier_csv": str(args.candidate_frontier_csv),
                "output_csv": str(args.output_csv),
                "report_path": str(args.report_path),
                "tcount_tolerance": args.tcount_tolerance,
                "mixed_slack": args.mixed_slack,
                "circuit_ids": list(circuit_ids),
                "num_rows": len(rows),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_report(rows, args.report_path, args.output_csv)
    print(
        json.dumps(
            {
                "output_csv": str(args.output_csv),
                "output_json": str(args.output_json),
                "report_path": str(args.report_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
