from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_best_objective_beam_ablation import inf_if_none
from scripts.run_best_objective_beam_ablation import parse_paths
from scripts.run_best_objective_beam_ablation import read_csv
from scripts.structural_target import coerce_float


DEFAULT_DECOMP_CSVS = (
    PROJECT_ROOT / "results" / "csv" / "alphaq_decomposition_objective_ablation.csv",
    PROJECT_ROOT / "results" / "csv" / "alphaq_decomposition_objective_holdout_ablation.csv",
)
DEFAULT_GRID_CSV = PROJECT_ROOT / "results" / "csv" / "alphaq_objective_beam_policy_grid.csv"
DEFAULT_EXTERNAL_DECOMP_CSV = (
    PROJECT_ROOT / "results" / "csv" / "alphaq_decomposition_objective_external_validation.csv"
)
DEFAULT_EXTERNAL_GRID_CSV = (
    PROJECT_ROOT / "results" / "csv" / "alphaq_objective_beam_policy_external_validation_grid.csv"
)
DEFAULT_NIGHT_DECOMP_CSV = (
    PROJECT_ROOT / "results" / "csv" / "alphaq_decomposition_objective_external_validation_night_long.csv"
)
DEFAULT_NIGHT_GRID_CSV = (
    PROJECT_ROOT
    / "results"
    / "csv"
    / "alphaq_objective_beam_policy_external_validation_night_long_grid.csv"
)
DEFAULT_EXTRA_EXTERNAL_RUNS = (
    "journal_repair_paircap",
    "journal_full_mod_mult_55",
    "journal_full_cuccaro_adder_n4",
    "journal_full_gf_2pow4_mult",
    "journal_full_hamming_weight_n6",
    "journal_full_hamming_weight_n7",
    "journal_full_nc_tof_5",
    "journal_full_gf_2pow5_mult",
    "journal_full_cuccaro_adder_n5",
)
DEFAULT_READINESS_CSV = PROJECT_ROOT / "results" / "csv" / "alphaq_external_validation_readiness.csv"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "results" / "csv" / "alphaq_objective_selection_dataset.csv"
DEFAULT_REPORT = PROJECT_ROOT / "results" / "reports" / "alphaq_objective_selection_dataset.md"

OBJECTIVES = ("factor_count", "factor_count_pair_cap", "mixed_pair")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a supervised AlphaQ objective-selection dataset from decomposition "
            "and beam materialization results."
        )
    )
    parser.add_argument(
        "--internal-decomposition-csvs",
        default=",".join(str(path) for path in DEFAULT_DECOMP_CSVS),
    )
    parser.add_argument("--internal-grid-csv", type=Path, default=DEFAULT_GRID_CSV)
    parser.add_argument("--external-decomposition-csv", type=Path, default=DEFAULT_EXTERNAL_DECOMP_CSV)
    parser.add_argument("--external-grid-csv", type=Path, default=DEFAULT_EXTERNAL_GRID_CSV)
    parser.add_argument("--night-decomposition-csv", type=Path, default=DEFAULT_NIGHT_DECOMP_CSV)
    parser.add_argument("--night-grid-csv", type=Path, default=DEFAULT_NIGHT_GRID_CSV)
    parser.add_argument(
        "--extra-external-runs",
        default=",".join(DEFAULT_EXTRA_EXTERNAL_RUNS),
        help=(
            "Comma-separated suffixed external-validation runs to include when both "
            "their decomposition CSV and beam-grid CSV exist."
        ),
    )
    parser.add_argument("--readiness-csv", type=Path, default=DEFAULT_READINESS_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def run_suffixes(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_dataset(
    *,
    internal_decomposition_csvs: list[Path],
    internal_grid_csv: Path,
    external_decomposition_csv: Path,
    external_grid_csv: Path,
    night_decomposition_csv: Path,
    night_grid_csv: Path,
    readiness_csv: Path,
    extra_external_runs: list[str] | None = None,
) -> list[dict[str, Any]]:
    metadata = target_metadata(read_csv(readiness_csv) if readiness_csv.exists() else [])
    groups = [
        (
            "internal",
            read_many(internal_decomposition_csvs),
            read_csv(internal_grid_csv),
        ),
    ]
    if external_decomposition_csv.exists() and external_grid_csv.exists():
        groups.append(("external_standard", read_csv(external_decomposition_csv), read_csv(external_grid_csv)))
    if night_decomposition_csv.exists() and night_grid_csv.exists():
        groups.append(("external_night_long", read_csv(night_decomposition_csv), read_csv(night_grid_csv)))
    for run in extra_external_runs or []:
        decomp_path, grid_path = external_run_paths(run)
        if decomp_path.exists() and grid_path.exists():
            groups.append((f"external_{run}", read_csv(decomp_path), read_csv(grid_path)))

    rows: list[dict[str, Any]] = []
    for split, decomp_rows, grid_rows in groups:
        rows.extend(dataset_rows_for_split(split, decomp_rows, grid_rows, metadata))
    return rows


def external_run_paths(run: str) -> tuple[Path, Path]:
    return (
        PROJECT_ROOT / "results" / "csv" / f"alphaq_decomposition_objective_external_validation_{run}.csv",
        PROJECT_ROOT / "results" / "csv" / f"alphaq_objective_beam_policy_external_validation_{run}_grid.csv",
    )


def read_many(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        if path.exists():
            rows.extend(read_csv(path))
    return rows


def target_metadata(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["target"]: row for row in rows if row.get("target")}


def dataset_rows_for_split(
    split: str,
    decomp_rows: list[dict[str, str]],
    grid_rows: list[dict[str, str]],
    metadata: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    decomp_by_key = {(row["target"], row["objective_variant"]): row for row in decomp_rows}
    best_beams = best_beam_rows(grid_rows)
    targets = sorted({target for target, _objective in set(decomp_by_key) | set(best_beams)})
    rows: list[dict[str, Any]] = []
    for target in targets:
        objective_rows = []
        for objective in OBJECTIVES:
            decomp = decomp_by_key.get((target, objective), {})
            beam = best_beams.get((target, objective), {})
            objective_rows.append(candidate_row(split, target, objective, decomp, beam, metadata.get(target, {})))
        annotate_oracle(objective_rows)
        rows.extend(objective_rows)
    return rows


def best_beam_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    best: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        materializer = row.get("materializer", "")
        if not materializer.startswith("selected-beam"):
            continue
        key = (row["target"], row["objective_variant"])
        current = best.get(key)
        if current is None or beam_key(row) < beam_key(current):
            best[key] = row
    return best


def beam_key(row: dict[str, str]) -> tuple[float, float, float, str]:
    return (
        inf_if_none(row.get("qasm_depth")),
        inf_if_none(row.get("primary_nc_depth_ratio")),
        inf_if_none(row.get("num_total_cnots")),
        row.get("materializer", ""),
    )


def candidate_row(
    split: str,
    target: str,
    objective: str,
    decomp: dict[str, str],
    beam: dict[str, str],
    meta: dict[str, str],
) -> dict[str, Any]:
    status = decomp.get("execution_status") or ("ok" if decomp else "missing")
    has_beam = bool(beam)
    return {
        "source_split": split,
        "target": target,
        "family": meta.get("family", ""),
        "n_qubits": meta.get("n_qubits", ""),
        "tensor_size": meta.get("tensor_size", ""),
        "original_tcount": meta.get("tcount_original", ""),
        "objective_variant": objective,
        "execution_status": status,
        "has_beam_candidate": has_beam,
        "factor_count": decomp.get("factor_count", ""),
        "factor_qubit_concentration_index": decomp.get("factor_qubit_concentration_index", ""),
        "factor_support_weight_mean": decomp.get("factor_support_weight_mean", ""),
        "factor_pairwise_support_overlap_mean": decomp.get("factor_pairwise_support_overlap_mean", ""),
        "factor_pairwise_jaccard_mean": decomp.get("factor_pairwise_jaccard_mean", ""),
        "decomp_tcount": decomp.get("tcount", ""),
        "decomp_tdepth": decomp.get("tdepth", ""),
        "decomp_qasm_depth": decomp.get("qasm_depth", ""),
        "decomp_qasm_depth_ratio": decomp.get("qasm_depth_ratio", ""),
        "decomp_structural_status": decomp.get("structural_target_status", ""),
        "best_beam_materializer": beam.get("materializer", ""),
        "best_beam_tcount": beam.get("tcount", ""),
        "best_beam_tdepth": beam.get("tdepth", ""),
        "best_beam_qasm_depth": beam.get("qasm_depth", ""),
        "best_beam_qasm_depth_ratio": beam.get("qasm_depth_ratio", ""),
        "best_beam_primary_nc_depth_ratio": beam.get("primary_nc_depth_ratio", ""),
        "best_beam_cnots": beam.get("num_total_cnots", ""),
        "best_beam_summary_path": beam.get("summary_path", ""),
    }


def annotate_oracle(rows: list[dict[str, Any]]) -> None:
    materialized = [row for row in rows if row["execution_status"] == "ok" and row["has_beam_candidate"]]
    train_ready = len(materialized) >= 2
    oracle = min(materialized, key=oracle_key) if materialized else None
    oracle_objective = oracle["objective_variant"] if oracle else ""
    sorted_rows = sorted(materialized, key=oracle_key)
    ranks = {row["objective_variant"]: index + 1 for index, row in enumerate(sorted_rows)}
    for row in rows:
        row["target_objective_count"] = len(materialized)
        row["train_ready"] = train_ready
        row["oracle_objective"] = oracle_objective
        row["objective_is_oracle"] = row["objective_variant"] == oracle_objective
        row["objective_rank"] = ranks.get(row["objective_variant"], "")


def oracle_key(row: dict[str, Any]) -> tuple[float, float, float, str]:
    return (
        inf_if_none(row.get("best_beam_tcount")),
        inf_if_none(row.get("best_beam_primary_nc_depth_ratio")),
        inf_if_none(row.get("best_beam_qasm_depth")),
        row.get("objective_variant", ""),
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_split",
        "target",
        "family",
        "n_qubits",
        "tensor_size",
        "original_tcount",
        "objective_variant",
        "execution_status",
        "has_beam_candidate",
        "factor_count",
        "factor_qubit_concentration_index",
        "factor_support_weight_mean",
        "factor_pairwise_support_overlap_mean",
        "factor_pairwise_jaccard_mean",
        "decomp_tcount",
        "decomp_tdepth",
        "decomp_qasm_depth",
        "decomp_qasm_depth_ratio",
        "decomp_structural_status",
        "best_beam_materializer",
        "best_beam_tcount",
        "best_beam_tdepth",
        "best_beam_qasm_depth",
        "best_beam_qasm_depth_ratio",
        "best_beam_primary_nc_depth_ratio",
        "best_beam_cnots",
        "target_objective_count",
        "train_ready",
        "oracle_objective",
        "objective_is_oracle",
        "objective_rank",
        "best_beam_summary_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, lineterminator="\n", fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def readiness_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups = grouped_targets(rows)
    train_ready_groups = [items for items in groups.values() if bool(items[0]["train_ready"])]
    oracle_counts = Counter(items[0]["oracle_objective"] for items in train_ready_groups)
    external_ready = [
        key
        for key, items in groups.items()
        if key[0].startswith("external") and bool(items[0]["train_ready"])
    ]
    decision = (
        "prototype-ready"
        if len(train_ready_groups) >= 8 and len(oracle_counts) >= 2
        else "not-ready"
    )
    return {
        "target_groups": len(groups),
        "train_ready_groups": len(train_ready_groups),
        "external_train_ready_groups": len(external_ready),
        "oracle_objective_counts": dict(oracle_counts),
        "decision": decision,
    }


def grouped_targets(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["source_split"], row["target"]), []).append(row)
    return grouped


def write_report(path: Path, rows: list[dict[str, Any]], csv_path: Path) -> None:
    summary = readiness_summary(rows)
    groups = grouped_targets(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# AlphaQ Objective-Selection Dataset",
        "",
        f"CSV: `{csv_path}`.",
        "",
        "This dataset reframes the current evidence as supervised objective selection: each target/run contains one row per AlphaQ objective and an oracle label derived from the best materialized beam candidate.",
        "",
        "## Bottom Line",
        "",
        f"Decision: `{summary['decision']}`.",
        "",
        f"Train-ready target groups: {summary['train_ready_groups']}/{summary['target_groups']}.",
        f"External train-ready target groups: {summary['external_train_ready_groups']}.",
        f"Oracle objective distribution: {format_counts(summary['oracle_objective_counts'])}.",
        "",
        (
            "We can proceed with a prototype selector and leave-one-target validation. "
            "The dataset is still too small for a high-capacity deep model or a broad journal-level generalization claim."
            if summary["decision"] == "prototype-ready"
            else "Do not train yet; objective coverage or label diversity is insufficient."
        ),
        "",
        "## Target Labels",
        "",
        "| split | target | train ready | materialized objectives | oracle objective |",
        "|---|---|---:|---:|---|",
    ]
    for (split, target), items in sorted(groups.items()):
        first = items[0]
        lines.append(
            f"| {split} | {target} | {first['train_ready']} | {first['target_objective_count']} | {first['oracle_objective'] or '-'} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "-"
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))


def main() -> int:
    args = parse_args()
    rows = build_dataset(
        internal_decomposition_csvs=parse_paths(args.internal_decomposition_csvs),
        internal_grid_csv=args.internal_grid_csv,
        external_decomposition_csv=args.external_decomposition_csv,
        external_grid_csv=args.external_grid_csv,
        night_decomposition_csv=args.night_decomposition_csv,
        night_grid_csv=args.night_grid_csv,
        readiness_csv=args.readiness_csv,
        extra_external_runs=run_suffixes(args.extra_external_runs),
    )
    write_csv(args.output_csv, rows)
    write_report(args.report_path, rows, args.output_csv)
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
