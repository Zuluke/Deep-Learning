from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_best_objective_beam_ablation import inf_if_none
from scripts.run_best_objective_beam_ablation import read_csv


DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "results" / "csv" / "alphaq_external_runs_consolidated.csv"
DEFAULT_REPORT = PROJECT_ROOT / "results" / "reports" / "alphaq_external_runs_consolidated.md"
DEFAULT_RUNS = (
    "standard",
    "night_long",
    "article_core",
    "article_extended",
    "journal_repair",
    "journal_repair_paircap",
    "journal_full_1_rerun",
    "journal_full_1_rerun2",
    "journal_full_2_rerun",
    "journal_full_2_rerun2",
    "journal_full_mod_mult_55",
    "journal_full_cuccaro_adder_n4",
    "journal_full_gf_2pow4_mult",
    "journal_full_hamming_weight_n6",
    "journal_full_hamming_weight_n7",
    "journal_full_gf_2pow5_mult",
    "journal_full_nc_tof_5",
    "journal_full_cuccaro_adder_n5",
    "article_repair2_barenco",
    "article_repair2_vbe",
    "journal_full_nc_tof_5_long",
    "journal_full_gf_2pow5_mult_long",
    "journal_k6_backfill",
    "journal_k6_frontier",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consolidate multiple suffixed AlphaQ external-validation runs."
    )
    parser.add_argument(
        "--runs",
        default=",".join(DEFAULT_RUNS),
        help="Comma-separated run suffixes. Use `standard` for unsuffixed external-validation CSVs.",
    )
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def run_suffixes(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def run_paths(run: str) -> tuple[Path, Path]:
    if run == "standard":
        return (
            PROJECT_ROOT / "results" / "csv" / "alphaq_decomposition_objective_external_validation.csv",
            PROJECT_ROOT / "results" / "csv" / "alphaq_objective_beam_policy_external_validation_grid.csv",
        )
    return (
        PROJECT_ROOT / "results" / "csv" / f"alphaq_decomposition_objective_external_validation_{run}.csv",
        PROJECT_ROOT / "results" / "csv" / f"alphaq_objective_beam_policy_external_validation_{run}_grid.csv",
    )


def consolidation_rows(runs: list[str]) -> list[dict[str, Any]]:
    data = []
    all_keys: set[tuple[str, str]] = set()
    for run in runs:
        decomp_path, grid_path = run_paths(run)
        decomp_rows = read_csv(decomp_path) if decomp_path.exists() else []
        grid_rows = read_csv(grid_path) if grid_path.exists() else []
        keyed_decomp = keyed(decomp_rows)
        beams = best_beam_rows(grid_rows)
        all_keys.update(keyed_decomp)
        data.append((run, keyed_decomp, beams, decomp_path.exists(), grid_path.exists()))
    rows = []
    for target, objective in sorted(all_keys):
        candidates = []
        completed_runs = []
        failed_runs = []
        missing_runs = []
        for run, keyed_decomp, beams, has_decomp, _has_grid in data:
            decomp = keyed_decomp.get((target, objective))
            if decomp is None:
                if has_decomp:
                    missing_runs.append(run)
                continue
            status = status_of(decomp)
            if status == "ok":
                completed_runs.append(run)
            elif status == "failed":
                failed_runs.append(run)
            beam = beams.get((target, objective), {})
            candidates.append((run, decomp, beam, status))
        best = best_candidate(candidates)
        rows.append(
            {
                "target": target,
                "objective_variant": objective,
                "best_run": "" if best is None else best[0],
                "best_status": "missing" if best is None else best[3],
                "best_tcount": "" if best is None else best[1].get("tcount", ""),
                "best_beam_qasm_depth": "" if best is None else best[2].get("qasm_depth", ""),
                "best_beam_tdepth": "" if best is None else best[2].get("tdepth", ""),
                "best_beam_materializer": "" if best is None else best[2].get("materializer", ""),
                "completed_runs": ",".join(completed_runs),
                "failed_runs": ",".join(failed_runs),
                "missing_runs": ",".join(missing_runs),
            }
        )
    return rows


def keyed(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {
        (row["target"], row["objective_variant"]): row
        for row in rows
        if row.get("target") and row.get("objective_variant")
    }


def status_of(row: dict[str, str]) -> str:
    return row.get("execution_status") or "ok"


def best_candidate(
    candidates: list[tuple[str, dict[str, str], dict[str, str], str]]
) -> tuple[str, dict[str, str], dict[str, str], str] | None:
    ok = [item for item in candidates if item[3] == "ok"]
    if ok:
        return min(ok, key=candidate_key)
    return candidates[0] if candidates else None


def candidate_key(item: tuple[str, dict[str, str], dict[str, str], str]) -> tuple[float, float, float, str]:
    run, decomp, beam, _status = item
    return (
        inf_if_none(decomp.get("tcount")),
        inf_if_none(beam.get("qasm_depth")),
        inf_if_none(beam.get("num_total_cnots")),
        run,
    )


def best_beam_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    best: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        if not row.get("materializer", "").startswith("selected-beam"):
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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "target",
        "objective_variant",
        "best_run",
        "best_status",
        "best_tcount",
        "best_beam_qasm_depth",
        "best_beam_tdepth",
        "best_beam_materializer",
        "completed_runs",
        "failed_runs",
        "missing_runs",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, lineterminator="\n", fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict[str, Any]], csv_path: Path, runs: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    targets = sorted({row["target"] for row in rows})
    complete_targets = [
        target
        for target in targets
        if all(
            row["best_status"] == "ok"
            for row in rows
            if row["target"] == target
        )
    ]
    lines = [
        "# AlphaQ External Runs Consolidated",
        "",
        f"CSV: `{csv_path}`.",
        f"Runs: `{','.join(runs)}`.",
        "",
        f"Complete targets: {len(complete_targets)}/{len(targets)}.",
        "",
        "| target | objective | best run | status | T-count | best QASM | completed runs | failed runs |",
        "|---|---|---|---|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {target} | {objective_variant} | {best_run} | {best_status} | {best_tcount} | {best_beam_qasm_depth} | {completed_runs} | {failed_runs} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    runs = run_suffixes(args.runs)
    rows = consolidation_rows(runs)
    write_csv(args.output_csv, rows)
    write_report(args.report_path, rows, args.output_csv, runs)
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
