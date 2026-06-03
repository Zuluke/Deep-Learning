from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_SWEEP_CSV = (
    PROJECT_ROOT
    / "results"
    / "csv"
    / "split_reward_target_sweep_split4_prior_norepeat_w3_m25.csv"
)
DEFAULT_OUTPUT_CSV = (
    PROJECT_ROOT
    / "results"
    / "csv"
    / "split_reward_target_sweep_split4_prior_norepeat_w3_m25_backfilled.csv"
)
DEFAULT_REPORT = (
    PROJECT_ROOT
    / "results"
    / "reports"
    / "split_reward_consolidation_norepeat_w3_m25.md"
)
DEFAULT_MATERIALIZED_ROOT = PROJECT_ROOT / "results" / "alphaq_split_reward_external"
REMOTE_PROJECT_PREFIXES = (
    "/home/CIN/cacl2/Deep-Learning/project",
    "/home/CIN/cacl2/Deep-Learning",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill solved split-reward materializations and produce a "
            "consolidated resolved/unsolved analysis report."
        )
    )
    parser.add_argument("--sweep-csv", type=Path, default=DEFAULT_SWEEP_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--materialized-root", type=Path, default=DEFAULT_MATERIALIZED_ROOT)
    parser.add_argument("--timeout-sec", type=int, default=900)
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


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def is_solved_row(row: dict[str, str]) -> bool:
    return (
        row.get("status") == "ok"
        and row.get("best_effective_t_cost") not in (None, "", "None")
    )


def needs_backfill(row: dict[str, str]) -> bool:
    return is_solved_row(row) and row.get("materialization_status") != "ok"


def output_root_for(row: dict[str, str], materialized_root: Path) -> Path:
    return materialized_root / f"{row['target']}_{row['mode']}_sweep"


def load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def apply_metrics(row: dict[str, str], metrics_path: Path, summary_path: Path) -> None:
    metrics = load_json(metrics_path)
    def clean(value: Any) -> str:
        return "" if value is None else str(value)

    row["materialization_status"] = "ok" if metrics_path.exists() else row.get(
        "materialization_status", ""
    )
    row["materialization_error"] = "" if metrics_path.exists() else row.get(
        "materialization_error", ""
    )
    row["materialized_summary_json"] = str(summary_path) if summary_path.exists() else ""
    row["materialized_metrics_json"] = str(metrics_path) if metrics_path.exists() else ""
    row["primary_nc_depth_ratio"] = clean(metrics.get("primary_nc_depth_ratio", ""))
    row["tcount_after"] = clean(metrics.get("tcount_after", ""))
    row["qasm_depth_ratio"] = clean(metrics.get("qasm_depth_ratio", ""))


def run_materialization(
    row: dict[str, str],
    *,
    materialized_root: Path,
    timeout_sec: int,
) -> None:
    manifest = resolve_project_path(row.get("candidate_manifest_path"))
    if manifest is None or not manifest.exists():
        row["materialization_status"] = "missing-manifest"
        row["materialization_error"] = "missing candidate manifest"
        return
    output_root = output_root_for(row, materialized_root)
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "backfill_materialize.log"
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "materialize_split_reward_candidate.py"),
        "--manifest-csv",
        str(manifest),
        "--target",
        row["target"],
        "--candidate-kind",
        "best_solved",
        "--split-reward-mode",
        row["mode"],
        "--output-root",
        str(output_root),
    ]
    with log_path.open("w", encoding="utf-8") as log_file:
        try:
            completed = subprocess.run(
                cmd,
                cwd=PROJECT_ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_sec,
                check=False,
            )
        except subprocess.TimeoutExpired:
            log_file.write(f"\nTIMEOUT after {timeout_sec} seconds\n")
            row["materialization_status"] = "timeout"
            row["materialization_error"] = f"timeout after {timeout_sec}s"
            return
    if completed.returncode != 0:
        row["materialization_status"] = "failed"
        row["materialization_error"] = f"exit code {completed.returncode}"
        return
    apply_metrics(
        row,
        output_root / "structural_metrics_from_qasm_original.json",
        output_root / "summary.json",
    )


def normalize_existing_materialization(row: dict[str, str]) -> None:
    metrics_path = resolve_project_path(row.get("materialized_metrics_json"))
    summary_path = resolve_project_path(row.get("materialized_summary_json"))
    if metrics_path is not None and metrics_path.exists():
        apply_metrics(row, metrics_path, summary_path or Path())


def numeric(value: str | None) -> float | None:
    if value in (None, "", "None"):
        return None
    return float(value)


def metric(row: dict[str, str], key: str) -> str:
    metrics = load_json(resolve_project_path(row.get("materialized_metrics_json")))
    value = metrics.get(key, "")
    return "" if value is None else str(value)


def resolved_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if is_solved_row(row) and row.get("materialization_status") == "ok"]


def write_report(path: Path, rows: list[dict[str, str]], output_csv: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    solved = resolved_rows(rows)
    baseline_by_target = {
        row["target"]: row for row in solved if row.get("mode") == "none"
    }
    lines = [
        "# Split Reward Consolidation",
        "",
        f"Backfilled CSV: `{output_csv}`.",
        "",
        "## Resolved Candidates",
        "",
        "| target | mode | effective cost | delta vs none | T-count | qasm depth ratio | nc core ratio | dependency core ratio | structural cost | primary ratio | primary status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in solved:
        baseline_cost = numeric(baseline_by_target.get(row["target"], {}).get("best_effective_t_cost"))
        cost = numeric(row.get("best_effective_t_cost"))
        delta = "" if baseline_cost is None or cost is None else f"{cost - baseline_cost:g}"
        lines.append(
            "| {target} | {mode} | {cost} | {delta} | {tcount} | {qasm} | {nc} | {dep} | {struct} | {primary_ratio} | {primary_status} |".format(
                target=row.get("target", ""),
                mode=row.get("mode", ""),
                cost=row.get("best_effective_t_cost", ""),
                delta=delta,
                tcount=row.get("tcount_after", ""),
                qasm=row.get("qasm_depth_ratio", ""),
                nc=metric(row, "alphaq_nc_core_depth_ratio"),
                dep=metric(row, "alphaq_dependency_core_depth_ratio"),
                struct=metric(row, "structural_cost"),
                primary_ratio=row.get("primary_nc_depth_ratio", ""),
                primary_status=metric(row, "structural_target_status") or "missing",
            )
        )
    lines.extend(
        [
            "",
            "## Unsolved Hamming Diagnostics",
            "",
            "| target | mode | moves used | residual final | residual vs none | split reward sum | mixed AUC | mixed mass |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    unsolved = [row for row in rows if row.get("materialization_status") == "not-solved"]
    none_residual = {
        row["target"]: numeric(row.get("best_return_residual_weight"))
        for row in unsolved
        if row.get("mode") == "none"
    }
    for row in unsolved:
        base = none_residual.get(row["target"])
        residual = numeric(row.get("best_return_residual_weight"))
        delta = "" if base is None or residual is None else f"{residual - base:g}"
        lines.append(
            "| {target} | {mode} | {moves} | {residual} | {delta} | {split} | {auc} | {mass} |".format(
                target=row.get("target", ""),
                mode=row.get("mode", ""),
                moves=row.get("best_return_num_moves", ""),
                residual=row.get("best_return_residual_weight", ""),
                delta=delta,
                split=row.get("avg_split_sum_rewards_final", ""),
                auc=row.get("avg_split_mixed_auc_sum_final", ""),
                mass=row.get("avg_split_mixed_mass_sum_final", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Does The Greedy Hypothesis Fit?",
            "",
            "Partially. The low-weight one-step audit found no immediate residual-improving actions, so every successful synthesis must tolerate an initial residual increase. The resolved targets show that this is possible under the current loop, but the hamming targets hit the 25-move horizon with residuals above their initial tensors. This points to horizon/action-space/search capacity before another circuit-level materialization change.",
            "",
            "## Next Experiment Recommendation",
            "",
            "Run a hamming-only feasibility sweep over `max_num_moves` and action dictionaries before changing the split reward again. Keep `none`, `v2_progress`, and `v3_frontier`; measure solved status, min residual, final residual, and per-step residual trajectories.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    rows, fieldnames = read_csv(args.sweep_csv)
    for row in rows:
        if needs_backfill(row):
            run_materialization(
                row,
                materialized_root=args.materialized_root,
                timeout_sec=args.timeout_sec,
            )
        elif row.get("materialization_status") == "ok":
            normalize_existing_materialization(row)
    write_csv(args.output_csv, rows, fieldnames)
    write_report(args.report, rows, args.output_csv)
    print(json.dumps({"output_csv": str(args.output_csv), "report": str(args.report)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
