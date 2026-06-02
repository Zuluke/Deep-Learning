from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._analysis_common import DEFAULT_CSV_ROOT
from scripts._analysis_common import DEFAULT_RESULTS_ROOT
from scripts._analysis_common import ensure_dir
from scripts._analysis_common import natural_sort_key
from scripts._analysis_common import write_csv_rows
from scripts._analysis_common import write_json
from scripts.run_formal_verification import circuit_to_tensor_binary
from scripts.run_formal_verification import feynver_path
from scripts.run_formal_verification import project_path
from scripts.run_formal_verification import run_verification_pair


DEFAULT_FRONTIER_CSV = (
    DEFAULT_RESULTS_ROOT / "public_resynth_structural" / "candidate_frontier.csv"
)
DEFAULT_FINAL_METRICS_CSV = DEFAULT_CSV_ROOT / "final_metrics.csv"
DEFAULT_OUTPUT_ROOT = DEFAULT_RESULTS_ROOT / "verification" / "frontier"
DEFAULT_SUMMARY_CSV = DEFAULT_OUTPUT_ROOT / "alphaq_candidate_frontier_verification.csv"
DEFAULT_SUMMARY_JSON = DEFAULT_OUTPUT_ROOT / "alphaq_candidate_frontier_verification.json"
DEFAULT_REPORT_PATH = DEFAULT_OUTPUT_ROOT / "alphaq_candidate_frontier_verification.md"


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def original_qasm_by_circuit(final_rows: list[dict[str, str]]) -> dict[str, Path]:
    originals: dict[str, Path] = {}
    for row in final_rows:
        if row.get("method") != "original":
            continue
        qasm_path = project_path(row.get("qasm_artifact_path"))
        if qasm_path is not None:
            originals[row["circuit_id"]] = qasm_path
    return originals


def verify_frontier(
    *,
    frontier_rows: list[dict[str, str]],
    originals: dict[str, Path],
    output_root: Path,
    timeout_sec: int,
    limit: int | None,
    circuit_ids: set[str] | None,
) -> list[dict[str, Any]]:
    proof_root = ensure_dir(output_root / "proofs")
    normalized_root = ensure_dir(output_root / "normalized_qasm")
    rows: list[dict[str, Any]] = []
    tasks = [
        row
        for row in frontier_rows
        if row.get("status") == "ok"
        and row.get("selection_status") == "ok"
        and row.get("candidate_qasm_path")
        and (circuit_ids is None or row.get("circuit_id") in circuit_ids)
    ]
    tasks = sorted(
        tasks,
        key=lambda row: (
            natural_sort_key(row["circuit_id"]),
            int(float(row.get("combo_index") or 0)),
        ),
    )
    if limit is not None:
        tasks = tasks[:limit]

    for task_index, row in enumerate(tasks, start=1):
        circuit_id = row["circuit_id"]
        candidate_id = row["candidate_id"]
        original_qasm = originals.get(circuit_id)
        candidate_qasm = project_path(row.get("candidate_qasm_path"))
        pair_dir = ensure_dir(proof_root / circuit_id)
        normalized_dir = ensure_dir(normalized_root / circuit_id / f"combo{row.get('combo_index')}")
        if original_qasm is None or not original_qasm.exists():
            result = {
                "verification_status": "missing-original",
                "verification_error": f"Missing original QASM for {circuit_id}.",
                "proof_path": None,
                "runtime_sec": None,
                "original_normalization_status": None,
                "candidate_normalization_status": None,
            }
        elif candidate_qasm is None or not candidate_qasm.exists():
            result = {
                "verification_status": "missing-candidate",
                "verification_error": f"Missing candidate QASM for {candidate_id}.",
                "proof_path": None,
                "runtime_sec": None,
                "original_normalization_status": None,
                "candidate_normalization_status": None,
            }
        else:
            result = run_verification_pair(
                original_qasm=original_qasm,
                candidate_qasm=candidate_qasm,
                proof_path=pair_dir / f"combo{row.get('combo_index')}.verify.txt",
                normalized_original=normalized_dir / "original.normalized.qasm",
                normalized_candidate=normalized_dir / "candidate.normalized.qasm",
                timeout_sec=timeout_sec,
            )
        print(
            f"[{task_index}/{len(tasks)}] {candidate_id}: "
            f"{result['verification_status']}",
            flush=True,
        )
        rows.append(
            {
                "task_index": task_index,
                "num_tasks": len(tasks),
                "circuit_id": circuit_id,
                "candidate_id": candidate_id,
                "combo_index": row.get("combo_index"),
                "candidate_qasm_path": row.get("candidate_qasm_path"),
                "structural_cost": row.get("structural_cost"),
                "tcount_after": row.get("tcount_after"),
                **result,
            }
        )
    return rows


def write_report(rows: list[dict[str, Any]], path: Path, summary_csv: Path) -> Path:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("verification_status"))
        counts[status] = counts.get(status, 0) + 1
    lines = [
        "# AlphaQ Candidate Frontier Verification",
        "",
        f"- CSV: `{summary_csv}`.",
        f"- Total candidates checked: {len(rows)}.",
        *[f"- {status}: {count}" for status, count in sorted(counts.items())],
        "",
        "## Non-equal Cases",
        "",
    ]
    problematic = [
        row
        for row in rows
        if row.get("verification_status") not in {"equal"}
    ]
    if not problematic:
        lines.append("- None.")
    else:
        for row in problematic:
            lines.append(
                f"- `{row['candidate_id']}`: {row.get('verification_status')} - "
                f"{row.get('verification_error') or 'see proof'}"
            )
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Formally verify candidate-frontier QASM files with feynver."
    )
    parser.add_argument("--frontier-csv", type=Path, default=DEFAULT_FRONTIER_CSV)
    parser.add_argument("--final-metrics-csv", type=Path, default=DEFAULT_FINAL_METRICS_CSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--timeout-sec", type=int, default=30)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--circuit-id", action="append", dest="circuit_ids", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if feynver_path() is None:
        raise SystemExit("feynver not found in PATH.")
    if not circuit_to_tensor_binary().exists():
        raise SystemExit(f"Missing circuit-to-tensor binary: {circuit_to_tensor_binary()}")

    frontier_rows = load_csv_rows(args.frontier_csv)
    originals = original_qasm_by_circuit(load_csv_rows(args.final_metrics_csv))
    rows = verify_frontier(
        frontier_rows=frontier_rows,
        originals=originals,
        output_root=args.output_root,
        timeout_sec=args.timeout_sec,
        limit=args.limit,
        circuit_ids=None if args.circuit_ids is None else set(args.circuit_ids),
    )
    write_csv_rows(rows, args.summary_csv)
    report_path = write_report(rows, args.report_path, args.summary_csv)
    write_json(
        {
            "frontier_csv": str(args.frontier_csv),
            "final_metrics_csv": str(args.final_metrics_csv),
            "summary_csv": str(args.summary_csv),
            "report_path": str(report_path),
            "timeout_sec": args.timeout_sec,
            "num_rows": len(rows),
            "circuit_ids": args.circuit_ids,
        },
        args.summary_json,
    )
    print(
        json.dumps(
            {
                "summary_csv": str(args.summary_csv),
                "summary_json": str(args.summary_json),
                "report_path": str(report_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
