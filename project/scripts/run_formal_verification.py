from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._analysis_common import DEFAULT_CSV_ROOT
from scripts._analysis_common import DEFAULT_REPORTS_ROOT
from scripts._analysis_common import DEFAULT_RESULTS_ROOT
from scripts._analysis_common import dump_qasm_v2
from scripts._analysis_common import ensure_dir
from scripts._analysis_common import load_qasm_circuit
from scripts._analysis_common import natural_sort_key
from scripts._analysis_common import normalize_circuit_to_basis
from scripts._analysis_common import write_csv_rows
from scripts._analysis_common import write_json


OK_METHOD_STATUSES = {"ok", "skipped-existing", "partial-log-only"}
DEFAULT_ENTREGA1_CIRCUITS = {
    "mod_5_4",
    "gf_2pow2_mult",
    "cuccaro_adder_n3",
    "qft_4",
    "vbe_adder_3",
    "hamming_15_low",
    "qcla_mod_7",
}


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def project_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def relative_or_absolute(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def circuit_to_tensor_binary() -> Path:
    return PROJECT_ROOT / "external" / "circuit-to-tensor" / "target" / "release" / "circuit-to-tensor"


def feynver_path() -> str | None:
    found = shutil.which("feynver")
    if found:
        return found
    local = Path.home() / ".local" / "bin" / "feynver"
    return str(local) if local.exists() else None


def verifier_env() -> dict[str, str]:
    env = os.environ.copy()
    prefix = [
        str(Path.home() / ".local" / "bin"),
        "/opt/homebrew/bin",
        "/usr/local/bin",
    ]
    env["PATH"] = ":".join([*prefix, env.get("PATH", "")])
    return env


def run_command_with_process_group_timeout(
    cmd: list[str],
    *,
    timeout_sec: int,
) -> tuple[int, str, str, bool]:
    proc = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=verifier_env(),
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_sec)
        return proc.returncode or 0, stdout, stderr, False
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = proc.communicate()
        return proc.returncode if proc.returncode is not None else -9, stdout, stderr, True


def normalize_qasm_for_verifier(source: Path, output: Path) -> tuple[str, str | None]:
    try:
        circuit = load_qasm_circuit(source)
        normalized, status, error = normalize_circuit_to_basis(circuit)
        if normalized is None:
            return status, error
        dump_qasm_v2(normalized, output)
        return status, error
    except Exception as exc:
        return "failed", str(exc)


def classify_proof(returncode: int, stdout: str, stderr: str) -> tuple[str, str | None]:
    combined = (stdout + "\n" + stderr).strip()
    if stdout.startswith("Equal"):
        return "equal", None
    if "Error:" in combined or "Unexpected:" in combined:
        return "parse-error", combined[:1200]
    if "Not equal" in combined or "Not Equal" in combined:
        return "not-equal", combined[:1200]
    if returncode != 0:
        return "tool-error", combined[:1200]
    if combined:
        return "inconclusive", combined[:1200]
    return "inconclusive", "Verifier returned no output."


def compact_message(value: Any, *, max_chars: int = 240) -> str:
    if value is None:
        return "see proof"
    text = str(value).strip()
    if not text:
        return "see proof"
    first_line = text.splitlines()[0].strip()
    if len(first_line) <= max_chars:
        return first_line
    return first_line[: max_chars - 3] + "..."


def run_verification_pair(
    *,
    original_qasm: Path,
    candidate_qasm: Path,
    proof_path: Path,
    normalized_original: Path,
    normalized_candidate: Path,
    timeout_sec: int,
) -> dict[str, Any]:
    original_norm_status, original_norm_error = normalize_qasm_for_verifier(
        original_qasm, normalized_original
    )
    candidate_norm_status, candidate_norm_error = normalize_qasm_for_verifier(
        candidate_qasm, normalized_candidate
    )
    if not normalized_original.exists() or not normalized_candidate.exists():
        error = (
            f"normalization original={original_norm_status}: {original_norm_error}; "
            f"candidate={candidate_norm_status}: {candidate_norm_error}"
        )
        proof_path.write_text(error + "\n", encoding="utf-8")
        return {
            "verification_status": "normalization-failed",
            "verification_error": error,
            "proof_path": relative_or_absolute(proof_path),
            "runtime_sec": None,
            "original_normalization_status": original_norm_status,
            "candidate_normalization_status": candidate_norm_status,
        }

    cmd = [
        str(circuit_to_tensor_binary()),
        "verify",
        str(normalized_original),
        str(normalized_candidate),
    ]
    start = time.time()
    returncode, stdout, stderr, timed_out = run_command_with_process_group_timeout(
        cmd,
        timeout_sec=timeout_sec,
    )
    if timed_out:
        runtime = time.time() - start
        output = (stdout + "\n" + stderr).strip()
        proof_path.write_text(output + "\n", encoding="utf-8")
        return {
            "verification_status": "timeout",
            "verification_error": f"Timed out after {timeout_sec}s.",
            "proof_path": relative_or_absolute(proof_path),
            "runtime_sec": runtime,
            "original_normalization_status": original_norm_status,
            "candidate_normalization_status": candidate_norm_status,
        }

    runtime = time.time() - start
    output = stdout + stderr
    proof_path.write_text(output, encoding="utf-8")
    status, error = classify_proof(returncode, stdout, stderr)
    return {
        "verification_status": status,
        "verification_error": error,
        "proof_path": relative_or_absolute(proof_path),
        "runtime_sec": runtime,
        "original_normalization_status": original_norm_status,
        "candidate_normalization_status": candidate_norm_status,
    }


def build_tasks(rows: list[dict[str, str]], scope: str) -> list[dict[str, str]]:
    originals = {row["circuit_id"]: row for row in rows if row["method"] == "original"}
    tasks = []
    for row in rows:
        if row["method"] == "original":
            continue
        if row.get("method_status") not in OK_METHOD_STATUSES:
            continue
        if not row.get("qasm_artifact_path"):
            continue
        if scope == "entrega1" and row["circuit_id"] not in DEFAULT_ENTREGA1_CIRCUITS:
            continue
        original = originals.get(row["circuit_id"])
        if not original or not original.get("qasm_artifact_path"):
            continue
        tasks.append({**row, "original_qasm_artifact_path": original["qasm_artifact_path"]})
    return sorted(
        tasks,
        key=lambda row: (
            0 if row["circuit_id"] in DEFAULT_ENTREGA1_CIRCUITS else 1,
            natural_sort_key(row["circuit_id"]),
            row["method"],
        ),
    )


def run_verifications(
    *,
    final_rows: list[dict[str, str]],
    scope: str,
    output_root: Path,
    timeout_sec: int,
    limit: int | None,
) -> list[dict[str, Any]]:
    ensure_dir(output_root)
    proof_root = ensure_dir(output_root / "proofs")
    normalized_root = ensure_dir(output_root / "normalized_qasm")
    tasks = build_tasks(final_rows, scope)
    if limit is not None:
        tasks = tasks[:limit]

    results = []
    for index, row in enumerate(tasks, start=1):
        circuit_id = row["circuit_id"]
        method = row["method"]
        original_qasm = project_path(row["original_qasm_artifact_path"])
        candidate_qasm = project_path(row["qasm_artifact_path"])
        base_result: dict[str, Any] = {
            "task_index": index,
            "num_tasks": len(tasks),
            "circuit_id": circuit_id,
            "method": method,
            "method_status": row.get("method_status"),
            "original_qasm_artifact_path": row["original_qasm_artifact_path"],
            "candidate_qasm_artifact_path": row["qasm_artifact_path"],
            "tcount_before": row.get("tcount_before"),
            "tcount_after": row.get("tcount_after"),
        }
        if original_qasm is None or not original_qasm.exists():
            results.append(
                {
                    **base_result,
                    "verification_status": "missing-original",
                    "verification_error": "Original QASM artifact is missing.",
                }
            )
            continue
        if candidate_qasm is None or not candidate_qasm.exists():
            results.append(
                {
                    **base_result,
                    "verification_status": "missing-candidate",
                    "verification_error": "Candidate QASM artifact is missing.",
                }
            )
            continue

        pair_dir = ensure_dir(proof_root / circuit_id)
        normalized_pair_dir = ensure_dir(normalized_root / circuit_id / method)
        verification = run_verification_pair(
            original_qasm=original_qasm,
            candidate_qasm=candidate_qasm,
            proof_path=pair_dir / f"{method}.verify.txt",
            normalized_original=normalized_pair_dir / "original.normalized.qasm",
            normalized_candidate=normalized_pair_dir / "candidate.normalized.qasm",
            timeout_sec=timeout_sec,
        )
        results.append({**base_result, **verification})
        print(
            f"[{index}/{len(tasks)}] {circuit_id}/{method}: "
            f"{verification['verification_status']}"
        )
    return results


def write_summary_report(rows: list[dict[str, Any]], report_path: Path, *, scope: str) -> Path:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["verification_status"]] = counts.get(row["verification_status"], 0) + 1
    equal_rows = [row for row in rows if row["verification_status"] == "equal"]
    failed_rows = [row for row in rows if row["verification_status"] != "equal"]
    lines = [
        "# Formal Verification Summary",
        "",
        f"- Scope: `{scope}`.",
        f"- Total verification tasks: {len(rows)}.",
        f"- Equal: {len(equal_rows)}.",
        *[f"- {status}: {count}" for status, count in sorted(counts.items()) if status != "equal"],
        "",
        "## Non-equal or inconclusive cases",
        "",
    ]
    if failed_rows:
        for row in failed_rows[:80]:
            lines.append(
                f"- `{row['circuit_id']}` / `{row['method']}`: "
                f"{row['verification_status']} - "
                f"{compact_message(row.get('verification_error'))}; "
                f"proof: `{row.get('proof_path', 'not written')}`"
            )
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Verification is run through `circuit-to-tensor verify`, which calls `feynver -postselect-ancillas -ignore-global-phase`.",
            "- QASM files are first normalized to the local Clifford+T comparison basis so PyZX `rz(k*pi/4)` output can be checked by the Feynman `.qc` backend.",
            "- A status of `equal` means Feynman returned an equality proof for the normalized original/candidate pair.",
            "",
        ]
    )
    ensure_dir(report_path.parent)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run formal equivalence checks with feynver.")
    parser.add_argument("--final-metrics-csv", type=Path, default=DEFAULT_CSV_ROOT / "final_metrics.csv")
    parser.add_argument("--scope", choices=("entrega1", "all"), default="entrega1")
    parser.add_argument("--timeout-sec", type=int, default=30)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT / "verification",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if feynver_path() is None:
        raise SystemExit("feynver not found in PATH. Install meamy/feynman first.")
    if not circuit_to_tensor_binary().exists():
        raise SystemExit(f"Missing circuit-to-tensor binary: {circuit_to_tensor_binary()}")

    scope_root = ensure_dir(args.output_root / args.scope)
    summary_csv = args.summary_csv or scope_root / "verification_summary.csv"
    summary_json = args.summary_json or scope_root / "verification_summary.json"
    rows = run_verifications(
        final_rows=load_csv_rows(args.final_metrics_csv),
        scope=args.scope,
        output_root=scope_root,
        timeout_sec=args.timeout_sec,
        limit=args.limit,
    )
    write_csv_rows(rows, summary_csv)
    write_json(
        {
            "scope": args.scope,
            "timeout_sec": args.timeout_sec,
            "num_rows": len(rows),
            "feynver_path": feynver_path(),
            "circuit_to_tensor_binary": str(circuit_to_tensor_binary()),
            "summary_csv": str(summary_csv),
        },
        summary_json,
    )
    report_path = write_summary_report(
        rows,
        args.report_path or DEFAULT_REPORTS_ROOT / f"formal_verification_{args.scope}_summary.md",
        scope=args.scope,
    )
    print(
        json.dumps(
            {
                "summary_csv": str(summary_csv),
                "summary_json": str(summary_json),
                "report_path": str(report_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
