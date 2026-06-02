from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from qiskit.quantum_info import Statevector

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._analysis_common import DEFAULT_CSV_ROOT
from scripts._analysis_common import DEFAULT_REPORTS_ROOT
from scripts._analysis_common import DEFAULT_RESULTS_ROOT
from scripts._analysis_common import ensure_dir
from scripts._analysis_common import load_qasm_circuit
from scripts._analysis_common import natural_sort_key
from scripts._analysis_common import write_csv_rows
from scripts._analysis_common import write_json
from scripts.run_formal_verification import project_path


DEFAULT_FRONTIER_VERIFICATION_CSV = (
    DEFAULT_RESULTS_ROOT
    / "verification"
    / "frontier"
    / "alphaq_candidate_frontier_verification.csv"
)
DEFAULT_FINAL_METRICS_CSV = DEFAULT_CSV_ROOT / "final_metrics.csv"
DEFAULT_OUTPUT_CSV = DEFAULT_CSV_ROOT / "postselection_equivalence_diagnostics.csv"
DEFAULT_OUTPUT_JSON = DEFAULT_CSV_ROOT / "postselection_equivalence_diagnostics.json"
DEFAULT_REPORT_PATH = DEFAULT_REPORTS_ROOT / "postselection_equivalence_diagnostics.md"


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
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


def basis_inputs(num_qubits: int, *, max_columns: int, seed: int) -> list[int]:
    dimension = 2**num_qubits
    if max_columns <= 0 or max_columns >= dimension:
        return list(range(dimension))
    rng = np.random.default_rng(seed)
    anchors = [0, dimension - 1]
    remaining = max(0, max_columns - len(set(anchors)))
    sampled = rng.choice(dimension, size=remaining, replace=False).tolist()
    return sorted(dict.fromkeys([*anchors, *sampled]))


def project_extra_pattern(
    state: np.ndarray,
    *,
    data_qubits: int,
    total_qubits: int,
    pattern: int,
) -> np.ndarray:
    """Project high-index extra qubits onto pattern, preserving q[0:data_qubits]."""
    del total_qubits
    data_dimension = 2**data_qubits
    projected = np.empty(data_dimension, dtype=complex)
    base = pattern << data_qubits
    for data_index in range(data_dimension):
        projected[data_index] = state[base | data_index]
    return projected


def alpha_phase_span(alphas: list[complex], *, zero_tol: float = 1e-12) -> float | None:
    nonzero = [alpha for alpha in alphas if abs(alpha) > zero_tol]
    if not nonzero:
        return None
    phases = np.unwrap(np.angle(np.asarray(nonzero)))
    return float(phases.max() - phases.min())


def diagnose_circuits(
    original: Any,
    candidate: Any,
    *,
    max_columns: int = 64,
    seed: int = 2026,
    tolerance: float = 1e-8,
    max_extra_qubits: int = 8,
) -> dict[str, Any]:
    data_qubits = int(original.num_qubits)
    total_qubits = int(candidate.num_qubits)
    if total_qubits < data_qubits:
        return {
            "diagnostic_status": "dimension-mismatch",
            "diagnostic_error": (
                f"candidate has {total_qubits} qubits but original has {data_qubits}"
            ),
            "data_qubits": data_qubits,
            "candidate_qubits": total_qubits,
        }
    extra_qubits = total_qubits - data_qubits
    if extra_qubits > max_extra_qubits:
        return {
            "diagnostic_status": "too-many-extra-qubits",
            "diagnostic_error": (
                f"{extra_qubits} extra qubits exceeds max_extra_qubits={max_extra_qubits}"
            ),
            "data_qubits": data_qubits,
            "candidate_qubits": total_qubits,
            "extra_qubits": extra_qubits,
        }

    inputs = basis_inputs(data_qubits, max_columns=max_columns, seed=seed)
    data_dimension = 2**data_qubits
    total_dimension = 2**total_qubits
    original_columns = [
        Statevector.from_int(index, data_dimension).evolve(original).data
        for index in inputs
    ]
    candidate_states = [
        Statevector.from_int(index, total_dimension).evolve(candidate).data
        for index in inputs
    ]

    best: dict[str, Any] | None = None
    for pattern in range(2**extra_qubits):
        alphas: list[complex] = []
        projected_columns: list[np.ndarray] = []
        column_residuals: list[float] = []
        success_norms: list[float] = []
        for original_column, candidate_state in zip(original_columns, candidate_states):
            projected = project_extra_pattern(
                candidate_state,
                data_qubits=data_qubits,
                total_qubits=total_qubits,
                pattern=pattern,
            )
            alpha = complex(np.vdot(original_column, projected))
            alphas.append(alpha)
            projected_columns.append(projected)
            column_residuals.append(float(np.linalg.norm(projected - alpha * original_column)))
            success_norms.append(float(np.linalg.norm(projected)))
        global_alpha = sum(alphas) / len(alphas)
        global_residuals = [
            float(np.linalg.norm(projected - global_alpha * original_column))
            for original_column, projected in zip(original_columns, projected_columns)
        ]
        row = {
            "best_postselection_pattern": pattern,
            "max_column_residual": max(column_residuals),
            "mean_column_residual": float(np.mean(column_residuals)),
            "max_global_residual": max(global_residuals),
            "mean_global_residual": float(np.mean(global_residuals)),
            "success_norm_min": min(success_norms),
            "success_norm_max": max(success_norms),
            "alpha_abs_min": min(abs(alpha) for alpha in alphas),
            "alpha_abs_max": max(abs(alpha) for alpha in alphas),
            "alpha_real_min": min(alpha.real for alpha in alphas),
            "alpha_real_max": max(alpha.real for alpha in alphas),
            "alpha_imag_min": min(alpha.imag for alpha in alphas),
            "alpha_imag_max": max(alpha.imag for alpha in alphas),
            "alpha_phase_span": alpha_phase_span(alphas),
            "global_alpha_real": global_alpha.real,
            "global_alpha_imag": global_alpha.imag,
        }
        has_nonzero_success = (
            row["alpha_abs_min"] > tolerance and row["success_norm_min"] > tolerance
        )
        key = (
            0 if has_nonzero_success else 1,
            row["max_column_residual"],
            row["max_global_residual"],
            -row["success_norm_min"],
            row["best_postselection_pattern"],
        )
        if best is None or key < best["_key"]:
            best = {**row, "_key": key}

    assert best is not None
    best.pop("_key", None)
    if best["max_global_residual"] <= tolerance and best["alpha_abs_min"] > tolerance:
        status = "sampled-global-match"
    elif best["max_column_residual"] <= tolerance and best["alpha_abs_min"] > tolerance:
        status = "sampled-columnwise-phase-mismatch"
    else:
        status = "sampled-mismatch"
    return {
        "diagnostic_status": status,
        "diagnostic_error": None,
        "data_qubits": data_qubits,
        "candidate_qubits": total_qubits,
        "extra_qubits": extra_qubits,
        "sampled_columns": len(inputs),
        "sampled_inputs": ";".join(str(item) for item in inputs),
        "tolerance": tolerance,
        **best,
    }


def diagnose_pair(
    *,
    original_qasm: Path,
    candidate_qasm: Path,
    max_columns: int,
    seed: int,
    tolerance: float,
    max_extra_qubits: int,
) -> dict[str, Any]:
    try:
        original = load_qasm_circuit(original_qasm)
        candidate = load_qasm_circuit(candidate_qasm)
        return diagnose_circuits(
            original,
            candidate,
            max_columns=max_columns,
            seed=seed,
            tolerance=tolerance,
            max_extra_qubits=max_extra_qubits,
        )
    except Exception as exc:
        return {
            "diagnostic_status": "diagnostic-error",
            "diagnostic_error": str(exc),
        }


def build_diagnostic_rows(
    *,
    frontier_verification_rows: list[dict[str, str]],
    originals: dict[str, Path],
    circuit_ids: set[str] | None,
    only_non_equal: bool,
    max_columns: int,
    seed: int,
    tolerance: float,
    max_extra_qubits: int,
) -> list[dict[str, Any]]:
    tasks = []
    for row in frontier_verification_rows:
        if circuit_ids is not None and row.get("circuit_id") not in circuit_ids:
            continue
        if only_non_equal and row.get("verification_status") == "equal":
            continue
        tasks.append(row)
    tasks = sorted(
        tasks,
        key=lambda row: (
            natural_sort_key(row.get("circuit_id", "")),
            int(float(row.get("combo_index") or 0)),
        ),
    )

    output_rows: list[dict[str, Any]] = []
    for row in tasks:
        circuit_id = row["circuit_id"]
        original_qasm = originals.get(circuit_id)
        candidate_qasm = project_path(row.get("candidate_qasm_path"))
        if original_qasm is None or not original_qasm.exists():
            diagnostic = {
                "diagnostic_status": "missing-original",
                "diagnostic_error": f"missing original QASM for {circuit_id}",
            }
        elif candidate_qasm is None or not candidate_qasm.exists():
            diagnostic = {
                "diagnostic_status": "missing-candidate",
                "diagnostic_error": f"missing candidate QASM for {row.get('candidate_id')}",
            }
        else:
            diagnostic = diagnose_pair(
                original_qasm=original_qasm,
                candidate_qasm=candidate_qasm,
                max_columns=max_columns,
                seed=seed,
                tolerance=tolerance,
                max_extra_qubits=max_extra_qubits,
            )
        output_rows.append(
            {
                "circuit_id": circuit_id,
                "candidate_id": row.get("candidate_id"),
                "combo_index": row.get("combo_index"),
                "verification_status": row.get("verification_status"),
                "structural_cost": row.get("structural_cost"),
                "tcount_after": row.get("tcount_after"),
                "candidate_qasm_path": row.get("candidate_qasm_path"),
                **diagnostic,
            }
        )
    return output_rows


def write_report(rows: list[dict[str, Any]], report_path: Path, csv_path: Path) -> Path:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("diagnostic_status"))
        counts[status] = counts.get(status, 0) + 1
    table_lines = [
        "| circuit | candidate | verifier | diagnostic | pattern | column residual | global residual | alpha abs | action |",
        "|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        status = row.get("diagnostic_status")
        if status == "sampled-global-match":
            action = "candidate looks consistent under sampled postselection; prioritize formal proof"
        elif status == "sampled-columnwise-phase-mismatch":
            action = "output action matches columnwise but relative phase varies; treat as suspect, not proof"
        elif status == "sampled-mismatch":
            action = "sampled postselection does not support equivalence"
        else:
            action = str(row.get("diagnostic_error") or "inspect diagnostic")
        alpha_abs = row.get("alpha_abs_min")
        table_lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row.get('circuit_id')}`",
                    f"`{row.get('candidate_id')}`",
                    f"`{row.get('verification_status')}`",
                    f"`{status}`",
                    "" if row.get("best_postselection_pattern") is None else str(row.get("best_postselection_pattern")),
                    _fmt(row.get("max_column_residual")),
                    _fmt(row.get("max_global_residual")),
                    _fmt(alpha_abs),
                    action,
                ]
            )
            + " |"
        )
    text = [
        "# Postselection Equivalence Diagnostics",
        "",
        f"- Diagnostic CSV: `{csv_path}`.",
        (
            "- Status counts: "
            + ", ".join(f"{key}={counts[key]}" for key in sorted(counts))
            + "."
        ),
        "",
        "## Reading",
        "",
        (
            "This is a sampled statevector diagnostic, not a replacement for formal "
            "verification. It checks whether projecting the candidate's extra high-index "
            "qubits onto a fixed bit pattern makes the sampled candidate columns match "
            "the original columns up to one global scalar. A columnwise-only match means "
            "the output basis action can look right while the relative phase still depends "
            "on the input, which is not a valid global equivalence claim."
        ),
        "",
        "## Candidate Diagnostics",
        "",
        *table_lines,
        "",
    ]
    ensure_dir(report_path.parent)
    report_path.write_text("\n".join(text), encoding="utf-8")
    return report_path


def _fmt(value: Any, digits: int = 3) -> str:
    if value in (None, ""):
        return ""
    try:
        return f"{float(value):.{digits}g}"
    except (TypeError, ValueError):
        return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample whether inconclusive frontier candidates match under fixed ancilla postselection."
    )
    parser.add_argument(
        "--frontier-verification-csv",
        type=Path,
        default=DEFAULT_FRONTIER_VERIFICATION_CSV,
    )
    parser.add_argument("--final-metrics-csv", type=Path, default=DEFAULT_FINAL_METRICS_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--circuit-id", action="append", dest="circuit_ids", default=None)
    parser.add_argument("--include-equal", action="store_true")
    parser.add_argument("--max-columns", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--max-extra-qubits", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = build_diagnostic_rows(
        frontier_verification_rows=load_csv_rows(args.frontier_verification_csv),
        originals=original_qasm_by_circuit(load_csv_rows(args.final_metrics_csv)),
        circuit_ids=None if args.circuit_ids is None else set(args.circuit_ids),
        only_non_equal=not args.include_equal,
        max_columns=args.max_columns,
        seed=args.seed,
        tolerance=args.tolerance,
        max_extra_qubits=args.max_extra_qubits,
    )
    write_csv_rows(rows, args.output_csv)
    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("diagnostic_status"))
        status_counts[status] = status_counts.get(status, 0) + 1
    write_json(
        {
            "frontier_verification_csv": str(args.frontier_verification_csv),
            "final_metrics_csv": str(args.final_metrics_csv),
            "output_csv": str(args.output_csv),
            "report_path": str(args.report_path),
            "circuit_ids": args.circuit_ids,
            "include_equal": args.include_equal,
            "max_columns": args.max_columns,
            "seed": args.seed,
            "tolerance": args.tolerance,
            "max_extra_qubits": args.max_extra_qubits,
            "num_rows": len(rows),
            "status_counts": status_counts,
        },
        args.output_json,
    )
    write_report(rows, args.report_path, args.output_csv)
    print(
        json.dumps(
            {
                "num_rows": len(rows),
                "output_csv": str(args.output_csv),
                "report_path": str(args.report_path),
                "status_counts": status_counts,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
