from __future__ import annotations

import argparse
from collections import Counter
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
DEFAULT_OUTPUT_CSV = DEFAULT_CSV_ROOT / "block_replacement_equivalence_diagnostics.csv"
DEFAULT_OUTPUT_JSON = DEFAULT_CSV_ROOT / "block_replacement_equivalence_diagnostics.json"
DEFAULT_REPORT_PATH = DEFAULT_REPORTS_ROOT / "block_replacement_equivalence_diagnostics.md"


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


def alpha_phase_span(alphas: list[complex], *, zero_tol: float = 1e-12) -> float | None:
    nonzero = [alpha for alpha in alphas if abs(alpha) > zero_tol]
    if not nonzero:
        return None
    phases = np.unwrap(np.angle(np.asarray(nonzero)))
    return float(phases.max() - phases.min())


def diagnose_block_circuits(
    reference: Any,
    candidate: Any,
    *,
    max_columns: int = 64,
    seed: int = 2026,
    tolerance: float = 1e-8,
    max_qubits: int = 20,
) -> dict[str, Any]:
    if reference.num_qubits != candidate.num_qubits:
        return {
            "block_diagnostic_status": "dimension-mismatch",
            "block_diagnostic_error": (
                f"reference has {reference.num_qubits} qubits but candidate has "
                f"{candidate.num_qubits}"
            ),
            "block_reference_qubits": reference.num_qubits,
            "block_candidate_qubits": candidate.num_qubits,
        }
    num_qubits = int(reference.num_qubits)
    if num_qubits > max_qubits:
        return {
            "block_diagnostic_status": "too-many-qubits",
            "block_diagnostic_error": (
                f"{num_qubits} qubits exceeds max_qubits={max_qubits}"
            ),
            "block_reference_qubits": num_qubits,
            "block_candidate_qubits": num_qubits,
        }
    dimension = 2**num_qubits
    inputs = basis_inputs(num_qubits, max_columns=max_columns, seed=seed)
    alphas: list[complex] = []
    column_residuals: list[float] = []
    candidate_columns: list[np.ndarray] = []
    reference_columns: list[np.ndarray] = []
    for index in inputs:
        reference_column = Statevector.from_int(index, dimension).evolve(reference).data
        candidate_column = Statevector.from_int(index, dimension).evolve(candidate).data
        alpha = complex(np.vdot(reference_column, candidate_column))
        alphas.append(alpha)
        reference_columns.append(reference_column)
        candidate_columns.append(candidate_column)
        column_residuals.append(
            float(np.linalg.norm(candidate_column - alpha * reference_column))
        )
    global_alpha = sum(alphas) / len(alphas)
    global_residuals = [
        float(np.linalg.norm(candidate_column - global_alpha * reference_column))
        for reference_column, candidate_column in zip(reference_columns, candidate_columns)
    ]
    max_column_residual = max(column_residuals)
    max_global_residual = max(global_residuals)
    alpha_abs_min = min(abs(alpha) for alpha in alphas)
    if max_global_residual <= tolerance and alpha_abs_min > tolerance:
        status = "sampled-global-match"
    elif max_column_residual <= tolerance and alpha_abs_min > tolerance:
        status = "sampled-columnwise-phase-mismatch"
    else:
        status = "sampled-mismatch"
    return {
        "block_diagnostic_status": status,
        "block_diagnostic_error": None,
        "block_reference_qubits": num_qubits,
        "block_candidate_qubits": num_qubits,
        "sampled_columns": len(inputs),
        "sampled_inputs": ";".join(str(item) for item in inputs),
        "tolerance": tolerance,
        "max_column_residual": max_column_residual,
        "mean_column_residual": float(np.mean(column_residuals)),
        "max_global_residual": max_global_residual,
        "mean_global_residual": float(np.mean(global_residuals)),
        "alpha_abs_min": alpha_abs_min,
        "alpha_abs_max": max(abs(alpha) for alpha in alphas),
        "alpha_real_min": min(alpha.real for alpha in alphas),
        "alpha_real_max": max(alpha.real for alpha in alphas),
        "alpha_imag_min": min(alpha.imag for alpha in alphas),
        "alpha_imag_max": max(alpha.imag for alpha in alphas),
        "alpha_phase_span": alpha_phase_span(alphas),
        "global_alpha_real": global_alpha.real,
        "global_alpha_imag": global_alpha.imag,
        **diagnose_monomial_phase_difference(
            reference,
            candidate,
            max_qubits=max_qubits,
        ),
    }


def diagnose_monomial_phase_difference(
    reference: Any,
    candidate: Any,
    *,
    max_qubits: int = 20,
    max_terms: int = 24,
) -> dict[str, Any]:
    if reference.num_qubits != candidate.num_qubits:
        return {
            "monomial_phase_status": "dimension-mismatch",
            "monomial_phase_error": (
                f"reference has {reference.num_qubits} qubits but candidate has "
                f"{candidate.num_qubits}"
            ),
        }
    num_qubits = int(reference.num_qubits)
    if num_qubits > max_qubits:
        return {
            "monomial_phase_status": "too-many-qubits",
            "monomial_phase_error": f"{num_qubits} qubits exceeds max_qubits={max_qubits}",
            "monomial_num_qubits": num_qubits,
        }

    reference_sim = simulate_monomial_circuit(reference, max_qubits=max_qubits)
    candidate_sim = simulate_monomial_circuit(candidate, max_qubits=max_qubits)
    if reference_sim["status"] != "ok":
        return {
            "monomial_phase_status": "unsupported-reference",
            "monomial_phase_error": reference_sim.get("error"),
            "monomial_num_qubits": num_qubits,
        }
    if candidate_sim["status"] != "ok":
        return {
            "monomial_phase_status": "unsupported-candidate",
            "monomial_phase_error": candidate_sim.get("error"),
            "monomial_num_qubits": num_qubits,
        }

    reference_output = reference_sim["output_index"]
    candidate_output = candidate_sim["output_index"]
    output_mismatch_count = int(np.count_nonzero(reference_output != candidate_output))
    delta = (candidate_sim["phase"] - reference_sim["phase"]) % 8
    counts = Counter(int(value) for value in delta.tolist())
    phase_values = ";".join(f"{key}:{counts[key]}" for key in sorted(counts))
    sign_only = all(value in {0, 4} for value in counts)
    global_phase_only = len(counts) == 1
    terms: list[str] = []
    degree: int | None = None
    num_terms: int | None = None
    if output_mismatch_count == 0 and sign_only:
        truth = (delta == 4).astype(np.uint8)
        masks = boolean_anf_masks(truth)
        num_terms = len(masks)
        degree = max((int(mask).bit_count() for mask in masks), default=0)
        terms = [format_anf_term(mask, num_qubits) for mask in masks[:max_terms]]

    return {
        "monomial_phase_status": "ok",
        "monomial_phase_error": None,
        "monomial_num_qubits": num_qubits,
        "monomial_num_inputs": int(2**num_qubits),
        "monomial_output_mismatch_count": output_mismatch_count,
        "monomial_phase_delta_values": phase_values,
        "monomial_phase_delta_sign_only": int(sign_only),
        "monomial_phase_delta_global_only": int(global_phase_only),
        "monomial_phase_delta_degree": degree,
        "monomial_phase_delta_num_terms": num_terms,
        "monomial_phase_delta_terms": ";".join(terms),
        "monomial_phase_delta_terms_truncated": int(
            num_terms is not None and len(terms) < num_terms
        ),
    }


def simulate_monomial_circuit(circuit: Any, *, max_qubits: int) -> dict[str, Any]:
    num_qubits = int(circuit.num_qubits)
    if num_qubits > max_qubits:
        return {
            "status": "too-many-qubits",
            "error": f"{num_qubits} qubits exceeds max_qubits={max_qubits}",
        }
    dimension = 2**num_qubits
    indexes = np.arange(dimension, dtype=np.uint64)
    bit_positions = np.arange(num_qubits, dtype=np.uint64)
    bits = ((indexes[:, None] >> bit_positions) & 1).astype(np.uint8)
    phase = np.zeros(dimension, dtype=np.uint8)

    def qindex(qubit: Any) -> int:
        return int(circuit.find_bit(qubit).index)

    for instruction in circuit.data:
        operation = instruction.operation
        name = operation.name
        qubits = [qindex(qubit) for qubit in instruction.qubits]
        if name in {"barrier", "id"}:
            continue
        if name == "x":
            bits[:, qubits[0]] ^= 1
        elif name in {"cx", "cnot"}:
            control, target = qubits
            bits[:, target] ^= bits[:, control]
        elif name == "ccx":
            control_a, control_b, target = qubits
            bits[:, target] ^= bits[:, control_a] & bits[:, control_b]
        elif name in {"t", "tdg", "s", "sdg", "z"}:
            exponent = {"t": 1, "s": 2, "z": 4, "sdg": 6, "tdg": 7}[name]
            phase = (phase + exponent * bits[:, qubits[0]]) % 8
        elif name == "cz":
            control, target = qubits
            phase = (phase + 4 * (bits[:, control] & bits[:, target])) % 8
        elif name == "ccz":
            control_a, control_b, target = qubits
            phase = (
                phase
                + 4 * (bits[:, control_a] & bits[:, control_b] & bits[:, target])
            ) % 8
        else:
            return {
                "status": "unsupported-gate",
                "error": f"unsupported monomial gate: {name}",
            }

    output_index = np.zeros(dimension, dtype=np.uint64)
    for bit in range(num_qubits):
        output_index |= bits[:, bit].astype(np.uint64) << np.uint64(bit)
    return {
        "status": "ok",
        "phase": phase,
        "output_index": output_index,
    }


def boolean_anf_masks(truth: np.ndarray) -> list[int]:
    coefficients = truth.copy()
    size = int(coefficients.shape[0])
    num_bits = int(np.log2(size))
    for bit in range(num_bits):
        step = 1 << bit
        for start in range(0, size, step << 1):
            coefficients[start + step : start + (step << 1)] ^= coefficients[
                start : start + step
            ]
    return [int(index) for index, value in enumerate(coefficients.tolist()) if value]


def format_anf_term(mask: int, num_qubits: int) -> str:
    if mask == 0:
        return "1"
    return "*".join(f"q{bit}" for bit in range(num_qubits) if mask & (1 << bit))


def block_paths_for_frontier_row(
    row: dict[str, str],
    original_qasm: Path,
) -> list[tuple[str, Path, Path]]:
    stems = [
        item.strip()
        for item in (row.get("decomposition_keys") or original_qasm.stem).split(";")
        if item.strip()
    ]
    if not stems:
        stems = [original_qasm.stem]
    candidate_qasm = project_path(row.get("candidate_qasm_path"))
    combo_dir = candidate_qasm.parent if candidate_qasm is not None else None
    output: list[tuple[str, Path, Path]] = []
    for stem in stems:
        reference = original_qasm.parent / f"{stem}.cnotphase.qasm"
        candidate = None if combo_dir is None else combo_dir / f"{stem}.qasm"
        if candidate is not None:
            output.append((stem, reference, candidate))
    return output


def diagnose_block_pair(
    *,
    reference_qasm: Path,
    candidate_qasm: Path,
    original_qasm: Path | None = None,
    max_columns: int,
    seed: int,
    tolerance: float,
    max_qubits: int,
) -> dict[str, Any]:
    if not reference_qasm.exists():
        return {
            "block_diagnostic_status": "missing-reference-block",
            "block_diagnostic_error": str(reference_qasm),
        }
    if not candidate_qasm.exists():
        return {
            "block_diagnostic_status": "missing-candidate-block",
            "block_diagnostic_error": str(candidate_qasm),
        }
    try:
        diagnostics = diagnose_block_circuits(
            load_qasm_circuit(reference_qasm),
            load_qasm_circuit(candidate_qasm),
            max_columns=max_columns,
            seed=seed,
            tolerance=tolerance,
            max_qubits=max_qubits,
        )
        diagnostics.update(
            mapped_phase_terms(
                diagnostics,
                reference_qasm=reference_qasm,
                original_qasm=original_qasm,
            )
        )
        return diagnostics
    except Exception as exc:
        return {
            "block_diagnostic_status": "diagnostic-error",
            "block_diagnostic_error": str(exc),
        }


def mapped_phase_terms(
    diagnostics: dict[str, Any],
    *,
    reference_qasm: Path,
    original_qasm: Path | None = None,
) -> dict[str, Any]:
    terms = str(diagnostics.get("monomial_phase_delta_terms") or "")
    if not terms:
        return {}
    mapping_path = mapping_path_for_reference_qasm(reference_qasm)
    if mapping_path is None or not mapping_path.exists():
        return {
            "monomial_phase_delta_terms_mapped_status": "missing-mapping",
            "monomial_phase_delta_terms_mapped_error": (
                "" if mapping_path is None else str(mapping_path)
            ),
        }
    try:
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        original_num_qubits = (
            None
            if original_qasm is None or not original_qasm.exists()
            else int(load_qasm_circuit(original_qasm).num_qubits)
        )
        return {
            "monomial_phase_delta_terms_mapped_status": "ok",
            "monomial_phase_delta_terms_mapped_error": None,
            "monomial_phase_delta_terms_mapped": format_mapped_terms(
                terms,
                mapping=mapping,
                original_num_qubits=original_num_qubits,
            ),
        }
    except Exception as exc:
        return {
            "monomial_phase_delta_terms_mapped_status": "failed",
            "monomial_phase_delta_terms_mapped_error": str(exc),
        }


def mapping_path_for_reference_qasm(reference_qasm: Path) -> Path | None:
    name = reference_qasm.name
    if name.endswith(".cnotphase.qasm"):
        return reference_qasm.with_name(name.replace(".cnotphase.qasm", ".mapping.txt"))
    if name.endswith(".qasm"):
        return reference_qasm.with_name(name.replace(".qasm", ".mapping.txt"))
    return None


def format_mapped_terms(
    terms: str,
    *,
    mapping: list[Any],
    original_num_qubits: int | None,
) -> str:
    return ";".join(
        format_mapped_term(
            term,
            mapping=mapping,
            original_num_qubits=original_num_qubits,
        )
        for term in terms.split(";")
        if term
    )


def format_mapped_term(
    term: str,
    *,
    mapping: list[Any],
    original_num_qubits: int | None,
) -> str:
    if term == "1":
        return "1"
    mapped_factors = []
    for factor in term.split("*"):
        if not factor.startswith("q"):
            mapped_factors.append(factor)
            continue
        index = int(factor[1:])
        mapped_value = int(mapping[index]) if index < len(mapping) else index
        if original_num_qubits is not None and mapped_value >= original_num_qubits:
            mapped_factors.append(f"work[{mapped_value}]")
        else:
            mapped_factors.append(f"orig[{mapped_value}]")
    return "*".join(mapped_factors)


def build_diagnostic_rows(
    *,
    frontier_verification_rows: list[dict[str, str]],
    originals: dict[str, Path],
    circuit_ids: set[str] | None,
    only_non_equal: bool,
    max_columns: int,
    seed: int,
    tolerance: float,
    max_qubits: int,
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
        if original_qasm is None or not original_qasm.exists():
            output_rows.append(
                {
                    "circuit_id": circuit_id,
                    "candidate_id": row.get("candidate_id"),
                    "combo_index": row.get("combo_index"),
                    "verification_status": row.get("verification_status"),
                    "block_stem": None,
                    "block_diagnostic_status": "missing-original",
                    "block_diagnostic_error": f"missing original QASM for {circuit_id}",
                }
            )
            continue
        for block_stem, reference_qasm, candidate_block_qasm in block_paths_for_frontier_row(
            row, original_qasm
        ):
            output_rows.append(
                {
                    "circuit_id": circuit_id,
                    "candidate_id": row.get("candidate_id"),
                    "combo_index": row.get("combo_index"),
                    "verification_status": row.get("verification_status"),
                    "structural_cost": row.get("structural_cost"),
                    "tcount_after": row.get("tcount_after"),
                    "block_stem": block_stem,
                    "reference_block_qasm": str(reference_qasm),
                    "candidate_block_qasm": str(candidate_block_qasm),
                    **diagnose_block_pair(
                        reference_qasm=reference_qasm,
                        candidate_qasm=candidate_block_qasm,
                        original_qasm=original_qasm,
                        max_columns=max_columns,
                        seed=seed,
                        tolerance=tolerance,
                        max_qubits=max_qubits,
                    ),
                }
            )
    return output_rows


def write_report(rows: list[dict[str, Any]], report_path: Path, csv_path: Path) -> Path:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("block_diagnostic_status"))
        counts[status] = counts.get(status, 0) + 1
    table_lines = [
        "| circuit | candidate | verifier | block | diagnostic | column residual | global residual | phase delta | ANF degree | ANF terms | mapped terms | action |",
        "|---|---|---|---|---|---:|---:|---|---:|---|---|---|",
    ]
    for row in rows:
        status = row.get("block_diagnostic_status")
        if status == "sampled-global-match":
            action = "block replacement matches sampled reference"
        elif status == "sampled-columnwise-phase-mismatch":
            action = "block maps columns correctly but has input-dependent phase"
        elif status == "sampled-mismatch":
            action = "block replacement does not match sampled reference"
        else:
            action = str(row.get("block_diagnostic_error") or "inspect diagnostic")
        table_lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row.get('circuit_id')}`",
                    f"`{row.get('candidate_id')}`",
                    f"`{row.get('verification_status')}`",
                    f"`{row.get('block_stem')}`",
                    f"`{status}`",
                    _fmt(row.get("max_column_residual")),
                    _fmt(row.get("max_global_residual")),
                    _phase_summary(row),
                    _fmt(row.get("monomial_phase_delta_degree"), digits=0),
                    _terms_summary(row),
                    _mapped_terms_summary(row),
                    action,
                ]
            )
            + " |"
        )
    text = [
        "# Block Replacement Equivalence Diagnostics",
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
            "This sampled diagnostic compares each ressynthesized block directly "
            "against the compiled `*.cnotphase.qasm` block it replaces, before the "
            "initial/final Clifford context and before original-circuit postselection. "
            "It is therefore a sharper assembly diagnostic than comparing the full "
            "candidate against the original QASM."
        ),
        (
            "When the compared blocks are monomial Clifford+T circuits, the diagnostic "
            "also enumerates all computational-basis inputs exactly and reports the "
            "phase-delta spectrum. For sign-only deltas (`0`/`4` modulo eighth-turns), "
            "it reconstructs the Boolean ANF of the input-dependent sign."
        ),
        "",
        "## Block Diagnostics",
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


def _phase_summary(row: dict[str, Any]) -> str:
    status = row.get("monomial_phase_status")
    if status == "ok":
        return str(row.get("monomial_phase_delta_values") or "")
    if status:
        return str(status)
    return _fmt(row.get("alpha_phase_span"))


def _terms_summary(row: dict[str, Any]) -> str:
    terms = str(row.get("monomial_phase_delta_terms") or "")
    if not terms:
        return ""
    if str(row.get("monomial_phase_delta_terms_truncated")) in {"1", "True", "true"}:
        return f"{terms};..."
    return terms


def _mapped_terms_summary(row: dict[str, Any]) -> str:
    terms = str(row.get("monomial_phase_delta_terms_mapped") or "")
    if terms:
        return terms
    status = row.get("monomial_phase_delta_terms_mapped_status")
    return "" if status in (None, "ok") else str(status)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample whether ressynthesized blocks match their cnotphase reference blocks."
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
    parser.add_argument("--max-qubits", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--tolerance", type=float, default=1e-8)
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
        max_qubits=args.max_qubits,
    )
    write_csv_rows(rows, args.output_csv)
    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("block_diagnostic_status"))
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
            "max_qubits": args.max_qubits,
            "seed": args.seed,
            "tolerance": args.tolerance,
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
