"""Numerically resolve inconclusive feynver proofs by exact isometry checking.

feynver's path-sum reduction can stall on resynthesized candidates that carry
postselected gadget ancillas, returning ``Inconclusive`` even though the
candidate is correct. This checker resolves those pairs numerically and
exactly up to floating point:

1. Build the original unitary ``U`` on ``n`` qubits.
2. For every computational-basis input ``|x>`` of the primary register,
   evolve the candidate on ``|x> (x) |0...0>`` and project the ancillas onto
   ``<0...0|`` (the postselection convention used by
   ``feynver -postselect-ancillas``). Column-stacking gives the candidate
   block ``W``.
3. The pair is equivalent iff ``W = c * U`` for a single nonzero constant
   ``c`` (global phase and postselection normalization). We report
   ``equal-numeric`` when ``max|W - c U| < 1e-8``.

Because the full input basis is enumerated, this is an exact functional
characterization of the candidate isometry, not statistical sampling. All
pairs in the current batteries have candidates of at most 14 qubits, so dense
statevector evolution is feasible.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._analysis_common import write_csv_rows

ATOL = 1e-8
MAX_CANDIDATE_QUBITS = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Numerically verify candidate/original QASM pairs whose feynver "
            "proof was inconclusive."
        )
    )
    parser.add_argument(
        "--verification-roots",
        required=True,
        help=(
            "Comma-separated verification output roots containing "
            "verification_summary.csv and normalized_qasm/ trees."
        ),
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    return parser.parse_args()


def load_unitary(path: Path) -> np.ndarray:
    from qiskit import qasm2
    from qiskit.quantum_info import Operator

    circuit = qasm2.load(
        str(path),
        custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
    )
    return Operator(circuit).data


def candidate_block(path: Path, primary_qubits: int) -> np.ndarray:
    """Return W[y, x] = <y, 0_a| V |x, 0_a> by full basis enumeration."""
    from qiskit import qasm2
    from qiskit.quantum_info import Statevector

    circuit = qasm2.load(
        str(path),
        custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
    )
    total_qubits = circuit.num_qubits
    if total_qubits > MAX_CANDIDATE_QUBITS:
        raise ValueError(f"Candidate too large for dense check: {total_qubits} qubits")
    dim = 2**primary_qubits
    block = np.zeros((dim, dim), dtype=complex)
    for x in range(dim):
        # Primary register occupies the low qubit indices in the resynthesized
        # candidates; ancillas are the appended high-index qubits, prepared in
        # and postselected on |0...0>.
        state = Statevector.from_int(x, dims=2**total_qubits)
        evolved = state.evolve(circuit).data
        block[:, x] = evolved[:dim]
    return block


def proportionality(block: np.ndarray, unitary: np.ndarray) -> tuple[float, complex]:
    constant = np.trace(unitary.conj().T @ block) / unitary.shape[0]
    residual = float(np.max(np.abs(block - constant * unitary)))
    return residual, constant


def anf_degree(values: np.ndarray) -> int:
    """Degree of the GF(2) polynomial with the given truth table (0/1)."""
    coefficients = values.astype(np.uint8).copy()
    n = int(round(np.log2(len(coefficients))))
    for q in range(n):
        step = 1 << q
        for base in range(0, len(coefficients), step << 1):
            for offset in range(step):
                coefficients[base + step + offset] ^= coefficients[base + offset]
    degree = 0
    for index, coefficient in enumerate(coefficients):
        if coefficient:
            degree = max(degree, bin(index).count("1"))
    return degree


def defect_digest(offset: int, basis_images: list[int], sign_bits: np.ndarray) -> str:
    """Short stable digest identifying a signed-permutation defect."""
    import hashlib

    payload = f"{offset}|{basis_images}|{sign_bits.tobytes().hex()}"
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def structured_correction(
    overlap: np.ndarray,
) -> dict[str, Any] | None:
    """Characterize ``overlap = c * K`` with K a signed basis permutation.

    K|x> = (-1)^{f(x)} |sigma(x)> with sigma affine-linear over GF(2) (an X
    frame plus a CNOT parity map). The GF(2) degree of f decides whether K is
    Clifford (degree <= 2) or needs non-Clifford corrections (degree >= 3).
    Returns None when the overlap does not have this structure.
    """
    dim = overlap.shape[0]
    n = int(round(np.log2(dim)))
    sigma = np.argmax(np.abs(overlap), axis=0)
    if len(set(sigma.tolist())) != dim:
        return None
    offset = int(sigma[0])
    basis_images = [int(sigma[1 << q]) ^ offset for q in range(n)]
    expected = np.zeros(dim, dtype=np.int64)
    for x in range(dim):
        image = offset
        for q in range(n):
            if (x >> q) & 1:
                image ^= basis_images[q]
        expected[x] = image
    if not np.array_equal(expected, sigma):
        return None
    matched = overlap[sigma, np.arange(dim)]
    constant = matched[0]
    if abs(constant) < 1e-9:
        return None
    signs = matched / constant
    if np.max(np.abs(np.abs(signs) - 1.0)) > 1e-6:
        return None
    sign_bits = (np.real(signs) < 0).astype(np.uint8)
    if np.max(np.abs(signs - (1.0 - 2.0 * sign_bits))) > 1e-6:
        return None
    kernel = np.zeros((dim, dim), dtype=complex)
    kernel[sigma, np.arange(dim)] = 1.0 - 2.0 * sign_bits.astype(float)
    return {
        "constant": constant,
        "kernel": kernel,
        "x_frame_weight": bin(offset).count("1"),
        "parity_changes": sum(1 for q in range(n) if basis_images[q] != (1 << q)),
        "phase_degree": anf_degree(sign_bits),
        "defect_signature": defect_digest(offset, basis_images, sign_bits),
    }


def correction_analysis(block: np.ndarray, unitary: np.ndarray) -> dict[str, Any] | None:
    """Try input-side then output-side signed-permutation corrections."""
    for side, overlap in (
        ("input", unitary.conj().T @ block),
        ("output", block @ unitary.conj().T),
    ):
        structured = structured_correction(overlap)
        if structured is None:
            continue
        kernel = structured["kernel"]
        constant = structured["constant"]
        if side == "input":
            reconstructed = constant * (unitary @ kernel)
        else:
            reconstructed = constant * (kernel @ unitary)
        residual = float(np.max(np.abs(block - reconstructed)))
        if residual < ATOL:
            return {**structured, "side": side, "residual": residual}
    return None


def check_pair(original: Path, candidate: Path) -> dict[str, Any]:
    start = time.time()
    unitary = load_unitary(original)
    primary_qubits = int(round(np.log2(unitary.shape[0])))
    block = candidate_block(candidate, primary_qubits)
    gram = block.conj().T @ block
    isometry_alpha = float(abs(gram[0, 0]))
    is_isometry = bool(np.allclose(gram, gram[0, 0] * np.eye(gram.shape[0]), atol=1e-8))
    residual, constant = proportionality(block, unitary)
    correction_fields: dict[str, Any] = {}
    if residual < ATOL and abs(constant) > 1e-9:
        status = "equal-numeric"
        error = None
    else:
        correction = correction_analysis(block, unitary)
        if correction is not None:
            residual = correction["residual"]
            constant = correction["constant"]
            if correction["phase_degree"] <= 2:
                status = "equal-up-to-clifford"
                error = None
            else:
                status = "nonclifford-correction"
                error = (
                    "Candidate equals the original composed with a signed basis "
                    f"permutation whose phase polynomial has degree "
                    f"{correction['phase_degree']} (non-Clifford)."
                )
            correction_fields = {
                "correction_side": correction["side"],
                "correction_x_frame_weight": correction["x_frame_weight"],
                "correction_parity_changes": correction["parity_changes"],
                "correction_phase_degree": correction["phase_degree"],
                "defect_signature": correction["defect_signature"],
            }
        else:
            status = "not-equal"
            error = f"Residual {residual:.3g} above tolerance with c={constant:.6g}."
    runtime = time.time() - start
    return {
        "verification_status": status,
        "verification_error": error,
        "postselection_constant_abs": f"{abs(constant):.6g}",
        "candidate_is_scaled_isometry": is_isometry,
        "isometry_alpha": f"{isometry_alpha:.6g}",
        "residual": f"{residual:.3g}",
        "runtime_sec": f"{runtime:.2f}",
        "correction_side": correction_fields.get("correction_side", ""),
        "correction_x_frame_weight": correction_fields.get("correction_x_frame_weight", ""),
        "correction_parity_changes": correction_fields.get("correction_parity_changes", ""),
        "correction_phase_degree": correction_fields.get("correction_phase_degree", ""),
        "defect_signature": correction_fields.get("defect_signature", ""),
    }


def inconclusive_rows(root: Path) -> list[dict[str, str]]:
    summary = root / "verification_summary.csv"
    with summary.open(encoding="utf-8", newline="") as handle:
        return [
            row
            for row in csv.DictReader(handle)
            if row.get("verification_status") == "inconclusive"
        ]


def normalized_pair(root: Path, row: dict[str, str]) -> tuple[Path, Path]:
    objective = row.get("objective_variant", "") or "default"
    base = root / "normalized_qasm" / row["target"] / objective / row["materializer"]
    return base / "original.normalized.qasm", base / "candidate.normalized.qasm"


def main() -> int:
    args = parse_args()
    roots = [Path(item.strip()) for item in args.verification_roots.split(",") if item.strip()]
    results = []
    for root in roots:
        for row in inconclusive_rows(root):
            original, candidate = normalized_pair(root, row)
            if not original.exists() or not candidate.exists():
                outcome: dict[str, Any] = {
                    "verification_status": "missing-normalized-qasm",
                    "verification_error": f"Missing {original} or {candidate}",
                }
            else:
                outcome = check_pair(original, candidate)
            print(
                f"{row['target']} {row.get('objective_variant', '')} "
                f"{row['materializer']}: {outcome['verification_status']}",
                flush=True,
            )
            results.append(
                {
                    "verification_root": str(root),
                    "target": row["target"],
                    "objective_variant": row.get("objective_variant", ""),
                    "materializer": row["materializer"],
                    "tcount": row.get("tcount", ""),
                    "qasm_depth": row.get("qasm_depth", ""),
                    **outcome,
                }
            )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv_rows(results, args.output_csv)
    counts: dict[str, int] = {}
    for row in results:
        counts[row["verification_status"]] = counts.get(row["verification_status"], 0) + 1
    lines = [
        "# Numeric Resolution Of Inconclusive feynver Proofs",
        "",
        f"CSV: `{args.output_csv}`.",
        "",
        "Each pair is checked by exact computational-basis enumeration of the "
        "candidate isometry with gadget ancillas prepared in and postselected "
        "on |0...0>, compared against the original unitary up to one global "
        "constant. `equal-numeric` means the full functional behaviour matches "
        f"with residual below {ATOL}. `equal-up-to-clifford` means the "
        "candidate equals the original composed with an explicitly extracted "
        "Clifford relabeling (X frame + CNOT parity map + degree-<=2 phase "
        "polynomial); such corrections leave T-count unchanged.",
        "",
        *[f"- {status}: {count}" for status, count in sorted(counts.items())],
        "",
        "| target | objective | materializer | status | |c| | residual | side | X frame | parity changes | phase degree | defect signature | runtime (s) |",
        "|---|---|---|---|---:|---:|---|---:|---:|---:|---|---:|",
    ]
    for row in results:
        lines.append(
            "| {target} | {objective} | {materializer} | {status} | {c} | {res} | {side} | {xf} | {pc} | {pd} | {sig} | {rt} |".format(
                target=row["target"],
                objective=row["objective_variant"],
                materializer=row["materializer"],
                status=row["verification_status"],
                c=row.get("postselection_constant_abs", ""),
                res=row.get("residual", ""),
                side=row.get("correction_side", ""),
                xf=row.get("correction_x_frame_weight", ""),
                pc=row.get("correction_parity_changes", ""),
                pd=row.get("correction_phase_degree", ""),
                sig=row.get("defect_signature", ""),
                rt=row.get("runtime_sec", ""),
            )
        )
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
