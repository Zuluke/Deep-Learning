from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._analysis_common import ensure_dir
from scripts._manifest import append_command

DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "results"
    / "alphaq_split_reward"
    / "mod_5_4_v1_tiebreak_budget2_lam0005_basis_mix_1000_b32_m16_with_cob"
    / "candidate_factors_manifest.csv"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "alphaq_split_reward_external"
    / "mod_5_4_v1_tiebreak_materialized"
)
REMOTE_PROJECT_PREFIXES = (
    "/home/CIN/cacl2/Deep-Learning/project",
    "/home/CIN/cacl2/Deep-Learning",
)


def resolve_project_path(raw_path: str | Path) -> Path:
    text = str(raw_path).strip()
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


def load_manifest_row(
    manifest_csv: Path,
    *,
    target: str,
    candidate_kind: str,
) -> dict[str, str]:
    with manifest_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    matches = [
        row
        for row in rows
        if row.get("target") == target
        and row.get("candidate_kind") == candidate_kind
    ]
    if not matches:
        raise ValueError(
            f"No manifest row for target={target!r} and "
            f"candidate_kind={candidate_kind!r} in {manifest_csv}."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous manifest rows for target={target!r} and "
            f"candidate_kind={candidate_kind!r} in {manifest_csv}."
        )
    return matches[0]


def gf2_inverse(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.uint8) % 2
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected a square GF(2) matrix, got {matrix.shape}.")
    size = matrix.shape[0]
    identity = np.eye(size, dtype=np.uint8)
    augmented = np.concatenate([matrix.copy(), identity], axis=1)
    pivot_row = 0
    for col in range(size):
        pivot_offsets = np.flatnonzero(augmented[pivot_row:, col])
        if len(pivot_offsets) == 0:
            raise ValueError("Change-of-basis matrix is singular over GF(2).")
        pivot = pivot_row + int(pivot_offsets[0])
        if pivot != pivot_row:
            augmented[[pivot_row, pivot]] = augmented[[pivot, pivot_row]]
        for row in range(size):
            if row != pivot_row and augmented[row, col]:
                augmented[row] ^= augmented[pivot_row]
        pivot_row += 1
    return augmented[:, size:]


def rank_one_tensor_sum(factors: np.ndarray) -> np.ndarray:
    factors = np.asarray(factors, dtype=np.uint8) % 2
    if factors.ndim != 2:
        raise ValueError(f"Expected factors with shape (rank, size), got {factors.shape}.")
    size = factors.shape[1]
    tensor = np.zeros((size, size, size), dtype=np.uint8)
    for factor in factors:
        tensor ^= np.einsum("i,j,k->ijk", factor, factor, factor, optimize=True).astype(
            np.uint8
        )
    return tensor.astype(bool)


def canonicalize_factors(
    factors: np.ndarray,
    change_of_basis: np.ndarray | None,
) -> np.ndarray:
    factors = np.asarray(factors, dtype=np.uint8) % 2
    if factors.ndim != 2:
        raise ValueError(f"Expected factors with shape (rank, size), got {factors.shape}.")
    if change_of_basis is None:
        return factors.astype(bool)
    inverse_basis = gf2_inverse(change_of_basis)
    if inverse_basis.shape[0] != factors.shape[1]:
        raise ValueError(
            "Change-of-basis size does not match factor width: "
            f"{inverse_basis.shape[0]} != {factors.shape[1]}."
        )
    return ((inverse_basis @ factors.T) % 2).T.astype(bool)


def materialized_factor_path(
    output_root: Path,
    *,
    target: str,
    mode: str,
    candidate_kind: str,
) -> Path:
    safe_mode = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in mode)
    safe_kind = "".join(
        char if char.isalnum() or char in {"_", "-"} else "_" for char in candidate_kind
    )
    return output_root / f"{target}.{safe_mode}.{safe_kind}.canonical.transposed.npy"


def find_benchmark_dir(target: str) -> Path:
    benchmark_root = PROJECT_ROOT / "external" / "circuit-to-tensor" / "benchmarks"
    matches = sorted(path for path in benchmark_root.glob(f"**/{target}") if path.is_dir())
    if not matches:
        raise FileNotFoundError(f"Could not find benchmark directory for {target}.")
    if len(matches) > 1:
        raise ValueError(f"Ambiguous benchmark directories for {target}: {matches}.")
    return matches[0]


def qasm_metrics(path: Path) -> dict[str, Any]:
    from scripts._analysis_common import compute_metrics_from_qasm_path

    return compute_metrics_from_qasm_path(path)


def structural_metrics(candidate_qasm: Path, original_qasm: Path) -> dict[str, Any]:
    from scripts.alphatensor_structural_cost import circuit_selection_row_from_qasm
    from scripts.alphatensor_structural_cost import compute_selection_metrics

    return compute_selection_metrics(
        circuit_selection_row_from_qasm(candidate_qasm),
        circuit_selection_row_from_qasm(original_qasm),
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Canonicalize, resynthesize, assemble, and audit a split-reward "
            "candidate exported by run_demo_train.py."
        )
    )
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--target", default="mod_5_4")
    parser.add_argument("--candidate-kind", default="best_solved")
    parser.add_argument("--split-reward-mode", default="v1_tiebreak")
    parser.add_argument("--benchmark-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--gadgets", choices=("on", "off"), default="on")
    parser.add_argument(
        "--skip-target-check",
        action="store_true",
        help="Skip exact tensor reconstruction check before resynthesis.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_csv = resolve_project_path(args.manifest_csv)
    output_root = resolve_project_path(args.output_root)
    benchmark_dir = (
        resolve_project_path(args.benchmark_dir)
        if args.benchmark_dir is not None
        else find_benchmark_dir(args.target)
    )
    ensure_dir(output_root)

    row = load_manifest_row(
        manifest_csv,
        target=args.target,
        candidate_kind=args.candidate_kind,
    )
    factors = np.load(resolve_project_path(row["factor_path"]))
    change_of_basis_path = row.get("change_of_basis_path") or ""
    change_of_basis = (
        np.load(resolve_project_path(change_of_basis_path))
        if change_of_basis_path
        else None
    )
    canonical_factors = canonicalize_factors(factors, change_of_basis)

    target_tensor_path = benchmark_dir / f"{args.target}.tensor.npy"
    target_tensor = np.load(target_tensor_path).astype(bool)
    reconstruction_ok = np.array_equal(
        rank_one_tensor_sum(canonical_factors),
        target_tensor,
    )
    if not reconstruction_ok and not args.skip_target_check:
        raise ValueError(
            "Canonicalized factors do not reconstruct the original target tensor. "
            "Use --skip-target-check only for diagnostic materialization."
        )

    materialized_path = materialized_factor_path(
        output_root,
        target=args.target,
        mode=args.split_reward_mode,
        candidate_kind=args.candidate_kind,
    )
    np.save(materialized_path, canonical_factors.T.astype(bool))

    from scripts.assemble_resynth_circuit import assemble_circuit
    from scripts.replay_public_decompositions import run_resynth

    status, error = run_resynth(
        decomposition_path=materialized_path,
        mapping_path=benchmark_dir / f"{args.target}.mapping.txt",
        original_path=benchmark_dir / f"{args.target}.matrix.npy",
        output_dir=output_root,
        use_gadgets=args.gadgets == "on",
    )
    if status != "ok":
        raise RuntimeError(f"Resynthesis failed: {error}")

    emitted_block_qasm = materialized_path.with_suffix(".qasm")
    canonical_block_qasm = output_root / f"{args.target}.qasm"
    if not emitted_block_qasm.exists():
        raise FileNotFoundError(f"Expected resynthesized QASM at {emitted_block_qasm}.")
    if emitted_block_qasm != canonical_block_qasm:
        shutil.copyfile(emitted_block_qasm, canonical_block_qasm)

    assembled_qasm, assembled_summary = assemble_circuit(benchmark_dir, output_root)
    block_qasm = canonical_block_qasm
    original_qasm = benchmark_dir / f"{args.target}.qasm"

    block_metrics = qasm_metrics(block_qasm)
    assembled_metrics = qasm_metrics(assembled_qasm)
    external_metrics = structural_metrics(assembled_qasm, original_qasm)

    write_json(output_root / "assembled_summary.json", assembled_summary)
    write_json(output_root / "structural_metrics_from_qasm_original.json", external_metrics)
    summary = {
        "status": "ok",
        "target": args.target,
        "candidate_kind": args.candidate_kind,
        "split_reward_mode": args.split_reward_mode,
        "manifest_csv": str(manifest_csv),
        "source_factor_path": str(resolve_project_path(row["factor_path"])),
        "source_change_of_basis_path": (
            str(resolve_project_path(change_of_basis_path)) if change_of_basis_path else None
        ),
        "source_is_canonical_basis": row.get("is_canonical_basis"),
        "reconstruction_ok": bool(reconstruction_ok),
        "materialized_factor_path": str(materialized_path),
        "materialized_factor_shape": list(canonical_factors.T.shape),
        "benchmark_dir": str(benchmark_dir),
        "original_qasm": str(original_qasm),
        "block_qasm": str(block_qasm),
        "assembled_qasm": str(assembled_qasm),
        "block_metrics": block_metrics,
        "assembled_metrics": assembled_metrics,
        "external_structural_metrics": external_metrics,
    }
    write_json(output_root / "summary.json", summary)
    append_command(
        {
            "tool": "materialize_split_reward_candidate.py",
            "command": " ".join(sys.argv),
            "cwd": str(PROJECT_ROOT),
            "manifest_csv": str(manifest_csv),
            "target": args.target,
            "candidate_kind": args.candidate_kind,
            "output_root": str(output_root),
            "assembled_qasm": str(assembled_qasm),
            "exit_code": 0,
        }
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
