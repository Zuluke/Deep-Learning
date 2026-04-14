from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._analysis_common import assembled_piece_paths
from scripts._analysis_common import dump_qasm_v2
from scripts._analysis_common import ensure_dir
from scripts._analysis_common import list_nonclifford_block_stems
from scripts._analysis_common import load_qasm_circuit
from scripts._manifest import append_command


def assemble_circuit(compile_dir: Path, resynth_root: Path) -> tuple[Path, dict[str, object]]:
    block_stems = list_nonclifford_block_stems(compile_dir)
    if not block_stems:
        raise ValueError(f"No non-Clifford blocks found in {compile_dir}")

    piece_paths = []
    for path in assembled_piece_paths(compile_dir, block_stems):
        if path.name.endswith(".initial.cliffords.qasm") or path.name.endswith(".cliffords.qasm"):
            piece_paths.append(path)
        else:
            replacement = resynth_root / path.name
            if not replacement.exists():
                raise FileNotFoundError(f"Missing ressynthesized block {replacement}")
            piece_paths.append(replacement)

    circuits = [load_qasm_circuit(path) for path in piece_paths]
    assembled = circuits[0].copy()
    for piece in circuits[1:]:
        if piece.num_qubits != assembled.num_qubits:
            raise ValueError(
                f"Qubit mismatch during assembly: {piece.num_qubits} != {assembled.num_qubits}"
            )
        assembled = assembled.compose(piece)

    output_path = resynth_root / "assembled.qasm"
    dump_qasm_v2(assembled, output_path)
    summary = {
        "compile_dir": str(compile_dir),
        "resynth_root": str(resynth_root),
        "assembled_qasm_path": str(output_path),
        "num_qubits": assembled.num_qubits,
        "depth": assembled.depth(),
        "size": assembled.size(),
        "piece_paths": [str(path) for path in piece_paths],
    }
    return output_path, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble a complete circuit from ressynthesized non-Clifford blocks.")
    parser.add_argument("--compile-dir", type=Path, required=True)
    parser.add_argument("--resynth-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_dir(args.resynth_root)
    output_path, summary = assemble_circuit(args.compile_dir, args.resynth_root)
    summary_path = args.resynth_root / "assembled_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    append_command(
        {
            "tool": "assemble_resynth_circuit.py",
            "command": (
                f"{sys.executable} scripts/assemble_resynth_circuit.py "
                f"--compile-dir {args.compile_dir} --resynth-root {args.resynth_root}"
            ),
            "cwd": str(PROJECT_ROOT),
            "compile_dir": str(args.compile_dir),
            "resynth_root": str(args.resynth_root),
            "assembled_qasm_path": str(output_path),
            "summary_path": str(summary_path),
            "exit_code": 0,
        }
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
