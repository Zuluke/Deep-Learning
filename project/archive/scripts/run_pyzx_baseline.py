from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._analysis_common import DEFAULT_PYZX_ROOT
from scripts._analysis_common import dump_qasm_v2
from scripts._analysis_common import ensure_dir
from scripts._analysis_common import load_qasm_circuit
from scripts._analysis_common import to_relative
from scripts._analysis_common import write_csv_rows
from scripts._analysis_common import write_json
from scripts._manifest import append_command


def _normalized_qasm_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    include_line = 'include "qelib1.inc";'
    if include_line in lines:
        lines = [line for line in lines if line != include_line]
        if lines and lines[0].strip().startswith("OPENQASM 2.0;"):
            lines.insert(1, include_line)
        else:
            lines.insert(0, include_line)
    return "\n".join(lines) + "\n"


def run_single(input_qasm: Path, output_dir: Path) -> dict[str, object]:
    import pyzx as zx

    ensure_dir(output_dir)
    read_start = time.time()
    try:
        circuit = zx.Circuit.load(str(input_qasm))
    except TypeError:
        circuit = zx.Circuit.from_qasm(_normalized_qasm_text(input_qasm))
    read_runtime = time.time() - read_start
    tcount_before = int(circuit.tcount())

    simplify_start = time.time()
    graph = circuit.to_graph()
    zx.simplify.full_reduce(graph)
    simplify_runtime = time.time() - simplify_start

    extract_start = time.time()
    extracted = zx.extract_circuit(graph.copy())
    basic = extracted.to_basic_gates()
    extract_runtime = time.time() - extract_start

    qasm_path = output_dir / "optimized.qasm"
    qasm_path.write_text(basic.to_qasm(), encoding="utf-8")

    # Round-trip through qiskit to guarantee QASM v2 serialization compatibility.
    qiskit_circuit = load_qasm_circuit(qasm_path)
    dump_qasm_v2(qiskit_circuit, qasm_path)

    summary = {
        "status": "ok",
        "runtime_sec": read_runtime + simplify_runtime + extract_runtime,
        "read_runtime_sec": read_runtime,
        "simplify_runtime_sec": simplify_runtime,
        "extract_runtime_sec": extract_runtime,
        "tcount_before_pyzx": tcount_before,
        "tcount_after_pyzx": int(basic.tcount()),
        "output_qasm_path": str(qasm_path),
    }
    write_json(summary, output_dir / "stats.json")
    return summary


def _run_single_subprocess(input_qasm: Path, output_dir: Path, timeout_sec: int) -> dict[str, object]:
    cmd = [
        sys.executable,
        __file__,
        "--single-input",
        str(input_qasm),
        "--output-dir",
        str(output_dir),
    ]
    start_time = time.time()
    try:
        completed = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        runtime_sec = time.time() - start_time
        if completed.returncode != 0:
            return {
                "status": "failed",
                "runtime_sec": runtime_sec,
                "output_dir": str(output_dir),
                "error": completed.stderr[-4000:] or completed.stdout[-4000:],
            }
        summary_path = output_dir / "stats.json"
        payload = json.loads(summary_path.read_text())
        payload["runtime_sec"] = runtime_sec
        return payload
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "timeout",
            "runtime_sec": time.time() - start_time,
            "output_dir": str(output_dir),
            "error": (exc.stderr or exc.stdout or "")[-4000:],
        }


def load_inventory_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a PyZX baseline over an inventory of circuits.")
    parser.add_argument("--inventory-csv", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_PYZX_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--timeout-sec", type=int, default=1200)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--single-input", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.single_input is not None:
        output_dir = args.output_dir or (
            args.output_root if args.output_root != DEFAULT_PYZX_ROOT else args.single_input.parent
        )
        summary = run_single(args.single_input, output_dir)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if args.inventory_csv is None:
        raise SystemExit("--inventory-csv is required unless --single-input is used.")

    ensure_dir(args.output_root)
    rows = load_inventory_rows(args.inventory_csv)
    if args.limit is not None:
        rows = rows[: args.limit]

    summary_rows: list[dict[str, object]] = []
    for row in rows:
        qasm_path_value = row.get("qasm_path")
        if not qasm_path_value:
            summary_rows.append(
                {
                    "circuit_id": row["circuit_id"],
                    "status": "not-run",
                    "runtime_sec": None,
                    "output_qasm_path": None,
                    "error": "Missing qasm_path in inventory row.",
                }
            )
            continue
        qasm_path = Path(qasm_path_value)
        if not qasm_path.is_absolute():
            qasm_path = PROJECT_ROOT / qasm_path_value
        output_dir = args.output_root / row["circuit_id"]
        if output_dir.exists() and (output_dir / "stats.json").exists() and not args.force:
            payload = json.loads((output_dir / "stats.json").read_text())
            payload["circuit_id"] = row["circuit_id"]
            summary_rows.append(payload)
            continue

        ensure_dir(output_dir)
        payload = _run_single_subprocess(qasm_path, output_dir, args.timeout_sec)
        payload["circuit_id"] = row["circuit_id"]
        if payload.get("output_qasm_path"):
            payload["output_qasm_path"] = to_relative(Path(payload["output_qasm_path"]))
        summary_rows.append(payload)

    summary_csv = args.output_root / "pyzx_summary.csv"
    summary_json = args.output_root / "pyzx_summary.json"
    write_csv_rows(summary_rows, summary_csv)
    write_json(
        {
            "inventory_csv": str(args.inventory_csv),
            "output_root": str(args.output_root),
            "timeout_sec": args.timeout_sec,
            "num_rows": len(summary_rows),
        },
        summary_json,
    )
    append_command(
        {
            "tool": "run_pyzx_baseline.py",
            "command": (
                f"{sys.executable} scripts/run_pyzx_baseline.py --inventory-csv {args.inventory_csv} "
                f"--output-root {args.output_root} --timeout-sec {args.timeout_sec}"
            ),
            "cwd": str(PROJECT_ROOT),
            "inventory_csv": str(args.inventory_csv),
            "summary_csv": str(summary_csv),
            "summary_json": str(summary_json),
            "exit_code": 0,
        }
    )
    print(json.dumps({"summary_csv": str(summary_csv), "summary_json": str(summary_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
