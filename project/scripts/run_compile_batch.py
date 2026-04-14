from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._analysis_common import DEFAULT_COMPILE_ROOT
from scripts._analysis_common import ensure_dir
from scripts._analysis_common import tensor_size_from_directory
from scripts._analysis_common import write_csv_rows
from scripts._analysis_common import write_json
from scripts._manifest import append_command


def load_inventory_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def compiled_binary() -> Path:
    return PROJECT_ROOT / "external" / "circuit-to-tensor" / "target" / "release" / "circuit-to-tensor"


def compile_one(
    *,
    input_qasm: Path,
    output_dir: Path,
    zx_preopt: bool,
    timeout_sec: int,
) -> dict[str, object]:
    cmd = [str(compiled_binary()), "compile", "-e", "tensor,matrix,circuit-qasm,log"]
    if zx_preopt:
        cmd.append("-z")
    cmd.extend([str(output_dir), str(input_qasm)])

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
        status = "ok" if completed.returncode == 0 else "failed"
        return {
            "status": status,
            "runtime_sec": runtime_sec,
            "exit_code": completed.returncode,
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-4000:],
            "tensor_size": tensor_size_from_directory(output_dir) if completed.returncode == 0 else None,
        }
    except subprocess.TimeoutExpired as exc:
        runtime_sec = time.time() - start_time
        return {
            "status": "timeout",
            "runtime_sec": runtime_sec,
            "exit_code": None,
            "stdout": (exc.stdout or "")[-4000:],
            "stderr": (exc.stderr or "")[-4000:],
            "tensor_size": None,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run circuit-to-tensor compile in batch.")
    parser.add_argument("--inventory-csv", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_COMPILE_ROOT,
    )
    parser.add_argument("--zx-preopt", choices=("on", "off"), required=True)
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_dir(args.output_root)
    rows = load_inventory_rows(args.inventory_csv)
    if args.limit is not None:
        rows = rows[: args.limit]

    summary_rows: list[dict[str, object]] = []
    zx_dir = args.output_root / ("quizx" if args.zx_preopt == "on" else "no_quizx")
    ensure_dir(zx_dir)

    for row in rows:
        qasm_path_value = row.get("qasm_path")
        if not qasm_path_value:
            summary_rows.append(
                {
                    "circuit_id": row["circuit_id"],
                    "zx_preopt": args.zx_preopt,
                    "status": "not-run",
                    "runtime_sec": None,
                    "exit_code": None,
                    "output_dir": None,
                    "tensor_size": None,
                    "error": "Missing qasm_path in inventory row.",
                }
            )
            continue

        qasm_path = Path(qasm_path_value)
        if not qasm_path.is_absolute():
            qasm_path = PROJECT_ROOT / qasm_path_value
        output_dir = zx_dir / row["circuit_id"]

        if output_dir.exists() and not args.force and any(output_dir.iterdir()):
            summary_rows.append(
                {
                    "circuit_id": row["circuit_id"],
                    "zx_preopt": args.zx_preopt,
                    "status": "skipped-existing",
                    "runtime_sec": None,
                    "exit_code": None,
                    "output_dir": str(output_dir),
                    "tensor_size": tensor_size_from_directory(output_dir),
                    "error": None,
                }
            )
            continue

        ensure_dir(output_dir)
        result = compile_one(
            input_qasm=qasm_path,
            output_dir=output_dir,
            zx_preopt=args.zx_preopt == "on",
            timeout_sec=args.timeout_sec,
        )
        summary_rows.append(
            {
                "circuit_id": row["circuit_id"],
                "zx_preopt": args.zx_preopt,
                "status": result["status"],
                "runtime_sec": result["runtime_sec"],
                "exit_code": result["exit_code"],
                "output_dir": str(output_dir),
                "tensor_size": result["tensor_size"],
                "error": result["stderr"] or None,
            }
        )

    summary_csv = args.output_root / f"compile_{args.zx_preopt}_summary.csv"
    summary_json = args.output_root / f"compile_{args.zx_preopt}_summary.json"
    write_csv_rows(summary_rows, summary_csv)
    write_json(
        {
            "inventory_csv": str(args.inventory_csv),
            "output_root": str(args.output_root),
            "zx_preopt": args.zx_preopt,
            "timeout_sec": args.timeout_sec,
            "num_rows": len(summary_rows),
        },
        summary_json,
    )
    append_command(
        {
            "tool": "run_compile_batch.py",
            "command": (
                f"{sys.executable} scripts/run_compile_batch.py --inventory-csv {args.inventory_csv} "
                f"--output-root {args.output_root} --zx-preopt {args.zx_preopt} "
                f"--timeout-sec {args.timeout_sec}"
            ),
            "cwd": str(PROJECT_ROOT),
            "inventory_csv": str(args.inventory_csv),
            "output_root": str(args.output_root),
            "summary_csv": str(summary_csv),
            "summary_json": str(summary_json),
            "exit_code": 0,
        }
    )
    print(json.dumps({"summary_csv": str(summary_csv), "summary_json": str(summary_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
