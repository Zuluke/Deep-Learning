from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._analysis_common import DEFAULT_CSV_ROOT
from scripts._analysis_common import ensure_dir
from scripts._analysis_common import list_vendored_benchmark_records
from scripts._analysis_common import parse_n_qubits_and_gate_count
from scripts._analysis_common import to_relative
from scripts._analysis_common import write_csv_rows
from scripts._analysis_common import write_json
from scripts._manifest import append_command


def _timestamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def prepare_vendored_local(output_csv: Path) -> dict[str, Any]:
    records = list_vendored_benchmark_records()
    rows = [record.as_dict() for record in records]
    write_csv_rows(rows, output_csv)
    return {
        "source": "vendored-local",
        "num_circuits": len(rows),
        "output_csv": str(output_csv),
    }


def _try_qasm_v2_export(qnode: Any, output_path: Path) -> bool:
    if hasattr(qnode, "openqasm"):
        qasm_text = qnode.openqasm
        if callable(qasm_text):
            qasm_text = qasm_text()
        if isinstance(qasm_text, str) and "OPENQASM 2.0" in qasm_text:
            output_path.write_text(qasm_text, encoding="utf-8")
            return True
    if hasattr(qnode, "to_openqasm"):
        qasm_text = qnode.to_openqasm()
        if isinstance(qasm_text, str) and "OPENQASM 2.0" in qasm_text:
            output_path.write_text(qasm_text, encoding="utf-8")
            return True
    return False


def prepare_op_t_mize(output_csv: Path, data_dir: Path) -> dict[str, Any]:
    import pennylane as qml

    ensure_dir(data_dir)
    datasets = qml.data.load(
        "op-t-mize",
        folder_path=data_dir,
        progress_bar=False,
    )
    rows: list[dict[str, Any]] = []
    output_qasm_root = ensure_dir(data_dir / "qasm_v2")

    for dataset in datasets:
        circuit_id = getattr(dataset, "name", None) or getattr(dataset, "id", None)
        if not circuit_id:
            circuit_id = getattr(dataset, "attrs", {}).get("id")
        if not circuit_id:
            circuit_id = f"op_t_mize_{len(rows):03d}"

        qasm_path = output_qasm_root / f"{circuit_id}.qasm"
        export_ok = False
        export_error = None
        for attr_name in ("circuit", "qnode", "operator", "tape"):
            if not hasattr(dataset, attr_name):
                continue
            try:
                candidate = getattr(dataset, attr_name)
                export_ok = _try_qasm_v2_export(candidate, qasm_path)
                if export_ok:
                    break
            except Exception as exc:  # pragma: no cover - depends on dataset API
                export_error = str(exc)

        if not export_ok:
            rows.append(
                {
                    "circuit_id": circuit_id,
                    "source": "op-t-mize",
                    "family": "op-t-mize",
                    "n_qubits": None,
                    "raw_gate_count": None,
                    "qasm_path": None,
                    "benchmark_dir": None,
                    "vendored_compile_dir": None,
                    "prepare_status": "not-normalized",
                    "prepare_error": export_error or "QASM v2 export not supported by dataset object.",
                }
            )
            continue

        n_qubits, raw_gate_count = parse_n_qubits_and_gate_count(qasm_path)
        rows.append(
            {
                "circuit_id": circuit_id,
                "source": "op-t-mize",
                "family": "op-t-mize",
                "n_qubits": n_qubits,
                "raw_gate_count": raw_gate_count,
                "qasm_path": str(qasm_path),
                "benchmark_dir": None,
                "vendored_compile_dir": None,
                "prepare_status": "ok",
                "prepare_error": None,
            }
        )

    write_csv_rows(rows, output_csv)
    return {
        "source": "op-t-mize",
        "num_datasets": len(rows),
        "output_csv": str(output_csv),
        "data_dir": str(data_dir),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare benchmark inventories for vendored-local data or the op-T-mize dataset."
    )
    parser.add_argument(
        "--source",
        choices=("vendored-local", "op-t-mize"),
        required=True,
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Destination CSV. Defaults to results/csv/<source>_catalog.csv.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "op_t_mize",
        help="Only used for op-T-mize downloads.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_csv = args.output_csv or (
        DEFAULT_CSV_ROOT / f"{args.source.replace('-', '_')}_catalog.csv"
    )
    if args.source == "vendored-local":
        summary = prepare_vendored_local(output_csv)
    else:
        summary = prepare_op_t_mize(output_csv, args.data_dir)

    summary["generated_at_utc"] = _timestamp()
    summary_json = output_csv.with_suffix(".json")
    write_json(summary, summary_json)
    append_command(
        {
            "tool": "download_or_prepare_benchmarks.py",
            "command": (
                f"{sys.executable} scripts/download_or_prepare_benchmarks.py "
                f"--source {args.source} --output-csv {output_csv}"
            ),
            "cwd": str(PROJECT_ROOT),
            "source": args.source,
            "output_csv": str(output_csv),
            "summary_json": str(summary_json),
            "exit_code": 0,
        }
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
