from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._analysis_common import DEFAULT_RESULTS_ROOT
from scripts._analysis_common import ensure_dir
from scripts._analysis_common import write_csv_rows
from scripts._analysis_common import write_json
from scripts.run_formal_verification import circuit_to_tensor_binary
from scripts.run_formal_verification import project_path
from scripts.run_formal_verification import relative_or_absolute
from scripts.run_formal_verification import run_verification_pair
from scripts.structural_target import coerce_float


DEFAULT_BEAM_CSV = PROJECT_ROOT / "results" / "csv" / "alphaq_beam_materializer_ablation.csv"
DEFAULT_OUTPUT_ROOT = DEFAULT_RESULTS_ROOT / "verification" / "alphaq_beam_materializer_ablation"
DEFAULT_SUMMARY_CSV = DEFAULT_OUTPUT_ROOT / "verification_summary.csv"
DEFAULT_SUMMARY_JSON = DEFAULT_OUTPUT_ROOT / "verification_summary.json"
DEFAULT_REPORT_PATH = DEFAULT_OUTPUT_ROOT / "verification_summary.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Formally verify best beam shared-parity materializer candidates."
    )
    parser.add_argument("--beam-csv", type=Path, default=DEFAULT_BEAM_CSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--materializer-prefix", default="beam-shared-parity")
    parser.add_argument("--objective-variant", default=None)
    parser.add_argument("--timeout-sec", type=int, default=60)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def best_beam_rows(rows: list[dict[str, str]], materializer_prefix: str = "beam-shared-parity") -> list[dict[str, str]]:
    best = []
    for target in sorted({row["target"] for row in rows}):
        candidates = [
            row
            for row in rows
            if row["target"] == target and row.get("materializer", "").startswith(materializer_prefix)
        ]
        if not candidates:
            continue
        best.append(
            min(
                candidates,
                key=lambda row: (
                    inf_if_none(row.get("qasm_depth_ratio")),
                    inf_if_none(row.get("primary_nc_depth_ratio")),
                    inf_if_none(row.get("num_total_cnots")),
                    row.get("materializer", ""),
                ),
            )
        )
    return best


def inf_if_none(value: Any) -> float:
    numeric = coerce_float(value)
    return float("inf") if numeric is None else numeric


def verify_rows(
    *,
    rows: list[dict[str, str]],
    output_root: Path,
    materializer_prefix: str,
    timeout_sec: int,
) -> list[dict[str, Any]]:
    proof_root = ensure_dir(output_root / "proofs")
    normalized_root = ensure_dir(output_root / "normalized_qasm")
    results = []
    for index, row in enumerate(best_beam_rows(rows, materializer_prefix), start=1):
        summary_path = project_path(row.get("summary_path"))
        if summary_path is None or not summary_path.exists():
            result = {
                "verification_status": "missing-summary",
                "verification_error": f"Missing summary: {row.get('summary_path')}",
                "proof_path": None,
                "runtime_sec": None,
                "original_normalization_status": None,
                "candidate_normalization_status": None,
            }
        else:
            summary = read_json(summary_path)
            target = str(row["target"])
            benchmark_dir = Path(str(summary["benchmark_dir"]))
            original_qasm = benchmark_dir / f"{target}.qasm"
            candidate_qasm = Path(str(summary["assembled_qasm"]))
            pair_dir = ensure_dir(proof_root / target)
            normalized_dir = ensure_dir(normalized_root / target / row["materializer"])
            result = run_verification_pair(
                original_qasm=original_qasm,
                candidate_qasm=candidate_qasm,
                proof_path=pair_dir / f"{row['materializer']}.verify.txt",
                normalized_original=normalized_dir / "original.normalized.qasm",
                normalized_candidate=normalized_dir / "candidate.normalized.qasm",
                timeout_sec=timeout_sec,
            )
        print(
            f"[{index}] {row['target']} {row['materializer']}: {result['verification_status']}",
            flush=True,
        )
        results.append(
            {
                "target": row["target"],
                "materializer": row["materializer"],
                "beam_width": row.get("beam_width"),
                "candidate_kind": row.get("candidate_kind", ""),
                "summary_path": row.get("summary_path"),
                "candidate_dir": row.get("candidate_dir"),
                "tcount": row.get("tcount"),
                "tdepth": row.get("tdepth"),
                "qasm_depth": row.get("qasm_depth"),
                "num_total_cnots": row.get("num_total_cnots"),
                "primary_nc_depth_ratio": row.get("primary_nc_depth_ratio"),
                "qasm_depth_ratio": row.get("qasm_depth_ratio"),
                **result,
            }
        )
    return results


def filter_rows(rows: list[dict[str, str]], objective_variant: str | None) -> list[dict[str, str]]:
    if objective_variant is None:
        return rows
    return [row for row in rows if row.get("objective_variant") == objective_variant]


def write_report(path: Path, rows: list[dict[str, Any]], summary_csv: Path) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("verification_status"))
        counts[status] = counts.get(status, 0) + 1
    lines = [
        "# Beam Materializer Formal Verification",
        "",
        f"CSV: `{summary_csv}`.",
        f"Total best-beam candidates checked: {len(rows)}.",
        *[f"- {status}: {count}" for status, count in sorted(counts.items())],
        "",
        "| target | materializer | verification | primary ratio | QASM ratio | proof |",
        "|---|---|---|---:|---:|---|",
    ]
    for row in rows:
        proof = row.get("proof_path") or ""
        lines.append(
            "| {target} | {materializer} | {status} | {primary} | {qasm} | `{proof}` |".format(
                target=row.get("target", ""),
                materializer=row.get("materializer", ""),
                status=row.get("verification_status", ""),
                primary=fmt(row.get("primary_nc_depth_ratio")),
                qasm=fmt(row.get("qasm_depth_ratio")),
                proof=proof,
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: Any) -> str:
    numeric = coerce_float(value)
    return "" if numeric is None else f"{numeric:.3g}"


def main() -> int:
    args = parse_args()
    if not circuit_to_tensor_binary().exists():
        raise SystemExit(f"Missing circuit-to-tensor binary: {circuit_to_tensor_binary()}")
    rows = verify_rows(
        rows=filter_rows(read_csv(args.beam_csv), args.objective_variant),
        output_root=args.output_root,
        materializer_prefix=args.materializer_prefix,
        timeout_sec=args.timeout_sec,
    )
    write_csv_rows(rows, args.summary_csv)
    write_json(rows, args.summary_json)
    write_report(args.report_path, rows, args.summary_csv)
    print(
        json.dumps(
            {
                "summary_csv": relative_or_absolute(args.summary_csv),
                "summary_json": relative_or_absolute(args.summary_json),
                "report_path": relative_or_absolute(args.report_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
