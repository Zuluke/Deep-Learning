from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._analysis_common import DEFAULT_CSV_ROOT
from scripts._analysis_common import DEFAULT_REPORTS_ROOT
from scripts._analysis_common import DEFAULT_RESULTS_ROOT
from scripts._analysis_common import ensure_dir
from scripts._analysis_common import natural_sort_key
from scripts._analysis_common import write_csv_rows


DEFAULT_CIRCUIT_IDS = (
    "mod_5_4",
    "gf_2pow2_mult",
    "qft_4",
    "hamming_weight_n4",
    "hamming_weight_n5",
)

COMPARISON_METHODS = (
    ("alphatensor_public", "AlphaTensor-public / T-count"),
    ("alphaq_tensor_v3", "AlphaQ tensor-v3"),
    ("frontier_primary_oracle", "Primary-target oracle"),
    ("public_resynth_structural", "AlphaQ structural selector"),
    ("alphaq_final", "AlphaQ-final"),
)


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def coerce_float(value: Any) -> float | None:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def coerce_int(value: Any) -> int | None:
    value_float = coerce_float(value)
    return None if value_float is None else int(value_float)


def verification_index(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(row["circuit_id"], row["method"]): row for row in rows}


def row_for_method(
    *,
    circuit_id: str,
    method: str,
    entrega_by_key: dict[tuple[str, str], dict[str, str]],
    final_by_key: dict[tuple[str, str], dict[str, str]],
    verification_by_key: dict[tuple[str, str], dict[str, str]],
    frontier_by_circuit: dict[str, list[dict[str, str]]],
) -> dict[str, Any] | None:
    if method == "frontier_primary_oracle":
        candidates = [
            row
            for row in frontier_by_circuit.get(circuit_id, [])
            if row.get("status") == "ok"
            and row.get("selection_status") == "ok"
            and coerce_float(row.get("primary_nc_depth_ratio")) is not None
        ]
        if not candidates:
            return None
        source = min(
            candidates,
            key=lambda row: (
                coerce_float(row.get("primary_nc_depth_ratio")) or float("inf"),
                coerce_float(row.get("tcount_after")) or float("inf"),
                coerce_float(row.get("qasm_depth_ratio")) or float("inf"),
                coerce_int(row.get("combo_index")) or 10**12,
            ),
        )
        formal_status = "not-run"
        source_method = method
    elif method == "public_resynth_structural":
        source = final_by_key.get((circuit_id, method))
        if source is None:
            return None
        verification_row = verification_by_key.get((circuit_id, method), {})
        formal_status = verification_row.get("verification_status", "not-run")
        source_method = method
    else:
        source = entrega_by_key.get((circuit_id, method))
        if source is None:
            return None
        source_method = source.get("source_method") or method
        formal_status = source.get("formal_verification_status") or verification_by_key.get(
            (circuit_id, source_method), {}
        ).get("verification_status", "not-run")

    return {
        "circuit_id": circuit_id,
        "comparison_method": method,
        "method_label": dict(COMPARISON_METHODS)[method],
        "source_method": source_method,
        "method_status": source.get("method_status") or source.get("status"),
        "formal_verification_status": formal_status,
        "selection_objective": source.get("selection_objective"),
        "tcount_after": coerce_int(source.get("tcount_after")),
        "tdepth_after": coerce_int(source.get("tdepth_after")),
        "primary_nc_depth_ratio": coerce_float(source.get("primary_nc_depth_ratio")),
        "zx_total_depth_ratio": coerce_float(source.get("zx_total_depth_ratio")),
        "qasm_depth_ratio": coerce_float(source.get("qasm_depth_ratio")),
        "structural_cost": coerce_float(source.get("structural_cost")),
        "tensor_v3_status": source.get("tensor_v3_status"),
        "tensor_v3_mixed_excess_norm": coerce_float(
            source.get("tensor_v3_mixed_excess_norm")
        ),
        "tensor_v3_mixed_auc_greedy_norm": coerce_float(
            source.get("tensor_v3_mixed_auc_greedy_norm")
        ),
        "tensor_v3_score_lex": source.get("tensor_v3_score_lex"),
        "qasm_artifact_path": source.get("qasm_artifact_path")
        or source.get("candidate_qasm_path"),
    }


def annotate_against_references(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_circuit: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_circuit.setdefault(row["circuit_id"], {})[row["comparison_method"]] = row

    annotated = []
    for row in rows:
        circuit_rows = by_circuit[row["circuit_id"]]
        baseline = circuit_rows.get("alphatensor_public")
        oracle = circuit_rows.get("frontier_primary_oracle")
        primary = row.get("primary_nc_depth_ratio")
        tcount = row.get("tcount_after")
        baseline_primary = None if baseline is None else baseline.get("primary_nc_depth_ratio")
        oracle_primary = None if oracle is None else oracle.get("primary_nc_depth_ratio")
        baseline_tcount = None if baseline is None else baseline.get("tcount_after")
        annotated.append(
            {
                **row,
                "primary_delta_vs_alphatensor_public": (
                    None
                    if primary is None or baseline_primary is None
                    else primary - baseline_primary
                ),
                "primary_delta_vs_primary_oracle": (
                    None
                    if primary is None or oracle_primary is None
                    else primary - oracle_primary
                ),
                "tcount_delta_vs_alphatensor_public": (
                    None if tcount is None or baseline_tcount is None else tcount - baseline_tcount
                ),
            }
        )
    return annotated


def relation(delta: float | None, *, eps: float = 1e-9) -> str:
    if delta is None:
        return "missing"
    if delta < -eps:
        return "better"
    if delta > eps:
        return "worse"
    return "tie"


def write_report(rows: list[dict[str, Any]], output_path: Path, csv_path: Path) -> Path:
    tensor_rows = [
        row for row in rows if row["comparison_method"] == "alphaq_tensor_v3"
    ]
    tensor_equal = [
        row for row in tensor_rows if row.get("formal_verification_status") == "equal"
    ]
    baseline_relations = [
        relation(row.get("primary_delta_vs_alphatensor_public")) for row in tensor_rows
    ]
    oracle_relations = [
        relation(row.get("primary_delta_vs_primary_oracle")) for row in tensor_rows
    ]
    counts = {
        "better_vs_baseline": baseline_relations.count("better"),
        "tie_vs_baseline": baseline_relations.count("tie"),
        "worse_vs_baseline": baseline_relations.count("worse"),
        "better_vs_oracle": oracle_relations.count("better"),
        "tie_vs_oracle": oracle_relations.count("tie"),
        "worse_vs_oracle": oracle_relations.count("worse"),
    }

    table_lines = [
        "| circuit | T-count baseline | tensor-v3 T | primary baseline | primary tensor-v3 | primary oracle | tensor-v3 vs baseline | tensor-v3 vs primary oracle | formal |",
        "|---|---:|---:|---:|---:|---:|---|---|---|",
    ]
    by_circuit: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_circuit.setdefault(row["circuit_id"], {})[row["comparison_method"]] = row
    for circuit_id in sorted(by_circuit, key=natural_sort_key):
        circuit_rows = by_circuit[circuit_id]
        baseline = circuit_rows.get("alphatensor_public", {})
        tensor = circuit_rows.get("alphaq_tensor_v3", {})
        oracle = circuit_rows.get("frontier_primary_oracle", {})
        table_lines.append(
            "| "
            + " | ".join(
                [
                    f"`{circuit_id}`",
                    fmt_value(baseline.get("tcount_after"), integer=True),
                    fmt_value(tensor.get("tcount_after"), integer=True),
                    fmt_value(baseline.get("primary_nc_depth_ratio")),
                    fmt_value(tensor.get("primary_nc_depth_ratio")),
                    fmt_value(oracle.get("primary_nc_depth_ratio")),
                    relation(tensor.get("primary_delta_vs_alphatensor_public")),
                    relation(tensor.get("primary_delta_vs_primary_oracle")),
                    str(tensor.get("formal_verification_status") or "missing"),
                ]
            )
            + " |"
        )

    text = [
        "# Tensor-v3 Selection Comparison",
        "",
        f"- Comparison CSV: `{csv_path}`.",
        f"- Tensor-v3 formally equal rows: {len(tensor_equal)}/{len(tensor_rows)}.",
        (
            "- Tensor-v3 vs AlphaTensor-public by `primary_nc_depth_ratio`: "
            f"better={counts['better_vs_baseline']}, "
            f"tie={counts['tie_vs_baseline']}, worse={counts['worse_vs_baseline']}."
        ),
        (
            "- Tensor-v3 vs primary-target oracle by `primary_nc_depth_ratio`: "
            f"better={counts['better_vs_oracle']}, "
            f"tie={counts['tie_vs_oracle']}, worse={counts['worse_vs_oracle']}."
        ),
        "",
        "## Circuit-Level Summary",
        "",
        *table_lines,
        "",
        "## Interpretation",
        "",
        (
            "`AlphaQ tensor-v3` is the AlphaQuantum-only candidate selector: it uses the "
            "tensor decomposition and semantic/graph partitions to rerank candidates, while "
            "`primary_nc_depth_ratio` remains an external audit target. The primary-target "
            "oracle is not a deployable selector because it chooses directly from the evaluated "
            "candidate frontier using the external target; it is included only to measure the "
            "gap left by tensor-v3. `AlphaQ structural selector` is the earlier circuit-level "
            "AlphaQ structural-cost selector, not the oracle."
        ),
        "",
    ]
    ensure_dir(output_path.parent)
    output_path.write_text("\n".join(text), encoding="utf-8")
    return output_path


def fmt_value(value: Any, *, integer: bool = False) -> str:
    if value is None or value == "":
        return ""
    if integer:
        return f"{int(value)}"
    return f"{float(value):.3f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare tensor-v3 selection against baselines.")
    parser.add_argument(
        "--final-metrics-csv",
        type=Path,
        default=DEFAULT_CSV_ROOT / "final_metrics.csv",
    )
    parser.add_argument(
        "--entrega-csv",
        type=Path,
        default=DEFAULT_CSV_ROOT / "entrega1_metrics.csv",
    )
    parser.add_argument(
        "--verification-csv",
        type=Path,
        default=DEFAULT_RESULTS_ROOT / "verification" / "entrega1" / "verification_summary.csv",
    )
    parser.add_argument(
        "--candidate-frontier-csv",
        type=Path,
        default=DEFAULT_RESULTS_ROOT
        / "public_resynth_tensor_v3"
        / "candidate_frontier.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_CSV_ROOT / "tensor_v3_selection_comparison.csv",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_CSV_ROOT / "tensor_v3_selection_comparison.json",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORTS_ROOT / "tensor_v3_selection_comparison.md",
    )
    parser.add_argument("--circuit-id", action="append", dest="circuit_ids", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    circuit_ids = tuple(args.circuit_ids) if args.circuit_ids else DEFAULT_CIRCUIT_IDS
    entrega_rows = load_csv_rows(args.entrega_csv)
    final_rows = load_csv_rows(args.final_metrics_csv)
    verification_rows = load_csv_rows(args.verification_csv)
    frontier_rows = load_csv_rows(args.candidate_frontier_csv)
    entrega_by_key = {(row["circuit_id"], row["method"]): row for row in entrega_rows}
    final_by_key = {(row["circuit_id"], row["method"]): row for row in final_rows}
    verification_by_key = verification_index(verification_rows)
    frontier_by_circuit: dict[str, list[dict[str, str]]] = {}
    for row in frontier_rows:
        frontier_by_circuit.setdefault(row["circuit_id"], []).append(row)

    rows = []
    for circuit_id in circuit_ids:
        for method, _label in COMPARISON_METHODS:
            row = row_for_method(
                circuit_id=circuit_id,
                method=method,
                entrega_by_key=entrega_by_key,
                final_by_key=final_by_key,
                verification_by_key=verification_by_key,
                frontier_by_circuit=frontier_by_circuit,
            )
            if row is not None:
                rows.append(row)
    rows = annotate_against_references(rows)

    ensure_dir(args.output_csv.parent)
    write_csv_rows(rows, args.output_csv)
    args.output_json.write_text(
        json.dumps(
            {
                "final_metrics_csv": str(args.final_metrics_csv),
                "entrega_csv": str(args.entrega_csv),
                "verification_csv": str(args.verification_csv),
                "candidate_frontier_csv": str(args.candidate_frontier_csv),
                "output_csv": str(args.output_csv),
                "report_path": str(args.report_path),
                "circuit_ids": list(circuit_ids),
                "num_rows": len(rows),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_report(rows, args.report_path, args.output_csv)
    print(
        json.dumps(
            {
                "output_csv": str(args.output_csv),
                "output_json": str(args.output_json),
                "report_path": str(args.report_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
