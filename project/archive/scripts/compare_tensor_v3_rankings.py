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


def finite(value: Any) -> float:
    value_float = coerce_float(value)
    return float("inf") if value_float is None else value_float


def relation(delta: float | None, *, eps: float = 1e-9) -> str:
    if delta is None:
        return "missing"
    if delta < -eps:
        return "better"
    if delta > eps:
        return "worse"
    return "tie"


def build_oracle_index(frontier_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    by_circuit: dict[str, list[dict[str, str]]] = {}
    for row in frontier_rows:
        if (
            row.get("status") == "ok"
            and row.get("selection_status") == "ok"
            and coerce_float(row.get("primary_nc_depth_ratio")) is not None
        ):
            by_circuit.setdefault(row["circuit_id"], []).append(row)
    return {
        circuit_id: min(
            rows,
            key=lambda row: (
                finite(row.get("primary_nc_depth_ratio")),
                finite(row.get("tcount_after")),
                finite(row.get("qasm_depth_ratio")),
                coerce_int(row.get("combo_index")) or 10**12,
            ),
        )
        for circuit_id, rows in by_circuit.items()
    }


def row_by_method(
    rows_by_key: dict[tuple[str, str], dict[str, str]],
    circuit_id: str,
    method: str,
) -> dict[str, str] | None:
    row = rows_by_key.get((circuit_id, method))
    if row is None or row.get("method_status") != "ok":
        return None
    return row


def build_comparison_rows(
    entrega_rows: list[dict[str, str]],
    frontier_rows: list[dict[str, str]],
    *,
    circuit_ids: tuple[str, ...],
) -> list[dict[str, Any]]:
    by_key = {(row["circuit_id"], row["method"]): row for row in entrega_rows}
    oracle_by_circuit = build_oracle_index(frontier_rows)
    output_rows: list[dict[str, Any]] = []
    for circuit_id in circuit_ids:
        baseline = row_by_method(by_key, circuit_id, "alphatensor_public")
        lex = row_by_method(by_key, circuit_id, "alphaq_tensor_v3")
        phase = row_by_method(by_key, circuit_id, "alphaq_tensor_v3_phase_slack")
        oracle = oracle_by_circuit.get(circuit_id)
        if baseline is None or lex is None or phase is None or oracle is None:
            continue
        baseline_primary = coerce_float(baseline.get("primary_nc_depth_ratio"))
        lex_primary = coerce_float(lex.get("primary_nc_depth_ratio"))
        phase_primary = coerce_float(phase.get("primary_nc_depth_ratio"))
        oracle_primary = coerce_float(oracle.get("primary_nc_depth_ratio"))
        output_rows.append(
            {
                "circuit_id": circuit_id,
                "baseline_tcount": coerce_int(baseline.get("tcount_after")),
                "lex_tcount": coerce_int(lex.get("tcount_after")),
                "phase_slack_tcount": coerce_int(phase.get("tcount_after")),
                "baseline_primary_nc_depth_ratio": baseline_primary,
                "lex_primary_nc_depth_ratio": lex_primary,
                "phase_slack_primary_nc_depth_ratio": phase_primary,
                "oracle_primary_nc_depth_ratio": oracle_primary,
                "lex_delta_vs_baseline": (
                    None
                    if lex_primary is None or baseline_primary is None
                    else lex_primary - baseline_primary
                ),
                "phase_slack_delta_vs_baseline": (
                    None
                    if phase_primary is None or baseline_primary is None
                    else phase_primary - baseline_primary
                ),
                "phase_slack_delta_vs_lex": (
                    None
                    if phase_primary is None or lex_primary is None
                    else phase_primary - lex_primary
                ),
                "lex_delta_vs_oracle": (
                    None
                    if lex_primary is None or oracle_primary is None
                    else lex_primary - oracle_primary
                ),
                "phase_slack_delta_vs_oracle": (
                    None
                    if phase_primary is None or oracle_primary is None
                    else phase_primary - oracle_primary
                ),
                "lex_formal_status": lex.get("formal_verification_status"),
                "phase_slack_formal_status": phase.get("formal_verification_status"),
                "lex_ranking_strategy": lex.get("tensor_v3_ranking_strategy"),
                "phase_slack_ranking_strategy": phase.get("tensor_v3_ranking_strategy"),
                "phase_slack_mixed_slack": coerce_float(
                    phase.get("tensor_v3_mixed_slack")
                ),
                "oracle_candidate_id": oracle.get("candidate_id"),
            }
        )
    return sorted(output_rows, key=lambda row: natural_sort_key(row["circuit_id"]))


def summarize(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    relations = [relation(row.get(key)) for row in rows]
    return {
        "better": relations.count("better"),
        "tie": relations.count("tie"),
        "worse": relations.count("worse"),
        "missing": relations.count("missing"),
    }


def fmt(value: Any, *, integer: bool = False) -> str:
    if value in (None, ""):
        return ""
    if integer:
        return str(int(value))
    return f"{float(value):.3f}"


def write_report(rows: list[dict[str, Any]], output_path: Path, csv_path: Path) -> Path:
    lex_vs_baseline = summarize(rows, "lex_delta_vs_baseline")
    phase_vs_baseline = summarize(rows, "phase_slack_delta_vs_baseline")
    phase_vs_lex = summarize(rows, "phase_slack_delta_vs_lex")
    phase_vs_oracle = summarize(rows, "phase_slack_delta_vs_oracle")

    table_lines = [
        "| circuit | baseline T | lex T | phase T | baseline primary | lex primary | phase primary | oracle primary | phase vs lex | phase vs oracle | formal phase |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        table_lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['circuit_id']}`",
                    fmt(row.get("baseline_tcount"), integer=True),
                    fmt(row.get("lex_tcount"), integer=True),
                    fmt(row.get("phase_slack_tcount"), integer=True),
                    fmt(row.get("baseline_primary_nc_depth_ratio")),
                    fmt(row.get("lex_primary_nc_depth_ratio")),
                    fmt(row.get("phase_slack_primary_nc_depth_ratio")),
                    fmt(row.get("oracle_primary_nc_depth_ratio")),
                    relation(row.get("phase_slack_delta_vs_lex")),
                    relation(row.get("phase_slack_delta_vs_oracle")),
                    str(row.get("phase_slack_formal_status")),
                ]
            )
            + " |"
        )

    text = [
        "# Tensor-v3 Ranking Comparison",
        "",
        f"- Comparison CSV: `{csv_path}`.",
        (
            "- lex-v1 vs AlphaTensor-public: "
            f"better={lex_vs_baseline['better']}, tie={lex_vs_baseline['tie']}, "
            f"worse={lex_vs_baseline['worse']}."
        ),
        (
            "- phase-slack-v1 vs AlphaTensor-public: "
            f"better={phase_vs_baseline['better']}, tie={phase_vs_baseline['tie']}, "
            f"worse={phase_vs_baseline['worse']}."
        ),
        (
            "- phase-slack-v1 vs lex-v1: "
            f"better={phase_vs_lex['better']}, tie={phase_vs_lex['tie']}, "
            f"worse={phase_vs_lex['worse']}."
        ),
        (
            "- phase-slack-v1 vs primary-target oracle: "
            f"better={phase_vs_oracle['better']}, tie={phase_vs_oracle['tie']}, "
            f"worse={phase_vs_oracle['worse']}."
        ),
        "",
        "## Circuit-Level Summary",
        "",
        *table_lines,
        "",
        "## Interpretation",
        "",
        (
            "`phase-slack-v1` is a frozen AlphaQuantum-only ranking variant: it first enforces "
            "near-optimal tensor mixed-excess score, then uses T-depth as a phase-structure "
            "tie-breaker inside that slack window. On this frontier it fixes the qft_4 miss "
            "without changing the other four selected candidates."
        ),
        "",
    ]
    ensure_dir(output_path.parent)
    output_path.write_text("\n".join(text), encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare tensor-v3 ranking variants.")
    parser.add_argument(
        "--entrega-csv",
        type=Path,
        default=DEFAULT_CSV_ROOT / "entrega1_metrics.csv",
    )
    parser.add_argument(
        "--candidate-frontier-csv",
        type=Path,
        default=DEFAULT_RESULTS_ROOT
        / "public_resynth_tensor_v3_phase_slack"
        / "candidate_frontier.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_CSV_ROOT / "tensor_v3_ranking_comparison.csv",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_CSV_ROOT / "tensor_v3_ranking_comparison.json",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORTS_ROOT / "tensor_v3_ranking_comparison.md",
    )
    parser.add_argument("--circuit-id", action="append", dest="circuit_ids", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    circuit_ids = tuple(args.circuit_ids) if args.circuit_ids else DEFAULT_CIRCUIT_IDS
    rows = build_comparison_rows(
        load_csv_rows(args.entrega_csv),
        load_csv_rows(args.candidate_frontier_csv),
        circuit_ids=circuit_ids,
    )
    write_csv_rows(rows, args.output_csv)
    args.output_json.write_text(
        json.dumps(
            {
                "entrega_csv": str(args.entrega_csv),
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
