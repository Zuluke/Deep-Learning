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


def ok_summary_by_circuit(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {
        row["circuit_id"]: row
        for row in rows
        if row.get("status") == "ok" and row.get("circuit_id")
    }


def frontier_rows_by_circuit(
    rows: list[dict[str, str]],
) -> dict[str, list[dict[str, str]]]:
    by_circuit: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if (
            row.get("status") == "ok"
            and row.get("selection_status") == "ok"
            and coerce_float(row.get("primary_nc_depth_ratio")) is not None
        ):
            by_circuit.setdefault(row["circuit_id"], []).append(row)
    return by_circuit


def primary_key(row: dict[str, str]) -> tuple[float, float, float]:
    return (
        finite(row.get("primary_nc_depth_ratio")),
        finite(row.get("tcount_after")),
        finite(row.get("tdepth_after")),
    )


def tcount_key(row: dict[str, str]) -> tuple[float, float, float]:
    return (
        finite(row.get("tcount_after")),
        finite(row.get("tdepth_after")),
        finite(row.get("primary_nc_depth_ratio")),
    )


def guarded_accepts(
    *,
    conservative_qasm_depth: float | None,
    aggressive_qasm_depth: float | None,
    conservative_mixed: float | None,
    aggressive_mixed: float | None,
    qasm_depth_gain: float,
    mixed_drop_fraction: float,
) -> bool:
    if (
        conservative_qasm_depth is None
        or aggressive_qasm_depth is None
        or conservative_mixed is None
        or aggressive_mixed is None
    ):
        return False
    qasm_gain = conservative_qasm_depth - aggressive_qasm_depth
    mixed_drop = conservative_mixed - aggressive_mixed
    mixed_drop_ratio = mixed_drop / max(abs(conservative_mixed), 1e-12)
    return (
        qasm_gain >= qasm_depth_gain
        and mixed_drop > 1e-12
        and mixed_drop_ratio >= mixed_drop_fraction
    )


def build_profile_comparison_rows(
    *,
    conservative_rows: list[dict[str, str]],
    aggressive_rows: list[dict[str, str]],
    frontier_rows: list[dict[str, str]],
    guarded_qasm_depth_gain: float,
    guarded_mixed_drop_fraction: float = 0.0,
) -> list[dict[str, Any]]:
    conservative_by_circuit = ok_summary_by_circuit(conservative_rows)
    aggressive_by_circuit = ok_summary_by_circuit(aggressive_rows)
    frontier_by_circuit = frontier_rows_by_circuit(frontier_rows)
    circuit_ids = sorted(
        set(conservative_by_circuit) | set(aggressive_by_circuit),
        key=natural_sort_key,
    )
    output_rows: list[dict[str, Any]] = []
    for circuit_id in circuit_ids:
        conservative = conservative_by_circuit.get(circuit_id)
        aggressive = aggressive_by_circuit.get(circuit_id)
        candidates = frontier_by_circuit.get(circuit_id, [])
        if conservative is None or aggressive is None:
            output_rows.append(
                {
                    "circuit_id": circuit_id,
                    "status": "missing-profile",
                    "has_conservative": int(conservative is not None),
                    "has_aggressive": int(aggressive is not None),
                }
            )
            continue

        global_oracle = min(candidates, key=primary_key) if candidates else None
        tcount_baseline = min(candidates, key=tcount_key) if candidates else None
        conservative_primary = coerce_float(conservative.get("primary_nc_depth_ratio"))
        aggressive_primary = coerce_float(aggressive.get("primary_nc_depth_ratio"))
        delta_primary = (
            None
            if conservative_primary is None or aggressive_primary is None
            else aggressive_primary - conservative_primary
        )
        conservative_tcount = coerce_int(conservative.get("tcount_after_selection"))
        aggressive_tcount = coerce_int(aggressive.get("tcount_after_selection"))
        delta_tcount = (
            None
            if conservative_tcount is None or aggressive_tcount is None
            else aggressive_tcount - conservative_tcount
        )
        conservative_qasm_depth = coerce_float(conservative.get("qasm_depth_ratio"))
        aggressive_qasm_depth = coerce_float(aggressive.get("qasm_depth_ratio"))
        conservative_mixed = coerce_float(conservative.get("tensor_v3_mixed_excess_norm"))
        aggressive_mixed = coerce_float(aggressive.get("tensor_v3_mixed_excess_norm"))
        qasm_depth_gain_observed = (
            None
            if conservative_qasm_depth is None or aggressive_qasm_depth is None
            else conservative_qasm_depth - aggressive_qasm_depth
        )
        mixed_drop_observed = (
            None
            if conservative_mixed is None or aggressive_mixed is None
            else conservative_mixed - aggressive_mixed
        )
        mixed_drop_fraction_observed = (
            None
            if mixed_drop_observed is None or conservative_mixed is None
            else mixed_drop_observed / max(abs(conservative_mixed), 1e-12)
        )
        guarded_uses_aggressive = guarded_accepts(
            conservative_qasm_depth=conservative_qasm_depth,
            aggressive_qasm_depth=aggressive_qasm_depth,
            conservative_mixed=conservative_mixed,
            aggressive_mixed=aggressive_mixed,
            qasm_depth_gain=guarded_qasm_depth_gain,
            mixed_drop_fraction=guarded_mixed_drop_fraction,
        )
        guarded = aggressive if guarded_uses_aggressive else conservative
        guarded_primary = coerce_float(guarded.get("primary_nc_depth_ratio"))
        guarded_tcount = coerce_int(guarded.get("tcount_after_selection"))
        guarded_candidate_id = f"{circuit_id}:combo{guarded.get('combo_index')}"
        global_oracle_primary = (
            coerce_float(global_oracle.get("primary_nc_depth_ratio"))
            if global_oracle
            else None
        )
        output_rows.append(
            {
                "circuit_id": circuit_id,
                "status": "ok",
                "conservative_candidate_id": f"{circuit_id}:combo{conservative.get('combo_index')}",
                "aggressive_candidate_id": f"{circuit_id}:combo{aggressive.get('combo_index')}",
                "tcount_baseline_candidate_id": (
                    tcount_baseline.get("candidate_id") if tcount_baseline else None
                ),
                "global_oracle_candidate_id": (
                    global_oracle.get("candidate_id") if global_oracle else None
                ),
                "conservative_tcount": conservative_tcount,
                "aggressive_tcount": aggressive_tcount,
                "delta_tcount": delta_tcount,
                "guarded_candidate_id": guarded_candidate_id,
                "guarded_uses_aggressive": int(guarded_uses_aggressive),
                "guarded_tcount": guarded_tcount,
                "guarded_delta_tcount_vs_conservative": (
                    None
                    if guarded_tcount is None or conservative_tcount is None
                    else guarded_tcount - conservative_tcount
                ),
                "conservative_primary_nc_depth_ratio": conservative_primary,
                "aggressive_primary_nc_depth_ratio": aggressive_primary,
                "guarded_primary_nc_depth_ratio": guarded_primary,
                "global_oracle_primary_nc_depth_ratio": global_oracle_primary,
                "delta_primary_vs_conservative": delta_primary,
                "aggressive_relation_vs_conservative": relation(delta_primary),
                "guarded_delta_primary_vs_conservative": (
                    None
                    if conservative_primary is None or guarded_primary is None
                    else guarded_primary - conservative_primary
                ),
                "guarded_relation_vs_conservative": relation(
                    None
                    if conservative_primary is None or guarded_primary is None
                    else guarded_primary - conservative_primary
                ),
                "aggressive_global_oracle_regret": (
                    None
                    if aggressive_primary is None or global_oracle_primary is None
                    else aggressive_primary - global_oracle_primary
                ),
                "guarded_global_oracle_regret": (
                    None
                    if guarded_primary is None or global_oracle_primary is None
                    else guarded_primary - global_oracle_primary
                ),
                "guarded_qasm_depth_gain_threshold": guarded_qasm_depth_gain,
                "guarded_mixed_drop_fraction_threshold": guarded_mixed_drop_fraction,
                "qasm_depth_gain_aggressive_vs_conservative": qasm_depth_gain_observed,
                "mixed_drop_aggressive_vs_conservative": mixed_drop_observed,
                "mixed_drop_fraction_aggressive_vs_conservative": (
                    mixed_drop_fraction_observed
                ),
                "conservative_profile": conservative.get("tensor_v3_profile"),
                "aggressive_profile": aggressive.get("tensor_v3_profile"),
                "conservative_tcount_tolerance": coerce_float(
                    conservative.get("tensor_v3_tcount_tolerance")
                ),
                "aggressive_tcount_tolerance": coerce_float(
                    aggressive.get("tensor_v3_tcount_tolerance")
                ),
                "conservative_mixed_slack": coerce_float(
                    conservative.get("tensor_v3_mixed_slack")
                ),
                "aggressive_mixed_slack": coerce_float(
                    aggressive.get("tensor_v3_mixed_slack")
                ),
                "aggressive_is_global_oracle": int(
                    global_oracle is not None
                    and f"{circuit_id}:combo{aggressive.get('combo_index')}"
                    == global_oracle.get("candidate_id")
                ),
                "guarded_is_global_oracle": int(
                    global_oracle is not None
                    and guarded_candidate_id == global_oracle.get("candidate_id")
                ),
            }
        )
    return output_rows


def fmt(value: Any, *, integer: bool = False) -> str:
    if value in (None, ""):
        return ""
    if integer:
        return str(int(value))
    return f"{float(value):.3f}"


def write_report(
    rows: list[dict[str, Any]],
    report_path: Path,
    csv_path: Path,
    *,
    guarded_qasm_depth_gain: float,
    guarded_mixed_drop_fraction: float,
) -> Path:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    relations = [
        str(row.get("aggressive_relation_vs_conservative")) for row in ok_rows
    ]
    guarded_relations = [
        str(row.get("guarded_relation_vs_conservative")) for row in ok_rows
    ]
    global_hits = sum(int(row.get("aggressive_is_global_oracle") or 0) for row in ok_rows)
    guarded_global_hits = sum(
        int(row.get("guarded_is_global_oracle") or 0) for row in ok_rows
    )
    changed = sum(
        row.get("conservative_candidate_id") != row.get("aggressive_candidate_id")
        for row in ok_rows
    )
    guarded_changed = sum(int(row.get("guarded_uses_aggressive") or 0) for row in ok_rows)
    tcount_overheads = [
        int(row.get("delta_tcount") or 0)
        for row in ok_rows
        if int(row.get("delta_tcount") or 0) > 0
    ]
    guarded_tcount_overheads = [
        int(row.get("guarded_delta_tcount_vs_conservative") or 0)
        for row in ok_rows
        if int(row.get("guarded_delta_tcount_vs_conservative") or 0) > 0
    ]

    table_lines = [
        "| circuit | conservative | aggressive | guarded | guarded delta T | conservative primary | aggressive primary | guarded primary | guarded vs conservative | guarded oracle regret |",
        "|---|---|---|---|---:|---:|---:|---:|---|---:|",
    ]
    for row in ok_rows:
        table_lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['circuit_id']}`",
                    f"`{row['conservative_candidate_id']}`",
                    f"`{row['aggressive_candidate_id']}`",
                    f"`{row['guarded_candidate_id']}`",
                    fmt(row.get("guarded_delta_tcount_vs_conservative"), integer=True),
                    fmt(row.get("conservative_primary_nc_depth_ratio")),
                    fmt(row.get("aggressive_primary_nc_depth_ratio")),
                    fmt(row.get("guarded_primary_nc_depth_ratio")),
                    str(row.get("guarded_relation_vs_conservative")),
                    fmt(row.get("guarded_global_oracle_regret")),
                ]
            )
            + " |"
        )

    text = [
        "# Tensor-v3 Profile Comparison",
        "",
        f"- Comparison CSV: `{csv_path}`.",
        (
            "- Aggressive vs conservative: "
            f"better={relations.count('better')}, tie={relations.count('tie')}, "
            f"worse={relations.count('worse')}."
        ),
        (
            "- Guarded aggressive vs conservative: "
            f"better={guarded_relations.count('better')}, "
            f"tie={guarded_relations.count('tie')}, "
            f"worse={guarded_relations.count('worse')}."
        ),
        f"- Aggressive global-oracle hits: {global_hits}/{len(ok_rows)}.",
        f"- Guarded aggressive global-oracle hits: {guarded_global_hits}/{len(ok_rows)}.",
        f"- Candidate changes: {changed}/{len(ok_rows)}.",
        f"- Guarded candidate changes: {guarded_changed}/{len(ok_rows)}.",
        (
            "- Positive T-count deltas: "
            f"{len(tcount_overheads)} circuits, total +{sum(tcount_overheads)} T gates."
        ),
        (
            "- Guarded positive T-count deltas: "
            f"{len(guarded_tcount_overheads)} circuits, "
            f"total +{sum(guarded_tcount_overheads)} T gates."
        ),
        "",
        "## Interpretation",
        "",
        (
            "The aggressive profile is an audit regime, not the default scientific claim. "
            "It exposes which improvements require relaxing the T-count guardrail enough "
            "to trade additional non-Clifford count for a smaller external splitting target."
        ),
        (
            "The guarded variant is a virtual selector: it accepts the aggressive candidate "
            "only when tensor mixed-excess improves and normalized QASM depth drops by the "
            "configured margin. In this run the margins are "
            f"`qasm_gain >= {guarded_qasm_depth_gain:g}` and "
            f"`mixed_drop_fraction >= {guarded_mixed_drop_fraction:g}`. It remains "
            "AlphaQuantum/QASM-only; external splitting metrics are used here only for audit."
        ),
        "",
        "## Circuit Summary",
        "",
        *table_lines,
        "",
    ]
    ensure_dir(report_path.parent)
    report_path.write_text("\n".join(text), encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare conservative and aggressive tensor-v3 phase-slack profiles."
    )
    parser.add_argument(
        "--conservative-summary-csv",
        type=Path,
        default=DEFAULT_RESULTS_ROOT
        / "public_resynth_tensor_v3_phase_slack_expanded"
        / "public_resynth_summary.csv",
    )
    parser.add_argument(
        "--aggressive-summary-csv",
        type=Path,
        default=DEFAULT_RESULTS_ROOT
        / "public_resynth_tensor_v3_phase_slack_aggressive"
        / "public_resynth_summary.csv",
    )
    parser.add_argument(
        "--candidate-frontier-csv",
        type=Path,
        default=DEFAULT_RESULTS_ROOT
        / "public_resynth_tensor_v3_phase_slack_aggressive"
        / "candidate_frontier.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_CSV_ROOT / "tensor_v3_profile_comparison.csv",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_CSV_ROOT / "tensor_v3_profile_comparison.json",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORTS_ROOT / "tensor_v3_profile_comparison.md",
    )
    parser.add_argument("--guarded-qasm-depth-gain", type=float, default=0.10)
    parser.add_argument("--guarded-mixed-drop-fraction", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = build_profile_comparison_rows(
        conservative_rows=load_csv_rows(args.conservative_summary_csv),
        aggressive_rows=load_csv_rows(args.aggressive_summary_csv),
        frontier_rows=load_csv_rows(args.candidate_frontier_csv),
        guarded_qasm_depth_gain=args.guarded_qasm_depth_gain,
        guarded_mixed_drop_fraction=args.guarded_mixed_drop_fraction,
    )
    write_csv_rows(rows, args.output_csv)
    args.output_json.write_text(
        json.dumps(
            {
                "conservative_summary_csv": str(args.conservative_summary_csv),
                "aggressive_summary_csv": str(args.aggressive_summary_csv),
                "candidate_frontier_csv": str(args.candidate_frontier_csv),
                "output_csv": str(args.output_csv),
                "report_path": str(args.report_path),
                "guarded_qasm_depth_gain": args.guarded_qasm_depth_gain,
                "guarded_mixed_drop_fraction": args.guarded_mixed_drop_fraction,
                "num_rows": len(rows),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_report(
        rows,
        args.report_path,
        args.output_csv,
        guarded_qasm_depth_gain=args.guarded_qasm_depth_gain,
        guarded_mixed_drop_fraction=args.guarded_mixed_drop_fraction,
    )
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
