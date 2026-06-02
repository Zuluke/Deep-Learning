from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._analysis_common import DEFAULT_RESULTS_ROOT
from scripts._analysis_common import ensure_dir
from scripts._analysis_common import natural_sort_key
from scripts._analysis_common import write_csv_rows
from scripts._analysis_common import write_json
from scripts.compare_tensor_v3_profiles import build_profile_comparison_rows
from scripts.compare_tensor_v3_profiles import load_csv_rows


GUARDED_METHOD = "public_resynth_tensor_v3_phase_slack_guarded"
GUARDED_PROFILE = "guarded-aggressive"


def rows_by_circuit(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {
        row["circuit_id"]: row
        for row in rows
        if row.get("status") == "ok" and row.get("circuit_id")
    }


def materialize_guarded_rows(
    *,
    conservative_rows: list[dict[str, str]],
    aggressive_rows: list[dict[str, str]],
    comparison_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    conservative_by_circuit = rows_by_circuit(conservative_rows)
    aggressive_by_circuit = rows_by_circuit(aggressive_rows)
    output_rows: list[dict[str, Any]] = []
    for comparison in sorted(
        comparison_rows,
        key=lambda row: natural_sort_key(str(row.get("circuit_id") or "")),
    ):
        if comparison.get("status") != "ok":
            continue
        circuit_id = str(comparison["circuit_id"])
        use_aggressive = bool(int(comparison.get("guarded_uses_aggressive") or 0))
        source = (
            aggressive_by_circuit.get(circuit_id)
            if use_aggressive
            else conservative_by_circuit.get(circuit_id)
        )
        if source is None:
            continue
        row = dict(source)
        row.update(
            {
                "method": GUARDED_METHOD,
                "tensor_v3_profile": GUARDED_PROFILE,
                "guarded_source_method": source.get("method"),
                "guarded_source_profile": source.get("tensor_v3_profile"),
                "guarded_uses_aggressive": int(use_aggressive),
                "guarded_qasm_depth_gain_threshold": comparison.get(
                    "guarded_qasm_depth_gain_threshold"
                ),
                "guarded_mixed_drop_fraction_threshold": comparison.get(
                    "guarded_mixed_drop_fraction_threshold"
                ),
                "qasm_depth_gain_aggressive_vs_conservative": comparison.get(
                    "qasm_depth_gain_aggressive_vs_conservative"
                ),
                "mixed_drop_aggressive_vs_conservative": comparison.get(
                    "mixed_drop_aggressive_vs_conservative"
                ),
                "mixed_drop_fraction_aggressive_vs_conservative": comparison.get(
                    "mixed_drop_fraction_aggressive_vs_conservative"
                ),
                "guarded_delta_tcount_vs_conservative": comparison.get(
                    "guarded_delta_tcount_vs_conservative"
                ),
                "guarded_delta_primary_vs_conservative": comparison.get(
                    "guarded_delta_primary_vs_conservative"
                ),
                "guarded_global_oracle_regret": comparison.get(
                    "guarded_global_oracle_regret"
                ),
                "guarded_is_global_oracle": comparison.get("guarded_is_global_oracle"),
            }
        )
        output_rows.append(row)
    return output_rows


def selected_manifest_by_circuit(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {
        row["circuit_id"]: row
        for row in rows
        if row.get("status") == "ok"
        and row.get("circuit_id")
        and str(row.get("tensor_v3_selected") or "").lower() in {"1", "true"}
    }


def materialize_guarded_selection_manifest_rows(
    *,
    conservative_manifest_rows: list[dict[str, str]],
    aggressive_manifest_rows: list[dict[str, str]],
    comparison_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    conservative_by_circuit = selected_manifest_by_circuit(conservative_manifest_rows)
    aggressive_by_circuit = selected_manifest_by_circuit(aggressive_manifest_rows)
    output_rows: list[dict[str, Any]] = []
    for comparison in sorted(
        comparison_rows,
        key=lambda row: natural_sort_key(str(row.get("circuit_id") or "")),
    ):
        if comparison.get("status") != "ok":
            continue
        circuit_id = str(comparison["circuit_id"])
        use_aggressive = bool(int(comparison.get("guarded_uses_aggressive") or 0))
        source = (
            aggressive_by_circuit.get(circuit_id)
            if use_aggressive
            else conservative_by_circuit.get(circuit_id)
        )
        if source is None:
            continue
        source_profile = source.get("tensor_v3_profile")
        row = dict(source)
        row.update(
            {
                "tensor_v3_selected": 1,
                "tensor_v3_profile": GUARDED_PROFILE,
                "guarded_source_profile": source_profile,
                "guarded_source_candidate_id": source.get("candidate_id"),
                "guarded_uses_aggressive": int(use_aggressive),
                "guarded_selection_rule": "qasm-depth-gain-and-mixed-drop",
                "guarded_qasm_depth_gain_threshold": comparison.get(
                    "guarded_qasm_depth_gain_threshold"
                ),
                "guarded_mixed_drop_fraction_threshold": comparison.get(
                    "guarded_mixed_drop_fraction_threshold"
                ),
                "qasm_depth_gain_aggressive_vs_conservative": comparison.get(
                    "qasm_depth_gain_aggressive_vs_conservative"
                ),
                "mixed_drop_aggressive_vs_conservative": comparison.get(
                    "mixed_drop_aggressive_vs_conservative"
                ),
                "mixed_drop_fraction_aggressive_vs_conservative": comparison.get(
                    "mixed_drop_fraction_aggressive_vs_conservative"
                ),
                "guarded_delta_tcount_vs_conservative": comparison.get(
                    "guarded_delta_tcount_vs_conservative"
                ),
            }
        )
        output_rows.append(row)
    return output_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize the guarded tensor-v3 profile as a summary CSV."
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
        "--conservative-selection-manifest-csv",
        type=Path,
        default=DEFAULT_RESULTS_ROOT
        / "public_resynth_tensor_v3_phase_slack_expanded"
        / "tensor_v3_selection_manifest.csv",
    )
    parser.add_argument(
        "--aggressive-selection-manifest-csv",
        type=Path,
        default=DEFAULT_RESULTS_ROOT
        / "public_resynth_tensor_v3_phase_slack_aggressive"
        / "tensor_v3_selection_manifest.csv",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT / "public_resynth_tensor_v3_phase_slack_guarded",
    )
    parser.add_argument("--guarded-qasm-depth-gain", type=float, default=0.10)
    parser.add_argument("--guarded-mixed-drop-fraction", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    conservative_rows = load_csv_rows(args.conservative_summary_csv)
    aggressive_rows = load_csv_rows(args.aggressive_summary_csv)
    comparison_rows = build_profile_comparison_rows(
        conservative_rows=conservative_rows,
        aggressive_rows=aggressive_rows,
        frontier_rows=load_csv_rows(args.candidate_frontier_csv),
        guarded_qasm_depth_gain=args.guarded_qasm_depth_gain,
        guarded_mixed_drop_fraction=args.guarded_mixed_drop_fraction,
    )
    guarded_rows = materialize_guarded_rows(
        conservative_rows=conservative_rows,
        aggressive_rows=aggressive_rows,
        comparison_rows=comparison_rows,
    )
    guarded_manifest_rows = materialize_guarded_selection_manifest_rows(
        conservative_manifest_rows=load_csv_rows(
            args.conservative_selection_manifest_csv
        ),
        aggressive_manifest_rows=load_csv_rows(args.aggressive_selection_manifest_csv),
        comparison_rows=comparison_rows,
    )
    ensure_dir(args.output_root)
    summary_csv = args.output_root / "public_resynth_summary.csv"
    summary_json = args.output_root / "public_resynth_summary.json"
    selection_manifest_csv = args.output_root / "tensor_v3_selection_manifest.csv"
    write_csv_rows(guarded_rows, summary_csv)
    if guarded_manifest_rows:
        write_csv_rows(guarded_manifest_rows, selection_manifest_csv)
    write_json(
        {
            "conservative_summary_csv": str(args.conservative_summary_csv),
            "aggressive_summary_csv": str(args.aggressive_summary_csv),
            "candidate_frontier_csv": str(args.candidate_frontier_csv),
            "conservative_selection_manifest_csv": str(
                args.conservative_selection_manifest_csv
            ),
            "aggressive_selection_manifest_csv": str(
                args.aggressive_selection_manifest_csv
            ),
            "output_root": str(args.output_root),
            "summary_csv": str(summary_csv),
            "selection_manifest_csv": (
                str(selection_manifest_csv) if guarded_manifest_rows else None
            ),
            "guarded_qasm_depth_gain": args.guarded_qasm_depth_gain,
            "guarded_mixed_drop_fraction": args.guarded_mixed_drop_fraction,
            "num_rows": len(guarded_rows),
            "num_selection_manifest_rows": len(guarded_manifest_rows),
            "num_aggressive_choices": sum(
                int(row.get("guarded_uses_aggressive") or 0)
                for row in guarded_rows
            ),
        },
        summary_json,
    )
    print(
        json.dumps(
            {
                "summary_csv": str(summary_csv),
                "summary_json": str(summary_json),
                "num_rows": len(guarded_rows),
                "selection_manifest_csv": (
                    str(selection_manifest_csv) if guarded_manifest_rows else None
                ),
                "num_selection_manifest_rows": len(guarded_manifest_rows),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
