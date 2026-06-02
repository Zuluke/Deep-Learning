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
from scripts._analysis_common import DEFAULT_FIGURES_ROOT
from scripts._analysis_common import DEFAULT_REPORTS_ROOT
from scripts._analysis_common import DEFAULT_RESULTS_ROOT
from scripts._analysis_common import ensure_dir
from scripts._analysis_common import natural_sort_key
from scripts._analysis_common import write_csv_rows
from scripts.analyze_tensor_v3_phase_slack_sensitivity import build_sensitivity_rows
from scripts.analyze_tensor_v3_phase_slack_sensitivity import fmt
from scripts.analyze_tensor_v3_phase_slack_sensitivity import load_csv_rows
from scripts.analyze_tensor_v3_phase_slack_sensitivity import parse_float_list
from scripts.analyze_tensor_v3_phase_slack_sensitivity import summarize_by_slack


DEFAULT_CIRCUIT_IDS = (
    "mod_5_4",
    "gf_2pow2_mult",
    "qft_4",
    "hamming_weight_n4",
    "hamming_weight_n5",
    "cuccaro_adder_n3",
    "vbe_adder_3",
    "barenco_tof_3",
    "barenco_tof_4",
    "hwb_6",
    "nc_tof_3",
    "nc_tof_4",
)
DEFAULT_SLACK_VALUES = (0.125, 0.15, 0.175, 0.20, 0.25, 0.30)
DEFAULT_TCOUNT_TOLERANCES = (0.0, 0.05, 0.10, 0.11, 0.12, 0.15, 0.20, 0.30, 0.40)


def build_surface_rows(
    frontier_rows: list[dict[str, str]],
    entrega_rows: list[dict[str, str]],
    *,
    circuit_ids: tuple[str, ...],
    slack_values: tuple[float, ...],
    tcount_tolerances: tuple[float, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for tcount_tolerance in tcount_tolerances:
        tolerance_detail_rows = build_sensitivity_rows(
            frontier_rows,
            entrega_rows,
            circuit_ids=circuit_ids,
            slack_values=slack_values,
            tcount_tolerance=tcount_tolerance,
        )
        for row in tolerance_detail_rows:
            row["tcount_tolerance"] = tcount_tolerance
        detail_rows.extend(tolerance_detail_rows)

        for row in summarize_by_slack(tolerance_detail_rows):
            row["tcount_tolerance"] = tcount_tolerance
            summary_rows.append(row)

    detail_rows.sort(
        key=lambda row: (
            float(row.get("tcount_tolerance") or 0.0),
            float(row.get("slack") or 0.0),
            natural_sort_key(str(row.get("circuit_id") or "")),
        )
    )
    summary_rows.sort(
        key=lambda row: (
            float(row.get("tcount_tolerance") or 0.0),
            float(row.get("slack") or 0.0),
        )
    )
    return detail_rows, summary_rows


def best_non_worse_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = [
        row for row in summary_rows if int(row.get("worse_vs_baseline") or 0) == 0
    ]
    if not candidates:
        return []
    best_global_hits = max(int(row.get("global_oracle_hits") or 0) for row in candidates)
    candidates = [
        row
        for row in candidates
        if int(row.get("global_oracle_hits") or 0) == best_global_hits
    ]
    best_global_regret = min(
        float(row.get("max_global_regret_vs_oracle") or 0.0) for row in candidates
    )
    candidates = [
        row
        for row in candidates
        if abs(
            float(row.get("max_global_regret_vs_oracle") or 0.0)
            - best_global_regret
        )
        <= 1e-12
    ]
    best_tcount_overhead = min(
        int(row.get("tcount_overhead_count") or 0) for row in candidates
    )
    return [
        row
        for row in candidates
        if int(row.get("tcount_overhead_count") or 0) == best_tcount_overhead
    ]


def write_figure(summary_rows: list[dict[str, Any]], figure_path: Path) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    if not summary_rows:
        return None
    slacks = sorted({float(row["slack"]) for row in summary_rows})
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for slack in slacks:
        rows = [
            row
            for row in summary_rows
            if abs(float(row["slack"]) - slack) <= 1e-12
        ]
        rows.sort(key=lambda row: float(row["tcount_tolerance"]))
        ax.plot(
            [float(row["tcount_tolerance"]) for row in rows],
            [float(row["max_global_regret_vs_oracle"]) for row in rows],
            marker="o",
            linewidth=1.4,
            label=f"slack={slack:g}",
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("T-count tolerance")
    ax.set_ylabel("max global primary regret")
    ax.set_title("Tensor-v3 tolerance surface")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    ensure_dir(figure_path.parent)
    fig.savefig(figure_path)
    plt.close(fig)
    return figure_path


def write_report(
    detail_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    *,
    output_path: Path,
    detail_csv_path: Path,
    summary_csv_path: Path,
    figure_path: Path | None,
) -> Path:
    best_rows = best_non_worse_rows(summary_rows)
    best_text = (
        ", ".join(
            f"Ttol={float(row['tcount_tolerance']):g}/slack={float(row['slack']):g}"
            for row in best_rows
        )
        or "none"
    )
    summary_lines = [
        "| T tolerance | slack | class | vs baseline B/T/W | constrained oracle | global oracle | max constrained regret | max global regret | T overhead |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        summary_lines.append(
            "| "
            + " | ".join(
                [
                    fmt(row["tcount_tolerance"]),
                    fmt(row["slack"]),
                    f"`{row['classification']}`",
                    (
                        f"{row['better_vs_baseline']}/"
                        f"{row['tie_vs_baseline']}/"
                        f"{row['worse_vs_baseline']}"
                    ),
                    f"{row['oracle_hits']}/{row['num_circuits']}",
                    f"{row['global_oracle_hits']}/{row['num_circuits']}",
                    fmt(row["max_regret_vs_oracle"]),
                    fmt(row["max_global_regret_vs_oracle"]),
                    str(row["tcount_overhead_count"]),
                ]
            )
            + " |"
        )

    focus_rows = [
        row
        for row in detail_rows
        if row.get("circuit_id") in {"hwb_6", "vbe_adder_3"}
        and abs(float(row.get("slack") or 0.0) - 0.20) <= 1e-12
    ]
    focus_lines = [
        "| circuit | T tolerance | selected | T | constrained oracle | global oracle | primary | constrained regret | global regret |",
        "|---|---:|---|---:|---|---|---:|---:|---:|",
    ]
    for row in focus_rows:
        focus_lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['circuit_id']}`",
                    fmt(row["tcount_tolerance"]),
                    f"`{row['selected_candidate_id']}`",
                    fmt(row["selected_tcount"], integer=True),
                    f"`{row['oracle_candidate_id']}`",
                    f"`{row['global_oracle_candidate_id']}`",
                    fmt(row["selected_primary_nc_depth_ratio"]),
                    fmt(row["regret_vs_oracle"]),
                    fmt(row["global_regret_vs_oracle"]),
                ]
            )
            + " |"
        )

    text = [
        "# Tensor-v3 T-count Tolerance Surface",
        "",
        f"- Detail CSV: `{detail_csv_path}`.",
        f"- Summary CSV: `{summary_csv_path}`.",
    ]
    if figure_path is not None:
        text.append(f"- Figure: `{figure_path}`.")
    text.extend(
        [
            "",
            "## Decision",
            "",
            (
                "- Best non-worse settings by global oracle hits, max global regret, "
                f"and T-count overhead: {best_text}."
            ),
            (
                "- `vbe_adder_3` reaches its global oracle when T-count tolerance is just above "
                "10%; `hwb_6` needs a much larger T-count relaxation and therefore remains a "
                "different regime rather than a small-threshold miss."
            ),
            (
                "- Use this as an audit surface, not as a new selector: the current default "
                "keeps T-count tolerance conservative while exposing where a relaxed regime "
                "would trade T-count for better splitting."
            ),
            "",
            "## Summary",
            "",
            *summary_lines,
            "",
            "## Focus Cases At Slack 0.20",
            "",
            *focus_lines,
            "",
        ]
    )
    ensure_dir(output_path.parent)
    output_path.write_text("\n".join(text), encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep tensor-v3 mixed slack and T-count tolerance together."
    )
    parser.add_argument(
        "--candidate-frontier-csv",
        type=Path,
        default=DEFAULT_RESULTS_ROOT
        / "public_resynth_tensor_v3_phase_slack_expanded"
        / "candidate_frontier.csv",
    )
    parser.add_argument(
        "--entrega-csv",
        type=Path,
        default=DEFAULT_CSV_ROOT / "entrega1_metrics.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_CSV_ROOT / "tensor_v3_tcount_tolerance_surface.csv",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=DEFAULT_CSV_ROOT / "tensor_v3_tcount_tolerance_surface_summary.csv",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_CSV_ROOT / "tensor_v3_tcount_tolerance_surface.json",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORTS_ROOT / "tensor_v3_tcount_tolerance_surface.md",
    )
    parser.add_argument(
        "--figure-path",
        type=Path,
        default=DEFAULT_FIGURES_ROOT / "tensor_v3_tcount_tolerance_surface.png",
    )
    parser.add_argument(
        "--slack-values",
        default=",".join(str(value) for value in DEFAULT_SLACK_VALUES),
    )
    parser.add_argument(
        "--tcount-tolerances",
        default=",".join(str(value) for value in DEFAULT_TCOUNT_TOLERANCES),
    )
    parser.add_argument("--circuit-id", action="append", dest="circuit_ids", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    circuit_ids = tuple(args.circuit_ids) if args.circuit_ids else DEFAULT_CIRCUIT_IDS
    slack_values = parse_float_list(args.slack_values)
    tcount_tolerances = parse_float_list(args.tcount_tolerances)
    detail_rows, summary_rows = build_surface_rows(
        load_csv_rows(args.candidate_frontier_csv),
        load_csv_rows(args.entrega_csv),
        circuit_ids=circuit_ids,
        slack_values=slack_values,
        tcount_tolerances=tcount_tolerances,
    )
    write_csv_rows(detail_rows, args.output_csv)
    write_csv_rows(summary_rows, args.summary_csv)
    figure_path = write_figure(summary_rows, args.figure_path)
    args.output_json.write_text(
        json.dumps(
            {
                "candidate_frontier_csv": str(args.candidate_frontier_csv),
                "entrega_csv": str(args.entrega_csv),
                "output_csv": str(args.output_csv),
                "summary_csv": str(args.summary_csv),
                "report_path": str(args.report_path),
                "figure_path": str(figure_path) if figure_path else None,
                "slack_values": list(slack_values),
                "tcount_tolerances": list(tcount_tolerances),
                "circuit_ids": list(circuit_ids),
                "num_detail_rows": len(detail_rows),
                "num_summary_rows": len(summary_rows),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_report(
        detail_rows,
        summary_rows,
        output_path=args.report_path,
        detail_csv_path=args.output_csv,
        summary_csv_path=args.summary_csv,
        figure_path=figure_path,
    )
    print(
        json.dumps(
            {
                "output_csv": str(args.output_csv),
                "summary_csv": str(args.summary_csv),
                "output_json": str(args.output_json),
                "report_path": str(args.report_path),
                "figure_path": str(figure_path) if figure_path else None,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
