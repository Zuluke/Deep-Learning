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
from scripts.analyze_tensor_v3_selection_ablation import coerce_float
from scripts.analyze_tensor_v3_selection_ablation import coerce_int
from scripts.analyze_tensor_v3_selection_ablation import finite
from scripts.analyze_tensor_v3_selection_ablation import objective_keys
from scripts.analyze_tensor_v3_selection_ablation import phase_slack_selection
from scripts.analyze_tensor_v3_selection_ablation import valid_rows_for_circuit


DEFAULT_CIRCUIT_IDS = (
    "mod_5_4",
    "gf_2pow2_mult",
    "qft_4",
    "hamming_weight_n4",
    "hamming_weight_n5",
)
DEFAULT_SLACK_VALUES = (
    0.00,
    0.01,
    0.025,
    0.05,
    0.075,
    0.10,
    0.125,
    0.15,
    0.175,
    0.20,
    0.25,
    0.30,
    0.50,
)


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_float_list(value: str) -> tuple[float, ...]:
    values = []
    for part in value.split(","):
        part = part.strip()
        if part:
            values.append(float(part))
    return tuple(sorted(dict.fromkeys(values)))


def relation(delta: float | None, *, eps: float = 1e-9) -> str:
    if delta is None:
        return "missing"
    if delta < -eps:
        return "better"
    if delta > eps:
        return "worse"
    return "tie"


def selected_method_row(
    rows: list[dict[str, str]],
    *,
    circuit_id: str,
    method: str,
) -> dict[str, str] | None:
    for row in rows:
        if (
            row.get("circuit_id") == circuit_id
            and row.get("method") == method
            and row.get("method_status") == "ok"
        ):
            return row
    return None


def primary_value(row: dict[str, Any] | None) -> float | None:
    if row is None:
        return None
    return coerce_float(row.get("primary_nc_depth_ratio"))


def bool_int(value: bool) -> int:
    return 1 if value else 0


def all_valid_rows_for_circuit(
    rows: list[dict[str, str]],
    *,
    circuit_id: str,
) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row.get("circuit_id") == circuit_id
        and row.get("status") == "ok"
        and row.get("selection_status") == "ok"
        and row.get("tensor_v3_status") == "ok"
        and coerce_float(row.get("primary_nc_depth_ratio")) is not None
        and coerce_float(row.get("tcount_after")) is not None
    ]


def build_sensitivity_rows(
    frontier_rows: list[dict[str, str]],
    entrega_rows: list[dict[str, str]],
    *,
    circuit_ids: tuple[str, ...],
    slack_values: tuple[float, ...],
    tcount_tolerance: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    keys = objective_keys()
    for circuit_id in circuit_ids:
        global_candidates = all_valid_rows_for_circuit(
            frontier_rows,
            circuit_id=circuit_id,
        )
        candidates = valid_rows_for_circuit(
            frontier_rows,
            circuit_id=circuit_id,
            tcount_tolerance=tcount_tolerance,
        )
        if not candidates:
            rows.append(
                {
                    "circuit_id": circuit_id,
                    "slack": None,
                    "selection_status": "missing-candidates",
                }
            )
            continue

        baseline = selected_method_row(
            entrega_rows,
            circuit_id=circuit_id,
            method="alphatensor_public",
        )
        oracle = min(candidates, key=keys["primary_oracle"])
        global_oracle = min(global_candidates or candidates, key=keys["primary_oracle"])
        lex = min(candidates, key=keys["current_lex_v1"])
        tcount_baseline = min(candidates, key=keys["tcount_only"])
        oracle_primary = finite(oracle.get("primary_nc_depth_ratio"))
        global_oracle_primary = finite(global_oracle.get("primary_nc_depth_ratio"))
        lex_primary = finite(lex.get("primary_nc_depth_ratio"))
        baseline_primary = primary_value(baseline)
        baseline_source = "entrega_alphatensor_public"
        baseline_candidate_id = None
        if baseline_primary is None:
            baseline_primary = finite(tcount_baseline.get("primary_nc_depth_ratio"))
            baseline_source = "frontier_tcount_only"
            baseline_candidate_id = tcount_baseline.get("candidate_id")
        best_tcount = min(finite(row.get("tcount_after")) for row in candidates)
        best_mixed = min(finite(row.get("tensor_v3_mixed_excess_norm")) for row in candidates)

        for slack in slack_values:
            selected = phase_slack_selection(candidates, mixed_slack=slack)
            selected_primary = finite(selected.get("primary_nc_depth_ratio"))
            selected_tcount = finite(selected.get("tcount_after"))
            selected_tdepth = finite(selected.get("tdepth_after"))
            allowed_mixed = best_mixed + max(abs(best_mixed) * slack, 1e-12)
            near_tensor_count = sum(
                finite(row.get("tensor_v3_mixed_excess_norm")) <= allowed_mixed
                for row in candidates
            )
            delta_vs_baseline = (
                None
                if baseline_primary is None
                else selected_primary - baseline_primary
            )
            rows.append(
                {
                    "circuit_id": circuit_id,
                    "slack": slack,
                    "selection_status": "ok",
                    "selected_candidate_id": selected.get("candidate_id"),
                    "selected_combo_index": coerce_int(selected.get("combo_index")),
                    "selected_tcount": coerce_int(selected.get("tcount_after")),
                    "selected_tdepth": coerce_int(selected.get("tdepth_after")),
                    "selected_primary_nc_depth_ratio": selected_primary,
                    "baseline_primary_nc_depth_ratio": baseline_primary,
                    "baseline_source": baseline_source,
                    "baseline_candidate_id": baseline_candidate_id,
                    "lex_candidate_id": lex.get("candidate_id"),
                    "lex_primary_nc_depth_ratio": lex_primary,
                    "oracle_candidate_id": oracle.get("candidate_id"),
                    "oracle_primary_nc_depth_ratio": oracle_primary,
                    "global_oracle_candidate_id": global_oracle.get("candidate_id"),
                    "global_oracle_primary_nc_depth_ratio": global_oracle_primary,
                    "delta_vs_baseline": delta_vs_baseline,
                    "delta_vs_lex": selected_primary - lex_primary,
                    "regret_vs_oracle": selected_primary - oracle_primary,
                    "global_regret_vs_oracle": selected_primary
                    - global_oracle_primary,
                    "selected_is_lex": bool_int(
                        selected.get("candidate_id") == lex.get("candidate_id")
                    ),
                    "selected_is_oracle": bool_int(
                        selected.get("candidate_id") == oracle.get("candidate_id")
                    ),
                    "selected_is_global_oracle": bool_int(
                        selected.get("candidate_id")
                        == global_oracle.get("candidate_id")
                    ),
                    "selected_tcount_over_best": selected_tcount - best_tcount,
                    "selected_tdepth_delta_vs_lex": selected_tdepth
                    - finite(lex.get("tdepth_after")),
                    "best_tcount": coerce_int(best_tcount),
                    "best_mixed_excess_norm": best_mixed,
                    "allowed_mixed_excess_norm": allowed_mixed,
                    "near_tensor_candidate_count": near_tensor_count,
                    "num_candidates_within_tcount_tolerance": len(candidates),
                    "tensor_v3_mixed_excess_norm": coerce_float(
                        selected.get("tensor_v3_mixed_excess_norm")
                    ),
                    "tensor_v3_mixed_auc_greedy_norm": coerce_float(
                        selected.get("tensor_v3_mixed_auc_greedy_norm")
                    ),
                    "tensor_v3_singleton_bridge_count_norm": coerce_float(
                        selected.get("tensor_v3_singleton_bridge_count_norm")
                    ),
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            float("inf") if row.get("slack") is None else float(row["slack"]),
            natural_sort_key(str(row["circuit_id"])),
        ),
    )


def summarize_by_slack(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    slacks = sorted(
        {
            float(row["slack"])
            for row in rows
            if row.get("selection_status") == "ok" and row.get("slack") is not None
        }
    )
    for slack in slacks:
        subset = [
            row
            for row in rows
            if row.get("selection_status") == "ok" and float(row["slack"]) == slack
        ]
        if not subset:
            continue
        baseline_relations = [relation(row.get("delta_vs_baseline")) for row in subset]
        lex_relations = [relation(row.get("delta_vs_lex")) for row in subset]
        regrets = [float(row["regret_vs_oracle"]) for row in subset]
        global_regrets = [
            float(row["global_regret_vs_oracle"]) for row in subset
        ]
        oracle_hits = sum(int(row.get("selected_is_oracle") or 0) for row in subset)
        global_oracle_hits = sum(
            int(row.get("selected_is_global_oracle") or 0) for row in subset
        )
        changed_from_lex = sum(
            1 - int(row.get("selected_is_lex") or 0) for row in subset
        )
        tcount_overhead = sum(
            float(row.get("selected_tcount_over_best") or 0.0) > 1e-9
            for row in subset
        )
        summary_rows.append(
            {
                "slack": slack,
                "num_circuits": len(subset),
                "better_vs_baseline": baseline_relations.count("better"),
                "tie_vs_baseline": baseline_relations.count("tie"),
                "worse_vs_baseline": baseline_relations.count("worse"),
                "better_vs_lex": lex_relations.count("better"),
                "tie_vs_lex": lex_relations.count("tie"),
                "worse_vs_lex": lex_relations.count("worse"),
                "oracle_hits": oracle_hits,
                "global_oracle_hits": global_oracle_hits,
                "changed_from_lex": changed_from_lex,
                "tcount_overhead_count": tcount_overhead,
                "mean_regret_vs_oracle": sum(regrets) / len(regrets),
                "max_regret_vs_oracle": max(regrets),
                "mean_global_regret_vs_oracle": sum(global_regrets)
                / len(global_regrets),
                "max_global_regret_vs_oracle": max(global_regrets),
                "classification": classify_slack(subset),
            }
        )
    return summary_rows


def classify_slack(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "missing"
    oracle_hits = sum(int(row.get("selected_is_oracle") or 0) for row in rows)
    worse_vs_baseline = sum(
        relation(row.get("delta_vs_baseline")) == "worse" for row in rows
    )
    if oracle_hits == len(rows) and worse_vs_baseline == 0:
        return "oracle-plateau"
    if worse_vs_baseline == 0:
        return "baseline-safe"
    return "mixed"


def candidate_intervals(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    by_circuit: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("selection_status") == "ok":
            by_circuit.setdefault(str(row["circuit_id"]), []).append(row)
    for circuit_id, circuit_rows in sorted(
        by_circuit.items(), key=lambda item: natural_sort_key(item[0])
    ):
        sorted_rows = sorted(circuit_rows, key=lambda row: float(row["slack"]))
        current_id = None
        start_slack = None
        end_slack = None
        representative: dict[str, Any] | None = None
        for row in sorted_rows:
            selected_id = row.get("selected_candidate_id")
            if selected_id != current_id:
                if current_id is not None and representative is not None:
                    output_rows.append(
                        {
                            "circuit_id": circuit_id,
                            "slack_min": start_slack,
                            "slack_max": end_slack,
                            "selected_candidate_id": current_id,
                            "selected_primary_nc_depth_ratio": representative.get(
                                "selected_primary_nc_depth_ratio"
                            ),
                            "selected_is_oracle": representative.get(
                                "selected_is_oracle"
                            ),
                            "selected_is_lex": representative.get("selected_is_lex"),
                        }
                    )
                current_id = selected_id
                start_slack = row.get("slack")
                representative = row
            end_slack = row.get("slack")
        if current_id is not None and representative is not None:
            output_rows.append(
                {
                    "circuit_id": circuit_id,
                    "slack_min": start_slack,
                    "slack_max": end_slack,
                    "selected_candidate_id": current_id,
                    "selected_primary_nc_depth_ratio": representative.get(
                        "selected_primary_nc_depth_ratio"
                    ),
                    "selected_is_oracle": representative.get("selected_is_oracle"),
                    "selected_is_lex": representative.get("selected_is_lex"),
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
    summary_rows: list[dict[str, Any]],
    interval_rows: list[dict[str, Any]],
    *,
    output_path: Path,
    csv_path: Path,
    summary_csv_path: Path,
    figure_path: Path | None,
) -> Path:
    plateau_slacks = [
        row["slack"]
        for row in summary_rows
        if row.get("classification") == "oracle-plateau"
    ]
    safe_slacks = [
        row["slack"]
        for row in summary_rows
        if row.get("classification") in {"oracle-plateau", "baseline-safe"}
    ]
    non_worse_rows = [
        row
        for row in summary_rows
        if int(row.get("worse_vs_baseline") or 0) == 0
    ]
    best_max_regret = (
        min(float(row["max_regret_vs_oracle"]) for row in non_worse_rows)
        if non_worse_rows
        else None
    )
    best_regret_slacks = [
        row["slack"]
        for row in non_worse_rows
        if best_max_regret is not None
        and abs(float(row["max_regret_vs_oracle"]) - best_max_regret) <= 1e-12
    ]
    summary_lines = [
        "| slack | class | vs baseline B/T/W | vs lex B/T/W | constrained oracle | global oracle | changed from lex | max constrained regret | max global regret | T overhead |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        summary_lines.append(
            "| "
            + " | ".join(
                [
                    fmt(row["slack"]),
                    f"`{row['classification']}`",
                    (
                        f"{row['better_vs_baseline']}/"
                        f"{row['tie_vs_baseline']}/"
                        f"{row['worse_vs_baseline']}"
                    ),
                    (
                        f"{row['better_vs_lex']}/"
                        f"{row['tie_vs_lex']}/"
                        f"{row['worse_vs_lex']}"
                    ),
                    f"{row['oracle_hits']}/{row['num_circuits']}",
                    f"{row['global_oracle_hits']}/{row['num_circuits']}",
                    str(row["changed_from_lex"]),
                    fmt(row["max_regret_vs_oracle"]),
                    fmt(row["max_global_regret_vs_oracle"]),
                    str(row["tcount_overhead_count"]),
                ]
            )
            + " |"
        )

    interval_lines = [
        "| circuit | slack interval | selected | primary | oracle? | lex? |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in interval_rows:
        interval_lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['circuit_id']}`",
                    f"{fmt(row['slack_min'])}-{fmt(row['slack_max'])}",
                    f"`{row['selected_candidate_id']}`",
                    fmt(row["selected_primary_nc_depth_ratio"]),
                    str(row["selected_is_oracle"]),
                    str(row["selected_is_lex"]),
                ]
            )
            + " |"
        )

    qft_rows = [
        row
        for row in rows
        if row.get("selection_status") == "ok" and row.get("circuit_id") == "qft_4"
    ]
    qft_lines = [
        "| slack | selected | primary | regret | mixed excess | near tensor candidates |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in sorted(qft_rows, key=lambda item: float(item["slack"])):
        qft_lines.append(
            "| "
            + " | ".join(
                [
                    fmt(row["slack"]),
                    f"`{row['selected_candidate_id']}`",
                    fmt(row["selected_primary_nc_depth_ratio"]),
                    fmt(row["regret_vs_oracle"]),
                    fmt(row["tensor_v3_mixed_excess_norm"]),
                    str(row["near_tensor_candidate_count"]),
                ]
            )
            + " |"
        )

    text = [
        "# Tensor-v3 Phase-Slack Sensitivity",
        "",
        f"- Per-circuit CSV: `{csv_path}`.",
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
                f"- Oracle plateau slacks: {', '.join(fmt(value) for value in plateau_slacks) or 'none'}."
            ),
            (
                f"- Baseline-safe slacks: {', '.join(fmt(value) for value in safe_slacks) or 'none'}."
            ),
            (
                "- Lowest max-regret slacks among non-worse settings: "
                f"{', '.join(fmt(value) for value in best_regret_slacks) or 'none'}."
            ),
            (
                "- Interpretation: a wide oracle plateau means the phase-slack rule is not tied "
                "to a single hand-picked threshold; a narrow or absent plateau means the next "
                "step should be broader frontier generation before changing the AlphaQuantum core."
            ),
        (
            "- Baseline convention: use `alphatensor_public` from Entrega when available; "
            "otherwise use the tcount-only selector on the same candidate frontier."
        ),
        (
            "- Constrained oracle means the best audited candidate after applying the configured "
            "T-count tolerance. Global oracle means the best audited candidate on the full frontier."
        ),
        (
            "- T overhead counts selections above the minimum T-count on the frontier, but still "
            "inside the configured T-count tolerance because the selector never sees candidates "
            "outside that filter."
        ),
        "",
        "## Slack Summary",
            "",
            *summary_lines,
            "",
            "## Selection Intervals",
            "",
            *interval_lines,
            "",
            "## qft_4 Threshold Detail",
            "",
            *qft_lines,
            "",
        ]
    )
    ensure_dir(output_path.parent)
    output_path.write_text("\n".join(text), encoding="utf-8")
    return output_path


def write_figure(rows: list[dict[str, Any]], figure_path: Path) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    ok_rows = [row for row in rows if row.get("selection_status") == "ok"]
    if not ok_rows:
        return None
    circuits = sorted({str(row["circuit_id"]) for row in ok_rows}, key=natural_sort_key)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for circuit_id in circuits:
        circuit_rows = sorted(
            [row for row in ok_rows if row["circuit_id"] == circuit_id],
            key=lambda item: float(item["slack"]),
        )
        ax.plot(
            [float(row["slack"]) for row in circuit_rows],
            [float(row["regret_vs_oracle"]) for row in circuit_rows],
            marker="o",
            linewidth=1.5,
            label=circuit_id,
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("tensor_v3_mixed_slack")
    ax.set_ylabel("primary regret vs audited oracle")
    ax.set_title("Phase-slack sensitivity")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    ensure_dir(figure_path.parent)
    fig.savefig(figure_path)
    plt.close(fig)
    return figure_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep tensor-v3 phase-slack thresholds on an audited frontier."
    )
    parser.add_argument(
        "--candidate-frontier-csv",
        type=Path,
        default=DEFAULT_RESULTS_ROOT
        / "public_resynth_tensor_v3_phase_slack"
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
        default=DEFAULT_CSV_ROOT / "tensor_v3_phase_slack_sensitivity.csv",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=DEFAULT_CSV_ROOT / "tensor_v3_phase_slack_sensitivity_summary.csv",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_CSV_ROOT / "tensor_v3_phase_slack_sensitivity.json",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORTS_ROOT / "tensor_v3_phase_slack_sensitivity.md",
    )
    parser.add_argument(
        "--figure-path",
        type=Path,
        default=DEFAULT_FIGURES_ROOT / "tensor_v3_phase_slack_sensitivity.png",
    )
    parser.add_argument(
        "--slack-values",
        default=",".join(str(value) for value in DEFAULT_SLACK_VALUES),
    )
    parser.add_argument("--tcount-tolerance", type=float, default=0.12)
    parser.add_argument("--circuit-id", action="append", dest="circuit_ids", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    circuit_ids = tuple(args.circuit_ids) if args.circuit_ids else DEFAULT_CIRCUIT_IDS
    slack_values = parse_float_list(args.slack_values)
    rows = build_sensitivity_rows(
        load_csv_rows(args.candidate_frontier_csv),
        load_csv_rows(args.entrega_csv),
        circuit_ids=circuit_ids,
        slack_values=slack_values,
        tcount_tolerance=args.tcount_tolerance,
    )
    summary_rows = summarize_by_slack(rows)
    interval_rows = candidate_intervals(rows)
    write_csv_rows(rows, args.output_csv)
    write_csv_rows(summary_rows, args.summary_csv)
    figure_path = write_figure(rows, args.figure_path)
    args.output_json.write_text(
        json.dumps(
            {
                "candidate_frontier_csv": str(args.candidate_frontier_csv),
                "entrega_csv": str(args.entrega_csv),
                "output_csv": str(args.output_csv),
                "summary_csv": str(args.summary_csv),
                "report_path": str(args.report_path),
                "figure_path": str(figure_path) if figure_path else None,
                "tcount_tolerance": args.tcount_tolerance,
                "slack_values": list(slack_values),
                "circuit_ids": list(circuit_ids),
                "num_rows": len(rows),
                "num_summary_rows": len(summary_rows),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_report(
        rows,
        summary_rows,
        interval_rows,
        output_path=args.report_path,
        csv_path=args.output_csv,
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
