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
from scripts._analysis_common import DEFAULT_FIGURES_ROOT
from scripts._analysis_common import DEFAULT_REPORTS_ROOT
from scripts._analysis_common import DEFAULT_RESULTS_ROOT
from scripts._analysis_common import ensure_dir
from scripts._analysis_common import natural_sort_key
from scripts._analysis_common import write_csv_rows
from scripts.compare_tensor_v3_profiles import coerce_float
from scripts.compare_tensor_v3_profiles import coerce_int
from scripts.compare_tensor_v3_profiles import finite
from scripts.compare_tensor_v3_profiles import frontier_rows_by_circuit
from scripts.compare_tensor_v3_profiles import guarded_accepts
from scripts.compare_tensor_v3_profiles import load_csv_rows
from scripts.compare_tensor_v3_profiles import ok_summary_by_circuit
from scripts.compare_tensor_v3_profiles import primary_key
from scripts.compare_tensor_v3_profiles import relation


DEFAULT_QASM_DEPTH_GAINS = (0.0, 0.025, 0.05, 0.075, 0.085, 0.10, 0.125, 0.15, 0.20, 0.30)
DEFAULT_MIXED_DROP_FRACTIONS = (0.0, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


def parse_float_list(value: str) -> tuple[float, ...]:
    values = []
    for part in value.split(","):
        part = part.strip()
        if part:
            values.append(float(part))
    return tuple(sorted(dict.fromkeys(values)))


def candidate_id(circuit_id: str, row: dict[str, str]) -> str:
    return f"{circuit_id}:combo{row.get('combo_index')}"


def build_guard_surface_rows(
    *,
    conservative_rows: list[dict[str, str]],
    aggressive_rows: list[dict[str, str]],
    frontier_rows: list[dict[str, str]],
    qasm_depth_gains: tuple[float, ...],
    mixed_drop_fractions: tuple[float, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    conservative_by_circuit = ok_summary_by_circuit(conservative_rows)
    aggressive_by_circuit = ok_summary_by_circuit(aggressive_rows)
    frontier_by_circuit = frontier_rows_by_circuit(frontier_rows)
    circuit_ids = sorted(
        set(conservative_by_circuit) | set(aggressive_by_circuit),
        key=natural_sort_key,
    )

    detail_rows: list[dict[str, Any]] = []
    for qasm_depth_gain in qasm_depth_gains:
        for mixed_drop_fraction in mixed_drop_fractions:
            for circuit_id in circuit_ids:
                conservative = conservative_by_circuit.get(circuit_id)
                aggressive = aggressive_by_circuit.get(circuit_id)
                if conservative is None or aggressive is None:
                    detail_rows.append(
                        {
                            "qasm_depth_gain": qasm_depth_gain,
                            "mixed_drop_fraction": mixed_drop_fraction,
                            "circuit_id": circuit_id,
                            "status": "missing-profile",
                            "has_conservative": int(conservative is not None),
                            "has_aggressive": int(aggressive is not None),
                        }
                    )
                    continue

                conservative_primary = coerce_float(conservative.get("primary_nc_depth_ratio"))
                aggressive_primary = coerce_float(aggressive.get("primary_nc_depth_ratio"))
                conservative_qasm = coerce_float(conservative.get("qasm_depth_ratio"))
                aggressive_qasm = coerce_float(aggressive.get("qasm_depth_ratio"))
                conservative_mixed = coerce_float(
                    conservative.get("tensor_v3_mixed_excess_norm")
                )
                aggressive_mixed = coerce_float(
                    aggressive.get("tensor_v3_mixed_excess_norm")
                )
                qasm_gain = (
                    None
                    if conservative_qasm is None or aggressive_qasm is None
                    else conservative_qasm - aggressive_qasm
                )
                mixed_drop = (
                    None
                    if conservative_mixed is None or aggressive_mixed is None
                    else conservative_mixed - aggressive_mixed
                )
                mixed_drop_ratio = (
                    None
                    if mixed_drop is None or conservative_mixed is None
                    else mixed_drop / max(abs(conservative_mixed), 1e-12)
                )
                use_aggressive = guarded_accepts(
                    conservative_qasm_depth=conservative_qasm,
                    aggressive_qasm_depth=aggressive_qasm,
                    conservative_mixed=conservative_mixed,
                    aggressive_mixed=aggressive_mixed,
                    qasm_depth_gain=qasm_depth_gain,
                    mixed_drop_fraction=mixed_drop_fraction,
                )
                selected = aggressive if use_aggressive else conservative
                selected_primary = coerce_float(selected.get("primary_nc_depth_ratio"))
                selected_tcount = coerce_int(selected.get("tcount_after_selection"))
                conservative_tcount = coerce_int(conservative.get("tcount_after_selection"))
                global_oracle = (
                    min(frontier_by_circuit[circuit_id], key=primary_key)
                    if frontier_by_circuit.get(circuit_id)
                    else None
                )
                global_oracle_primary = (
                    coerce_float(global_oracle.get("primary_nc_depth_ratio"))
                    if global_oracle
                    else None
                )
                selected_id = candidate_id(circuit_id, selected)
                detail_rows.append(
                    {
                        "qasm_depth_gain": qasm_depth_gain,
                        "mixed_drop_fraction": mixed_drop_fraction,
                        "circuit_id": circuit_id,
                        "status": "ok",
                        "selected_candidate_id": selected_id,
                        "selected_source_profile": selected.get("tensor_v3_profile"),
                        "guarded_uses_aggressive": int(use_aggressive),
                        "conservative_candidate_id": candidate_id(circuit_id, conservative),
                        "aggressive_candidate_id": candidate_id(circuit_id, aggressive),
                        "global_oracle_candidate_id": (
                            global_oracle.get("candidate_id") if global_oracle else None
                        ),
                        "conservative_primary_nc_depth_ratio": conservative_primary,
                        "aggressive_primary_nc_depth_ratio": aggressive_primary,
                        "selected_primary_nc_depth_ratio": selected_primary,
                        "global_oracle_primary_nc_depth_ratio": global_oracle_primary,
                        "selected_delta_primary_vs_conservative": (
                            None
                            if conservative_primary is None or selected_primary is None
                            else selected_primary - conservative_primary
                        ),
                        "selected_relation_vs_conservative": relation(
                            None
                            if conservative_primary is None or selected_primary is None
                            else selected_primary - conservative_primary
                        ),
                        "selected_global_oracle_regret": (
                            None
                            if selected_primary is None or global_oracle_primary is None
                            else selected_primary - global_oracle_primary
                        ),
                        "selected_is_global_oracle": int(
                            global_oracle is not None
                            and selected_id == global_oracle.get("candidate_id")
                        ),
                        "conservative_tcount": conservative_tcount,
                        "aggressive_tcount": coerce_int(
                            aggressive.get("tcount_after_selection")
                        ),
                        "selected_tcount": selected_tcount,
                        "selected_delta_tcount_vs_conservative": (
                            None
                            if selected_tcount is None or conservative_tcount is None
                            else selected_tcount - conservative_tcount
                        ),
                        "qasm_depth_gain_aggressive_vs_conservative": qasm_gain,
                        "mixed_drop_aggressive_vs_conservative": mixed_drop,
                        "mixed_drop_fraction_aggressive_vs_conservative": mixed_drop_ratio,
                    }
                )

    summary_rows: list[dict[str, Any]] = []
    for qasm_depth_gain in qasm_depth_gains:
        for mixed_drop_fraction in mixed_drop_fractions:
            subset = [
                row
                for row in detail_rows
                if row.get("status") == "ok"
                and abs(float(row["qasm_depth_gain"]) - qasm_depth_gain) <= 1e-12
                and abs(float(row["mixed_drop_fraction"]) - mixed_drop_fraction) <= 1e-12
            ]
            if not subset:
                continue
            relations = [str(row["selected_relation_vs_conservative"]) for row in subset]
            regrets = [
                float(row["selected_global_oracle_regret"])
                for row in subset
                if row.get("selected_global_oracle_regret") not in (None, "")
            ]
            tcount_deltas = [
                int(row.get("selected_delta_tcount_vs_conservative") or 0)
                for row in subset
            ]
            accepted_rows = [
                row for row in subset if int(row.get("guarded_uses_aggressive") or 0)
            ]
            summary_rows.append(
                {
                    "qasm_depth_gain": qasm_depth_gain,
                    "mixed_drop_fraction": mixed_drop_fraction,
                    "num_circuits": len(subset),
                    "accepted_aggressive_count": len(accepted_rows),
                    "better_vs_conservative": relations.count("better"),
                    "tie_vs_conservative": relations.count("tie"),
                    "worse_vs_conservative": relations.count("worse"),
                    "global_oracle_hits": sum(
                        int(row.get("selected_is_global_oracle") or 0)
                        for row in subset
                    ),
                    "mean_global_oracle_regret": (
                        sum(regrets) / len(regrets) if regrets else None
                    ),
                    "max_global_oracle_regret": max(regrets) if regrets else None,
                    "tcount_overhead_count": sum(delta > 0 for delta in tcount_deltas),
                    "tcount_overhead_total": sum(
                        delta for delta in tcount_deltas if delta > 0
                    ),
                    "accepted_circuits": ",".join(
                        str(row["circuit_id"]) for row in accepted_rows
                    ),
                    "regressed_circuits": ",".join(
                        str(row["circuit_id"])
                        for row in subset
                        if row["selected_relation_vs_conservative"] == "worse"
                    ),
                    "improved_circuits": ",".join(
                        str(row["circuit_id"])
                        for row in subset
                        if row["selected_relation_vs_conservative"] == "better"
                    ),
                }
            )

    detail_rows.sort(
        key=lambda row: (
            float(row.get("qasm_depth_gain") or 0.0),
            float(row.get("mixed_drop_fraction") or 0.0),
            natural_sort_key(str(row.get("circuit_id") or "")),
        )
    )
    summary_rows.sort(
        key=lambda row: (
            float(row.get("qasm_depth_gain") or 0.0),
            float(row.get("mixed_drop_fraction") or 0.0),
        )
    )
    return detail_rows, summary_rows


def best_non_worse_guard_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = [
        row for row in summary_rows if int(row.get("worse_vs_conservative") or 0) == 0
    ]
    if not candidates:
        return []
    best_better = max(int(row.get("better_vs_conservative") or 0) for row in candidates)
    candidates = [
        row
        for row in candidates
        if int(row.get("better_vs_conservative") or 0) == best_better
    ]
    best_hits = max(int(row.get("global_oracle_hits") or 0) for row in candidates)
    candidates = [
        row for row in candidates if int(row.get("global_oracle_hits") or 0) == best_hits
    ]
    best_regret = min(finite(row.get("max_global_oracle_regret")) for row in candidates)
    candidates = [
        row
        for row in candidates
        if abs(finite(row.get("max_global_oracle_regret")) - best_regret) <= 1e-12
    ]
    best_overhead = min(int(row.get("tcount_overhead_total") or 0) for row in candidates)
    return [
        row
        for row in candidates
        if int(row.get("tcount_overhead_total") or 0) == best_overhead
    ]


def robust_qasm_guard_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_qasm: dict[float, list[dict[str, Any]]] = {}
    for row in summary_rows:
        by_qasm.setdefault(float(row["qasm_depth_gain"]), []).append(row)
    robust_rows: list[dict[str, Any]] = []
    for qasm_depth_gain, rows in by_qasm.items():
        if any(int(row.get("worse_vs_conservative") or 0) > 0 for row in rows):
            continue
        best_row = max(
            rows,
            key=lambda row: (
                int(row.get("better_vs_conservative") or 0),
                int(row.get("global_oracle_hits") or 0),
                -finite(row.get("max_global_oracle_regret")),
                -int(row.get("tcount_overhead_total") or 0),
            ),
        )
        robust_rows.append({**best_row, "robust_qasm_depth_gain": qasm_depth_gain})
    robust_rows.sort(key=lambda row: float(row["robust_qasm_depth_gain"]))
    return robust_rows


def fmt(value: Any, *, integer: bool = False) -> str:
    if value in (None, ""):
        return ""
    if integer:
        return str(int(value))
    return f"{float(value):.3f}"


def write_figure(summary_rows: list[dict[str, Any]], figure_path: Path) -> Path | None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return None

    if not summary_rows:
        return None
    x_values = sorted({float(row["qasm_depth_gain"]) for row in summary_rows})
    y_values = sorted({float(row["mixed_drop_fraction"]) for row in summary_rows})
    grid = np.full((len(y_values), len(x_values)), np.nan)
    labels: dict[tuple[int, int], str] = {}
    for row in summary_rows:
        x_idx = x_values.index(float(row["qasm_depth_gain"]))
        y_idx = y_values.index(float(row["mixed_drop_fraction"]))
        score = int(row["better_vs_conservative"]) - int(row["worse_vs_conservative"])
        grid[y_idx, x_idx] = score
        labels[(y_idx, x_idx)] = (
            f"{row['better_vs_conservative']}/"
            f"{row['tie_vs_conservative']}/"
            f"{row['worse_vs_conservative']}"
        )

    fig, ax = plt.subplots(figsize=(9, 4.8))
    image = ax.imshow(grid, cmap="RdYlGn", aspect="auto", vmin=-2, vmax=2)
    ax.set_xticks(range(len(x_values)))
    ax.set_xticklabels([fmt(value) for value in x_values], rotation=45, ha="right")
    ax.set_yticks(range(len(y_values)))
    ax.set_yticklabels([fmt(value) for value in y_values])
    ax.set_xlabel("minimum normalized QASM depth gain")
    ax.set_ylabel("minimum mixed-excess drop fraction")
    ax.set_title("Guarded aggressive profile calibration")
    for (y_idx, x_idx), label in labels.items():
        ax.text(x_idx, y_idx, label, ha="center", va="center", fontsize=7)
    cbar = fig.colorbar(image, ax=ax, shrink=0.85)
    cbar.set_label("better minus worse vs conservative")
    fig.tight_layout()
    ensure_dir(figure_path.parent)
    fig.savefig(figure_path)
    plt.close(fig)
    return figure_path


def write_report(
    detail_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    *,
    report_path: Path,
    detail_csv_path: Path,
    summary_csv_path: Path,
    figure_path: Path | None,
) -> Path:
    best_rows = best_non_worse_guard_rows(summary_rows)
    robust_rows = robust_qasm_guard_rows(summary_rows)
    best_text = "none"
    if best_rows:
        qasm_values = [float(row["qasm_depth_gain"]) for row in best_rows]
        mixed_values = [float(row["mixed_drop_fraction"]) for row in best_rows]
        best_text = (
            f"{len(best_rows)} settings "
            f"(qasm_gain {fmt(min(qasm_values))}-{fmt(max(qasm_values))}, "
            f"mixed_drop {fmt(min(mixed_values))}-{fmt(max(mixed_values))})"
        )
    robust_text = "none"
    if robust_rows:
        first_robust = robust_rows[0]
        robust_text = (
            f"minimum qasm_gain={fmt(first_robust['robust_qasm_depth_gain'])} "
            f"with B/T/W={first_robust['better_vs_conservative']}/"
            f"{first_robust['tie_vs_conservative']}/"
            f"{first_robust['worse_vs_conservative']} and "
            f"{first_robust['global_oracle_hits']}/{first_robust['num_circuits']} "
            "global-oracle hits"
        )
    non_worse_rows = [
        row for row in summary_rows if int(row.get("worse_vs_conservative") or 0) == 0
    ]
    safe_qasm_min = (
        min(float(row["qasm_depth_gain"]) for row in non_worse_rows)
        if non_worse_rows
        else None
    )
    safe_qasm_max = (
        max(float(row["qasm_depth_gain"]) for row in non_worse_rows)
        if non_worse_rows
        else None
    )

    summary_lines = [
        "| QASM gain | mixed drop | accepted | B/T/W vs conservative | global oracle | max global regret | T overhead | accepted circuits | regressed circuits |",
        "|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in summary_rows:
        summary_lines.append(
            "| "
            + " | ".join(
                [
                    fmt(row["qasm_depth_gain"]),
                    fmt(row["mixed_drop_fraction"]),
                    str(row["accepted_aggressive_count"]),
                    (
                        f"{row['better_vs_conservative']}/"
                        f"{row['tie_vs_conservative']}/"
                        f"{row['worse_vs_conservative']}"
                    ),
                    f"{row['global_oracle_hits']}/{row['num_circuits']}",
                    fmt(row["max_global_oracle_regret"]),
                    str(row["tcount_overhead_total"]),
                    f"`{row['accepted_circuits']}`" if row["accepted_circuits"] else "",
                    f"`{row['regressed_circuits']}`" if row["regressed_circuits"] else "",
                ]
            )
            + " |"
        )

    threshold_rows = [
        row
        for row in detail_rows
        if row.get("status") == "ok"
        and abs(float(row["mixed_drop_fraction"]) - 0.0) <= 1e-12
    ]
    by_circuit: dict[str, list[dict[str, Any]]] = {}
    for row in threshold_rows:
        if int(row.get("guarded_uses_aggressive") or 0):
            by_circuit.setdefault(str(row["circuit_id"]), []).append(row)
    threshold_lines = [
        "| circuit | QASM gain range accepting aggressive | relation | aggressive primary | conservative primary | observed QASM gain | observed mixed drop |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for circuit_id, rows in sorted(by_circuit.items(), key=lambda item: natural_sort_key(item[0])):
        qasm_values = [float(row["qasm_depth_gain"]) for row in rows]
        representative = rows[0]
        threshold_lines.append(
            "| "
            + " | ".join(
                [
                    f"`{circuit_id}`",
                    f"{fmt(min(qasm_values))}-{fmt(max(qasm_values))}",
                    f"`{representative['selected_relation_vs_conservative']}`",
                    fmt(representative["aggressive_primary_nc_depth_ratio"]),
                    fmt(representative["conservative_primary_nc_depth_ratio"]),
                    fmt(representative["qasm_depth_gain_aggressive_vs_conservative"]),
                    fmt(representative["mixed_drop_fraction_aggressive_vs_conservative"]),
                ]
            )
            + " |"
        )

    text = [
        "# Tensor-v3 Guard Surface",
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
                "- Best non-worse guard settings by better-count, global-oracle hits, "
                f"max regret, and T overhead: {best_text}."
            ),
            f"- Robust QASM-only guard plateau: {robust_text}.",
            (
                "- Non-worse QASM gain range across the sweep: "
                f"{fmt(safe_qasm_min)}-{fmt(safe_qasm_max)}."
            ),
            (
                "- Practical reading: `qft_4` has a large tensor mixed-drop but only a small "
                "normalized QASM-depth gain, and it is exactly the observed regression. This "
                "makes the QASM-depth gate the more reliable safety condition than a nearly "
                "complete mixed-drop threshold."
            ),
            (
                "- Interpretation: the guard is doing real work when the aggressive profile "
                "contains both improvements and regressions. A stable non-worse plateau means "
                "the next selector can expose a tunable splitting regime instead of relying on "
                "one hand-picked threshold."
            ),
            "",
            "## Summary",
            "",
            *summary_lines,
            "",
            "## Aggressive Acceptance Windows",
            "",
            *threshold_lines,
            "",
        ]
    )
    ensure_dir(report_path.parent)
    report_path.write_text("\n".join(text), encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep guarded-aggressive thresholds for tensor-v3 profile selection."
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
        default=DEFAULT_CSV_ROOT / "tensor_v3_guard_surface.csv",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=DEFAULT_CSV_ROOT / "tensor_v3_guard_surface_summary.csv",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_CSV_ROOT / "tensor_v3_guard_surface.json",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORTS_ROOT / "tensor_v3_guard_surface.md",
    )
    parser.add_argument(
        "--figure-path",
        type=Path,
        default=DEFAULT_FIGURES_ROOT / "tensor_v3_guard_surface.png",
    )
    parser.add_argument(
        "--qasm-depth-gains",
        default=",".join(str(value) for value in DEFAULT_QASM_DEPTH_GAINS),
    )
    parser.add_argument(
        "--mixed-drop-fractions",
        default=",".join(str(value) for value in DEFAULT_MIXED_DROP_FRACTIONS),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    qasm_depth_gains = parse_float_list(args.qasm_depth_gains)
    mixed_drop_fractions = parse_float_list(args.mixed_drop_fractions)
    detail_rows, summary_rows = build_guard_surface_rows(
        conservative_rows=load_csv_rows(args.conservative_summary_csv),
        aggressive_rows=load_csv_rows(args.aggressive_summary_csv),
        frontier_rows=load_csv_rows(args.candidate_frontier_csv),
        qasm_depth_gains=qasm_depth_gains,
        mixed_drop_fractions=mixed_drop_fractions,
    )
    write_csv_rows(detail_rows, args.output_csv)
    write_csv_rows(summary_rows, args.summary_csv)
    figure_path = write_figure(summary_rows, args.figure_path)
    args.output_json.write_text(
        json.dumps(
            {
                "conservative_summary_csv": str(args.conservative_summary_csv),
                "aggressive_summary_csv": str(args.aggressive_summary_csv),
                "candidate_frontier_csv": str(args.candidate_frontier_csv),
                "output_csv": str(args.output_csv),
                "summary_csv": str(args.summary_csv),
                "report_path": str(args.report_path),
                "figure_path": str(figure_path) if figure_path else None,
                "qasm_depth_gains": list(qasm_depth_gains),
                "mixed_drop_fractions": list(mixed_drop_fractions),
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
        report_path=args.report_path,
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
