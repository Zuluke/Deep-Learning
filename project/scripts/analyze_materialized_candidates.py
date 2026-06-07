from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "results" / "csv" / "materialized_candidate_analysis.csv"
DEFAULT_REPORT = PROJECT_ROOT / "results" / "reports" / "materialized_candidate_analysis.md"
DEFAULT_FIGURE = PROJECT_ROOT / "results" / "figures" / "materialized_candidate_analysis.png"


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed


def read_summary(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    structural = data.get("external_structural_metrics") or {}
    assembled = data.get("assembled_metrics") or {}
    return {
        "summary_path": str(path),
        "candidate_dir": str(path.parent),
        "target": data.get("target"),
        "candidate_kind": data.get("candidate_kind"),
        "synthesis": data.get("synthesis", "circuit_to_tensor_resynth"),
        "factor_order": data.get("factor_order", ""),
        "target_strategy": data.get("target_strategy", ""),
        "status": data.get("status"),
        "reconstruction_ok": data.get("reconstruction_ok"),
        "num_forward_cnots": parse_float(data.get("num_forward_cnots")),
        "num_total_cnots": parse_float(data.get("num_total_cnots")),
        "num_shared_forward_cnots": parse_float(data.get("num_shared_forward_cnots")),
        "num_shared_total_cnots": parse_float(data.get("num_shared_total_cnots")),
        "tcount": parse_float(assembled.get("tcount", assembled.get("t_count"))),
        "tdepth": parse_float(assembled.get("tdepth")),
        "qasm_depth": parse_float(assembled.get("normalized_qasm_depth")),
        "primary_nc_depth_ratio": parse_float(structural.get("primary_nc_depth_ratio")),
        "alphaq_nc_core_depth_ratio": parse_float(
            structural.get("alphaq_nc_core_depth_ratio")
        ),
        "alphaq_dependency_core_depth_ratio": parse_float(
            structural.get("alphaq_dependency_core_depth_ratio")
        ),
        "qasm_depth_ratio": parse_float(structural.get("qasm_depth_ratio")),
        "zx_total_depth_ratio": parse_float(structural.get("zx_total_depth_ratio")),
        "tcount_ratio": parse_float(structural.get("tcount_ratio")),
        "structural_cost": parse_float(structural.get("structural_cost")),
        "structural_target_status": structural.get("structural_target_status"),
        "verification_status": "",
        "verification_proof_path": "",
    }


def read_verification_summaries(paths: list[Path]) -> dict[tuple[str, str], dict[str, Any]]:
    by_candidate: dict[tuple[str, str], dict[str, Any]] = {}
    for path in paths:
        rows = json.loads(path.read_text(encoding="utf-8"))
        for row in rows:
            target = str(row.get("target") or "")
            kind = str(row.get("candidate_kind") or "")
            if not target or not kind:
                continue
            normalized = dict(row)
            proof_path = normalized.get("proof_path")
            if normalized.get("verification_status") == "failed" and proof_path:
                proof = Path(proof_path)
                if not proof.is_absolute():
                    proof = PROJECT_ROOT / proof
                if proof.exists() and proof.read_text(encoding="utf-8").startswith("Inconclusive"):
                    normalized["verification_status"] = "inconclusive"
            by_candidate[(target, kind)] = normalized
    return by_candidate


def attach_verification(
    rows: list[dict[str, Any]],
    verification_rows: dict[tuple[str, str], dict[str, Any]],
) -> None:
    for row in rows:
        key = (str(row.get("target") or ""), str(row.get("candidate_kind") or ""))
        verification = verification_rows.get(key)
        if verification is None:
            continue
        row["verification_status"] = verification.get("verification_status", "")
        row["verification_proof_path"] = verification.get("proof_path", "")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "target",
        "candidate_kind",
        "synthesis",
        "factor_order",
        "target_strategy",
        "status",
        "reconstruction_ok",
        "tcount",
        "tdepth",
        "qasm_depth",
        "tcount_ratio",
        "primary_nc_depth_ratio",
        "alphaq_nc_core_depth_ratio",
        "alphaq_dependency_core_depth_ratio",
        "qasm_depth_ratio",
        "zx_total_depth_ratio",
        "structural_cost",
        "structural_target_status",
        "verification_status",
        "verification_proof_path",
        "candidate_dir",
        "summary_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict[str, Any]], output_csv: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    formal_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("verification_status") or "unverified")
        formal_counts[status] = formal_counts.get(status, 0) + 1
    structural_wins = [
        row
        for row in rows
        if row.get("primary_nc_depth_ratio") is not None
        and float(row["primary_nc_depth_ratio"]) < 1.0
    ]
    t_wins = [
        row for row in rows
        if row.get("tcount_ratio") is not None and float(row["tcount_ratio"]) < 1.0
    ]
    qasm_inflated = [
        row
        for row in rows
        if row.get("qasm_depth_ratio") is not None and float(row["qasm_depth_ratio"]) > 1.0
    ]
    lines = [
        "# Materialized candidate analysis",
        "",
        f"CSV: `{output_csv}`.",
        "",
        "## Aggregate behavior",
        "",
        f"- Candidates analyzed: {len(rows)}.",
        f"- Formal status: {format_counts(formal_counts)}.",
        f"- T-count improves in {len(t_wins)}/{len(rows)} candidates.",
        f"- Primary NC depth ratio improves in {len(structural_wins)}/{len(rows)} candidates.",
        f"- QASM depth inflates in {len(qasm_inflated)}/{len(rows)} candidates.",
        "",
        "The primary structural signal is favorable when the ratio is below one. "
        "QASM depth is reported separately because the current shared-parity realization "
        "can buy a smaller non-Clifford core by adding a large Clifford routing layer.",
        "",
        "## Best candidates by target",
        "",
        "| target | best T-count row | best structural row | interpretation |",
        "|---|---:|---:|---|",
    ]
    for target in sorted({str(row["target"]) for row in rows}):
        target_rows = [row for row in rows if str(row["target"]) == target]
        best_t = min(
            target_rows,
            key=lambda row: (
                row["tcount"] if row["tcount"] is not None else float("inf"),
                row["primary_nc_depth_ratio"]
                if row["primary_nc_depth_ratio"] is not None
                else float("inf"),
            ),
        )
        best_structural = min(
            target_rows,
            key=lambda row: (
                row["primary_nc_depth_ratio"]
                if row["primary_nc_depth_ratio"] is not None
                else float("inf"),
                row["tcount"] if row["tcount"] is not None else float("inf"),
            ),
        )
        interpretation = interpret_target_rows(target_rows)
        lines.append(
            "| {target} | {best_t} | {best_structural} | {interpretation} |".format(
                target=target,
                best_t=short_candidate_label(best_t),
                best_structural=short_candidate_label(best_structural),
                interpretation=interpretation,
            )
        )
    lines.extend(
        [
            "",
            "## Candidate table",
            "",
        "| target | formal status | kind | synthesis | factor order | target strategy | T-count | T-ratio | primary NC depth ratio | QASM depth ratio | structural cost |",
        "|---|---|---|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(rows, key=lambda item: (str(item["target"]), item["tcount"] or 1e9)):
        lines.append(
            "| {target} | {verification_status} | {candidate_kind} | {synthesis} | {factor_order} | {target_strategy} | {tcount} | {tcount_ratio} | {primary_nc_depth_ratio} | {qasm_depth_ratio} | {structural_cost} |".format(
                **{
                    key: format_value(value)
                    for key, value in row.items()
                }
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def short_candidate_label(row: dict[str, Any]) -> str:
    return (
        f"{format_value(row.get('candidate_kind'))} "
        f"[{format_value(row.get('synthesis'))}] "
        f"{format_variant(row)} "
        f"(T={format_value(row.get('tcount'))}, "
        f"primary={format_value(row.get('primary_nc_depth_ratio'))})"
    )


def format_variant(row: dict[str, Any]) -> str:
    order = format_value(row.get("factor_order"))
    target = format_value(row.get("target_strategy"))
    if not order and not target:
        return ""
    return f"{order}/{target}"


def interpret_target_rows(rows: list[dict[str, Any]]) -> str:
    improving_t = [
        row for row in rows
        if row.get("tcount_ratio") is not None and float(row["tcount_ratio"]) < 1.0
    ]
    improving_structural = [
        row
        for row in rows
        if row.get("primary_nc_depth_ratio") is not None
        and float(row["primary_nc_depth_ratio"]) < 1.0
    ]
    if improving_t and improving_structural:
        qasm_inflated = any(
            row.get("qasm_depth_ratio") is not None and float(row["qasm_depth_ratio"]) > 1.0
            for row in rows
        )
        if qasm_inflated:
            return "improves T-count and structural target, with QASM depth inflation"
        return "improves T-count and structural target"
    if improving_t:
        return "improves T-count, but structural target still worsens"
    if improving_structural:
        return "improves structural target without T-count gain"
    return "no candidate beats either normalized target"


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return "" if value is None else str(value)


def format_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))


def write_plot(path: Path, rows: list[dict[str, Any]]) -> bool:
    plot_rows = [
        row
        for row in rows
        if row.get("tcount") is not None and row.get("primary_nc_depth_ratio") is not None
    ]
    if len(plot_rows) < 2:
        return False
    if len({row["primary_nc_depth_ratio"] for row in plot_rows}) < 2:
        return False

    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [
        "\n".join(
            item
            for item in [
                str(row["target"]).replace("hamming_weight_", "hw "),
                short_plot_label(row),
            ]
            if item
        )
        for row in plot_rows
    ]
    primary_values = [float(row["primary_nc_depth_ratio"]) for row in plot_rows]
    qasm_values = [
        parse_float(row.get("qasm_depth_ratio")) or 0.0
        for row in plot_rows
    ]
    colors = ["#2e7d59" if value < 1.0 else "#9d4f4f" for value in primary_values]
    fig, (ax, qasm_ax) = plt.subplots(
        2,
        1,
        figsize=(max(7.2, 0.8 * len(plot_rows)), 7.0),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.2]},
    )
    x_values = list(range(len(plot_rows)))
    bars = ax.bar(x_values, primary_values, color=colors, width=0.72)
    ax.axhline(1.0, color="#333333", linewidth=1.0, linestyle="--")
    ax.set_ylabel("primary NC depth ratio")
    ax.set_title("Materialized AlphaQ candidates: structural target")
    ax.set_ylim(0, max(primary_values) * 1.18)
    ax.grid(axis="y", alpha=0.25)
    for bar, value, row in zip(bars, primary_values, plot_rows):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(primary_values) * 0.025,
            f"{value:.2f}\nT={format_value(row.get('tcount'))}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    qasm_ax.bar(x_values, qasm_values, color="#5278a8", width=0.72)
    qasm_ax.axhline(1.0, color="#333333", linewidth=1.0, linestyle="--")
    qasm_ax.set_ylabel("QASM depth ratio")
    qasm_ax.set_xticks(x_values)
    qasm_ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    qasm_ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def short_plot_label(row: dict[str, Any]) -> str:
    synthesis = str(row.get("synthesis") or "")
    order = str(row.get("factor_order") or "")
    target = str(row.get("target_strategy") or "")
    if synthesis == "shared_parity_network" and order:
        return f"shared {order}/{target}"
    if synthesis == "shared_parity_network":
        return "shared default"
    kind = str(row.get("candidate_kind") or "")
    if "greedy" in kind:
        return "resynth greedy"
    return "resynth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze materialized AlphaQ candidate summaries.")
    parser.add_argument("summary_paths", nargs="+", type=Path)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--figure-path", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument(
        "--verification-summary",
        action="append",
        type=Path,
        default=[],
        help="Optional verification_summary.json files to annotate candidates.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = [read_summary(path) for path in args.summary_paths]
    attach_verification(rows, read_verification_summaries(args.verification_summary))
    write_csv(args.output_csv, rows)
    write_report(args.report_path, rows, args.output_csv)
    plotted = write_plot(args.figure_path, rows)
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.report_path}")
    if plotted:
        print(f"Wrote {args.figure_path}")
    else:
        print("No figure generated; not enough distinct plotted points.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
