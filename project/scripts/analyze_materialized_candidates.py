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
        "status": data.get("status"),
        "reconstruction_ok": data.get("reconstruction_ok"),
        "tcount": parse_float(assembled.get("tcount")),
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
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "target",
        "candidate_kind",
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
        "candidate_dir",
        "summary_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict[str, Any]], output_csv: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Materialized candidate analysis",
        "",
        f"CSV: `{output_csv}`.",
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
        "| target | kind | T-count | T-ratio | primary NC depth ratio | QASM depth ratio | structural cost |",
        "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(rows, key=lambda item: (str(item["target"]), item["tcount"] or 1e9)):
        lines.append(
            "| {target} | {candidate_kind} | {tcount} | {tcount_ratio} | {primary_nc_depth_ratio} | {qasm_depth_ratio} | {structural_cost} |".format(
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
        f"(T={format_value(row.get('tcount'))}, "
        f"primary={format_value(row.get('primary_nc_depth_ratio'))})"
    )


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


def write_plot(path: Path, rows: list[dict[str, Any]]) -> bool:
    plot_rows = [
        row
        for row in rows
        if row.get("tcount") is not None and row.get("primary_nc_depth_ratio") is not None
    ]
    if len(plot_rows) < 2:
        return False
    if len({(row["tcount"], row["primary_nc_depth_ratio"]) for row in plot_rows}) < 2:
        return False

    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [
        f"{row['target']}\n{row['candidate_kind']}"
        for row in plot_rows
    ]
    x_values = [float(row["tcount"]) for row in plot_rows]
    y_values = [float(row["primary_nc_depth_ratio"]) for row in plot_rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.scatter(x_values, y_values, color="#2f6f8f", s=70)
    for label, x_value, y_value in zip(labels, x_values, y_values):
        ax.annotate(label, (x_value, y_value), textcoords="offset points", xytext=(6, 5), fontsize=8)
    ax.set_xlabel("T-count")
    ax.set_ylabel("primary NC depth ratio")
    ax.set_title("Materialized AlphaQ candidates")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze materialized AlphaQ candidate summaries.")
    parser.add_argument("summary_paths", nargs="+", type=Path)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--figure-path", type=Path, default=DEFAULT_FIGURE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = [read_summary(path) for path in args.summary_paths]
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
