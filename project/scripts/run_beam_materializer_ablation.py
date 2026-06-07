from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_materialized_candidates import read_summary
from scripts.run_shared_parity_ordering_ablation import find_manifest
from scripts.run_shared_parity_study import CORE_TARGETS
from scripts.run_shared_parity_study import STUDY_CASES
from scripts.structural_target import coerce_float


BASELINE_MATERIALIZER = "shared-parity"
DEFAULT_BEAM_WIDTHS = (4, 16)
DEFAULT_SEARCH_ROOTS = (
    PROJECT_ROOT / "results" / "alphaq_shared_parity_study",
    PROJECT_ROOT / "results" / "alphaq_shared_parity_study_expansion_pilot",
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "alphaq_beam_materializer_ablation"
DEFAULT_CSV = PROJECT_ROOT / "results" / "csv" / "alphaq_beam_materializer_ablation.csv"
DEFAULT_REPORT = PROJECT_ROOT / "results" / "reports" / "alphaq_beam_materializer_ablation.md"
DEFAULT_FIGURE = PROJECT_ROOT / "results" / "figures" / "alphaq_beam_materializer_ablation.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare current shared-parity materialization against beam shared-parity search."
    )
    parser.add_argument("--targets", default=",".join(CORE_TARGETS))
    parser.add_argument(
        "--beam-widths",
        default=",".join(str(width) for width in DEFAULT_BEAM_WIDTHS),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--figure-path", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_targets(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_widths(value: str) -> list[int]:
    widths = [int(item.strip()) for item in value.split(",") if item.strip()]
    if any(width <= 0 for width in widths):
        raise ValueError("Beam widths must be positive.")
    return widths


def materializers(beam_widths: list[int]) -> list[tuple[str, int | None]]:
    return [(BASELINE_MATERIALIZER, None), *[("beam-shared-parity", width) for width in beam_widths]]


def materializer_label(synthesis: str, beam_width: int | None) -> str:
    if synthesis == BASELINE_MATERIALIZER:
        return synthesis
    return f"{synthesis}-w{beam_width}"


def materialize(
    *,
    target: str,
    synthesis: str,
    beam_width: int | None,
    output_root: Path,
    force: bool,
) -> Path:
    case = STUDY_CASES[target]
    manifest = find_manifest(target, case.candidate_kind, list(DEFAULT_SEARCH_ROOTS))
    label = materializer_label(synthesis, beam_width)
    destination = output_root / target / label
    summary_path = destination / "summary.json"
    if summary_path.exists() and not force:
        return summary_path
    destination.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "scripts/materialize_shared_parity_candidate.py",
        "--target",
        target,
        "--manifest-csv",
        str(manifest),
        "--candidate-kind",
        case.candidate_kind,
        "--factor-order",
        case.factor_order,
        "--target-strategy",
        case.target_strategy,
        "--synthesis",
        synthesis,
        "--output-root",
        str(destination),
    ]
    if beam_width is not None:
        cmd.extend(["--beam-width", str(beam_width)])
    print(f"+ materialize {target} {label}", flush=True)
    completed = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr, file=sys.stderr)
        completed.check_returncode()
    return summary_path


def collect_rows(
    *,
    targets: list[str],
    beam_widths: list[int],
    output_root: Path,
    force: bool,
) -> list[dict[str, Any]]:
    rows = []
    for target in targets:
        for synthesis, beam_width in materializers(beam_widths):
            summary_path = materialize(
                target=target,
                synthesis=synthesis,
                beam_width=beam_width,
                output_root=output_root,
                force=force,
            )
            row = read_summary(summary_path)
            row["materializer"] = materializer_label(synthesis, beam_width)
            row["beam_width"] = "" if beam_width is None else beam_width
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "target",
        "materializer",
        "beam_width",
        "synthesis",
        "factor_order",
        "target_strategy",
        "tcount",
        "tdepth",
        "qasm_depth",
        "num_total_cnots",
        "primary_nc_depth_ratio",
        "qasm_depth_ratio",
        "structural_cost",
        "candidate_dir",
        "summary_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, lineterminator="\n", fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def paired_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs = []
    for target in sorted({str(row["target"]) for row in rows}):
        by_materializer = {row["materializer"]: row for row in rows if row["target"] == target}
        baseline = by_materializer.get(BASELINE_MATERIALIZER)
        if baseline is None:
            continue
        for label, candidate in sorted(by_materializer.items()):
            if label == BASELINE_MATERIALIZER:
                continue
            pairs.append(
                {
                    "target": target,
                    "materializer": label,
                    "primary_ratio": safe_ratio(
                        candidate.get("primary_nc_depth_ratio"),
                        baseline.get("primary_nc_depth_ratio"),
                    ),
                    "qasm_ratio": safe_ratio(
                        candidate.get("qasm_depth_ratio"),
                        baseline.get("qasm_depth_ratio"),
                    ),
                    "tdepth_ratio": safe_ratio(candidate.get("tdepth"), baseline.get("tdepth")),
                    "cnot_ratio": safe_ratio(
                        candidate.get("num_total_cnots"),
                        baseline.get("num_total_cnots"),
                    ),
                }
            )
    return pairs


def best_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for target in sorted({str(row["target"]) for row in rows}):
        pairs = [row for row in paired_rows(rows) if row["target"] == target]
        if not pairs:
            continue
        result.append(
            min(
                pairs,
                key=lambda row: (
                    inf_if_none(row.get("qasm_ratio")),
                    inf_if_none(row.get("primary_ratio")),
                    inf_if_none(row.get("cnot_ratio")),
                    row["materializer"],
                ),
            )
        )
    return result


def safe_ratio(numerator: Any, denominator: Any) -> float | None:
    numerator_f = coerce_float(numerator)
    denominator_f = coerce_float(denominator)
    if numerator_f is None or denominator_f is None or denominator_f <= 0:
        return None
    return numerator_f / denominator_f


def inf_if_none(value: Any) -> float:
    numeric = coerce_float(value)
    return float("inf") if numeric is None else numeric


def write_report(path: Path, rows: list[dict[str, Any]], csv_path: Path, figure_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pairs = paired_rows(rows)
    best = best_pairs(rows)
    qasm_wins = [row for row in best if is_win(row.get("qasm_ratio"))]
    primary_preserved = [row for row in best if is_preserved(row.get("primary_ratio"))]
    cnot_wins = [row for row in best if is_win(row.get("cnot_ratio"))]
    lines = [
        "# Beam shared-parity materializer ablation",
        "",
        f"CSV: `{csv_path}`.",
        f"Figure: `{figure_path}`.",
        "",
        "This ablation keeps tensor factors fixed and replaces the current greedy shared-parity materialization with a beam search over factor order and parity targets.",
        "",
        f"- Best beam variant improves QASM depth ratio in {len(qasm_wins)}/{len(best)} targets.",
        f"- Best beam variant preserves primary NC ratio in {len(primary_preserved)}/{len(best)} targets.",
        f"- Best beam variant improves CNOT count in {len(cnot_wins)}/{len(best)} targets.",
        "",
        "| target | best beam | primary beam/shared | QASM beam/shared | T-depth beam/shared | CNOT beam/shared |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in best:
        lines.append(
            "| {target} | {materializer} | {primary} | {qasm} | {tdepth} | {cnots} |".format(
                target=row["target"],
                materializer=row["materializer"],
                primary=fmt(row.get("primary_ratio")),
                qasm=fmt(row.get("qasm_ratio")),
                tdepth=fmt(row.get("tdepth_ratio")),
                cnots=fmt(row.get("cnot_ratio")),
            )
        )
    lines.extend(
        [
            "",
            "## All beam variants",
            "",
            "| target | materializer | primary beam/shared | QASM beam/shared | T-depth beam/shared | CNOT beam/shared |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in pairs:
        lines.append(
            "| {target} | {materializer} | {primary} | {qasm} | {tdepth} | {cnots} |".format(
                target=row["target"],
                materializer=row["materializer"],
                primary=fmt(row.get("primary_ratio")),
                qasm=fmt(row.get("qasm_ratio")),
                tdepth=fmt(row.get("tdepth_ratio")),
                cnots=fmt(row.get("cnot_ratio")),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figure(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    best = best_pairs(rows)
    labels = [short_label(row["target"]) for row in best]
    x = range(len(best))
    fields = [
        ("qasm_ratio", "QASM depth"),
        ("primary_ratio", "primary NC depth"),
        ("tdepth_ratio", "T-depth"),
        ("cnot_ratio", "CNOT count"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14.2, 3.8), constrained_layout=True)
    for ax, (field, title) in zip(axes, fields):
        values = [coerce_float(row.get(field)) for row in best]
        ax.bar(
            list(x),
            [0.0 if value is None else value for value in values],
            color=["#1b9e77" if value is not None and value < 1.0 else "#d95f02" for value in values],
            width=0.68,
        )
        ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1.0)
        ax.set_title(title)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
        finite = [value for value in values if value is not None]
        ax.set_ylim(0, max([1.1, *(value * 1.18 for value in finite)]))
        for index, value in enumerate(values):
            if value is not None:
                ax.text(index, value + max(finite) * 0.035, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Best beam shared-parity variant versus current shared-parity", fontsize=12)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def is_win(value: Any) -> bool:
    numeric = coerce_float(value)
    return numeric is not None and numeric < 1.0


def is_preserved(value: Any) -> bool:
    numeric = coerce_float(value)
    return numeric is not None and numeric <= 1.0


def short_label(target: str) -> str:
    return target.replace("hamming_weight_", "hw ").replace("gf_2pow", "gf2^").replace("_mult", " mult")


def fmt(value: Any) -> str:
    numeric = coerce_float(value)
    return "" if numeric is None else f"{numeric:.3g}"


def main() -> int:
    args = parse_args()
    rows = collect_rows(
        targets=parse_targets(args.targets),
        beam_widths=parse_widths(args.beam_widths),
        output_root=args.output_root,
        force=args.force,
    )
    write_csv(args.output_csv, rows)
    write_figure(args.figure_path, rows)
    write_report(args.report_path, rows, args.output_csv, args.figure_path)
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.report_path}")
    print(f"Wrote {args.figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
