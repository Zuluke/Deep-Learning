from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_materialized_candidates import read_summary
from scripts.alphaq_portfolio_common import linear_span_manifest_path
from scripts.run_decomposition_objective_ablation import OBJECTIVE_VARIANTS
from scripts.run_decomposition_objective_ablation import parse_targets
from scripts.run_shared_parity_study import CORE_TARGETS
from scripts.run_shared_parity_study import STUDY_CASES
from scripts.structural_target import coerce_float


DEFAULT_DECOMP_CSVS = (
    PROJECT_ROOT / "results" / "csv" / "alphaq_decomposition_objective_ablation.csv",
    PROJECT_ROOT / "results" / "csv" / "alphaq_decomposition_objective_holdout_ablation.csv",
)
DEFAULT_DECOMP_ROOTS = (
    PROJECT_ROOT / "results" / "alphaq_decomposition_objective_ablation",
    PROJECT_ROOT / "results" / "alphaq_decomposition_objective_holdout_ablation",
)
DEFAULT_CURRENT_BEAM_CSVS = (
    PROJECT_ROOT / "results" / "csv" / "alphaq_beam_materializer_ablation.csv",
    PROJECT_ROOT / "results" / "csv" / "alphaq_beam_materializer_holdout_ablation.csv",
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "alphaq_best_objective_beam_ablation"
DEFAULT_CSV = PROJECT_ROOT / "results" / "csv" / "alphaq_best_objective_beam_ablation.csv"
DEFAULT_MATERIALIZED_CSV = (
    PROJECT_ROOT / "results" / "csv" / "alphaq_best_objective_beam_materialized.csv"
)
DEFAULT_REPORT = PROJECT_ROOT / "results" / "reports" / "alphaq_best_objective_beam_ablation.md"
DEFAULT_FIGURE = PROJECT_ROOT / "results" / "figures" / "alphaq_best_objective_beam_ablation.png"
DEFAULT_BEAM_WIDTHS = (4, 16)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select the best tensor objective per target, then apply beam shared-parity "
            "materialization and compare against the current checkpoint beam candidates."
        )
    )
    parser.add_argument("--targets", default=None)
    parser.add_argument(
        "--decomposition-csvs",
        default=",".join(str(path) for path in DEFAULT_DECOMP_CSVS),
    )
    parser.add_argument(
        "--decomposition-roots",
        default=",".join(str(path) for path in DEFAULT_DECOMP_ROOTS),
    )
    parser.add_argument(
        "--current-beam-csvs",
        default=",".join(str(path) for path in DEFAULT_CURRENT_BEAM_CSVS),
    )
    parser.add_argument(
        "--beam-widths",
        default=",".join(str(width) for width in DEFAULT_BEAM_WIDTHS),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--materialized-csv", type=Path, default=DEFAULT_MATERIALIZED_CSV)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--figure-path", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_paths(value: str) -> list[Path]:
    return [Path(item.strip()) for item in value.split(",") if item.strip()]


def parse_widths(value: str) -> list[int]:
    widths = [int(item.strip()) for item in value.split(",") if item.strip()]
    if any(width <= 0 for width in widths):
        raise ValueError("Beam widths must be positive.")
    return widths


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def selected_objective_rows(rows: list[dict[str, str]], targets: list[str] | None) -> list[dict[str, str]]:
    target_names = targets or sorted({row["target"] for row in rows})
    selected = []
    for target in target_names:
        candidates = [row for row in rows if row["target"] == target]
        if not candidates:
            continue
        selected.append(min(candidates, key=objective_selection_key))
    return selected


def objective_selection_key(row: dict[str, str]) -> tuple[float, float, float, str]:
    return (
        inf_if_none(row.get("tcount")),
        inf_if_none(row.get("primary_nc_depth_ratio")),
        inf_if_none(row.get("qasm_depth_ratio")),
        row.get("objective_variant", ""),
    )


def variant_by_name(name: str):
    variants = {variant.name: variant for variant in OBJECTIVE_VARIANTS}
    if name not in variants:
        raise KeyError(f"Unknown objective variant {name!r}.")
    return variants[name]


def find_manifest(target: str, objective_variant: str, roots: list[Path]) -> Path:
    case = STUDY_CASES[target]
    variant = variant_by_name(objective_variant)
    for root in roots:
        path = linear_span_manifest_path(
            root / "linear_span",
            target,
            objective_variant=variant.name,
            max_action_weight=case.max_action_weight,
            objective=variant.objective,
        )
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing manifest for {target} {objective_variant} in {roots}.")


def materializer_label(synthesis: str, beam_width: int | None) -> str:
    if synthesis == "shared-parity":
        return "selected-shared-parity"
    return f"selected-beam-shared-parity-w{beam_width}"


def materialize(
    *,
    target: str,
    objective_variant: str,
    manifest: Path,
    synthesis: str,
    beam_width: int | None,
    output_root: Path,
    force: bool,
) -> Path:
    case = STUDY_CASES[target]
    variant = variant_by_name(objective_variant)
    label = materializer_label(synthesis, beam_width)
    destination = output_root / target / objective_variant / label
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
        variant.candidate_kind,
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
    print(f"+ materialize {target} {objective_variant} {label}", flush=True)
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


def collect_materialized_rows(
    *,
    selected_rows: list[dict[str, str]],
    roots: list[Path],
    beam_widths: list[int],
    output_root: Path,
    force: bool,
) -> list[dict[str, Any]]:
    rows = []
    for selected in selected_rows:
        target = selected["target"]
        objective_variant = selected["objective_variant"]
        manifest = find_manifest(target, objective_variant, roots)
        for synthesis, beam_width in [("shared-parity", None), *[("beam-shared-parity", width) for width in beam_widths]]:
            summary_path = materialize(
                target=target,
                objective_variant=objective_variant,
                manifest=manifest,
                synthesis=synthesis,
                beam_width=beam_width,
                output_root=output_root,
                force=force,
            )
            row = read_summary(summary_path)
            row["objective_variant"] = objective_variant
            row["materializer"] = materializer_label(synthesis, beam_width)
            row["beam_width"] = "" if beam_width is None else beam_width
            row["selected_objective_tcount"] = selected.get("tcount")
            row["selected_objective_primary_nc_depth_ratio"] = selected.get("primary_nc_depth_ratio")
            row["selected_objective_qasm_depth_ratio"] = selected.get("qasm_depth_ratio")
            rows.append(row)
    return rows


def best_beam_by_target(rows: list[dict[str, Any]], *, prefix: str) -> dict[str, dict[str, Any]]:
    result = {}
    for target in sorted({str(row["target"]) for row in rows}):
        beams = [
            row
            for row in rows
            if row["target"] == target and str(row.get("materializer", "")).startswith(prefix)
        ]
        if not beams:
            continue
        result[target] = min(
            beams,
            key=lambda row: (
                inf_if_none(row.get("qasm_depth_ratio")),
                inf_if_none(row.get("primary_nc_depth_ratio")),
                inf_if_none(row.get("num_total_cnots")),
                row.get("materializer", ""),
            ),
        )
    return result


def comparison_rows(
    *,
    selected_rows: list[dict[str, Any]],
    current_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    selected_beams = best_beam_by_target(selected_rows, prefix="selected-beam-shared-parity")
    selected_shared = {
        row["target"]: row
        for row in selected_rows
        if row.get("materializer") == "selected-shared-parity"
    }
    current_beams = best_beam_by_target(current_rows, prefix="beam-shared-parity")
    rows = []
    for target in sorted(selected_beams):
        selected = selected_beams[target]
        shared = selected_shared.get(target)
        current = current_beams.get(target)
        if shared is None or current is None:
            continue
        rows.append(
            {
                "target": target,
                "selected_objective": selected.get("objective_variant"),
                "selected_beam_materializer": selected.get("materializer"),
                "current_beam_materializer": current.get("materializer"),
                "selected_beam_tcount": selected.get("tcount"),
                "current_beam_tcount": current.get("tcount"),
                "selected_vs_current_tcount_ratio": safe_ratio(selected.get("tcount"), current.get("tcount")),
                "selected_beam_tdepth": selected.get("tdepth"),
                "current_beam_tdepth": current.get("tdepth"),
                "selected_vs_current_tdepth_ratio": safe_ratio(selected.get("tdepth"), current.get("tdepth")),
                "selected_beam_qasm_depth": selected.get("qasm_depth"),
                "current_beam_qasm_depth": current.get("qasm_depth"),
                "selected_vs_current_qasm_depth_ratio": safe_ratio(selected.get("qasm_depth"), current.get("qasm_depth")),
                "selected_beam_cnots": selected.get("num_total_cnots"),
                "current_beam_cnots": current.get("num_total_cnots"),
                "selected_vs_current_cnot_ratio": safe_ratio(selected.get("num_total_cnots"), current.get("num_total_cnots")),
                "selected_beam_primary_nc_depth_ratio": selected.get("primary_nc_depth_ratio"),
                "current_beam_primary_nc_depth_ratio": current.get("primary_nc_depth_ratio"),
                "selected_vs_current_primary_nc_ratio": safe_ratio(
                    selected.get("primary_nc_depth_ratio"),
                    current.get("primary_nc_depth_ratio"),
                ),
                "selected_vs_selected_shared_qasm_depth_ratio": safe_ratio(
                    selected.get("qasm_depth"),
                    shared.get("qasm_depth"),
                ),
                "selected_vs_selected_shared_primary_nc_ratio": safe_ratio(
                    selected.get("primary_nc_depth_ratio"),
                    shared.get("primary_nc_depth_ratio"),
                ),
                "selected_summary_path": selected.get("summary_path"),
                "selected_candidate_dir": selected.get("candidate_dir"),
            }
        )
    return rows


def safe_ratio(numerator: Any, denominator: Any) -> float | None:
    numerator_f = coerce_float(numerator)
    denominator_f = coerce_float(denominator)
    if numerator_f is None or denominator_f is None or denominator_f <= 0:
        return None
    return numerator_f / denominator_f


def inf_if_none(value: Any) -> float:
    numeric = coerce_float(value)
    return float("inf") if numeric is None else numeric


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "target",
        "selected_objective",
        "selected_beam_materializer",
        "current_beam_materializer",
        "selected_beam_tcount",
        "current_beam_tcount",
        "selected_vs_current_tcount_ratio",
        "selected_beam_tdepth",
        "current_beam_tdepth",
        "selected_vs_current_tdepth_ratio",
        "selected_beam_qasm_depth",
        "current_beam_qasm_depth",
        "selected_vs_current_qasm_depth_ratio",
        "selected_beam_cnots",
        "current_beam_cnots",
        "selected_vs_current_cnot_ratio",
        "selected_beam_primary_nc_depth_ratio",
        "current_beam_primary_nc_depth_ratio",
        "selected_vs_current_primary_nc_ratio",
        "selected_vs_selected_shared_qasm_depth_ratio",
        "selected_vs_selected_shared_primary_nc_ratio",
        "selected_summary_path",
        "selected_candidate_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, lineterminator="\n", fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_materialized_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "target",
        "objective_variant",
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


def write_report(path: Path, rows: list[dict[str, Any]], csv_path: Path, figure_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t_wins = count(rows, "selected_vs_current_tcount_ratio", strict=False)
    primary_wins = count(rows, "selected_vs_current_primary_nc_ratio", strict=True)
    qasm_wins = count(rows, "selected_vs_current_qasm_depth_ratio", strict=True)
    shared_qasm_wins = count(rows, "selected_vs_selected_shared_qasm_depth_ratio", strict=True)
    lines = [
        "# Best-objective beam ablation",
        "",
        f"CSV: `{csv_path}`.",
        f"Figure: `{figure_path}`.",
        "",
        "This ablation first selects the best tensor objective per target by lexicographic `(T-count, primary NC ratio, QASM ratio)`, then applies the beam shared-parity materializer. It compares that candidate against the current checkpoint beam candidate for the same target.",
        "",
        f"- Selected-objective beam has T-count <= current beam in {t_wins}/{len(rows)} targets.",
        f"- Selected-objective beam improves primary NC ratio vs current beam in {primary_wins}/{len(rows)} targets.",
        f"- Selected-objective beam improves QASM depth vs current beam in {qasm_wins}/{len(rows)} targets.",
        f"- Beam improves QASM depth vs its selected shared materialization in {shared_qasm_wins}/{len(rows)} targets.",
        "",
        "| target | selected objective | T selected/current | primary selected/current | QASM selected/current | selected beam/shared QASM |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {target} | {objective} | {t} | {primary} | {qasm} | {shared_qasm} |".format(
                target=row["target"],
                objective=row["selected_objective"],
                t=fmt(row.get("selected_vs_current_tcount_ratio")),
                primary=fmt(row.get("selected_vs_current_primary_nc_ratio")),
                qasm=fmt(row.get("selected_vs_current_qasm_depth_ratio")),
                shared_qasm=fmt(row.get("selected_vs_selected_shared_qasm_depth_ratio")),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def count(rows: list[dict[str, Any]], key: str, *, strict: bool) -> int:
    result = 0
    for row in rows:
        value = coerce_float(row.get(key))
        if value is None:
            continue
        if value < 1.0 or (not strict and value <= 1.0):
            result += 1
    return result


def write_figure(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [short_label(str(row["target"])) for row in rows]
    x = range(len(rows))
    panels = [
        ("selected_vs_current_tcount_ratio", "T-count"),
        ("selected_vs_current_primary_nc_ratio", "primary NC"),
        ("selected_vs_current_qasm_depth_ratio", "QASM depth"),
        ("selected_vs_selected_shared_qasm_depth_ratio", "beam/shared QASM"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14.4, 4.0), constrained_layout=True)
    for ax, (field, title) in zip(axes, panels):
        values = [coerce_float(row.get(field)) for row in rows]
        finite = [value for value in values if value is not None]
        ax.bar(
            list(x),
            [0.0 if value is None else value for value in values],
            color=["#1b9e77" if value is not None and value <= 1.0 else "#d95f02" for value in values],
            width=0.68,
        )
        ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1.0)
        ax.set_title(title)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=28, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(0, max([1.1, *(value * 1.16 for value in finite)]))
        offset = max(finite) * 0.035 if finite else 0.04
        for index, value in enumerate(values):
            if value is not None:
                ax.text(index, value + offset, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Best tensor objective plus beam versus current checkpoint beam", fontsize=12)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def short_label(target: str) -> str:
    return target.replace("hamming_weight_", "hw ").replace("gf_2pow", "gf2^").replace("_mult", " mult")


def fmt(value: Any) -> str:
    numeric = coerce_float(value)
    return "" if numeric is None else f"{numeric:.3g}"


def main() -> int:
    args = parse_args()
    decomposition_rows = [row for path in parse_paths(args.decomposition_csvs) for row in read_csv(path)]
    targets = parse_targets(args.targets) if args.targets else None
    selected = selected_objective_rows(decomposition_rows, targets)
    materialized_rows = collect_materialized_rows(
        selected_rows=selected,
        roots=parse_paths(args.decomposition_roots),
        beam_widths=parse_widths(args.beam_widths),
        output_root=args.output_root,
        force=args.force,
    )
    current_beam_rows = [row for path in parse_paths(args.current_beam_csvs) for row in read_csv(path)]
    comparisons = comparison_rows(selected_rows=materialized_rows, current_rows=current_beam_rows)
    write_materialized_csv(args.materialized_csv, materialized_rows)
    write_csv(args.output_csv, comparisons)
    write_report(args.report_path, comparisons, args.output_csv, args.figure_path)
    write_figure(args.figure_path, comparisons)
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.materialized_csv}")
    print(f"Wrote {args.report_path}")
    print(f"Wrote {args.figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
