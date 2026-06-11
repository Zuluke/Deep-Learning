from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_TARGETS = "nc_tof_4,barenco_tof_4,vbe_adder_3"
DEFAULT_EXTERNAL_ROOT = PROJECT_ROOT / "results" / "alphaq_decomposition_objective_external_validation"
DEFAULT_EXTERNAL_CSV = PROJECT_ROOT / "results" / "csv" / "alphaq_decomposition_objective_external_validation.csv"
DEFAULT_EXTERNAL_REPORT = PROJECT_ROOT / "results" / "reports" / "alphaq_decomposition_objective_external_validation.md"
DEFAULT_EXTERNAL_FIGURE = PROJECT_ROOT / "results" / "figures" / "alphaq_decomposition_objective_external_validation.png"
DEFAULT_EXTERNAL_BEAM_ROOT = PROJECT_ROOT / "results" / "alphaq_objective_beam_policy_external_validation"
DEFAULT_EXTERNAL_GRID = PROJECT_ROOT / "results" / "csv" / "alphaq_objective_beam_policy_external_validation_grid.csv"
DEFAULT_EXTERNAL_POLICY = PROJECT_ROOT / "results" / "csv" / "alphaq_objective_beam_policy_external_validation_summary.csv"
DEFAULT_EXTERNAL_BEAM_REPORT = PROJECT_ROOT / "results" / "reports" / "alphaq_objective_beam_policy_external_validation.md"
DEFAULT_EXTERNAL_BEAM_FIGURE = PROJECT_ROOT / "results" / "figures" / "alphaq_objective_beam_policy_external_validation.png"
DEFAULT_BASE_GRID = PROJECT_ROOT / "results" / "csv" / "alphaq_objective_beam_policy_grid.csv"
DEFAULT_COMBINED_GRID = PROJECT_ROOT / "results" / "csv" / "alphaq_objective_beam_policy_grid_plus_external_validation.csv"
DEFAULT_TRAIN_DECOMP_CSVS = (
    PROJECT_ROOT / "results" / "csv" / "alphaq_decomposition_objective_ablation.csv",
    PROJECT_ROOT / "results" / "csv" / "alphaq_decomposition_objective_holdout_ablation.csv",
)
DEFAULT_CURRENT_BEAM_CSVS = (
    PROJECT_ROOT / "results" / "csv" / "alphaq_beam_materializer_ablation.csv",
    PROJECT_ROOT / "results" / "csv" / "alphaq_beam_materializer_holdout_ablation.csv",
)
DEFAULT_TRANSFER_SUMMARY = PROJECT_ROOT / "results" / "csv" / "alphaq_external_selector_transfer_summary.csv"
DEFAULT_TRANSFER_DETAILS = PROJECT_ROOT / "results" / "csv" / "alphaq_external_selector_transfer_details.csv"
DEFAULT_TRANSFER_REPORT = PROJECT_ROOT / "results" / "reports" / "alphaq_external_selector_transfer.md"
DEFAULT_STATUS_CSV = PROJECT_ROOT / "results" / "csv" / "alphaq_external_validation_status.csv"
DEFAULT_STATUS_REPORT = PROJECT_ROOT / "results" / "reports" / "alphaq_external_validation_status.md"
DEFAULT_READINESS_CSV = PROJECT_ROOT / "results" / "csv" / "alphaq_external_validation_readiness.csv"
DEFAULT_PAPER_ZX_CSV = PROJECT_ROOT / "results" / "csv" / "alphaq_external_validation_paper_zx_audit.csv"
DEFAULT_PAPER_ZX_REPORT = PROJECT_ROOT / "results" / "reports" / "alphaq_external_validation_paper_zx_audit.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run external AlphaQ objective validation: decomposition objectives, "
            "beam materialization grid, combined grid, and guarded selector transfer."
        )
    )
    parser.add_argument("--targets", default=DEFAULT_TARGETS)
    parser.add_argument("--time-limit-sec", type=float, default=900.0)
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=1,
        help=(
            "Run up to N (target, objective) MILP optimizations concurrently "
            "in the decomposition step."
        ),
    )
    parser.add_argument(
        "--objective-variants",
        default=None,
        help=(
            "Comma-separated objective variants to pass to "
            "run_decomposition_objective_ablation.py. Defaults to all variants."
        ),
    )
    parser.add_argument("--beam-widths", default="4,16")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_EXTERNAL_ROOT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_EXTERNAL_CSV)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_EXTERNAL_REPORT)
    parser.add_argument("--figure-path", type=Path, default=DEFAULT_EXTERNAL_FIGURE)
    parser.add_argument("--beam-output-root", type=Path, default=DEFAULT_EXTERNAL_BEAM_ROOT)
    parser.add_argument("--beam-grid-csv", type=Path, default=DEFAULT_EXTERNAL_GRID)
    parser.add_argument("--beam-policy-csv", type=Path, default=DEFAULT_EXTERNAL_POLICY)
    parser.add_argument("--beam-report-path", type=Path, default=DEFAULT_EXTERNAL_BEAM_REPORT)
    parser.add_argument("--beam-figure-path", type=Path, default=DEFAULT_EXTERNAL_BEAM_FIGURE)
    parser.add_argument("--base-grid-csv", type=Path, default=DEFAULT_BASE_GRID)
    parser.add_argument("--combined-grid-csv", type=Path, default=DEFAULT_COMBINED_GRID)
    parser.add_argument(
        "--train-decomposition-csvs",
        default=",".join(str(path) for path in DEFAULT_TRAIN_DECOMP_CSVS),
    )
    parser.add_argument(
        "--decomposition-roots",
        default=None,
        help=(
            "Comma-separated decomposition roots for beam materialization. "
            "Defaults to --output-root, so suffixed runs do not accidentally "
            "look for manifests in the standard external-validation root."
        ),
    )
    parser.add_argument(
        "--current-beam-csvs",
        default=",".join(str(path) for path in DEFAULT_CURRENT_BEAM_CSVS),
    )
    parser.add_argument("--transfer-summary-csv", type=Path, default=DEFAULT_TRANSFER_SUMMARY)
    parser.add_argument("--transfer-detail-csv", type=Path, default=DEFAULT_TRANSFER_DETAILS)
    parser.add_argument("--transfer-report-path", type=Path, default=DEFAULT_TRANSFER_REPORT)
    parser.add_argument("--readiness-csv", type=Path, default=DEFAULT_READINESS_CSV)
    parser.add_argument("--status-csv", type=Path, default=DEFAULT_STATUS_CSV)
    parser.add_argument("--status-report-path", type=Path, default=DEFAULT_STATUS_REPORT)
    parser.add_argument("--paper-zx-csv", type=Path, default=DEFAULT_PAPER_ZX_CSV)
    parser.add_argument("--paper-zx-report-path", type=Path, default=DEFAULT_PAPER_ZX_REPORT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-decomposition", action="store_true")
    parser.add_argument("--skip-beam-grid", action="store_true")
    parser.add_argument(
        "--skip-paper-zx-audit",
        action="store_true",
        help="Skip the paper-style ZX audit over the external beam-grid candidates.",
    )
    return parser.parse_args()


def run_command(args: list[str]) -> None:
    print("+ " + " ".join(args), flush=True)
    subprocess.run(args, cwd=PROJECT_ROOT, check=True)


def combine_csvs(paths: list[Path], output_path: Path) -> None:
    rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    for path in paths:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for name in reader.fieldnames or []:
                if name not in fieldnames:
                    fieldnames.append(name)
            rows.extend(reader)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, lineterminator="\n", fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output_path} ({len(rows)} rows)")


def main() -> int:
    args = parse_args()
    decomposition_roots = args.decomposition_roots or str(args.output_root)
    if not args.skip_decomposition:
        command = [
            sys.executable,
            "scripts/run_decomposition_objective_ablation.py",
            "--targets",
            args.targets,
            "--output-root",
            str(args.output_root),
            "--output-csv",
            str(args.output_csv),
            "--report-path",
            str(args.report_path),
            "--figure-path",
            str(args.figure_path),
            "--time-limit-sec",
            str(args.time_limit_sec),
            "--continue-on-error",
        ]
        if args.max_parallel > 1:
            command.extend(["--max-parallel", str(args.max_parallel)])
        if args.force:
            command.append("--force")
        if args.objective_variants:
            command.extend(["--objective-variants", args.objective_variants])
        run_command(command)

    if not args.skip_beam_grid:
        command = [
            sys.executable,
            "scripts/run_objective_beam_policy_grid.py",
            "--decomposition-csvs",
            str(args.output_csv),
            "--decomposition-roots",
            decomposition_roots,
            "--current-beam-csvs",
            args.current_beam_csvs,
            "--beam-widths",
            args.beam_widths,
            "--output-root",
            str(args.beam_output_root),
            "--grid-csv",
            str(args.beam_grid_csv),
            "--policy-csv",
            str(args.beam_policy_csv),
            "--report-path",
            str(args.beam_report_path),
            "--figure-path",
            str(args.beam_figure_path),
        ]
        if args.force:
            command.append("--force")
        run_command(command)

    combine_csvs([args.base_grid_csv, args.beam_grid_csv], args.combined_grid_csv)

    run_command(
        [
            sys.executable,
            "scripts/analyze_alphaq_external_selector_transfer.py",
            "--train-decomposition-csvs",
            args.train_decomposition_csvs,
            "--external-decomposition-csv",
            str(args.output_csv),
            "--grid-csv",
            str(args.combined_grid_csv),
            "--output-csv",
            str(args.transfer_summary_csv),
            "--detail-csv",
            str(args.transfer_detail_csv),
            "--report-path",
            str(args.transfer_report_path),
        ]
    )
    run_command(
        [
            sys.executable,
            "scripts/analyze_alphaq_external_validation_status.py",
            "--readiness-csv",
            str(args.readiness_csv),
            "--decomposition-csv",
            str(args.output_csv),
            "--transfer-detail-csv",
            str(args.transfer_detail_csv),
            "--output-csv",
            str(args.status_csv),
            "--report-path",
            str(args.status_report_path),
        ]
    )
    if not args.skip_paper_zx_audit:
        run_command(
            [
                sys.executable,
                "scripts/compare_zx_border_detectors.py",
                "--input-csv",
                str(args.beam_grid_csv),
                "--output-csv",
                str(args.paper_zx_csv),
                "--report-path",
                str(args.paper_zx_report_path),
            ]
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
