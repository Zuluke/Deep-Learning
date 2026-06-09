from __future__ import annotations

import csv

from scripts.run_alphaq_external_validation_pipeline import combine_csvs
from scripts.run_alphaq_external_validation_pipeline import parse_args


def test_combine_csvs_preserves_union_of_columns(tmp_path) -> None:
    base = tmp_path / "base.csv"
    external = tmp_path / "external.csv"
    output = tmp_path / "combined.csv"
    base.write_text("target,metric\ncore,1\n", encoding="utf-8")
    external.write_text("target,extra\nexternal,2\n", encoding="utf-8")

    combine_csvs([base, external], output)

    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows == [
        {"target": "core", "metric": "1", "extra": ""},
        {"target": "external", "metric": "", "extra": "2"},
    ]


def test_decomposition_roots_default_to_output_root(monkeypatch, tmp_path) -> None:
    output_root = tmp_path / "external_night_long"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_alphaq_external_validation_pipeline.py",
            "--output-root",
            str(output_root),
        ],
    )

    args = parse_args()
    decomposition_roots = args.decomposition_roots or str(args.output_root)

    assert decomposition_roots == str(output_root)


def test_objective_variants_argument_is_parsed(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_alphaq_external_validation_pipeline.py",
            "--objective-variants",
            "factor_count_pair_cap",
        ],
    )

    args = parse_args()

    assert args.objective_variants == "factor_count_pair_cap"


def test_paper_zx_audit_paths_are_parsed(monkeypatch, tmp_path) -> None:
    paper_csv = tmp_path / "paper.csv"
    paper_report = tmp_path / "paper.md"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_alphaq_external_validation_pipeline.py",
            "--paper-zx-csv",
            str(paper_csv),
            "--paper-zx-report-path",
            str(paper_report),
            "--skip-paper-zx-audit",
        ],
    )

    args = parse_args()

    assert args.paper_zx_csv == paper_csv
    assert args.paper_zx_report_path == paper_report
    assert args.skip_paper_zx_audit is True
