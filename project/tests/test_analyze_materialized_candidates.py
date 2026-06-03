from __future__ import annotations

import csv
import json


def test_analyze_materialized_candidates_writes_csv_and_report(tmp_path):
    from scripts import analyze_materialized_candidates

    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "target": "toy",
                "candidate_kind": "unit",
                "status": "ok",
                "reconstruction_ok": True,
                "assembled_metrics": {
                    "t_count": 3,
                    "tdepth": 2,
                    "normalized_qasm_depth": 7,
                },
                "external_structural_metrics": {
                    "primary_nc_depth_ratio": 0.5,
                    "qasm_depth_ratio": 1.2,
                    "tcount_ratio": 0.75,
                    "structural_cost": 0.9,
                    "structural_target_status": "ok",
                },
            }
        ),
        encoding="utf-8",
    )
    output_csv = tmp_path / "analysis.csv"
    report_path = tmp_path / "analysis.md"
    figure_path = tmp_path / "analysis.png"

    rows = [analyze_materialized_candidates.read_summary(summary)]
    analyze_materialized_candidates.write_csv(output_csv, rows)
    analyze_materialized_candidates.write_report(report_path, rows, output_csv)
    plotted = analyze_materialized_candidates.write_plot(figure_path, rows)

    with output_csv.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["target"] == "toy"
    assert row["tcount"] == "3.0"
    assert "verification_status" in row
    assert "toy" in report_path.read_text(encoding="utf-8")
    assert not plotted


def test_analyze_materialized_candidates_normalizes_inconclusive_verification(tmp_path):
    from scripts import analyze_materialized_candidates

    proof = tmp_path / "proof.txt"
    proof.write_text("Inconclusive (took 0.010s)\n", encoding="utf-8")
    verification_summary = tmp_path / "verification_summary.json"
    verification_summary.write_text(
        json.dumps(
            [
                {
                    "target": "toy",
                    "candidate_kind": "unit",
                    "verification_status": "failed",
                    "proof_path": str(proof),
                }
            ]
        ),
        encoding="utf-8",
    )
    rows = [{"target": "toy", "candidate_kind": "unit"}]

    verification = analyze_materialized_candidates.read_verification_summaries(
        [verification_summary]
    )
    analyze_materialized_candidates.attach_verification(rows, verification)

    assert rows[0]["verification_status"] == "inconclusive"
