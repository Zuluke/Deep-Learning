from __future__ import annotations

from scripts.consolidate_alphaq_external_runs import best_candidate
from scripts.consolidate_alphaq_external_runs import consolidation_rows


def test_best_candidate_prefers_successful_low_tcount_then_qasm() -> None:
    failed = ("old", {"tcount": ""}, {}, "failed")
    worse = ("run_a", {"tcount": "10"}, {"qasm_depth": "80"}, "ok")
    better = ("run_b", {"tcount": "8"}, {"qasm_depth": "100"}, "ok")

    assert best_candidate([failed, worse, better]) == better


def test_best_candidate_keeps_explicit_failure_when_no_success() -> None:
    failed = ("run_a", {"tcount": ""}, {}, "failed")

    assert best_candidate([failed]) == failed


def test_consolidation_rows_merge_requested_runs(monkeypatch, tmp_path) -> None:
    import scripts.consolidate_alphaq_external_runs as module

    write_csv(
        tmp_path / "decomp_standard.csv",
        [
            {"target": "toy", "objective_variant": "factor_count", "execution_status": "failed", "tcount": ""},
        ],
    )
    write_csv(tmp_path / "grid_standard.csv", [])
    write_csv(
        tmp_path / "decomp_journal.csv",
        [
            {"target": "toy", "objective_variant": "factor_count", "execution_status": "ok", "tcount": "5"},
        ],
    )
    write_csv(
        tmp_path / "grid_journal.csv",
        [
            {
                "target": "toy",
                "objective_variant": "factor_count",
                "materializer": "selected-beam-shared-parity-w4",
                "qasm_depth": "33",
                "tdepth": "2",
                "num_total_cnots": "1",
            }
        ],
    )

    def fake_run_paths(run: str):
        if run == "standard":
            return tmp_path / "decomp_standard.csv", tmp_path / "grid_standard.csv"
        return tmp_path / "decomp_journal.csv", tmp_path / "grid_journal.csv"

    monkeypatch.setattr(module, "run_paths", fake_run_paths)

    rows = consolidation_rows(["standard", "journal"])

    assert rows == [
        {
            "target": "toy",
            "objective_variant": "factor_count",
            "best_run": "journal",
            "best_status": "ok",
            "best_tcount": "5",
            "best_beam_qasm_depth": "33",
            "best_beam_tdepth": "2",
            "best_beam_materializer": "selected-beam-shared-parity-w4",
            "completed_runs": "journal",
            "failed_runs": "standard",
            "missing_runs": "",
        }
    ]


def write_csv(path, rows):
    import csv

    fieldnames = sorted({key for row in rows for key in row}) if rows else ["target"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
