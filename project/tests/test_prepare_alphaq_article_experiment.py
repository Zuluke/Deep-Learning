from __future__ import annotations

import csv

from scripts.prepare_alphaq_article_experiment import ARTICLE_OBJECTIVES
from scripts.prepare_alphaq_article_experiment import DEFAULT_BATCHES
from scripts.prepare_alphaq_article_experiment import pipeline_command
from scripts.prepare_alphaq_article_experiment import protocol_rows
from scripts.prepare_alphaq_article_experiment import shell_line
from scripts.prepare_alphaq_article_experiment import write_protocol_csv


def test_article_protocol_marks_readiness_and_includes_frontier_pair() -> None:
    batch = DEFAULT_BATCHES[0]
    readiness = {
        batch.targets[0]: {
            "target": batch.targets[0],
            "family": "toy-family",
            "tensor_size": "5",
            "readiness_status": "ready-full-action",
        }
    }

    rows = protocol_rows((batch,), readiness)
    first = rows[0]

    assert first["target_status"] == "ready-new-run"
    assert first["family"] == "toy-family"
    assert "frontier_pair" in first["objectives"].split(",")
    assert tuple(first["objectives"].split(",")) == ARTICLE_OBJECTIVES
    assert rows[1]["target_status"] == "needs-readiness-entry"


def test_article_protocol_classifies_control_targets_as_ready_refresh() -> None:
    batch = DEFAULT_BATCHES[0]
    readiness = {
        batch.targets[0]: {
            "target": batch.targets[0],
            "readiness_status": "current-grid-control",
        }
    }

    rows = protocol_rows((batch,), readiness)

    assert rows[0]["target_status"] == "ready-control-refresh"


def test_pipeline_command_is_executable_shell_and_has_paper_zx_audit() -> None:
    command = pipeline_command(DEFAULT_BATCHES[0])
    rendered = shell_line(command)

    assert rendered.startswith("PYTHONPATH=.:external .venv/bin/python ")
    assert "--objective-variants factor_count,factor_count_pair_cap,mixed_pair,frontier_pair" in rendered
    assert "--paper-zx-csv" in rendered
    assert "--paper-zx-report-path" in rendered


def test_write_protocol_csv_preserves_expected_columns(tmp_path) -> None:
    path = tmp_path / "protocol.csv"
    rows = protocol_rows((DEFAULT_BATCHES[0],), {})

    write_protocol_csv(path, rows)

    with path.open(encoding="utf-8", newline="") as handle:
        loaded = list(csv.DictReader(handle))

    assert loaded[0]["batch"] == DEFAULT_BATCHES[0].name
    assert loaded[0]["primary_external_metric"] == "paper_primary_nc_depth_ratio"
