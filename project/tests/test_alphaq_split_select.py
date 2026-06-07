from __future__ import annotations

import csv
import math
import subprocess
import sys
from pathlib import Path

from scripts.analyze_alphaq_split_select import candidate_weights
from scripts.analyze_alphaq_split_select import evaluation_rows
from scripts.analyze_alphaq_split_select import linear_score
from scripts.analyze_alphaq_split_select import normalize_rows
from scripts.analyze_alphaq_split_select import policy_row
from scripts.analyze_alphaq_split_select import summary_rows
from scripts.analyze_alphaq_split_select import train_ready_rows


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_candidate_weights_use_sparse_grid_for_large_feature_sets() -> None:
    weights = list(candidate_weights((-1, 0, 1), 6))

    assert weights
    assert all(sum(value != 0 for value in item) <= 2 for item in weights)
    assert (1, 0, 0, 0, 0, 0) in weights
    assert (1, -1, 0, 0, 0, 0) in weights


def test_evaluation_rows_train_split_select_with_loto_dataset() -> None:
    rows = [
        *make_group("internal", "a", oracle="mixed_pair"),
        *make_group("internal", "b", oracle="mixed_pair"),
        *make_group("external", "c", oracle="mixed_pair"),
    ]

    details = evaluation_rows(rows)
    summaries = {row["policy"]: row for row in summary_rows(details)}

    assert summaries["baseline_factor_count"]["exact_oracle_matches"] == 0
    assert summaries["split_select_linear_alphaq"]["exact_oracle_matches"] == 3
    assert summaries["split_select_linear_alphaq"]["tcount_wins_vs_baseline"] == 3
    assert summaries["oracle_posthoc"]["exact_oracle_matches"] == 3


def test_missing_fixed_objective_is_reported_without_fallback() -> None:
    items = make_group("external", "partial", oracle="factor_count")[:1]

    row = policy_row(
        "best_fixed_loto",
        "fixed-objective",
        "partial",
        ("external", "partial"),
        None,
        items,
        fixed_objective="mixed_pair",
    )

    assert row["selection_status"] == "missing-selected-objective"
    assert row["fixed_objective"] == "mixed_pair"
    assert "selected_objective" not in row


def test_train_ready_rows_filter_failed_unready_and_missing_beam_rows() -> None:
    valid = make_row("internal", "valid", "factor_count", tcount=1, qasm=1, primary=1, overlap=1, oracle="factor_count")
    failed = dict(valid, target="failed", execution_status="failed")
    unready = dict(valid, target="unready", train_ready="False")
    no_beam = dict(valid, target="no_beam", has_beam_candidate="False")

    rows = train_ready_rows([valid, failed, unready, no_beam])

    assert rows == [valid]


def test_normalize_rows_handles_missing_features_without_nan() -> None:
    rows = [
        make_row("internal", "toy", "factor_count", tcount=1, qasm=10, primary=1, overlap=1, oracle="factor_count"),
        make_row("internal", "toy", "mixed_pair", tcount=2, qasm=20, primary=1, overlap=2, oracle="factor_count"),
    ]
    rows[1]["factor_pairwise_jaccard_mean"] = ""

    normalized = normalize_rows(rows, ("factor_pairwise_jaccard_mean",))
    scores = [linear_score(row, ("factor_pairwise_jaccard_mean",), (1,)) for row in normalized]

    assert normalized[0]["norm_factor_pairwise_jaccard_mean"] == 0.0
    assert normalized[1]["norm_factor_pairwise_jaccard_mean"] == 1.0
    assert all(math.isfinite(score) for score in scores)


def test_single_target_dataset_reports_no_training_data_for_split_select() -> None:
    rows = make_group("internal", "only", oracle="mixed_pair")

    details = evaluation_rows(rows)
    split_rows = [row for row in details if row["policy"].startswith("split_select_")]

    assert split_rows
    assert all(row["selection_status"] == "no-training-data" for row in split_rows)


def test_cli_writes_summary_detail_report_and_figure(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.csv"
    summary = tmp_path / "summary.csv"
    details = tmp_path / "details.csv"
    report = tmp_path / "report.md"
    figure = tmp_path / "figure.png"
    rows = [
        *make_group("internal", "a", oracle="mixed_pair"),
        *make_group("internal", "b", oracle="mixed_pair"),
        *make_group("external", "c", oracle="mixed_pair"),
    ]
    write_dataset(dataset, rows)

    subprocess.run(
        [
            sys.executable,
            "scripts/analyze_alphaq_split_select.py",
            "--dataset-csv",
            str(dataset),
            "--summary-csv",
            str(summary),
            "--detail-csv",
            str(details),
            "--report-path",
            str(report),
            "--figure-path",
            str(figure),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )

    assert summary.exists()
    assert details.exists()
    assert "Decision:" in report.read_text(encoding="utf-8")
    assert figure.stat().st_size > 0


def make_group(source_split: str, target: str, *, oracle: str) -> list[dict[str, str]]:
    return [
        make_row(
            source_split,
            target,
            "factor_count",
            tcount=10,
            qasm=100,
            primary=1.0,
            overlap=10,
            oracle=oracle,
        ),
        make_row(
            source_split,
            target,
            "mixed_pair",
            tcount=5,
            qasm=90,
            primary=0.8,
            overlap=1,
            oracle=oracle,
        ),
    ]


def make_row(
    source_split: str,
    target: str,
    objective: str,
    *,
    tcount: int,
    qasm: int,
    primary: float,
    overlap: int,
    oracle: str,
) -> dict[str, str]:
    return {
        "source_split": source_split,
        "target": target,
        "family": "toy",
        "objective_variant": objective,
        "execution_status": "ok",
        "has_beam_candidate": "True",
        "train_ready": "True",
        "factor_count": str(tcount),
        "factor_qubit_concentration_index": str(overlap),
        "factor_support_weight_mean": str(overlap),
        "factor_pairwise_support_overlap_mean": str(overlap),
        "factor_pairwise_jaccard_mean": str(overlap),
        "decomp_tcount": str(tcount),
        "decomp_tdepth": str(tcount),
        "decomp_qasm_depth_ratio": str(qasm / 100),
        "best_beam_tcount": str(tcount),
        "best_beam_primary_nc_depth_ratio": str(primary),
        "best_beam_qasm_depth": str(qasm),
        "oracle_objective": oracle,
        "objective_is_oracle": str(objective == oracle),
    }


def write_dataset(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
