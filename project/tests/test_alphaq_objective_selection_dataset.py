from __future__ import annotations

from pathlib import Path

import scripts.build_alphaq_objective_selection_dataset as dataset
from scripts.build_alphaq_objective_selection_dataset import build_dataset
from scripts.build_alphaq_objective_selection_dataset import dataset_rows_for_split
from scripts.build_alphaq_objective_selection_dataset import readiness_summary
from scripts.build_alphaq_objective_selection_dataset import run_suffixes


def test_dataset_marks_oracle_from_best_materialized_beam() -> None:
    decomp_rows = [
        make_decomp("toy", "factor_count", "ok", tcount=5),
        make_decomp("toy", "factor_count_pair_cap", "ok", tcount=4),
        make_decomp("toy", "mixed_pair", "failed", tcount=""),
    ]
    grid_rows = [
        make_grid("toy", "factor_count", tcount=5, primary=1.0, qasm=50),
        make_grid("toy", "factor_count_pair_cap", tcount=4, primary=2.0, qasm=60),
    ]

    rows = dataset_rows_for_split("unit", decomp_rows, grid_rows, {})
    by_objective = {row["objective_variant"]: row for row in rows}

    assert by_objective["factor_count_pair_cap"]["objective_is_oracle"] is True
    assert by_objective["factor_count_pair_cap"]["objective_rank"] == 1
    assert by_objective["mixed_pair"]["train_ready"] is True
    assert by_objective["mixed_pair"]["objective_rank"] == ""


def test_constrained_oracle_rejects_t_unsafe_candidate() -> None:
    decomp_rows = [
        make_decomp("toy", "factor_count", "ok", tcount=10),
        make_decomp("toy", "mixed_pair", "ok", tcount=20),
    ]
    grid_rows = [
        make_grid("toy", "factor_count", tcount=10, primary=2.0, qasm=100),
        make_grid("toy", "mixed_pair", tcount=20, primary=0.1, qasm=80),
    ]

    rows = dataset_rows_for_split("unit", decomp_rows, grid_rows, {})
    by_objective = {row["objective_variant"]: row for row in rows}

    assert by_objective["factor_count"]["objective_is_oracle"] is True
    assert by_objective["factor_count"]["oracle_selection_status"] == "constrained"
    assert by_objective["mixed_pair"]["objective_t_safe"] is False


def test_constrained_oracle_can_select_safe_nonbaseline_objective() -> None:
    decomp_rows = [
        make_decomp("toy", "factor_count", "ok", tcount=10),
        make_decomp("toy", "factor_count_pair_cap", "ok", tcount=10),
    ]
    grid_rows = [
        make_grid("toy", "factor_count", tcount=10, primary=2.0, qasm=100),
        make_grid("toy", "factor_count_pair_cap", tcount=9, primary=1.5, qasm=90),
    ]

    rows = dataset_rows_for_split("unit", decomp_rows, grid_rows, {})
    by_objective = {row["objective_variant"]: row for row in rows}

    assert by_objective["factor_count_pair_cap"]["objective_is_oracle"] is True
    assert by_objective["factor_count_pair_cap"]["objective_t_safe"] is True
    assert by_objective["factor_count_pair_cap"]["objective_qasm_safe"] is True


def test_readiness_requires_enough_groups_and_label_diversity() -> None:
    rows = []
    for index in range(8):
        target = f"target_{index}"
        oracle = "factor_count" if index < 4 else "mixed_pair"
        group = dataset_rows_for_split(
            "unit",
            [
                make_decomp(target, "factor_count", "ok", tcount=1 if oracle == "factor_count" else 2),
                make_decomp(target, "mixed_pair", "ok", tcount=1 if oracle == "mixed_pair" else 2),
            ],
            [
                make_grid(target, "factor_count", tcount=1 if oracle == "factor_count" else 2, primary=1, qasm=10),
                make_grid(target, "mixed_pair", tcount=1 if oracle == "mixed_pair" else 2, primary=1, qasm=10),
            ],
            {},
        )
        rows.extend(group)

    summary = readiness_summary(rows)

    assert summary["decision"] == "prototype-ready"
    assert summary["train_ready_groups"] == 8
    assert summary["oracle_objective_counts"] == {"factor_count": 4, "mixed_pair": 4}


def test_build_dataset_uses_available_sources(tmp_path: Path) -> None:
    internal_decomp = tmp_path / "internal_decomp.csv"
    internal_grid = tmp_path / "internal_grid.csv"
    readiness = tmp_path / "readiness.csv"
    write_rows(
        internal_decomp,
        [
            make_decomp("toy", "factor_count", "ok", tcount=2),
            make_decomp("toy", "mixed_pair", "ok", tcount=1),
        ],
    )
    write_rows(
        internal_grid,
        [
            make_grid("toy", "factor_count", tcount=2, primary=1, qasm=20),
            make_grid("toy", "mixed_pair", tcount=1, primary=1, qasm=20),
        ],
    )
    readiness.write_text(
        "target,family,n_qubits,tensor_size,tcount_original\n"
        "toy,unit,3,4,7\n",
        encoding="utf-8",
    )

    rows = build_dataset(
        internal_decomposition_csvs=[internal_decomp],
        internal_grid_csv=internal_grid,
        external_decomposition_csv=tmp_path / "missing_ext.csv",
        external_grid_csv=tmp_path / "missing_ext_grid.csv",
        night_decomposition_csv=tmp_path / "missing_night.csv",
        night_grid_csv=tmp_path / "missing_night_grid.csv",
        readiness_csv=readiness,
    )

    toy_rows = [row for row in rows if row["target"] == "toy"]
    assert len(toy_rows) == len(dataset.OBJECTIVES)
    assert toy_rows[0]["family"] == "unit"
    assert {row["oracle_objective"] for row in toy_rows} == {"mixed_pair"}


def test_build_dataset_includes_extra_external_run_suffixes(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(dataset, "PROJECT_ROOT", tmp_path)
    csv_root = tmp_path / "results" / "csv"
    csv_root.mkdir(parents=True)
    run = "journal_full_toy"
    decomp = csv_root / f"alphaq_decomposition_objective_external_validation_{run}.csv"
    grid = csv_root / f"alphaq_objective_beam_policy_external_validation_{run}_grid.csv"
    write_rows(
        decomp,
        [
            make_decomp("toy_external", "factor_count", "ok", tcount=4),
            make_decomp("toy_external", "mixed_pair", "ok", tcount=3),
        ],
    )
    write_rows(
        grid,
        [
            make_grid("toy_external", "factor_count", tcount=4, primary=1, qasm=30),
            make_grid("toy_external", "mixed_pair", tcount=3, primary=1, qasm=25),
        ],
    )
    readiness = tmp_path / "readiness.csv"
    readiness.write_text(
        "target,family,n_qubits,tensor_size,tcount_original\n"
        "toy_external,external-family,5,8,9\n",
        encoding="utf-8",
    )
    internal_grid = tmp_path / "internal_grid.csv"
    internal_grid.write_text("target,objective_variant,materializer\n", encoding="utf-8")

    rows = build_dataset(
        internal_decomposition_csvs=[],
        internal_grid_csv=internal_grid,
        external_decomposition_csv=tmp_path / "missing_ext.csv",
        external_grid_csv=tmp_path / "missing_ext_grid.csv",
        night_decomposition_csv=tmp_path / "missing_night.csv",
        night_grid_csv=tmp_path / "missing_night_grid.csv",
        readiness_csv=readiness,
        extra_external_runs=[run],
    )

    assert {row["source_split"] for row in rows} == {f"external_{run}"}
    assert {row["oracle_objective"] for row in rows} == {"mixed_pair"}
    assert rows[0]["family"] == "external-family"


def test_run_suffixes_strips_empty_items() -> None:
    assert run_suffixes("a,, b ,") == ["a", "b"]


def make_decomp(target: str, objective: str, status: str, *, tcount: int | str) -> dict[str, str]:
    return {
        "target": target,
        "objective_variant": objective,
        "execution_status": status,
        "factor_count": str(tcount),
        "factor_qubit_concentration_index": "0.1",
        "factor_support_weight_mean": "2",
        "factor_pairwise_support_overlap_mean": "0.5",
        "factor_pairwise_jaccard_mean": "0.2",
        "tcount": str(tcount),
        "tdepth": str(tcount),
        "qasm_depth": "100",
        "qasm_depth_ratio": "1.0",
        "structural_target_status": "ok",
    }


def make_grid(target: str, objective: str, *, tcount: int, primary: float, qasm: int) -> dict[str, str]:
    return {
        "target": target,
        "objective_variant": objective,
        "materializer": "selected-beam-shared-parity-w4",
        "tcount": str(tcount),
        "tdepth": str(tcount),
        "qasm_depth": str(qasm),
        "qasm_depth_ratio": "1.0",
        "primary_nc_depth_ratio": str(primary),
        "num_total_cnots": "3",
        "summary_path": "summary.json",
    }


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    lines = [",".join(fieldnames)]
    for row in rows:
        lines.append(",".join(row.get(field, "") for field in fieldnames))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
