from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from atq_repro.csvio import write_csv_rows
from atq_repro.csvio import write_json
from atq_repro.expected import PAPER_BENCHMARK_GADGETS
from atq_repro.expected import PAPER_BENCHMARK_NO_GADGETS
from atq_repro.expected import expected_binary_addition_effective_tcount
from atq_repro.expected import expected_gf_gadget_effective_tcount
from atq_repro.expected import expected_gf_no_gadget_tcount
from atq_repro.gadgets import analyze_gadgetization
from atq_repro.io import block_label_from_key
from atq_repro.io import circuit_id_from_key
from atq_repro.io import iter_decompositions
from atq_repro.io import load_target_tensor
from atq_repro.paths import PAPER_REPRO_ROOT
from atq_repro.paths import REPORTS_ROOT
from atq_repro.paths import ensure_dir
from atq_repro.paths import relative_to_project
from atq_repro.plots import generate_paper_figures
from atq_repro.tensor import validate_decomposition


def _bool_text(value: bool | None) -> str:
    if value is None:
        return ""
    return "true" if value else "false"


def build_long_results() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family, npz_path, key, candidate_index, factors in iter_decompositions():
        tensor_path, target_tensor = load_target_tensor(key)
        validation = validate_decomposition(factors, target_tensor)
        gadget_summary = analyze_gadgetization(factors, use_gadgets=family.use_gadgets)
        rows.append(
            {
                "family_file": family.file_name,
                "family_label": family.family_label,
                "method": family.method,
                "use_gadgets": family.use_gadgets,
                "decomposition_key": key,
                "circuit_id": circuit_id_from_key(key),
                "block_label": block_label_from_key(key),
                "candidate_index": candidate_index,
                "num_factors": gadget_summary.num_factors,
                "tensor_size": int(np.asarray(factors).shape[1]),
                "n_toffoli_gadgets": gadget_summary.num_toffoli,
                "n_cs_gadgets": gadget_summary.num_cs,
                "n_t_remaining": gadget_summary.num_t_remaining,
                "effective_tcount": gadget_summary.effective_tcount,
                "tensor_path": relative_to_project(tensor_path),
                "tensor_equal": _bool_text(validation.equal),
                "tensor_mismatch_count": validation.mismatch_count,
                "tensor_target_shape": validation.target_shape,
                "tensor_reconstructed_shape": validation.reconstructed_shape,
                "tensor_validation_error": validation.error,
                "npz_path": relative_to_project(npz_path),
            }
        )
    return rows


def best_rows_by_key(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["family_file"], row["decomposition_key"])].append(row)

    best_rows = []
    for group_rows in groups.values():
        best_rows.append(
            min(
                group_rows,
                key=lambda row: (
                    int(row["effective_tcount"]),
                    int(row["num_factors"]),
                    int(row["candidate_index"]),
                ),
            )
        )
    return sorted(best_rows, key=lambda row: (row["family_file"], row["circuit_id"], row["decomposition_key"]))


def aggregate_best_totals(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best = best_rows_by_key(rows)
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in best:
        groups[(row["family_file"], row["method"], row["circuit_id"])].append(row)

    aggregate_rows = []
    for (family_file, method, circuit_id), group_rows in sorted(groups.items()):
        aggregate_rows.append(
            {
                "family_file": family_file,
                "method": method,
                "family_label": group_rows[0]["family_label"],
                "circuit_id": circuit_id,
                "num_blocks": len(group_rows),
                "best_effective_tcount": sum(int(row["effective_tcount"]) for row in group_rows),
                "best_num_factors": sum(int(row["num_factors"]) for row in group_rows),
                "best_n_toffoli_gadgets": sum(int(row["n_toffoli_gadgets"]) for row in group_rows),
                "best_n_cs_gadgets": sum(int(row["n_cs_gadgets"]) for row in group_rows),
                "best_n_t_remaining": sum(int(row["n_t_remaining"]) for row in group_rows),
                "all_tensor_equal": all(row["tensor_equal"] == "true" for row in group_rows),
                "best_decomposition_keys": ";".join(row["decomposition_key"] for row in group_rows),
                "best_candidate_indices": ";".join(str(row["candidate_index"]) for row in group_rows),
            }
        )
    return aggregate_rows


def tensor_validation_rows(long_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "family_file": row["family_file"],
            "decomposition_key": row["decomposition_key"],
            "candidate_index": row["candidate_index"],
            "circuit_id": row["circuit_id"],
            "tensor_path": row["tensor_path"],
            "tensor_equal": row["tensor_equal"],
            "tensor_mismatch_count": row["tensor_mismatch_count"],
            "tensor_validation_error": row["tensor_validation_error"],
        }
        for row in long_rows
    ]


def family_summary_rows(long_rows: list[dict[str, Any]], aggregate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for family_file in sorted({row["family_file"] for row in long_rows}):
        family_long = [row for row in long_rows if row["family_file"] == family_file]
        family_agg = [row for row in aggregate_rows if row["family_file"] == family_file]
        rows.append(
            {
                "family_file": family_file,
                "family_label": family_long[0]["family_label"],
                "method": family_long[0]["method"],
                "num_decomposition_keys": len({row["decomposition_key"] for row in family_long}),
                "num_candidates": len(family_long),
                "num_circuits": len({row["circuit_id"] for row in family_long}),
                "num_best_totals": len(family_agg),
                "num_tensor_equal_candidates": sum(row["tensor_equal"] == "true" for row in family_long),
                "num_missing_tensor_candidates": sum(bool(row["tensor_validation_error"]) for row in family_long),
                "mean_best_effective_tcount": float(np.mean([row["best_effective_tcount"] for row in family_agg])) if family_agg else None,
            }
        )
    return rows


def _comparison_rows_for_expected(
    *,
    aggregate_index: dict[tuple[str, str, str], dict[str, Any]],
    source: str,
    family_file: str,
    method: str,
    expected: dict[str, int],
) -> list[dict[str, Any]]:
    rows = []
    for circuit_id, expected_tcount in sorted(expected.items()):
        reproduced = aggregate_index.get((family_file, method, circuit_id))
        reproduced_tcount = None if reproduced is None else int(reproduced["best_effective_tcount"])
        rows.append(
            {
                "source": source,
                "family_file": family_file,
                "method": method,
                "circuit_id": circuit_id,
                "paper_effective_tcount": expected_tcount,
                "reproduced_effective_tcount": reproduced_tcount,
                "delta": None if reproduced_tcount is None else reproduced_tcount - expected_tcount,
                "match": _bool_text(reproduced_tcount == expected_tcount if reproduced_tcount is not None else False),
                "all_tensor_equal": "" if reproduced is None else _bool_text(bool(reproduced["all_tensor_equal"])),
                "best_decomposition_keys": "" if reproduced is None else reproduced["best_decomposition_keys"],
                "best_candidate_indices": "" if reproduced is None else reproduced["best_candidate_indices"],
            }
        )
    return rows


def benchmark_comparison_rows(aggregate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregate_index = {
        (row["family_file"], row["method"], row["circuit_id"]): row
        for row in aggregate_rows
    }
    benchmark_no_gf = {
        key: value
        for key, value in PAPER_BENCHMARK_NO_GADGETS.items()
        if not key.startswith("gf_2pow")
    }
    benchmark_gf_no = {
        key: value
        for key, value in PAPER_BENCHMARK_NO_GADGETS.items()
        if key.startswith("gf_2pow")
    }
    benchmark_with_gf = {
        key: value
        for key, value in PAPER_BENCHMARK_GADGETS.items()
        if not key.startswith("gf_2pow")
    }
    benchmark_gf_with = {
        key: value
        for key, value in PAPER_BENCHMARK_GADGETS.items()
        if key.startswith("gf_2pow")
    }
    rows: list[dict[str, Any]] = []
    rows.extend(
        _comparison_rows_for_expected(
            aggregate_index=aggregate_index,
            source="paper_table_benchmark_no_gadgets",
            family_file="benchmarks_no_gadgets.npz",
            method="paper_no_gadgets",
            expected=benchmark_no_gf,
        )
    )
    rows.extend(
        _comparison_rows_for_expected(
            aggregate_index=aggregate_index,
            source="paper_table_benchmark_no_gadgets",
            family_file="multiplication_finite_fields_no_gadgets.npz",
            method="paper_no_gadgets",
            expected=benchmark_gf_no,
        )
    )
    rows.extend(
        _comparison_rows_for_expected(
            aggregate_index=aggregate_index,
            source="paper_table_benchmark_gadgets",
            family_file="benchmarks_gadgets.npz",
            method="paper_gadgets",
            expected=benchmark_with_gf,
        )
    )
    rows.extend(
        _comparison_rows_for_expected(
            aggregate_index=aggregate_index,
            source="paper_table_benchmark_gadgets",
            family_file="multiplication_finite_fields_gadgets.npz",
            method="paper_gadgets",
            expected=benchmark_gf_with,
        )
    )
    rows.extend(
        _comparison_rows_for_expected(
            aggregate_index=aggregate_index,
            source="paper_fig4_gf_no_gadgets",
            family_file="multiplication_finite_fields_no_gadgets.npz",
            method="paper_no_gadgets",
            expected=expected_gf_no_gadget_tcount(),
        )
    )
    rows.extend(
        _comparison_rows_for_expected(
            aggregate_index=aggregate_index,
            source="paper_fig4_gf_gadgets",
            family_file="multiplication_finite_fields_gadgets.npz",
            method="paper_gadgets",
            expected=expected_gf_gadget_effective_tcount(),
        )
    )
    rows.extend(
        _comparison_rows_for_expected(
            aggregate_index=aggregate_index,
            source="paper_fig4_binary_addition",
            family_file="binary_addition.npz",
            method="paper_gadgets",
            expected=expected_binary_addition_effective_tcount(),
        )
    )
    return rows


def write_reproduction_report(
    *,
    report_path: Path,
    comparison_rows: list[dict[str, Any]],
    family_rows: list[dict[str, Any]],
    figure_paths: dict[str, Path],
) -> Path:
    total = len(comparison_rows)
    matches = sum(row["match"] == "true" for row in comparison_rows)
    tensor_ok = sum(row["all_tensor_equal"] == "true" for row in comparison_rows)
    lines = [
        "# Reproducao do artigo AlphaTensor-Quantum",
        "",
        "## Cobertura",
        "",
        f"- Linhas comparadas com o artigo: {total}.",
        f"- Matches exatos reproduzidos: {matches}/{total}.",
        f"- Linhas cujas decomposicoes selecionadas validam tensorialmente: {tensor_ok}/{total}.",
        "",
        "## Familias",
        "",
    ]
    for row in family_rows:
        lines.append(
            f"- `{row['family_file']}`: {row['num_decomposition_keys']} chaves de decomposicao, "
            f"{row['num_candidates']} candidatos, {row['num_tensor_equal_candidates']} candidatos tensor-equal."
        )
    lines.extend(
        [
            "",
            "## Figuras",
            "",
            *[f"- `{name}`: `{path}`" for name, path in sorted(figure_paths.items())],
            "",
            "## Interpretacao para a Entrega 1",
            "",
            "Use esta reproducao como a replicacao principal do baseline. Ela valida as decomposicoes oficiais do AlphaTensor-Quantum recomputando seus tensores assinatura e recalculando os T-counts efetivos com gadgets reportados no artigo.",
            "",
            "A reconstrucao QASM separada e as checagens com `feynver` continuam uteis como camada de auditoria, mas a verificacao tensorial e o alvo fiel de reproducao das decomposicoes otimizadas porque o artigo original publica decomposicoes otimizadas, nao circuitos reconstruidos completos para todos os casos.",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def reproduce_paper_results(output_dir: Path = PAPER_REPRO_ROOT) -> dict[str, Path]:
    ensure_dir(output_dir)
    long_rows = build_long_results()
    aggregate_rows = aggregate_best_totals(long_rows)
    validation_rows = tensor_validation_rows(long_rows)
    family_rows = family_summary_rows(long_rows, aggregate_rows)
    comparison_rows = benchmark_comparison_rows(aggregate_rows)

    paths = {
        "paper_results_long_csv": output_dir / "paper_results_long.csv",
        "paper_tensor_validation_csv": output_dir / "paper_tensor_validation.csv",
        "paper_benchmark_comparison_csv": output_dir / "paper_benchmark_comparison.csv",
        "paper_family_summary_csv": output_dir / "paper_family_summary.csv",
        "paper_aggregate_best_csv": output_dir / "paper_aggregate_best.csv",
        "paper_summary_json": output_dir / "paper_summary.json",
    }
    write_csv_rows(long_rows, paths["paper_results_long_csv"])
    write_csv_rows(validation_rows, paths["paper_tensor_validation_csv"])
    write_csv_rows(comparison_rows, paths["paper_benchmark_comparison_csv"])
    write_csv_rows(family_rows, paths["paper_family_summary_csv"])
    write_csv_rows(aggregate_rows, paths["paper_aggregate_best_csv"])

    figure_paths = generate_paper_figures(
        comparison_rows=comparison_rows,
        aggregate_rows=aggregate_rows,
        output_dir=output_dir / "figures",
    )
    report_path = write_reproduction_report(
        report_path=REPORTS_ROOT / "paper_reproduction_summary.md",
        comparison_rows=comparison_rows,
        family_rows=family_rows,
        figure_paths=figure_paths,
    )
    paths["paper_reproduction_report"] = report_path
    paths.update({f"figure_{name}": path for name, path in figure_paths.items()})
    write_json(
        {
            "num_long_rows": len(long_rows),
            "num_comparison_rows": len(comparison_rows),
            "num_exact_matches": sum(row["match"] == "true" for row in comparison_rows),
            "num_tensor_equal_candidates": sum(row["tensor_equal"] == "true" for row in long_rows),
            "outputs": {name: str(path) for name, path in paths.items()},
        },
        paths["paper_summary_json"],
    )
    return paths
