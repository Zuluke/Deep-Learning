from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._analysis_common import DEFAULT_RESYNTH_ROOT
from scripts._analysis_common import DECOMPOSITIONS_ROOT
from scripts._analysis_common import artifact_stem_candidates
from scripts._analysis_common import compute_metrics_from_qasm_path
from scripts._analysis_common import ensure_dir
from scripts._analysis_common import list_nonclifford_block_stems
from scripts._analysis_common import natural_sort_key
from scripts._analysis_common import write_csv_rows
from scripts._analysis_common import write_json
from scripts.alphatensor_structural_cost import circuit_selection_row_from_qasm
from scripts.alphatensor_structural_cost import compute_selection_metrics_from_qasm
from scripts.alphatensor_structural_cost import structural_selection_key
from scripts.assemble_resynth_circuit import assemble_circuit
from scripts._manifest import append_command


GADGET_METHOD = "public_resynth_gadgets"
NO_GADGET_METHOD = "public_resynth_no_gadgets"
STRUCTURAL_METHOD = "public_resynth_structural"
DEFAULT_STRUCTURAL_MAX_CANDIDATES_PER_KEY = 10
DEFAULT_STRUCTURAL_CIRCUIT_IDS = (
    "mod_5_4",
    "gf_2pow2_mult",
    "cuccaro_adder_n3",
    "qft_4",
    "vbe_adder_3",
)
GADGET_FAMILIES = (
    "benchmarks_gadgets.npz",
    "binary_addition.npz",
    "hamming_weight_phase_gradient.npz",
    "multiplication_finite_fields_gadgets.npz",
    "quantum_chemistry.npz",
    "unary_iteration_gadgets.npz",
)
NO_GADGET_FAMILIES = (
    "benchmarks_no_gadgets.npz",
    "multiplication_finite_fields_no_gadgets.npz",
    "unary_iteration_no_gadgets.npz",
)
STRUCTURAL_FAMILY_SPECS = tuple(
    [(family, False) for family in NO_GADGET_FAMILIES]
    + [(family, True) for family in GADGET_FAMILIES]
)


def compiled_binary() -> Path:
    return PROJECT_ROOT / "external" / "circuit-to-tensor" / "target" / "release" / "circuit-to-tensor"


def load_inventory_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_decomposition_index(family_files: tuple[str, ...]) -> dict[str, list[tuple[Path, str]]]:
    index: dict[str, list[tuple[Path, str]]] = {}
    for family_file in family_files:
        npz_path = DECOMPOSITIONS_ROOT / family_file
        if not npz_path.exists():
            continue
        with np.load(npz_path, allow_pickle=True) as data:
            for key in data.files:
                for candidate in artifact_stem_candidates(key):
                    index.setdefault(candidate, []).append((npz_path, key))
    return index


def load_structural_decomposition_index() -> dict[str, list[tuple[Path, str, bool]]]:
    index: dict[str, list[tuple[Path, str, bool]]] = {}
    for family_file, use_gadgets in STRUCTURAL_FAMILY_SPECS:
        npz_path = DECOMPOSITIONS_ROOT / family_file
        if not npz_path.exists():
            continue
        with np.load(npz_path, allow_pickle=True) as data:
            for key in data.files:
                for candidate in artifact_stem_candidates(key):
                    index.setdefault(candidate, []).append((npz_path, key, use_gadgets))
    return index


def run_resynth(
    *,
    decomposition_path: Path,
    mapping_path: Path,
    original_path: Path,
    output_dir: Path,
    use_gadgets: bool,
) -> tuple[str, str | None]:
    ensure_dir(output_dir)
    cmd = [
        str(compiled_binary()),
        "resynth",
        "-e",
        "circuit-qasm,log",
        "-m",
        str(mapping_path),
        "-O",
        str(original_path),
    ]
    if use_gadgets:
        cmd.append("-g")
    cmd.extend([str(output_dir), str(decomposition_path)])
    completed = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return "failed", completed.stderr[-4000:] or completed.stdout[-4000:]
    return "ok", None


def public_source_method(use_gadgets: bool) -> str:
    return GADGET_METHOD if use_gadgets else NO_GADGET_METHOD


def safe_source_stem(npz_path: Path, key: str) -> str:
    clean_key = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in key)
    return f"{npz_path.stem}.{clean_key}"


def bool_as_int(value: bool) -> int:
    return 1 if value else 0


def choose_best_block_candidate(
    *,
    artifact_stem: str,
    compile_dir: Path,
    matches: list[tuple[Path, str]],
    method_dir: Path,
    use_gadgets: bool,
) -> dict[str, Any]:
    mapping_path = compile_dir / f"{artifact_stem}.mapping.txt"
    original_path = compile_dir / f"{artifact_stem}.matrix.npy"
    if not mapping_path.exists() or not original_path.exists():
        return {"status": "missing-compile-artifacts", "error": f"Missing mapping/original for {artifact_stem}"}

    best_candidate: dict[str, Any] | None = None
    candidates_root = ensure_dir(method_dir / "candidates" / artifact_stem)
    for npz_path, key in matches:
        with np.load(npz_path, allow_pickle=True) as data:
            decompositions = np.asarray(data[key])
        for candidate_index, candidate in enumerate(decompositions[:3]):
            candidate_npy = candidates_root / f"{artifact_stem}.candidate{candidate_index}.npy"
            # AlphaTensor-Quantum stores decompositions as (num_factors, tensor_size),
            # while circuit-to-tensor resynth expects (tensor_size, num_factors).
            np.save(candidate_npy, np.asarray(candidate).T)
            candidate_output = ensure_dir(candidates_root / f"candidate{candidate_index}")
            status, error = run_resynth(
                decomposition_path=candidate_npy,
                mapping_path=mapping_path,
                original_path=original_path,
                output_dir=candidate_output,
                use_gadgets=use_gadgets,
            )
            qasm_path = candidate_output / f"{candidate_npy.stem}.qasm"
            if status != "ok" or not qasm_path.exists():
                continue
            metrics = compute_metrics_from_qasm_path(qasm_path)
            tcount = metrics.get("tcount")
            if tcount is None:
                continue
            candidate_summary = {
                "status": "ok",
                "npz_path": str(npz_path.relative_to(PROJECT_ROOT)),
                "decomposition_key": key,
                "candidate_index": candidate_index,
                "candidate_npy": str(candidate_npy),
                "candidate_qasm_path": str(qasm_path),
                "tcount": tcount,
                "tdepth": metrics.get("tdepth"),
            }
            if best_candidate is None or (
                candidate_summary["tcount"],
                candidate_summary["tdepth"],
                candidate_summary["candidate_index"],
            ) < (
                best_candidate["tcount"],
                best_candidate["tdepth"],
                best_candidate["candidate_index"],
            ):
                best_candidate = candidate_summary

    if best_candidate is None:
        return {
            "status": "no-successful-candidate",
            "error": f"No successful resynth candidate for {artifact_stem}",
        }

    final_qasm_path = method_dir / f"{artifact_stem}.qasm"
    shutil.copyfile(best_candidate["candidate_qasm_path"], final_qasm_path)
    best_candidate["final_qasm_path"] = str(final_qasm_path)
    return best_candidate


def collect_structural_block_candidates(
    *,
    artifact_stem: str,
    compile_dir: Path,
    matches: list[tuple[Path, str, bool]],
    method_dir: Path,
    max_candidates_per_key: int,
) -> list[dict[str, Any]]:
    mapping_path = compile_dir / f"{artifact_stem}.mapping.txt"
    original_path = compile_dir / f"{artifact_stem}.matrix.npy"
    if not mapping_path.exists() or not original_path.exists():
        return []

    block_candidates = []
    candidates_root = ensure_dir(method_dir / "candidates" / artifact_stem)
    for npz_path, key, use_gadgets in matches:
        with np.load(npz_path, allow_pickle=True) as data:
            decompositions = np.asarray(data[key])
        source_stem = safe_source_stem(npz_path, key)
        for candidate_index, candidate in enumerate(decompositions[:max_candidates_per_key]):
            candidate_npy = (
                candidates_root
                / f"{artifact_stem}.{source_stem}.candidate{candidate_index}.npy"
            )
            np.save(candidate_npy, np.asarray(candidate).T)
            candidate_output = ensure_dir(
                candidates_root / source_stem / f"candidate{candidate_index}"
            )
            status, error = run_resynth(
                decomposition_path=candidate_npy,
                mapping_path=mapping_path,
                original_path=original_path,
                output_dir=candidate_output,
                use_gadgets=use_gadgets,
            )
            qasm_path = candidate_output / f"{candidate_npy.stem}.qasm"
            if status != "ok" or not qasm_path.exists():
                continue
            metrics = compute_metrics_from_qasm_path(qasm_path)
            tcount = metrics.get("tcount")
            if tcount is None:
                continue
            block_candidates.append(
                {
                    "status": "ok",
                    "npz_path": str(npz_path.relative_to(PROJECT_ROOT)),
                    "decomposition_key": key,
                    "source_method": public_source_method(use_gadgets),
                    "use_gadgets": use_gadgets,
                    "candidate_index": candidate_index,
                    "candidate_npy": str(candidate_npy),
                    "candidate_qasm_path": str(qasm_path),
                    "tcount": tcount,
                    "tdepth": metrics.get("tdepth"),
                }
            )
    return block_candidates


def choose_structural_circuit_candidate(
    *,
    circuit_row: dict[str, str],
    compile_dir: Path,
    block_stems: list[str],
    decomposition_index: dict[str, list[tuple[Path, str, bool]]],
    method_dir: Path,
    max_candidates_per_key: int,
) -> dict[str, Any]:
    original_qasm = PROJECT_ROOT / circuit_row["qasm_path"]
    if not original_qasm.exists():
        return {
            "status": "missing-original",
            "error": f"Missing original QASM for {circuit_row['circuit_id']}",
        }

    original_row = circuit_selection_row_from_qasm(original_qasm)
    candidates_by_block: list[tuple[str, list[dict[str, Any]]]] = []
    for artifact_stem in sorted(block_stems, key=natural_sort_key):
        matches = decomposition_index.get(artifact_stem, [])
        if not matches:
            return {
                "status": "not-available",
                "error": f"Missing public decomposition for {artifact_stem}",
            }
        block_candidates = collect_structural_block_candidates(
            artifact_stem=artifact_stem,
            compile_dir=compile_dir,
            matches=matches,
            method_dir=method_dir,
            max_candidates_per_key=max_candidates_per_key,
        )
        if not block_candidates:
            return {
                "status": "no-successful-candidate",
                "error": f"No successful resynth candidate for {artifact_stem}",
            }
        candidates_by_block.append((artifact_stem, block_candidates))

    best_combo: dict[str, Any] | None = None
    frontier_rows: list[dict[str, Any]] = []
    combinations_root = ensure_dir(method_dir / "structural_combinations")
    ordered_block_stems = [artifact_stem for artifact_stem, _ in candidates_by_block]
    candidate_lists = [candidates for _, candidates in candidates_by_block]
    for combo_index, combo in enumerate(itertools.product(*candidate_lists)):
        combo_dir = ensure_dir(combinations_root / f"combo{combo_index}")
        for artifact_stem, candidate in zip(ordered_block_stems, combo):
            shutil.copyfile(
                candidate["candidate_qasm_path"],
                combo_dir / f"{artifact_stem}.qasm",
            )
        try:
            assembled_qasm, assembled_summary = assemble_circuit(compile_dir, combo_dir)
            selection_metrics = compute_selection_metrics_from_qasm(
                candidate_qasm=assembled_qasm,
                original_row=original_row,
            )
        except Exception as exc:
            selection_metrics = {
                "selection_status": "failed",
                "selection_error": str(exc),
            }
            assembled_qasm = None
            assembled_summary = None

        combo_summary = {
            "status": "ok" if selection_metrics.get("selection_status") == "ok" else "failed",
            "combo_index": combo_index,
            "combo_dir": str(combo_dir),
            "candidate_qasm_path": None if assembled_qasm is None else str(assembled_qasm),
            "block_choices": [
                {
                    "artifact_stem": artifact_stem,
                    "source_method": candidate["source_method"],
                    "npz_path": candidate["npz_path"],
                    "decomposition_key": candidate["decomposition_key"],
                    "candidate_index": candidate["candidate_index"],
                    "candidate_qasm_path": candidate["candidate_qasm_path"],
                    "tcount": candidate["tcount"],
                    "tdepth": candidate["tdepth"],
                }
                for artifact_stem, candidate in zip(ordered_block_stems, combo)
            ],
            "assembled_summary": assembled_summary,
            **selection_metrics,
        }
        frontier_rows.append(
            candidate_frontier_row(
                circuit_id=circuit_row["circuit_id"],
                combo_summary=combo_summary,
            )
        )
        if combo_summary["status"] != "ok":
            continue
        if best_combo is None or structural_selection_key(
            combo_summary, combo_index
        ) < structural_selection_key(best_combo, best_combo["combo_index"]):
            best_combo = combo_summary

    if best_combo is None:
        return {
            "status": "no-structural-candidate",
            "error": "No assembled candidate had valid structural target metrics.",
            "candidate_frontier_rows": frontier_rows,
        }

    for block_choice in best_combo["block_choices"]:
        shutil.copyfile(
            block_choice["candidate_qasm_path"],
            method_dir / f"{block_choice['artifact_stem']}.qasm",
        )
    assembled_qasm, assembled_summary = assemble_circuit(compile_dir, method_dir)
    write_json(assembled_summary, method_dir / "assembled_summary.json")
    return {
        **best_combo,
        "status": "ok",
        "error": None,
        "final_qasm_path": str(assembled_qasm),
        "assembled_qasm_path": str(assembled_qasm),
        "assembled_summary": assembled_summary,
        "candidate_frontier_rows": frontier_rows,
    }


def candidate_frontier_row(
    *,
    circuit_id: str,
    combo_summary: dict[str, Any],
) -> dict[str, Any]:
    block_choices = combo_summary.get("block_choices", [])
    num_blocks = len(block_choices)
    num_gadget_blocks = sum(
        1 for block in block_choices if block.get("source_method") == GADGET_METHOD
    )
    block_tcounts = [
        value for block in block_choices if (value := block.get("tcount")) is not None
    ]
    block_tdepths = [
        value for block in block_choices if (value := block.get("tdepth")) is not None
    ]
    return {
        "circuit_id": circuit_id,
        "candidate_id": f"{circuit_id}:combo{combo_summary.get('combo_index')}",
        "combo_index": combo_summary.get("combo_index"),
        "status": combo_summary.get("status"),
        "selection_status": combo_summary.get("selection_status"),
        "selection_error": combo_summary.get("selection_error"),
        "source_methods": ";".join(
            str(block.get("source_method")) for block in block_choices
        ),
        "decomposition_keys": ";".join(
            str(block.get("decomposition_key")) for block in block_choices
        ),
        "candidate_indices": ";".join(
            str(block.get("candidate_index")) for block in block_choices
        ),
        "num_blocks": num_blocks,
        "num_gadget_blocks": num_gadget_blocks,
        "num_no_gadget_blocks": num_blocks - num_gadget_blocks,
        "gadget_block_fraction": 0.0 if num_blocks == 0 else num_gadget_blocks / num_blocks,
        "uses_any_gadget_source": bool_as_int(num_gadget_blocks > 0),
        "block_tcount_sum": sum(block_tcounts) if block_tcounts else None,
        "block_tdepth_sum": sum(block_tdepths) if block_tdepths else None,
        "structural_cost": combo_summary.get("structural_cost"),
        "primary_nc_depth_ratio": combo_summary.get("primary_nc_depth_ratio"),
        "primary_nc_depth_delta_vs_original": combo_summary.get(
            "primary_nc_depth_delta_vs_original"
        ),
        "zx_total_depth_ratio": combo_summary.get("zx_total_depth_ratio"),
        "qasm_depth_ratio": combo_summary.get("qasm_depth_ratio"),
        "tcount_ratio": combo_summary.get("tcount_ratio"),
        "tdepth_ratio": combo_summary.get("tdepth_ratio"),
        "gate_count_ratio": combo_summary.get("gate_count_ratio"),
        "tcount_after": combo_summary.get("tcount_after"),
        "tdepth_after": combo_summary.get("tdepth_after"),
        "depth_after": combo_summary.get("depth_after"),
        "gate_count_after": combo_summary.get("gate_count_after"),
        "rho_t": combo_summary.get("rho_t"),
        "rho_w": combo_summary.get("rho_w"),
        "n_clifford_blocks": combo_summary.get("n_clifford_blocks"),
        "n_nonclifford_blocks": combo_summary.get("n_nonclifford_blocks"),
        "avg_nonclifford_block_len": combo_summary.get("avg_nonclifford_block_len"),
        "hadamard_boundary_density": combo_summary.get("hadamard_boundary_density"),
        "tdepth_over_tcount": combo_summary.get("tdepth_over_tcount"),
        "candidate_qasm_path": combo_summary.get("candidate_qasm_path"),
    }


def replay_for_method(
    *,
    inventory_rows: list[dict[str, str]],
    method_name: str,
    family_files: tuple[str, ...],
    output_root: Path,
) -> list[dict[str, Any]]:
    decomposition_index = load_decomposition_index(family_files)
    summary_rows: list[dict[str, Any]] = []
    use_gadgets = method_name == GADGET_METHOD

    for row in inventory_rows:
        if not row.get("vendored_compile_dir"):
            continue
        compile_dir = PROJECT_ROOT / row["vendored_compile_dir"]
        method_dir = ensure_dir(output_root / method_name / row["circuit_id"])
        block_stems = list_nonclifford_block_stems(compile_dir)
        if not block_stems:
            summary_rows.append(
                {
                    "circuit_id": row["circuit_id"],
                    "method": method_name,
                    "status": "not-available",
                    "assembled_qasm_path": None,
                    "error": "No non-Clifford block stems available.",
                }
            )
            continue

        block_results: list[dict[str, Any]] = []
        method_status = "ok"
        method_error = None
        for artifact_stem in sorted(block_stems, key=natural_sort_key):
            matches = decomposition_index.get(artifact_stem, [])
            if not matches:
                method_status = "not-available"
                method_error = f"Missing public decomposition for {artifact_stem}"
                break
            block_result = choose_best_block_candidate(
                artifact_stem=artifact_stem,
                compile_dir=compile_dir,
                matches=matches,
                method_dir=method_dir,
                use_gadgets=use_gadgets,
            )
            block_results.append({"artifact_stem": artifact_stem, **block_result})
            if block_result["status"] != "ok":
                method_status = "failed"
                method_error = block_result.get("error")
                break

        assembled_qasm_path = None
        if method_status == "ok":
            try:
                assembled_qasm, assembled_summary = assemble_circuit(compile_dir, method_dir)
                assembled_qasm_path = str(assembled_qasm)
                write_json(assembled_summary, method_dir / "assembled_summary.json")
            except Exception as exc:
                method_status = "failed"
                method_error = str(exc)

        write_json(
            {
                "circuit_id": row["circuit_id"],
                "method": method_name,
                "status": method_status,
                "error": method_error,
                "block_results": block_results,
                "assembled_qasm_path": assembled_qasm_path,
            },
            method_dir / "summary.json",
        )
        summary_rows.append(
            {
                "circuit_id": row["circuit_id"],
                "method": method_name,
                "status": method_status,
                "assembled_qasm_path": assembled_qasm_path,
                "error": method_error,
            }
        )
    return summary_rows


def replay_structural_method(
    *,
    inventory_rows: list[dict[str, str]],
    output_root: Path,
    max_candidates_per_key: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    decomposition_index = load_structural_decomposition_index()
    summary_rows: list[dict[str, Any]] = []
    frontier_rows: list[dict[str, Any]] = []

    for row in inventory_rows:
        if not row.get("vendored_compile_dir"):
            continue
        compile_dir = PROJECT_ROOT / row["vendored_compile_dir"]
        method_dir = ensure_dir(output_root / STRUCTURAL_METHOD / row["circuit_id"])
        block_stems = list_nonclifford_block_stems(compile_dir)
        if not block_stems:
            summary_rows.append(
                {
                    "circuit_id": row["circuit_id"],
                    "method": STRUCTURAL_METHOD,
                    "status": "not-available",
                    "selection_objective": "structural",
                    "assembled_qasm_path": None,
                    "error": "No non-Clifford block stems available.",
                }
            )
            continue

        structural_result = choose_structural_circuit_candidate(
            circuit_row=row,
            compile_dir=compile_dir,
            block_stems=block_stems,
            decomposition_index=decomposition_index,
            method_dir=method_dir,
            max_candidates_per_key=max_candidates_per_key,
        )
        candidate_frontier_rows = structural_result.pop("candidate_frontier_rows", [])
        frontier_rows.extend(candidate_frontier_rows)
        write_csv_rows(candidate_frontier_rows, method_dir / "candidate_frontier.csv")
        write_json(
            {
                "circuit_id": row["circuit_id"],
                "method": STRUCTURAL_METHOD,
                "selection_objective": "structural",
                **structural_result,
            },
            method_dir / "summary.json",
        )
        summary_rows.append(
            {
                "circuit_id": row["circuit_id"],
                "method": STRUCTURAL_METHOD,
                "status": structural_result["status"],
                "selection_objective": "structural",
                "selection_status": structural_result.get("selection_status"),
                "selection_error": structural_result.get("selection_error"),
                "structural_cost": structural_result.get("structural_cost"),
                "primary_nc_depth_ratio": structural_result.get("primary_nc_depth_ratio"),
                "zx_total_depth_ratio": structural_result.get("zx_total_depth_ratio"),
                "qasm_depth_ratio": structural_result.get("qasm_depth_ratio"),
                "tcount_ratio": structural_result.get("tcount_ratio"),
                "tcount_after_selection": structural_result.get("tcount_after"),
                "tdepth_after_selection": structural_result.get("tdepth_after"),
                "combo_index": structural_result.get("combo_index"),
                "assembled_qasm_path": structural_result.get("assembled_qasm_path"),
                "error": structural_result.get("error"),
            }
        )
    return summary_rows, frontier_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay public AlphaTensor-Quantum decompositions and assemble full circuits.")
    parser.add_argument("--inventory-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_RESYNTH_ROOT)
    parser.add_argument(
        "--selection-objective",
        choices=("tcount", "structural"),
        default="tcount",
    )
    parser.add_argument(
        "--max-candidates-per-key",
        type=int,
        default=DEFAULT_STRUCTURAL_MAX_CANDIDATES_PER_KEY,
    )
    parser.add_argument("--circuit-id", action="append", dest="circuit_ids", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inventory_rows = load_inventory_rows(args.inventory_csv)
    circuit_ids = args.circuit_ids
    if args.selection_objective == "structural" and circuit_ids is None:
        circuit_ids = list(DEFAULT_STRUCTURAL_CIRCUIT_IDS)
    if circuit_ids is not None:
        circuit_ids = tuple(circuit_ids)
        circuit_id_set = set(circuit_ids)
        inventory_rows = [
            row for row in inventory_rows if row["circuit_id"] in circuit_id_set
        ]
    ensure_dir(args.output_root)

    summary_rows = []
    frontier_rows = []
    if args.selection_objective == "structural":
        structural_summary_rows, frontier_rows = replay_structural_method(
            inventory_rows=inventory_rows,
            output_root=args.output_root,
            max_candidates_per_key=args.max_candidates_per_key,
        )
        summary_rows.extend(structural_summary_rows)
    else:
        summary_rows.extend(
            replay_for_method(
                inventory_rows=inventory_rows,
                method_name=NO_GADGET_METHOD,
                family_files=NO_GADGET_FAMILIES,
                output_root=args.output_root,
            )
        )
        summary_rows.extend(
            replay_for_method(
                inventory_rows=inventory_rows,
                method_name=GADGET_METHOD,
                family_files=GADGET_FAMILIES,
                output_root=args.output_root,
            )
        )

    summary_csv = args.output_root / "public_resynth_summary.csv"
    summary_json = args.output_root / "public_resynth_summary.json"
    frontier_csv = args.output_root / "candidate_frontier.csv"
    write_csv_rows(summary_rows, summary_csv)
    if frontier_rows:
        write_csv_rows(frontier_rows, frontier_csv)
    write_json(
        {
            "inventory_csv": str(args.inventory_csv),
            "output_root": str(args.output_root),
            "selection_objective": args.selection_objective,
            "max_candidates_per_key": args.max_candidates_per_key,
            "circuit_ids": [row["circuit_id"] for row in inventory_rows],
            "num_rows": len(summary_rows),
            "candidate_frontier_csv": str(frontier_csv) if frontier_rows else None,
            "num_candidate_frontier_rows": len(frontier_rows),
        },
        summary_json,
    )
    append_command(
        {
            "tool": "replay_public_decompositions.py",
            "command": (
                f"{sys.executable} scripts/replay_public_decompositions.py "
                f"--inventory-csv {args.inventory_csv} --output-root {args.output_root} "
                f"--selection-objective {args.selection_objective} "
                f"--max-candidates-per-key {args.max_candidates_per_key}"
                + "".join(
                    f" --circuit-id {circuit_id}"
                    for circuit_id in (args.circuit_ids or [])
                )
            ),
            "cwd": str(PROJECT_ROOT),
            "inventory_csv": str(args.inventory_csv),
            "selection_objective": args.selection_objective,
            "circuit_ids": [row["circuit_id"] for row in inventory_rows],
            "summary_csv": str(summary_csv),
            "summary_json": str(summary_json),
            "candidate_frontier_csv": str(frontier_csv) if frontier_rows else None,
            "exit_code": 0,
        }
    )
    print(json.dumps({"summary_csv": str(summary_csv), "summary_json": str(summary_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
