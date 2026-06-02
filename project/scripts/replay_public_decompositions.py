from __future__ import annotations

import argparse
import csv
import hashlib
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
from scripts.tensor_split_core import default_partitions
from scripts.tensor_split_core import group_multiset_gadgets
from scripts.tensor_split_core import semantic_partitions_from_qasm
from scripts.tensor_split_core import tensor_split_v3_stats


GADGET_METHOD = "public_resynth_gadgets"
NO_GADGET_METHOD = "public_resynth_no_gadgets"
STRUCTURAL_METHOD = "public_resynth_structural"
TENSOR_V3_METHOD = "public_resynth_tensor_v3"
TENSOR_V3_PHASE_SLACK_METHOD = "public_resynth_tensor_v3_phase_slack"
TENSOR_V3_PHASE_SLACK_AGGRESSIVE_METHOD = (
    "public_resynth_tensor_v3_phase_slack_aggressive"
)
DEFAULT_STRUCTURAL_MAX_CANDIDATES_PER_KEY = 10
DEFAULT_TENSOR_V3_TCOUNT_TOLERANCE = 0.12
DEFAULT_TENSOR_V3_RANKING = "lex-v1"
DEFAULT_TENSOR_V3_MIXED_SLACK = 0.20
TENSOR_V3_PROFILE_SPECS = {
    "conservative": {
        "method": TENSOR_V3_PHASE_SLACK_METHOD,
        "tcount_tolerance": 0.12,
        "mixed_slack": 0.20,
    },
    "splitting-aggressive": {
        "method": TENSOR_V3_PHASE_SLACK_AGGRESSIVE_METHOD,
        "tcount_tolerance": 0.40,
        "mixed_slack": 0.20,
    },
}
DEFAULT_STRUCTURAL_CIRCUIT_IDS = (
    "mod_5_4",
    "gf_2pow2_mult",
    "cuccaro_adder_n3",
    "qft_4",
    "vbe_adder_3",
    "hamming_weight_n4",
    "hamming_weight_n5",
    "barenco_tof_3",
    "barenco_tof_4",
    "hwb_6",
    "nc_tof_3",
    "nc_tof_4",
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
TENSOR_V3_PARTITION_PRIORITY = (
    "semantic_role_k2",
    "semantic_family_k2",
    "semantic_register_k2",
    "tensor_graph_spectral_k2",
    "balanced_contiguous_k2",
)
TENSOR_V3_FIELDS = (
    "tensor_v3_status",
    "tensor_v3_error",
    "tensor_v3_partition_id",
    "tensor_v3_partition_kind",
    "tensor_v3_tcount_tolerance",
    "tensor_v3_target_mixed_weight",
    "tensor_v3_gadget_mixed_weight",
    "tensor_v3_gadget_mixed_weight_norm",
    "tensor_v3_mixed_excess_norm",
    "tensor_v3_mixed_auc_original",
    "tensor_v3_mixed_auc_original_norm",
    "tensor_v3_mixed_auc_greedy",
    "tensor_v3_mixed_auc_greedy_norm",
    "tensor_v3_singleton_bridge_count",
    "tensor_v3_singleton_bridge_count_norm",
    "tensor_v3_mixed_group_count",
    "tensor_v3_mixed_group_count_norm",
    "tensor_v3_num_factor_groups",
    "tensor_v3_factor_count",
    "tensor_v3_score_lex",
    "tensor_v3_stable_factor_hash",
    "tensor_v3_partition_metrics_json",
)
TENSOR_V3_PARTITION_METRIC_KEYS = (
    "target_mixed_weight",
    "gadget_mixed_weight",
    "gadget_mixed_weight_norm",
    "mixed_excess_norm",
    "mixed_auc_original",
    "mixed_auc_original_norm",
    "mixed_auc_greedy",
    "mixed_auc_greedy_norm",
    "singleton_bridge_count",
    "singleton_bridge_count_norm",
    "mixed_group_count",
    "mixed_group_count_norm",
    "stable_factor_hash",
    "score_v3_lex",
)
TENSOR_V3_SELECTION_FIELDS = (
    "tensor_v3_selection_status",
    "tensor_v3_selection_error",
    "tensor_v3_ranking_strategy",
    "tensor_v3_profile",
    "tensor_v3_mixed_slack",
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


def stable_hash_from_parts(parts: list[str]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def finite_or_inf(value: Any) -> float:
    try:
        if value in (None, "", "None"):
            return float("inf")
        return float(value)
    except (TypeError, ValueError):
        return float("inf")


def tensor_v3_metric_key(metrics: dict[str, Any]) -> tuple[float, float, float, str]:
    return (
        finite_or_inf(metrics.get("tensor_v3_mixed_excess_norm")),
        finite_or_inf(metrics.get("tensor_v3_mixed_auc_greedy_norm")),
        finite_or_inf(metrics.get("tensor_v3_singleton_bridge_count_norm")),
        str(metrics.get("tensor_v3_stable_factor_hash") or ""),
    )


def tensor_v3_profile_settings(
    *,
    profile: str | None,
    tcount_tolerance: float,
    mixed_slack: float,
) -> tuple[str | None, float, float]:
    if profile is None:
        return None, tcount_tolerance, mixed_slack
    if profile not in TENSOR_V3_PROFILE_SPECS:
        raise ValueError(f"Unknown tensor-v3 profile: {profile}")
    spec = TENSOR_V3_PROFILE_SPECS[profile]
    return (
        profile,
        float(spec["tcount_tolerance"]),
        float(spec["mixed_slack"]),
    )


def tensor_v3_method_name(
    *,
    selection_objective: str,
    ranking_strategy: str,
    profile: str | None,
) -> str:
    if selection_objective == "structural":
        return STRUCTURAL_METHOD
    if ranking_strategy != "phase-slack-v1":
        return TENSOR_V3_METHOD
    if profile is not None:
        return str(TENSOR_V3_PROFILE_SPECS[profile]["method"])
    return TENSOR_V3_PHASE_SLACK_METHOD


def qasm_only_circuit_row_from_qasm(path: Path) -> dict[str, Any]:
    qasm_metrics = compute_metrics_from_qasm_path(path)
    return {
        "qasm_path": str(path),
        "tcount_after": qasm_metrics.get("tcount"),
        "tdepth_after": qasm_metrics.get("tdepth"),
        "depth_after": qasm_metrics.get("normalized_qasm_depth"),
        "gate_count_after": qasm_metrics.get("normalized_qasm_size"),
        "rho_t": qasm_metrics.get("rho_t"),
        "rho_w": qasm_metrics.get("rho_w"),
        "n_clifford_blocks": qasm_metrics.get("n_clifford_blocks"),
        "n_nonclifford_blocks": qasm_metrics.get("n_nonclifford_blocks"),
        "avg_nonclifford_block_len": qasm_metrics.get("avg_nonclifford_block_len"),
        "hadamard_boundary_density": qasm_metrics.get("hadamard_boundary_density"),
        "tdepth_over_tcount": qasm_metrics.get("tdepth_over_tcount"),
    }


def compute_tensor_v3_pure_selection_metrics(
    *,
    candidate_qasm: Path,
    original_row: dict[str, Any],
    ranking_strategy: str,
    profile: str | None,
    mixed_slack: float,
) -> dict[str, Any]:
    candidate_row = qasm_only_circuit_row_from_qasm(candidate_qasm)
    if candidate_row.get("tcount_after") is None:
        return {
            "selection_status": "missing-qasm-tcount",
            "selection_error": "Candidate QASM metrics did not contain T-count.",
            "tensor_v3_selection_status": "missing-qasm-tcount",
            "tensor_v3_selection_error": "Candidate QASM metrics did not contain T-count.",
            "tensor_v3_ranking_strategy": ranking_strategy,
            "tensor_v3_profile": profile,
            "tensor_v3_mixed_slack": mixed_slack,
        }
    return {
        "selection_status": "ok",
        "selection_error": None,
        "tensor_v3_selection_status": "ok",
        "tensor_v3_selection_error": None,
        "tensor_v3_ranking_strategy": ranking_strategy,
        "tensor_v3_profile": profile,
        "tensor_v3_mixed_slack": mixed_slack,
        "tcount_ratio": _safe_ratio(
            candidate_row.get("tcount_after"), original_row.get("tcount_after")
        ),
        "tdepth_ratio": _safe_ratio(
            candidate_row.get("tdepth_after"), original_row.get("tdepth_after")
        ),
        "gate_count_ratio": _safe_ratio(
            candidate_row.get("gate_count_after"), original_row.get("gate_count_after")
        ),
        "qasm_depth_ratio": _safe_ratio(
            candidate_row.get("depth_after"), original_row.get("depth_after")
        ),
        **candidate_row,
    }


def _safe_ratio(numerator: Any, denominator: Any) -> float | None:
    try:
        if numerator in (None, "", "None") or denominator in (None, "", "None"):
            return None
        denominator_float = float(denominator)
        if denominator_float == 0:
            return None
        return float(numerator) / denominator_float
    except (TypeError, ValueError):
        return None


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
    circuit_id: str,
    original_qasm: Path,
    artifact_stem: str,
    compile_dir: Path,
    matches: list[tuple[Path, str, bool]],
    method_dir: Path,
    max_candidates_per_key: int,
    selection_objective: str,
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
            candidate_row = {
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
            if selection_objective == "tensor-v3":
                candidate_row.update(
                    tensor_v3_block_metrics(
                        circuit_id=circuit_id,
                        artifact_stem=artifact_stem,
                        compile_dir=compile_dir,
                        original_qasm=original_qasm,
                        factors=np.asarray(candidate),
                    )
                )
            block_candidates.append(candidate_row)
    return block_candidates


def tensor_v3_block_metrics(
    *,
    circuit_id: str,
    artifact_stem: str,
    compile_dir: Path,
    original_qasm: Path,
    factors: np.ndarray,
) -> dict[str, Any]:
    tensor_path = compile_dir / f"{artifact_stem}.tensor.npy"
    mapping_path = compile_dir / f"{artifact_stem}.mapping.txt"
    if not tensor_path.exists():
        return {
            "tensor_v3_status": "missing-tensor",
            "tensor_v3_error": str(tensor_path),
        }
    try:
        tensor = np.load(tensor_path, allow_pickle=True).astype(np.uint8) % 2
        groups = group_multiset_gadgets(factors)
        semantic_partitions = (
            semantic_partitions_from_qasm(
                circuit_id=circuit_id,
                qasm_path=original_qasm,
                mapping_path=mapping_path,
                tensor_size=tensor.shape[0],
            )
            if original_qasm.exists()
            else []
        )
        partitions = [*semantic_partitions, *default_partitions(tensor, random_seeds=())]
        partitions_by_id = {partition.partition_id: partition for partition in partitions}
        partition = next(
            (
                partitions_by_id[partition_id]
                for partition_id in TENSOR_V3_PARTITION_PRIORITY
                if partition_id in partitions_by_id
            ),
            partitions[0],
        )
        partition_metric_rows = []
        for candidate_partition in partitions:
            candidate_stats = tensor_split_v3_stats(
                tensor, factors, candidate_partition, groups=groups
            )
            partition_metric_rows.append(
                {
                    "partition_id": candidate_partition.partition_id,
                    "partition_kind": candidate_partition.kind,
                    "num_factor_groups": len(groups),
                    "factor_count": int(np.asarray(factors).shape[0]),
                    **{
                        key: candidate_stats.get(key)
                        for key in TENSOR_V3_PARTITION_METRIC_KEYS
                    },
                }
            )
        stats = next(
            row
            for row in partition_metric_rows
            if row["partition_id"] == partition.partition_id
        )
    except Exception as exc:
        return {
            "tensor_v3_status": "failed",
            "tensor_v3_error": str(exc),
        }
    return {
        "tensor_v3_status": "ok",
        "tensor_v3_error": None,
        "tensor_v3_partition_id": partition.partition_id,
        "tensor_v3_partition_kind": partition.kind,
        "tensor_v3_target_mixed_weight": stats.get("target_mixed_weight"),
        "tensor_v3_gadget_mixed_weight": stats.get("gadget_mixed_weight"),
        "tensor_v3_gadget_mixed_weight_norm": stats.get(
            "gadget_mixed_weight_norm"
        ),
        "tensor_v3_mixed_excess_norm": stats.get("mixed_excess_norm"),
        "tensor_v3_mixed_auc_original": stats.get("mixed_auc_original"),
        "tensor_v3_mixed_auc_original_norm": stats.get(
            "mixed_auc_original_norm"
        ),
        "tensor_v3_mixed_auc_greedy": stats.get("mixed_auc_greedy"),
        "tensor_v3_mixed_auc_greedy_norm": stats.get("mixed_auc_greedy_norm"),
        "tensor_v3_singleton_bridge_count": stats.get("singleton_bridge_count"),
        "tensor_v3_singleton_bridge_count_norm": stats.get(
            "singleton_bridge_count_norm"
        ),
        "tensor_v3_mixed_group_count": stats.get("mixed_group_count"),
        "tensor_v3_mixed_group_count_norm": stats.get("mixed_group_count_norm"),
        "tensor_v3_score_lex": stats.get("score_v3_lex"),
        "tensor_v3_stable_factor_hash": stats.get("stable_factor_hash"),
        "tensor_v3_num_factor_groups": len(groups),
        "tensor_v3_factor_count": int(np.asarray(factors).shape[0]),
        "tensor_v3_partition_metrics_json": json.dumps(
            partition_metric_rows,
            sort_keys=True,
            separators=(",", ":"),
        ),
    }


def choose_structural_circuit_candidate(
    *,
    circuit_row: dict[str, str],
    compile_dir: Path,
    block_stems: list[str],
    decomposition_index: dict[str, list[tuple[Path, str, bool]]],
    method_dir: Path,
    max_candidates_per_key: int,
    max_combinations: int | None,
    selection_objective: str,
    tensor_v3_tcount_tolerance: float,
    tensor_v3_ranking_strategy: str,
    tensor_v3_profile: str | None,
    tensor_v3_mixed_slack: float,
) -> dict[str, Any]:
    original_qasm = PROJECT_ROOT / circuit_row["qasm_path"]
    if not original_qasm.exists():
        return {
            "status": "missing-original",
            "error": f"Missing original QASM for {circuit_row['circuit_id']}",
        }

    original_row = (
        qasm_only_circuit_row_from_qasm(original_qasm)
        if selection_objective == "tensor-v3"
        else circuit_selection_row_from_qasm(original_qasm)
    )
    candidates_by_block: list[tuple[str, list[dict[str, Any]]]] = []
    for artifact_stem in sorted(block_stems, key=natural_sort_key):
        matches = decomposition_index.get(artifact_stem, [])
        if not matches:
            return {
                "status": "not-available",
                "error": f"Missing public decomposition for {artifact_stem}",
            }
        block_candidates = collect_structural_block_candidates(
            circuit_id=circuit_row["circuit_id"],
            original_qasm=original_qasm,
            artifact_stem=artifact_stem,
            compile_dir=compile_dir,
            matches=matches,
            method_dir=method_dir,
            max_candidates_per_key=max_candidates_per_key,
            selection_objective=selection_objective,
        )
        if not block_candidates:
            return {
                "status": "no-successful-candidate",
                "error": f"No successful resynth candidate for {artifact_stem}",
            }
        candidates_by_block.append((artifact_stem, block_candidates))

    best_combo: dict[str, Any] | None = None
    valid_combos: list[dict[str, Any]] = []
    combo_summaries: list[dict[str, Any]] = []
    tensor_v3_candidate_manifest_rows: list[dict[str, Any]] = []
    combinations_root = ensure_dir(method_dir / "structural_combinations")
    ordered_block_stems = [artifact_stem for artifact_stem, _ in candidates_by_block]
    candidate_lists = [candidates for _, candidates in candidates_by_block]
    combinations = itertools.product(*candidate_lists)
    if max_combinations is not None:
        combinations = itertools.islice(combinations, max_combinations)
    for combo_index, combo in enumerate(combinations):
        combo_dir = ensure_dir(combinations_root / f"combo{combo_index}")
        for artifact_stem, candidate in zip(ordered_block_stems, combo):
            shutil.copyfile(
                candidate["candidate_qasm_path"],
                combo_dir / f"{artifact_stem}.qasm",
            )
        try:
            assembled_qasm, assembled_summary = assemble_circuit(compile_dir, combo_dir)
            if selection_objective == "tensor-v3":
                selection_metrics = compute_tensor_v3_pure_selection_metrics(
                    candidate_qasm=assembled_qasm,
                    original_row=original_row,
                    ranking_strategy=tensor_v3_ranking_strategy,
                    profile=tensor_v3_profile,
                    mixed_slack=tensor_v3_mixed_slack,
                )
            else:
                selection_metrics = compute_selection_metrics_from_qasm(
                    candidate_qasm=assembled_qasm,
                    original_row=original_row,
                )
        except Exception as exc:
            selection_metrics = {
                "selection_status": "failed",
                "selection_error": str(exc),
                "tensor_v3_selection_status": "failed",
                "tensor_v3_selection_error": str(exc),
                "tensor_v3_ranking_strategy": tensor_v3_ranking_strategy,
                "tensor_v3_profile": tensor_v3_profile,
                "tensor_v3_mixed_slack": tensor_v3_mixed_slack,
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
                    **{
                        field: candidate.get(field)
                        for field in TENSOR_V3_FIELDS
                        if field in candidate
                    },
                }
                for artifact_stem, candidate in zip(ordered_block_stems, combo)
            ],
            "assembled_summary": assembled_summary,
            **selection_metrics,
        }
        if selection_objective == "tensor-v3":
            combo_summary.update(
                aggregate_tensor_v3_combo_metrics(
                    combo, tensor_v3_tcount_tolerance=tensor_v3_tcount_tolerance
                )
            )
            tensor_v3_candidate_manifest_rows.append(
                tensor_v3_manifest_row(
                    circuit_id=circuit_row["circuit_id"],
                    combo_summary=combo_summary,
                    selected=False,
                )
            )
        combo_summaries.append(combo_summary)
        if combo_summary["status"] != "ok":
            continue
        valid_combos.append(combo_summary)

    if valid_combos and selection_objective == "tensor-v3":
        best_tcount = min(
            finite_or_inf(combo.get("tcount_after")) for combo in valid_combos
        )
        best_combo = select_tensor_v3_combo(
            valid_combos,
            best_tcount=best_tcount,
            ranking_strategy=tensor_v3_ranking_strategy,
            mixed_slack=tensor_v3_mixed_slack,
        )
    elif valid_combos:
        best_combo = min(
            valid_combos,
            key=lambda combo: structural_selection_key(combo, int(combo["combo_index"])),
        )

    if selection_objective == "tensor-v3" and best_combo is not None:
        for row in tensor_v3_candidate_manifest_rows:
            row["tensor_v3_selected"] = bool_as_int(
                row.get("combo_index") == best_combo.get("combo_index")
            )
        original_audit_row = circuit_selection_row_from_qasm(original_qasm)
        for combo_summary in combo_summaries:
            candidate_qasm = combo_summary.get("candidate_qasm_path")
            if combo_summary.get("status") != "ok" or not candidate_qasm:
                continue
            try:
                combo_summary.update(
                    compute_selection_metrics_from_qasm(
                        candidate_qasm=Path(candidate_qasm),
                        original_row=original_audit_row,
                    )
                )
                combo_summary.setdefault("tensor_v3_selection_status", "ok")
                combo_summary.setdefault("tensor_v3_selection_error", None)
                combo_summary.setdefault(
                    "tensor_v3_ranking_strategy", tensor_v3_ranking_strategy
                )
                combo_summary.setdefault("tensor_v3_profile", tensor_v3_profile)
                combo_summary.setdefault("tensor_v3_mixed_slack", tensor_v3_mixed_slack)
            except Exception as exc:
                combo_summary.update(
                    {
                        "selection_status": "audit-failed",
                        "selection_error": str(exc),
                    }
                )

    frontier_rows = [
        candidate_frontier_row(
            circuit_id=circuit_row["circuit_id"],
            combo_summary=combo_summary,
        )
        for combo_summary in combo_summaries
    ]

    if best_combo is None:
        return {
            "status": "no-structural-candidate",
            "error": "No assembled candidate had valid structural target metrics.",
            "candidate_frontier_rows": frontier_rows,
            "tensor_v3_candidate_manifest_rows": tensor_v3_candidate_manifest_rows,
            "tensor_v3_selection_manifest_rows": [],
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
        "tensor_v3_candidate_manifest_rows": tensor_v3_candidate_manifest_rows,
        "tensor_v3_selection_manifest_rows": (
            [
                tensor_v3_manifest_row(
                    circuit_id=circuit_row["circuit_id"],
                    combo_summary=best_combo,
                    selected=True,
                )
            ]
            if selection_objective == "tensor-v3"
            else []
        ),
    }


def aggregate_tensor_v3_combo_metrics(
    combo: tuple[dict[str, Any], ...],
    *,
    tensor_v3_tcount_tolerance: float,
) -> dict[str, Any]:
    if any(candidate.get("tensor_v3_status") != "ok" for candidate in combo):
        return {
            "tensor_v3_status": "failed",
            "tensor_v3_error": ";".join(
                str(candidate.get("tensor_v3_error") or "unknown")
                for candidate in combo
                if candidate.get("tensor_v3_status") != "ok"
            ),
            "tensor_v3_tcount_tolerance": tensor_v3_tcount_tolerance,
        }

    target_mixed_weight = sum(
        finite_or_inf(candidate.get("tensor_v3_target_mixed_weight"))
        for candidate in combo
    )
    gadget_mixed_weight = sum(
        finite_or_inf(candidate.get("tensor_v3_gadget_mixed_weight"))
        for candidate in combo
    )
    mixed_auc_original = sum(
        finite_or_inf(candidate.get("tensor_v3_mixed_auc_original"))
        for candidate in combo
    )
    mixed_auc_greedy = sum(
        finite_or_inf(candidate.get("tensor_v3_mixed_auc_greedy"))
        for candidate in combo
    )
    singleton_bridge_count = sum(
        finite_or_inf(candidate.get("tensor_v3_singleton_bridge_count"))
        for candidate in combo
    )
    mixed_group_count = sum(
        finite_or_inf(candidate.get("tensor_v3_mixed_group_count"))
        for candidate in combo
    )
    num_factor_groups = sum(
        finite_or_inf(candidate.get("tensor_v3_num_factor_groups"))
        for candidate in combo
    )
    factor_count = sum(
        finite_or_inf(candidate.get("tensor_v3_factor_count")) for candidate in combo
    )
    target_denominator = max(target_mixed_weight, 1.0)
    auc_denominator = (num_factor_groups + len(combo)) * target_denominator
    mixed_excess_norm = (
        gadget_mixed_weight - target_mixed_weight
    ) / target_denominator
    mixed_auc_greedy_norm = mixed_auc_greedy / max(auc_denominator, 1.0)
    singleton_bridge_count_norm = singleton_bridge_count / max(factor_count, 1.0)
    stable_hash = stable_hash_from_parts(
        [str(candidate.get("tensor_v3_stable_factor_hash") or "") for candidate in combo]
    )
    partition_metrics_json = aggregate_tensor_v3_partition_metrics_json(combo)
    return {
        "tensor_v3_status": "ok",
        "tensor_v3_error": None,
        "tensor_v3_partition_id": ";".join(
            str(candidate.get("tensor_v3_partition_id")) for candidate in combo
        ),
        "tensor_v3_partition_kind": ";".join(
            str(candidate.get("tensor_v3_partition_kind")) for candidate in combo
        ),
        "tensor_v3_tcount_tolerance": tensor_v3_tcount_tolerance,
        "tensor_v3_target_mixed_weight": target_mixed_weight,
        "tensor_v3_gadget_mixed_weight": gadget_mixed_weight,
        "tensor_v3_gadget_mixed_weight_norm": gadget_mixed_weight
        / target_denominator,
        "tensor_v3_mixed_excess_norm": mixed_excess_norm,
        "tensor_v3_mixed_auc_original": mixed_auc_original,
        "tensor_v3_mixed_auc_original_norm": mixed_auc_original
        / max(auc_denominator, 1.0),
        "tensor_v3_mixed_auc_greedy": mixed_auc_greedy,
        "tensor_v3_mixed_auc_greedy_norm": mixed_auc_greedy_norm,
        "tensor_v3_singleton_bridge_count": singleton_bridge_count,
        "tensor_v3_singleton_bridge_count_norm": singleton_bridge_count_norm,
        "tensor_v3_mixed_group_count": mixed_group_count,
        "tensor_v3_mixed_group_count_norm": mixed_group_count
        / max(num_factor_groups, 1.0),
        "tensor_v3_num_factor_groups": num_factor_groups,
        "tensor_v3_factor_count": factor_count,
        "tensor_v3_score_lex": (
            f"{mixed_excess_norm:.12g};"
            f"{mixed_auc_greedy_norm:.12g};"
            f"{singleton_bridge_count_norm:.12g};"
            f"{stable_hash}"
        ),
        "tensor_v3_stable_factor_hash": stable_hash,
        "tensor_v3_partition_metrics_json": partition_metrics_json,
    }


def aggregate_tensor_v3_partition_metrics_json(
    combo: tuple[dict[str, Any], ...],
) -> str:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for candidate in combo:
        raw_json = candidate.get("tensor_v3_partition_metrics_json")
        if not raw_json:
            continue
        try:
            rows = json.loads(str(raw_json))
        except json.JSONDecodeError:
            continue
        for row in rows:
            partition_id = str(row.get("partition_id") or "")
            if partition_id:
                grouped.setdefault(partition_id, []).append(row)

    aggregate_rows = []
    for partition_id, rows in grouped.items():
        target_mixed_weight = sum(
            finite_or_inf(row.get("target_mixed_weight")) for row in rows
        )
        gadget_mixed_weight = sum(
            finite_or_inf(row.get("gadget_mixed_weight")) for row in rows
        )
        mixed_auc_original = sum(
            finite_or_inf(row.get("mixed_auc_original")) for row in rows
        )
        mixed_auc_greedy = sum(
            finite_or_inf(row.get("mixed_auc_greedy")) for row in rows
        )
        singleton_bridge_count = sum(
            finite_or_inf(row.get("singleton_bridge_count")) for row in rows
        )
        mixed_group_count = sum(
            finite_or_inf(row.get("mixed_group_count")) for row in rows
        )
        num_factor_groups = sum(
            finite_or_inf(row.get("num_factor_groups")) for row in rows
        )
        factor_count = sum(finite_or_inf(row.get("factor_count")) for row in rows)
        target_denominator = max(target_mixed_weight, 1.0)
        auc_denominator = (num_factor_groups + len(rows)) * target_denominator
        stable_hash = stable_hash_from_parts(
            [str(row.get("stable_factor_hash") or "") for row in rows]
        )
        mixed_excess_norm = (
            gadget_mixed_weight - target_mixed_weight
        ) / target_denominator
        mixed_auc_original_norm = mixed_auc_original / max(auc_denominator, 1.0)
        mixed_auc_greedy_norm = mixed_auc_greedy / max(auc_denominator, 1.0)
        singleton_bridge_count_norm = singleton_bridge_count / max(factor_count, 1.0)
        aggregate_rows.append(
            {
                "partition_id": partition_id,
                "partition_kind": rows[0].get("partition_kind"),
                "num_block_rows": len(rows),
                "num_factor_groups": num_factor_groups,
                "factor_count": factor_count,
                "target_mixed_weight": target_mixed_weight,
                "gadget_mixed_weight": gadget_mixed_weight,
                "gadget_mixed_weight_norm": gadget_mixed_weight
                / target_denominator,
                "mixed_excess_norm": mixed_excess_norm,
                "mixed_auc_original": mixed_auc_original,
                "mixed_auc_original_norm": mixed_auc_original_norm,
                "mixed_auc_greedy": mixed_auc_greedy,
                "mixed_auc_greedy_norm": mixed_auc_greedy_norm,
                "singleton_bridge_count": singleton_bridge_count,
                "singleton_bridge_count_norm": singleton_bridge_count_norm,
                "mixed_group_count": mixed_group_count,
                "mixed_group_count_norm": mixed_group_count
                / max(num_factor_groups, 1.0),
                "stable_factor_hash": stable_hash,
                "score_v3_lex": (
                    f"{mixed_excess_norm:.12g};"
                    f"{mixed_auc_greedy_norm:.12g};"
                    f"{singleton_bridge_count_norm:.12g};"
                    f"{stable_hash}"
                ),
            }
        )
    aggregate_rows.sort(key=lambda row: str(row["partition_id"]))
    return json.dumps(aggregate_rows, sort_keys=True, separators=(",", ":"))


def tensor_v3_selection_key(
    metrics: dict[str, Any],
    *,
    best_tcount: float,
    tie_breaker: int,
) -> tuple[float, float, float, float, str, float, int]:
    selection_status = metrics.get("tensor_v3_selection_status") or metrics.get(
        "selection_status"
    )
    if selection_status != "ok" or metrics.get("tensor_v3_status") != "ok":
        return (
            float("inf"),
            float("inf"),
            float("inf"),
            float("inf"),
            "",
            float("inf"),
            tie_breaker,
        )
    tolerance = finite_or_inf(metrics.get("tensor_v3_tcount_tolerance"))
    tcount = finite_or_inf(metrics.get("tcount_after"))
    allowed_tcount = best_tcount * (1.0 + (0.0 if tolerance == float("inf") else tolerance))
    outside_tolerance = 1.0 if tcount > allowed_tcount else 0.0
    return (
        outside_tolerance,
        finite_or_inf(metrics.get("tensor_v3_mixed_excess_norm")),
        finite_or_inf(metrics.get("tensor_v3_mixed_auc_greedy_norm")),
        finite_or_inf(metrics.get("tensor_v3_singleton_bridge_count_norm")),
        str(metrics.get("tensor_v3_stable_factor_hash") or ""),
        tcount,
        tie_breaker,
    )


def select_tensor_v3_combo(
    combos: list[dict[str, Any]],
    *,
    best_tcount: float,
    ranking_strategy: str,
    mixed_slack: float,
) -> dict[str, Any]:
    if ranking_strategy == "phase-slack-v1":
        tolerance = finite_or_inf(combos[0].get("tensor_v3_tcount_tolerance"))
        allowed_tcount = best_tcount * (
            1.0 + (0.0 if tolerance == float("inf") else tolerance)
        )
        eligible_mixed = [
            finite_or_inf(combo.get("tensor_v3_mixed_excess_norm"))
            for combo in combos
            if finite_or_inf(combo.get("tcount_after")) <= allowed_tcount
            and (combo.get("tensor_v3_selection_status") or combo.get("selection_status"))
            == "ok"
            and combo.get("tensor_v3_status") == "ok"
        ]
        best_mixed = min(eligible_mixed or [float("inf")])
        return min(
            combos,
            key=lambda combo: tensor_v3_phase_slack_selection_key(
                combo,
                best_tcount=best_tcount,
                best_mixed=best_mixed,
                mixed_slack=mixed_slack,
                tie_breaker=int(combo["combo_index"]),
            ),
        )
    return min(
        combos,
        key=lambda combo: tensor_v3_selection_key(
            combo,
            best_tcount=best_tcount,
            tie_breaker=int(combo["combo_index"]),
        ),
    )


def tensor_v3_phase_slack_selection_key(
    metrics: dict[str, Any],
    *,
    best_tcount: float,
    best_mixed: float,
    mixed_slack: float,
    tie_breaker: int,
) -> tuple[float, float, float, float, float, float, str, float, int]:
    selection_status = metrics.get("tensor_v3_selection_status") or metrics.get(
        "selection_status"
    )
    if selection_status != "ok" or metrics.get("tensor_v3_status") != "ok":
        return (
            float("inf"),
            float("inf"),
            float("inf"),
            float("inf"),
            float("inf"),
            float("inf"),
            "",
            float("inf"),
            tie_breaker,
        )

    tolerance = finite_or_inf(metrics.get("tensor_v3_tcount_tolerance"))
    tcount = finite_or_inf(metrics.get("tcount_after"))
    allowed_tcount = best_tcount * (1.0 + (0.0 if tolerance == float("inf") else tolerance))
    outside_tolerance = 1.0 if tcount > allowed_tcount else 0.0
    allowed_mixed = best_mixed + max(abs(best_mixed) * mixed_slack, 1e-12)
    mixed_excess = finite_or_inf(metrics.get("tensor_v3_mixed_excess_norm"))
    outside_mixed_slack = 1.0 if mixed_excess > allowed_mixed else 0.0
    return (
        outside_tolerance,
        outside_mixed_slack,
        finite_or_inf(metrics.get("tdepth_after")),
        mixed_excess,
        finite_or_inf(metrics.get("tensor_v3_mixed_auc_greedy_norm")),
        finite_or_inf(metrics.get("tensor_v3_singleton_bridge_count_norm")),
        str(metrics.get("tensor_v3_stable_factor_hash") or ""),
        tcount,
        tie_breaker,
    )


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
        **{
            field: combo_summary.get(field)
            for field in TENSOR_V3_SELECTION_FIELDS
        },
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
        "alphaq_target_status": combo_summary.get("alphaq_target_status"),
        "alphaq_target_error": combo_summary.get("alphaq_target_error"),
        "alphaq_nc_core_area_ratio": combo_summary.get("alphaq_nc_core_area_ratio"),
        "alphaq_dependency_core_area_ratio": combo_summary.get(
            "alphaq_dependency_core_area_ratio"
        ),
        "alphaq_nc_core_area_delta_vs_original": combo_summary.get(
            "alphaq_nc_core_area_delta_vs_original"
        ),
        "alphaq_dependency_core_area_delta_vs_original": combo_summary.get(
            "alphaq_dependency_core_area_delta_vs_original"
        ),
        "alphaq_nc_core_depth_ratio": combo_summary.get(
            "alphaq_nc_core_depth_ratio"
        ),
        "alphaq_nc_core_width_ratio": combo_summary.get(
            "alphaq_nc_core_width_ratio"
        ),
        "alphaq_dependency_core_depth_ratio": combo_summary.get(
            "alphaq_dependency_core_depth_ratio"
        ),
        "alphaq_dependency_core_width_ratio": combo_summary.get(
            "alphaq_dependency_core_width_ratio"
        ),
        "alphaq_total_depth_ratio": combo_summary.get("alphaq_total_depth_ratio"),
        "alphaq_total_area_ratio": combo_summary.get("alphaq_total_area_ratio"),
        "alphaq_border_status": combo_summary.get("alphaq_border_status"),
        "alphaq_border_error": combo_summary.get("alphaq_border_error"),
        "alphaq_total_depth": combo_summary.get("alphaq_total_depth"),
        "alphaq_total_width": combo_summary.get("alphaq_total_width"),
        "alphaq_total_area": combo_summary.get("alphaq_total_area"),
        "alphaq_nc_core_depth": combo_summary.get("alphaq_nc_core_depth"),
        "alphaq_nc_core_width": combo_summary.get("alphaq_nc_core_width"),
        "alphaq_nc_core_area": combo_summary.get("alphaq_nc_core_area"),
        "alphaq_core_tcount": combo_summary.get("alphaq_core_tcount"),
        "alphaq_crossing_closure_count": combo_summary.get(
            "alphaq_crossing_closure_count"
        ),
        "alphaq_dependency_core_depth": combo_summary.get(
            "alphaq_dependency_core_depth"
        ),
        "alphaq_dependency_core_width": combo_summary.get(
            "alphaq_dependency_core_width"
        ),
        "alphaq_dependency_core_area": combo_summary.get(
            "alphaq_dependency_core_area"
        ),
        "alphaq_dependency_core_size": combo_summary.get(
            "alphaq_dependency_core_size"
        ),
        "alphaq_dependency_core_tcount": combo_summary.get(
            "alphaq_dependency_core_tcount"
        ),
        "alphaq_dependency_entangling_count": combo_summary.get(
            "alphaq_dependency_entangling_count"
        ),
        "alphaq_dependency_closure_rounds": combo_summary.get(
            "alphaq_dependency_closure_rounds"
        ),
        "alphaq_dependency_internal_edge_count": combo_summary.get(
            "alphaq_dependency_internal_edge_count"
        ),
        "alphaq_dependency_boundary_edge_count": combo_summary.get(
            "alphaq_dependency_boundary_edge_count"
        ),
        "alphaq_dependency_component_count": combo_summary.get(
            "alphaq_dependency_component_count"
        ),
        "alphaq_dependency_largest_component_size": combo_summary.get(
            "alphaq_dependency_largest_component_size"
        ),
        "alphaq_dependency_largest_component_fraction": combo_summary.get(
            "alphaq_dependency_largest_component_fraction"
        ),
        "alphaq_dependency_chain_depth": combo_summary.get(
            "alphaq_dependency_chain_depth"
        ),
        "alphaq_dependency_edge_density": combo_summary.get(
            "alphaq_dependency_edge_density"
        ),
        "alphaq_left_nc_core_depth": combo_summary.get("alphaq_left_nc_core_depth"),
        "alphaq_left_nc_core_width": combo_summary.get("alphaq_left_nc_core_width"),
        "alphaq_left_nc_core_area": combo_summary.get("alphaq_left_nc_core_area"),
        "alphaq_left_crossing_closure_count": combo_summary.get(
            "alphaq_left_crossing_closure_count"
        ),
        "alphaq_right_nc_core_depth": combo_summary.get(
            "alphaq_right_nc_core_depth"
        ),
        "alphaq_right_nc_core_width": combo_summary.get(
            "alphaq_right_nc_core_width"
        ),
        "alphaq_right_nc_core_area": combo_summary.get("alphaq_right_nc_core_area"),
        "alphaq_right_crossing_closure_count": combo_summary.get(
            "alphaq_right_crossing_closure_count"
        ),
        "alphaq_prefix_clifford_depth": combo_summary.get(
            "alphaq_prefix_clifford_depth"
        ),
        "alphaq_suffix_clifford_depth": combo_summary.get(
            "alphaq_suffix_clifford_depth"
        ),
        "alphaq_clifford_shaved_depth_fraction": combo_summary.get(
            "alphaq_clifford_shaved_depth_fraction"
        ),
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
        **{field: combo_summary.get(field) for field in TENSOR_V3_FIELDS},
    }


def tensor_v3_manifest_row(
    *,
    circuit_id: str,
    combo_summary: dict[str, Any],
    selected: bool,
) -> dict[str, Any]:
    block_choices = combo_summary.get("block_choices", [])
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
        "tensor_v3_selected": bool_as_int(selected),
        "status": combo_summary.get("status"),
        "source_methods": ";".join(
            str(block.get("source_method")) for block in block_choices
        ),
        "decomposition_keys": ";".join(
            str(block.get("decomposition_key")) for block in block_choices
        ),
        "candidate_indices": ";".join(
            str(block.get("candidate_index")) for block in block_choices
        ),
        "num_blocks": len(block_choices),
        "block_tcount_sum": sum(block_tcounts) if block_tcounts else None,
        "block_tdepth_sum": sum(block_tdepths) if block_tdepths else None,
        "tcount_after": combo_summary.get("tcount_after"),
        "tdepth_after": combo_summary.get("tdepth_after"),
        "depth_after": combo_summary.get("depth_after"),
        "gate_count_after": combo_summary.get("gate_count_after"),
        "candidate_qasm_path": combo_summary.get("candidate_qasm_path"),
        **{
            field: combo_summary.get(field)
            for field in TENSOR_V3_SELECTION_FIELDS
        },
        **{field: combo_summary.get(field) for field in TENSOR_V3_FIELDS},
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


def replay_selection_method(
    *,
    inventory_rows: list[dict[str, str]],
    output_root: Path,
    max_candidates_per_key: int,
    max_combinations_per_circuit: int | None,
    selection_objective: str,
    method_name: str,
    tensor_v3_tcount_tolerance: float,
    tensor_v3_ranking_strategy: str,
    tensor_v3_profile: str | None,
    tensor_v3_mixed_slack: float,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    decomposition_index = load_structural_decomposition_index()
    summary_rows: list[dict[str, Any]] = []
    frontier_rows: list[dict[str, Any]] = []
    tensor_v3_candidate_manifest_rows: list[dict[str, Any]] = []
    tensor_v3_selection_manifest_rows: list[dict[str, Any]] = []

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
                    "selection_objective": selection_objective,
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
            max_combinations=max_combinations_per_circuit,
            selection_objective=selection_objective,
            tensor_v3_tcount_tolerance=tensor_v3_tcount_tolerance,
            tensor_v3_ranking_strategy=tensor_v3_ranking_strategy,
            tensor_v3_profile=tensor_v3_profile,
            tensor_v3_mixed_slack=tensor_v3_mixed_slack,
        )
        candidate_frontier_rows = structural_result.pop("candidate_frontier_rows", [])
        candidate_manifest_rows = structural_result.pop(
            "tensor_v3_candidate_manifest_rows", []
        )
        selection_manifest_rows = structural_result.pop(
            "tensor_v3_selection_manifest_rows", []
        )
        frontier_rows.extend(candidate_frontier_rows)
        tensor_v3_candidate_manifest_rows.extend(candidate_manifest_rows)
        tensor_v3_selection_manifest_rows.extend(selection_manifest_rows)
        write_csv_rows(candidate_frontier_rows, method_dir / "candidate_frontier.csv")
        if candidate_manifest_rows:
            write_csv_rows(
                candidate_manifest_rows,
                method_dir / "tensor_v3_candidate_manifest.csv",
            )
        if selection_manifest_rows:
            write_csv_rows(
                selection_manifest_rows,
                method_dir / "tensor_v3_selection_manifest.csv",
            )
        write_json(
            {
                "circuit_id": row["circuit_id"],
                "method": method_name,
                "selection_objective": selection_objective,
                **structural_result,
            },
            method_dir / "summary.json",
        )
        summary_rows.append(
            {
                "circuit_id": row["circuit_id"],
                "method": method_name,
                "status": structural_result["status"],
                "selection_objective": selection_objective,
                "selection_status": structural_result.get("selection_status"),
                "selection_error": structural_result.get("selection_error"),
                **{
                    field: structural_result.get(field)
                    for field in TENSOR_V3_SELECTION_FIELDS
                },
                "structural_cost": structural_result.get("structural_cost"),
                "alphaq_target_status": structural_result.get("alphaq_target_status"),
                "alphaq_target_error": structural_result.get("alphaq_target_error"),
                "alphaq_nc_core_area_ratio": structural_result.get(
                    "alphaq_nc_core_area_ratio"
                ),
                "alphaq_dependency_core_area_ratio": structural_result.get(
                    "alphaq_dependency_core_area_ratio"
                ),
                "alphaq_nc_core_area_delta_vs_original": structural_result.get(
                    "alphaq_nc_core_area_delta_vs_original"
                ),
                "alphaq_dependency_core_area_delta_vs_original": structural_result.get(
                    "alphaq_dependency_core_area_delta_vs_original"
                ),
                "alphaq_nc_core_depth_ratio": structural_result.get(
                    "alphaq_nc_core_depth_ratio"
                ),
                "alphaq_nc_core_width_ratio": structural_result.get(
                    "alphaq_nc_core_width_ratio"
                ),
                "alphaq_dependency_core_depth_ratio": structural_result.get(
                    "alphaq_dependency_core_depth_ratio"
                ),
                "alphaq_dependency_core_width_ratio": structural_result.get(
                    "alphaq_dependency_core_width_ratio"
                ),
                "alphaq_total_depth_ratio": structural_result.get(
                    "alphaq_total_depth_ratio"
                ),
                "alphaq_total_area_ratio": structural_result.get(
                    "alphaq_total_area_ratio"
                ),
                "alphaq_border_status": structural_result.get("alphaq_border_status"),
                "alphaq_border_error": structural_result.get("alphaq_border_error"),
                "alphaq_total_depth": structural_result.get("alphaq_total_depth"),
                "alphaq_total_width": structural_result.get("alphaq_total_width"),
                "alphaq_total_area": structural_result.get("alphaq_total_area"),
                "alphaq_nc_core_depth": structural_result.get(
                    "alphaq_nc_core_depth"
                ),
                "alphaq_nc_core_width": structural_result.get("alphaq_nc_core_width"),
                "alphaq_nc_core_area": structural_result.get("alphaq_nc_core_area"),
                "alphaq_core_tcount": structural_result.get("alphaq_core_tcount"),
                "alphaq_crossing_closure_count": structural_result.get(
                    "alphaq_crossing_closure_count"
                ),
                "alphaq_dependency_core_depth": structural_result.get(
                    "alphaq_dependency_core_depth"
                ),
                "alphaq_dependency_core_width": structural_result.get(
                    "alphaq_dependency_core_width"
                ),
                "alphaq_dependency_core_area": structural_result.get(
                    "alphaq_dependency_core_area"
                ),
                "alphaq_dependency_core_size": structural_result.get(
                    "alphaq_dependency_core_size"
                ),
                "alphaq_dependency_core_tcount": structural_result.get(
                    "alphaq_dependency_core_tcount"
                ),
                "alphaq_dependency_entangling_count": structural_result.get(
                    "alphaq_dependency_entangling_count"
                ),
                "alphaq_dependency_closure_rounds": structural_result.get(
                    "alphaq_dependency_closure_rounds"
                ),
                "alphaq_dependency_internal_edge_count": structural_result.get(
                    "alphaq_dependency_internal_edge_count"
                ),
                "alphaq_dependency_boundary_edge_count": structural_result.get(
                    "alphaq_dependency_boundary_edge_count"
                ),
                "alphaq_dependency_component_count": structural_result.get(
                    "alphaq_dependency_component_count"
                ),
                "alphaq_dependency_largest_component_size": structural_result.get(
                    "alphaq_dependency_largest_component_size"
                ),
                "alphaq_dependency_largest_component_fraction": structural_result.get(
                    "alphaq_dependency_largest_component_fraction"
                ),
                "alphaq_dependency_chain_depth": structural_result.get(
                    "alphaq_dependency_chain_depth"
                ),
                "alphaq_dependency_edge_density": structural_result.get(
                    "alphaq_dependency_edge_density"
                ),
                "alphaq_left_nc_core_depth": structural_result.get(
                    "alphaq_left_nc_core_depth"
                ),
                "alphaq_left_nc_core_width": structural_result.get(
                    "alphaq_left_nc_core_width"
                ),
                "alphaq_left_nc_core_area": structural_result.get(
                    "alphaq_left_nc_core_area"
                ),
                "alphaq_left_crossing_closure_count": structural_result.get(
                    "alphaq_left_crossing_closure_count"
                ),
                "alphaq_right_nc_core_depth": structural_result.get(
                    "alphaq_right_nc_core_depth"
                ),
                "alphaq_right_nc_core_width": structural_result.get(
                    "alphaq_right_nc_core_width"
                ),
                "alphaq_right_nc_core_area": structural_result.get(
                    "alphaq_right_nc_core_area"
                ),
                "alphaq_right_crossing_closure_count": structural_result.get(
                    "alphaq_right_crossing_closure_count"
                ),
                "alphaq_prefix_clifford_depth": structural_result.get(
                    "alphaq_prefix_clifford_depth"
                ),
                "alphaq_suffix_clifford_depth": structural_result.get(
                    "alphaq_suffix_clifford_depth"
                ),
                "alphaq_clifford_shaved_depth_fraction": structural_result.get(
                    "alphaq_clifford_shaved_depth_fraction"
                ),
                "primary_nc_depth_ratio": structural_result.get("primary_nc_depth_ratio"),
                "zx_total_depth_ratio": structural_result.get("zx_total_depth_ratio"),
                "qasm_depth_ratio": structural_result.get("qasm_depth_ratio"),
                "tcount_ratio": structural_result.get("tcount_ratio"),
                "tcount_after_selection": structural_result.get("tcount_after"),
                "tdepth_after_selection": structural_result.get("tdepth_after"),
                "combo_index": structural_result.get("combo_index"),
                "assembled_qasm_path": structural_result.get("assembled_qasm_path"),
                "error": structural_result.get("error"),
                **{field: structural_result.get(field) for field in TENSOR_V3_FIELDS},
            }
        )
    return (
        summary_rows,
        frontier_rows,
        tensor_v3_candidate_manifest_rows,
        tensor_v3_selection_manifest_rows,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay public AlphaTensor-Quantum decompositions and assemble full circuits.")
    parser.add_argument("--inventory-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_RESYNTH_ROOT)
    parser.add_argument(
        "--selection-objective",
        choices=("tcount", "structural", "tensor-v3"),
        default="tcount",
    )
    parser.add_argument(
        "--max-candidates-per-key",
        type=int,
        default=DEFAULT_STRUCTURAL_MAX_CANDIDATES_PER_KEY,
    )
    parser.add_argument("--max-combinations-per-circuit", type=int, default=250)
    parser.add_argument(
        "--tensor-v3-tcount-tolerance",
        type=float,
        default=DEFAULT_TENSOR_V3_TCOUNT_TOLERANCE,
    )
    parser.add_argument(
        "--tensor-v3-ranking",
        choices=("lex-v1", "phase-slack-v1"),
        default=DEFAULT_TENSOR_V3_RANKING,
    )
    parser.add_argument(
        "--tensor-v3-mixed-slack",
        type=float,
        default=DEFAULT_TENSOR_V3_MIXED_SLACK,
    )
    parser.add_argument(
        "--tensor-v3-profile",
        choices=tuple(TENSOR_V3_PROFILE_SPECS),
        default=None,
        help=(
            "Optional named tensor-v3 profile. Overrides T-count tolerance and "
            "mixed slack; splitting-aggressive writes a distinct method name."
        ),
    )
    parser.add_argument("--circuit-id", action="append", dest="circuit_ids", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    (
        tensor_v3_profile,
        tensor_v3_tcount_tolerance,
        tensor_v3_mixed_slack,
    ) = tensor_v3_profile_settings(
        profile=args.tensor_v3_profile,
        tcount_tolerance=args.tensor_v3_tcount_tolerance,
        mixed_slack=args.tensor_v3_mixed_slack,
    )
    inventory_rows = load_inventory_rows(args.inventory_csv)
    circuit_ids = args.circuit_ids
    if args.selection_objective in {"structural", "tensor-v3"} and circuit_ids is None:
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
    tensor_v3_candidate_manifest_rows = []
    tensor_v3_selection_manifest_rows = []
    if args.selection_objective in {"structural", "tensor-v3"}:
        method_name = tensor_v3_method_name(
            selection_objective=args.selection_objective,
            ranking_strategy=args.tensor_v3_ranking,
            profile=tensor_v3_profile,
        )
        (
            structural_summary_rows,
            frontier_rows,
            tensor_v3_candidate_manifest_rows,
            tensor_v3_selection_manifest_rows,
        ) = replay_selection_method(
            inventory_rows=inventory_rows,
            output_root=args.output_root,
            max_candidates_per_key=args.max_candidates_per_key,
            max_combinations_per_circuit=args.max_combinations_per_circuit,
            selection_objective=args.selection_objective,
            method_name=method_name,
            tensor_v3_tcount_tolerance=tensor_v3_tcount_tolerance,
            tensor_v3_ranking_strategy=args.tensor_v3_ranking,
            tensor_v3_profile=tensor_v3_profile,
            tensor_v3_mixed_slack=tensor_v3_mixed_slack,
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
    tensor_v3_candidate_manifest_csv = args.output_root / "tensor_v3_candidate_manifest.csv"
    tensor_v3_selection_manifest_csv = args.output_root / "tensor_v3_selection_manifest.csv"
    write_csv_rows(summary_rows, summary_csv)
    if frontier_rows:
        write_csv_rows(frontier_rows, frontier_csv)
    if tensor_v3_candidate_manifest_rows:
        write_csv_rows(
            tensor_v3_candidate_manifest_rows,
            tensor_v3_candidate_manifest_csv,
        )
    if tensor_v3_selection_manifest_rows:
        write_csv_rows(
            tensor_v3_selection_manifest_rows,
            tensor_v3_selection_manifest_csv,
        )
    write_json(
        {
            "inventory_csv": str(args.inventory_csv),
            "output_root": str(args.output_root),
            "selection_objective": args.selection_objective,
            "max_candidates_per_key": args.max_candidates_per_key,
            "max_combinations_per_circuit": args.max_combinations_per_circuit,
            "tensor_v3_tcount_tolerance": tensor_v3_tcount_tolerance,
            "tensor_v3_ranking": args.tensor_v3_ranking,
            "tensor_v3_profile": tensor_v3_profile,
            "tensor_v3_mixed_slack": tensor_v3_mixed_slack,
            "circuit_ids": [row["circuit_id"] for row in inventory_rows],
            "num_rows": len(summary_rows),
            "candidate_frontier_csv": str(frontier_csv) if frontier_rows else None,
            "num_candidate_frontier_rows": len(frontier_rows),
            "tensor_v3_candidate_manifest_csv": (
                str(tensor_v3_candidate_manifest_csv)
                if tensor_v3_candidate_manifest_rows
                else None
            ),
            "tensor_v3_selection_manifest_csv": (
                str(tensor_v3_selection_manifest_csv)
                if tensor_v3_selection_manifest_rows
                else None
            ),
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
                f"--max-candidates-per-key {args.max_candidates_per_key} "
                f"--max-combinations-per-circuit {args.max_combinations_per_circuit}"
                f" --tensor-v3-tcount-tolerance {tensor_v3_tcount_tolerance}"
                f" --tensor-v3-ranking {args.tensor_v3_ranking}"
                f" --tensor-v3-mixed-slack {tensor_v3_mixed_slack}"
                + (
                    f" --tensor-v3-profile {tensor_v3_profile}"
                    if tensor_v3_profile
                    else ""
                )
                + "".join(
                    f" --circuit-id {circuit_id}"
                    for circuit_id in (args.circuit_ids or [])
                )
            ),
            "cwd": str(PROJECT_ROOT),
            "inventory_csv": str(args.inventory_csv),
            "selection_objective": args.selection_objective,
            "tensor_v3_ranking": args.tensor_v3_ranking,
            "tensor_v3_profile": tensor_v3_profile,
            "tensor_v3_tcount_tolerance": tensor_v3_tcount_tolerance,
            "tensor_v3_mixed_slack": tensor_v3_mixed_slack,
            "circuit_ids": [row["circuit_id"] for row in inventory_rows],
            "summary_csv": str(summary_csv),
            "summary_json": str(summary_json),
            "candidate_frontier_csv": str(frontier_csv) if frontier_rows else None,
            "tensor_v3_candidate_manifest_csv": (
                str(tensor_v3_candidate_manifest_csv)
                if tensor_v3_candidate_manifest_rows
                else None
            ),
            "tensor_v3_selection_manifest_csv": (
                str(tensor_v3_selection_manifest_csv)
                if tensor_v3_selection_manifest_rows
                else None
            ),
            "exit_code": 0,
        }
    )
    print(json.dumps({"summary_csv": str(summary_csv), "summary_json": str(summary_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
