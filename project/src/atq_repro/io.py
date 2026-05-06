from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np

from atq_repro.paths import BENCHMARKS_ROOT
from atq_repro.paths import DECOMPOSITIONS_ROOT
from atq_repro.paths import PUBLIC_DECOMPOSITION_FILES


@dataclass(frozen=True)
class DecompositionFamily:
    file_name: str
    method: str
    family_label: str
    use_gadgets: bool


DECOMPOSITION_FAMILIES: tuple[DecompositionFamily, ...] = (
    DecompositionFamily("benchmarks_no_gadgets.npz", "paper_no_gadgets", "benchmark", False),
    DecompositionFamily("benchmarks_gadgets.npz", "paper_gadgets", "benchmark", True),
    DecompositionFamily("binary_addition.npz", "paper_gadgets", "binary_addition", True),
    DecompositionFamily("hamming_weight_phase_gradient.npz", "paper_gadgets", "hamming_weight_phase_gradient", True),
    DecompositionFamily("multiplication_finite_fields_no_gadgets.npz", "paper_no_gadgets", "finite_field_multiplication", False),
    DecompositionFamily("multiplication_finite_fields_gadgets.npz", "paper_gadgets", "finite_field_multiplication", True),
    DecompositionFamily("quantum_chemistry.npz", "paper_gadgets", "quantum_chemistry", True),
    DecompositionFamily("unary_iteration_no_gadgets.npz", "paper_no_gadgets", "unary_iteration", False),
    DecompositionFamily("unary_iteration_gadgets.npz", "paper_gadgets", "unary_iteration", True),
)


def available_decomposition_families() -> tuple[DecompositionFamily, ...]:
    configured = {family.file_name: family for family in DECOMPOSITION_FAMILIES}
    return tuple(
        configured[file_name]
        for file_name in PUBLIC_DECOMPOSITION_FILES
        if file_name in configured and (DECOMPOSITIONS_ROOT / file_name).exists()
    )


def normalize_stem_name(value: str) -> str:
    return value.replace("_toff_", "_tof_")


def artifact_stem_candidates(decomposition_key: str) -> list[str]:
    key = normalize_stem_name(decomposition_key)
    candidates = [key]
    if key.endswith("_block10"):
        # Historical circuit-to-tensor artifact naming in the vendored benchmark
        # uses `mod_adder_10240` for the tenth block.
        candidates.append(key.removesuffix("_block10") + "0")
    if key.endswith("_comp2"):
        candidates.append(key.removesuffix("_comp2"))
        candidates.append(f"{key.removesuffix('_comp2')}_comp1")
    elif key.endswith("_comp1"):
        candidates.append(key.removesuffix("_comp1"))
    return list(dict.fromkeys(candidates))


def block_label_from_key(decomposition_key: str) -> str | None:
    match = re.search(r"_block(\d+)$", decomposition_key)
    return None if match is None else f"block{match.group(1)}"


def circuit_id_from_key(decomposition_key: str) -> str:
    key = normalize_stem_name(decomposition_key)
    key = re.sub(r"_block\d+$", "", key)
    key = re.sub(r"_comp[12]$", "", key)
    return key


def benchmark_dir_for_circuit(circuit_id: str) -> Path | None:
    for family_dir in sorted(path for path in BENCHMARKS_ROOT.iterdir() if path.is_dir()):
        candidate = family_dir / circuit_id
        if candidate.is_dir():
            return candidate
    return None


def tensor_path_for_key(decomposition_key: str) -> Path | None:
    circuit_id = circuit_id_from_key(decomposition_key)
    benchmark_dir = benchmark_dir_for_circuit(circuit_id)
    if benchmark_dir is None:
        return None

    for stem in artifact_stem_candidates(decomposition_key):
        candidate = benchmark_dir / f"{stem}.tensor.npy"
        if candidate.exists():
            return candidate
    return None


def load_target_tensor(decomposition_key: str) -> tuple[Path | None, np.ndarray | None]:
    tensor_path = tensor_path_for_key(decomposition_key)
    if tensor_path is None:
        return None, None
    return tensor_path, np.load(tensor_path)


def iter_decompositions(
    families: tuple[DecompositionFamily, ...] | None = None,
):
    for family in families or available_decomposition_families():
        npz_path = DECOMPOSITIONS_ROOT / family.file_name
        with np.load(npz_path, allow_pickle=True) as data:
            for key in sorted(data.files):
                decompositions = np.asarray(data[key], dtype=bool)
                for candidate_index, factors in enumerate(decompositions):
                    yield family, npz_path, key, candidate_index, factors
