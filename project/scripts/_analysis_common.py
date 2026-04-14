from __future__ import annotations

import csv
import dataclasses
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS_ROOT = (
    PROJECT_ROOT / "external" / "circuit-to-tensor" / "benchmarks"
)
DECOMPOSITIONS_ROOT = (
    PROJECT_ROOT / "external" / "alphatensor_quantum" / "decompositions"
)
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results"
DEFAULT_CSV_ROOT = DEFAULT_RESULTS_ROOT / "csv"
DEFAULT_FIGURES_ROOT = DEFAULT_RESULTS_ROOT / "figures"
DEFAULT_REPORTS_ROOT = DEFAULT_RESULTS_ROOT / "reports"
DEFAULT_PYZX_ROOT = DEFAULT_RESULTS_ROOT / "pyzx"
DEFAULT_COMPILE_ROOT = DEFAULT_RESULTS_ROOT / "compile_stage1"
DEFAULT_RESYNTH_ROOT = DEFAULT_RESULTS_ROOT / "public_resynth"

CLIFFORD_GATES = {"h", "s", "sdg", "x", "y", "z", "cx"}
NON_CLIFFORD_GATES = {"t", "tdg"}
COMPARISON_BASIS_GATES = sorted(CLIFFORD_GATES | NON_CLIFFORD_GATES)
DEFAULT_METRIC_LAMBDAS = (5, 10, 20)
DEFAULT_BOUNDARY_WINDOWS = (1, 2)
DEFAULT_BOOTSTRAP_SAMPLES = 1_000
DEFAULT_BOOTSTRAP_SEED = 2024

PUBLIC_DECOMPOSITION_FILES = (
    "benchmarks_gadgets.npz",
    "benchmarks_no_gadgets.npz",
    "binary_addition.npz",
    "hamming_weight_phase_gradient.npz",
    "multiplication_finite_fields_gadgets.npz",
    "multiplication_finite_fields_no_gadgets.npz",
    "quantum_chemistry.npz",
    "unary_iteration_gadgets.npz",
    "unary_iteration_no_gadgets.npz",
)


@dataclasses.dataclass(frozen=True)
class InventoryRecord:
    circuit_id: str
    source: str
    family: str
    n_qubits: int
    raw_gate_count: int
    qasm_path: str
    benchmark_dir: str | None
    vendored_compile_dir: str | None

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def natural_sort_key(text: str) -> list[Any]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]


def write_csv_rows(rows: list[dict[str, Any]], path: Path) -> Path:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_json(payload: dict[str, Any], path: Path) -> Path:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _qiskit_modules() -> tuple[Any, Any, Any]:
    from qiskit import QuantumCircuit, transpile
    try:
        from qiskit import qasm2
    except ImportError:  # pragma: no cover
        qasm2 = None
    return QuantumCircuit, transpile, qasm2


def load_qasm_circuit(qasm_path: Path) -> Any:
    QuantumCircuit, _, qasm2 = _qiskit_modules()
    if qasm2 is None:  # pragma: no cover
        return QuantumCircuit.from_qasm_file(str(qasm_path))

    from qiskit.circuit.library import CCZGate
    from qiskit.circuit.library import CSGate

    custom_instructions = list(qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
    custom_instructions.append(
        qasm2.CustomInstruction(
            name="ccz",
            num_params=0,
            num_qubits=3,
            constructor=CCZGate,
            builtin=True,
        )
    )
    custom_instructions.append(
        qasm2.CustomInstruction(
            name="cs",
            num_params=0,
            num_qubits=2,
            constructor=CSGate,
            builtin=True,
        )
    )
    return qasm2.load(str(qasm_path), custom_instructions=custom_instructions)


def dump_qasm_v2(circuit: Any, output_path: Path) -> Path:
    _, _, qasm2 = _qiskit_modules()
    ensure_dir(output_path.parent)
    if qasm2 is not None:
        output_path.write_text(qasm2.dumps(circuit), encoding="utf-8")
    else:  # pragma: no cover
        output_path.write_text(circuit.qasm(), encoding="utf-8")
    return output_path


def sanitize_circuit(circuit: Any) -> Any:
    sanitized = circuit.remove_final_measurements(inplace=False)
    sanitized.data = [
        instruction
        for instruction in sanitized.data
        if instruction.operation.name != "barrier"
    ]
    return sanitized


def _quarter_turn_ops(angle: float) -> list[str] | None:
    quarter_turn = math.pi / 4.0
    multiple = round(angle / quarter_turn)
    if not math.isclose(angle, multiple * quarter_turn, abs_tol=1e-9):
        return None
    multiple %= 8
    lookup = {
        0: [],
        1: ["t"],
        2: ["s"],
        3: ["s", "t"],
        4: ["z"],
        5: ["z", "t"],
        6: ["sdg"],
        7: ["tdg"],
    }
    return lookup[multiple]


def _append_phase_ops(normalized: Any, qubit: Any, ops: list[str]) -> None:
    for op_name in ops:
        getattr(normalized, op_name)(qubit)


def _append_ccx_decomposition(normalized: Any, control_a: Any, control_b: Any, target: Any) -> None:
    normalized.h(target)
    normalized.cx(control_b, target)
    normalized.tdg(target)
    normalized.cx(control_a, target)
    normalized.t(target)
    normalized.cx(control_b, target)
    normalized.tdg(target)
    normalized.cx(control_a, target)
    normalized.t(control_b)
    normalized.t(target)
    normalized.h(target)
    normalized.cx(control_a, control_b)
    normalized.t(control_a)
    normalized.tdg(control_b)
    normalized.cx(control_a, control_b)


def _append_cs_decomposition(normalized: Any, control: Any, target: Any) -> None:
    normalized.t(control)
    normalized.cx(control, target)
    normalized.tdg(target)
    normalized.cx(control, target)
    normalized.t(target)


def _deterministic_normalize(circuit: Any) -> tuple[Any | None, str, str | None]:
    QuantumCircuit, _, _ = _qiskit_modules()
    normalized = QuantumCircuit(*circuit.qregs, *circuit.cregs, name=circuit.name)
    unsupported: list[str] = []

    for instruction in circuit.data:
        operation = instruction.operation
        name = operation.name
        qubits = instruction.qubits
        if name in COMPARISON_BASIS_GATES:
            normalized.append(operation.copy(), qubits, instruction.clbits)
            continue
        if name == "cz":
            control, target = qubits
            normalized.h(target)
            normalized.cx(control, target)
            normalized.h(target)
            continue
        if name == "cs":
            control, target = qubits
            _append_cs_decomposition(normalized, control, target)
            continue
        if name == "ccx":
            control_a, control_b, target = qubits
            _append_ccx_decomposition(normalized, control_a, control_b, target)
            continue
        if name == "ccz":
            control_a, control_b, target = qubits
            normalized.h(target)
            _append_ccx_decomposition(normalized, control_a, control_b, target)
            normalized.h(target)
            continue
        if name == "rz":
            if len(operation.params) != 1:
                unsupported.append(name)
                continue
            try:
                angle = float(operation.params[0])
            except (TypeError, ValueError):
                unsupported.append(name)
                continue
            ops = _quarter_turn_ops(angle)
            if ops is None:
                unsupported.append(f"rz({operation.params[0]})")
                continue
            _append_phase_ops(normalized, qubits[0], ops)
            continue
        unsupported.append(name)

    if unsupported:
        return normalized, "partial", ", ".join(sorted(set(unsupported)))
    return normalized, "ok", None


def normalize_circuit_to_basis(circuit: Any) -> tuple[Any | None, str, str | None]:
    _, transpile, _ = _qiskit_modules()
    sanitized = sanitize_circuit(circuit)
    deterministic, deterministic_status, deterministic_error = _deterministic_normalize(sanitized)
    if deterministic is not None and deterministic_status == "ok":
        return deterministic, deterministic_status, deterministic_error
    try:
        normalized = transpile(
            sanitized,
            basis_gates=COMPARISON_BASIS_GATES,
            optimization_level=0,
        )
    except Exception as exc:  # pragma: no cover - exercised in integration
        return None, "failed", str(exc)
    unsupported = sorted(
        {
            instruction.operation.name
            for instruction in normalized.data
            if instruction.operation.name not in COMPARISON_BASIS_GATES
        }
    )
    if unsupported:
        return normalized, "partial", ", ".join(unsupported)
    return normalized, "ok", None


def gate_name_sequence(circuit: Any) -> list[str]:
    return [
        instruction.operation.name
        for instruction in circuit.data
        if instruction.operation.name in COMPARISON_BASIS_GATES
    ]


def count_t_depth(circuit: Any) -> int:
    t_layers: dict[int, int] = {}
    for instruction in circuit.data:
        name = instruction.operation.name
        if name not in NON_CLIFFORD_GATES:
            continue
        qubit = circuit.find_bit(instruction.qubits[0]).index
        t_layers[qubit] = t_layers.get(qubit, 0) + 1
    return max(t_layers.values(), default=0)


def _gate_class(name: str) -> str | None:
    if name in CLIFFORD_GATES:
        return "C"
    if name in NON_CLIFFORD_GATES:
        return "NC"
    return None


def count_hadamards_near_boundaries(sequence: list[str], window: int) -> int:
    boundaries = [
        index
        for index in range(len(sequence) - 1)
        if _gate_class(sequence[index]) != _gate_class(sequence[index + 1])
    ]
    if not boundaries:
        return 0
    hadamard_indices: set[int] = set()
    for boundary in boundaries:
        for position, gate_name in enumerate(sequence):
            if gate_name != "h":
                continue
            if min(abs(position - boundary), abs(position - (boundary + 1))) <= window:
                hadamard_indices.add(position)
    return len(hadamard_indices)


def compute_structural_metrics(
    circuit: Any,
    *,
    lambdas: tuple[int, ...] = DEFAULT_METRIC_LAMBDAS,
    boundary_windows: tuple[int, ...] = DEFAULT_BOUNDARY_WINDOWS,
) -> dict[str, Any]:
    sequence = gate_name_sequence(circuit)
    total_gates = len(sequence)
    n_t = sum(name in NON_CLIFFORD_GATES for name in sequence)
    n_clifford = sum(name in CLIFFORD_GATES for name in sequence)

    blocks: list[tuple[str, int]] = []
    for name in sequence:
        gate_class = _gate_class(name)
        if gate_class is None:
            continue
        if blocks and blocks[-1][0] == gate_class:
            blocks[-1] = (gate_class, blocks[-1][1] + 1)
        else:
            blocks.append((gate_class, 1))

    nonclifford_block_lengths = [length for gate_class, length in blocks if gate_class == "NC"]
    n_clifford_blocks = sum(gate_class == "C" for gate_class, _ in blocks)
    n_nonclifford_blocks = sum(gate_class == "NC" for gate_class, _ in blocks)
    num_boundaries = max(len(blocks) - 1, 1)

    metrics: dict[str, Any] = {
        "n_total_gates": total_gates,
        "n_t_gates": n_t,
        "n_clifford_gates": n_clifford,
        "tcount": n_t,
        "tdepth": count_t_depth(circuit),
        "rho_t": 0.0 if total_gates == 0 else n_t / total_gates,
        "n_clifford_blocks": n_clifford_blocks,
        "n_nonclifford_blocks": n_nonclifford_blocks,
        "avg_nonclifford_block_len": (
            0.0
            if not nonclifford_block_lengths
            else float(np.mean(nonclifford_block_lengths))
        ),
    }
    metrics["tdepth_over_tcount"] = (
        0.0 if metrics["tcount"] == 0 else metrics["tdepth"] / metrics["tcount"]
    )

    for lambda_weight in lambdas:
        denominator = n_clifford + lambda_weight * n_t
        value = 0.0 if denominator == 0 else (lambda_weight * n_t) / denominator
        metrics[f"rho_w_lambda_{lambda_weight}"] = value
        if lambda_weight == 10:
            metrics["rho_w"] = value

    for window in boundary_windows:
        value = count_hadamards_near_boundaries(sequence, window) / num_boundaries
        metrics[f"hadamard_boundary_density_w{window}"] = value
        if window == 1:
            metrics["hadamard_boundary_density"] = value

    return metrics


def compute_metrics_from_qasm_path(qasm_path: Path) -> dict[str, Any]:
    circuit = load_qasm_circuit(qasm_path)
    normalized, normalization_status, normalization_error = normalize_circuit_to_basis(circuit)
    if normalized is None:
        return {
            "normalization_status": normalization_status,
            "normalization_error": normalization_error,
            "comparability_status": "none",
        }
    metrics = compute_structural_metrics(normalized)
    metrics.update(
        {
            "normalization_status": normalization_status,
            "normalization_error": normalization_error,
            "comparability_status": (
                "full" if normalization_status == "ok" else "partial"
            ),
            "normalized_qasm_num_qubits": normalized.num_qubits,
            "normalized_qasm_depth": normalized.depth(),
            "normalized_qasm_size": normalized.size(),
        }
    )
    return metrics


def parse_n_qubits_and_gate_count(qasm_path: Path) -> tuple[int, int]:
    circuit = load_qasm_circuit(qasm_path)
    sanitized = sanitize_circuit(circuit)
    return sanitized.num_qubits, sanitized.size()


def tensor_size_from_directory(directory: Path) -> int | None:
    tensor_sizes: list[int] = []
    for tensor_path in sorted(directory.glob("*.tensor.npy")):
        try:
            tensor = np.load(tensor_path)
        except Exception:
            continue
        if tensor.ndim >= 1:
            tensor_sizes.append(int(tensor.shape[-1]))
    return max(tensor_sizes) if tensor_sizes else None


def list_nonclifford_block_stems(directory: Path) -> list[str]:
    stems = []
    for mapping_path in directory.glob("*.mapping.txt"):
        stems.append(mapping_path.name[: -len(".mapping.txt")])
    return sorted(stems, key=natural_sort_key)


def block_sort_key(stem: str) -> list[Any]:
    return natural_sort_key(stem)


def assembled_piece_paths(directory: Path, block_stems: list[str]) -> list[Path]:
    if not block_stems:
        raise ValueError(f"No non-Clifford blocks found in {directory}.")
    first_stem = block_stems[0]
    root_name = first_stem.split("_block", 1)[0] if "_block" in first_stem else first_stem
    paths = [directory / f"{root_name}.initial.cliffords.qasm"]
    for stem in block_stems:
        paths.append(directory / f"{stem}.qasm")
        paths.append(directory / f"{stem}.cliffords.qasm")
    return [path for path in paths if path.exists()]


def list_vendored_benchmark_records() -> list[InventoryRecord]:
    records: list[InventoryRecord] = []
    for family_dir in sorted(
        [path for path in BENCHMARKS_ROOT.iterdir() if path.is_dir()],
        key=lambda path: natural_sort_key(path.name),
    ):
        for circuit_dir in sorted(
            [path for path in family_dir.iterdir() if path.is_dir()],
            key=lambda path: natural_sort_key(path.name),
        ):
            qasm_path = circuit_dir / f"{circuit_dir.name}.qasm"
            if not qasm_path.exists():
                continue
            n_qubits, raw_gate_count = parse_n_qubits_and_gate_count(qasm_path)
            records.append(
                InventoryRecord(
                    circuit_id=circuit_dir.name,
                    source="vendored-local",
                    family=family_dir.name,
                    n_qubits=n_qubits,
                    raw_gate_count=raw_gate_count,
                    qasm_path=str(qasm_path.relative_to(PROJECT_ROOT)),
                    benchmark_dir=str(circuit_dir.relative_to(PROJECT_ROOT)),
                    vendored_compile_dir=str(circuit_dir.relative_to(PROJECT_ROOT)),
                )
            )
    return records


def available_public_decomposition_files() -> list[Path]:
    return [
        DECOMPOSITIONS_ROOT / file_name
        for file_name in PUBLIC_DECOMPOSITION_FILES
        if (DECOMPOSITIONS_ROOT / file_name).exists()
    ]


def load_public_decomposition_index() -> dict[str, list[dict[str, str]]]:
    index: dict[str, list[dict[str, str]]] = {}
    for npz_path in available_public_decomposition_files():
        with np.load(npz_path, allow_pickle=True) as data:
            for key in data.files:
                index.setdefault(key, []).append(
                    {
                        "npz_path": str(npz_path.relative_to(PROJECT_ROOT)),
                        "decomposition_key": key,
                    }
                )
    return index


def artifact_stem_candidates(decomposition_key: str) -> list[str]:
    candidates = [decomposition_key]
    if decomposition_key.endswith("_comp2"):
        candidates.append(decomposition_key.removesuffix("_comp2"))
        candidates.append(f"{decomposition_key.removesuffix('_comp2')}_comp1")
    elif decomposition_key.endswith("_comp1"):
        candidates.append(decomposition_key.removesuffix("_comp1"))
    return list(dict.fromkeys(candidates))


def resolve_artifact_stem(directory: Path, decomposition_key: str) -> str | None:
    available = set(list_nonclifford_block_stems(directory))
    for candidate in artifact_stem_candidates(decomposition_key):
        if candidate in available:
            return candidate
    return None


def to_relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))
