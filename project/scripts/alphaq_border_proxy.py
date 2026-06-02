from __future__ import annotations

"""Circuit-level Clifford/non-Clifford border proxy.

The paper's ZX procedure pushes non-Clifford spiders and then recursively assigns
two-qubit gates crossing the Clifford/non-Clifford border to the non-Clifford
side. This module keeps that invariant at QASM level: it peels Clifford prefix
and suffix regions, closes over crossing entangling Clifford gates, and scores
the remaining middle core without using ZX/PyZX in the selection target.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from scripts._analysis_common import CLIFFORD_GATES
from scripts._analysis_common import NON_CLIFFORD_GATES
from scripts._analysis_common import load_qasm_circuit
from scripts._analysis_common import normalize_circuit_to_basis
from scripts.structural_target import coerce_float


ALPHAQ_BORDER_STATUS_KEY = "alphaq_border_status"
ALPHAQ_BORDER_COST_KEY = "alphaq_nc_core_area_ratio"
ALPHAQ_DEPENDENCY_COST_KEY = "alphaq_dependency_core_area_ratio"


@dataclass(frozen=True)
class GateRecord:
    index: int
    name: str
    qubits: tuple[int, ...]

    @property
    def is_nonclifford(self) -> bool:
        return self.name in NON_CLIFFORD_GATES

    @property
    def is_clifford(self) -> bool:
        return self.name in CLIFFORD_GATES

    @property
    def is_entangling_clifford(self) -> bool:
        return self.is_clifford and len(self.qubits) > 1


def compute_alphaq_border_metrics_from_qasm(path: Path) -> dict[str, Any]:
    return compute_alphaq_border_metrics(load_qasm_circuit(path))


def compute_alphaq_border_metrics(circuit: Any) -> dict[str, Any]:
    normalized, normalization_status, normalization_error = normalize_circuit_to_basis(
        circuit
    )
    if normalized is None:
        return _border_status_metrics(
            "normalization-failed",
            normalization_error or "Could not normalize circuit to comparison basis.",
        )
    if normalization_status != "ok":
        return _border_status_metrics(
            "unsupported-gates",
            normalization_error or "Circuit contains gates outside the comparison basis.",
        )

    records = _gate_records(normalized)
    total_depth = int(normalized.depth() or 0)
    total_width = int(normalized.num_qubits)
    total_area = total_depth * total_width
    total_gates = len(records)
    tcount = sum(record.is_nonclifford for record in records)

    if tcount == 0:
        return {
            **_empty_core_metrics(),
            ALPHAQ_BORDER_STATUS_KEY: "ok",
            "alphaq_border_error": None,
            "alphaq_total_depth": total_depth,
            "alphaq_total_width": total_width,
            "alphaq_total_area": total_area,
            "alphaq_total_gates": total_gates,
            "alphaq_tcount": 0,
            "alphaq_has_nonclifford": False,
        }

    forward_core = _forward_nonclifford_closure(records, total_width)
    backward_core = _backward_nonclifford_closure(records, total_width)
    sandwich_core = forward_core & backward_core
    dependency_core, dependency_closure_rounds = _nonclifford_dependency_closure(
        records,
        sandwich_core,
    )
    prefix_clifford = {record.index for record in records} - forward_core
    suffix_clifford = {record.index for record in records} - backward_core

    left_depth, left_width = _subset_depth_width(records, forward_core, total_width)
    right_depth, right_width = _subset_depth_width(records, backward_core, total_width)
    core_depth, core_width = _subset_depth_width(records, sandwich_core, total_width)
    dependency_depth, dependency_width = _subset_depth_width(
        records,
        dependency_core,
        total_width,
    )
    prefix_depth, _ = _subset_depth_width(records, prefix_clifford, total_width)
    suffix_depth, _ = _subset_depth_width(records, suffix_clifford, total_width)

    core_tcount = sum(
        record.is_nonclifford and record.index in sandwich_core for record in records
    )
    dependency_profile = _dependency_profile(records, dependency_core)
    crossing_closure_count = _crossing_closure_count(records, sandwich_core)
    left_crossing_closure_count = _crossing_closure_count(records, forward_core)
    right_crossing_closure_count = _crossing_closure_count(records, backward_core)
    shaved_depth = prefix_depth + suffix_depth

    return {
        ALPHAQ_BORDER_STATUS_KEY: "ok",
        "alphaq_border_error": None,
        "alphaq_total_depth": total_depth,
        "alphaq_total_width": total_width,
        "alphaq_total_area": total_area,
        "alphaq_total_gates": total_gates,
        "alphaq_tcount": tcount,
        "alphaq_has_nonclifford": True,
        "alphaq_prefix_clifford_depth": prefix_depth,
        "alphaq_suffix_clifford_depth": suffix_depth,
        "alphaq_clifford_shaved_depth": shaved_depth,
        "alphaq_clifford_shaved_depth_fraction": _safe_ratio(
            shaved_depth, total_depth
        ),
        "alphaq_left_nc_core_depth": left_depth,
        "alphaq_left_nc_core_width": left_width,
        "alphaq_left_nc_core_area": left_depth * left_width,
        "alphaq_left_crossing_closure_count": left_crossing_closure_count,
        "alphaq_right_nc_core_depth": right_depth,
        "alphaq_right_nc_core_width": right_width,
        "alphaq_right_nc_core_area": right_depth * right_width,
        "alphaq_right_crossing_closure_count": right_crossing_closure_count,
        "alphaq_nc_core_depth": core_depth,
        "alphaq_nc_core_width": core_width,
        "alphaq_nc_core_area": core_depth * core_width,
        "alphaq_nc_core_size": len(sandwich_core),
        "alphaq_core_tcount": core_tcount,
        "alphaq_crossing_closure_count": crossing_closure_count,
        "alphaq_dependency_core_depth": dependency_depth,
        "alphaq_dependency_core_width": dependency_width,
        "alphaq_dependency_core_area": dependency_depth * dependency_width,
        "alphaq_dependency_core_size": len(dependency_core),
        "alphaq_dependency_core_tcount": sum(
            record.is_nonclifford and record.index in dependency_core
            for record in records
        ),
        "alphaq_dependency_entangling_count": sum(
            record.is_entangling_clifford and record.index in dependency_core
            for record in records
        ),
        "alphaq_dependency_closure_rounds": dependency_closure_rounds,
        **dependency_profile,
    }


def compute_alphaq_border_target_metrics(
    candidate_row: Mapping[str, Any],
    original_row: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if original_row is None:
        return _target_status_metrics("missing-original", "Missing original row.")
    if (
        candidate_row.get(ALPHAQ_BORDER_STATUS_KEY) != "ok"
        or original_row.get(ALPHAQ_BORDER_STATUS_KEY) != "ok"
    ):
        return _target_status_metrics(
            "missing-alphaq-border",
            "AlphaQuantum border metrics are not available.",
        )

    candidate_area = coerce_float(candidate_row.get("alphaq_nc_core_area"))
    original_area = coerce_float(original_row.get("alphaq_nc_core_area"))
    candidate_dependency_area = coerce_float(
        candidate_row.get("alphaq_dependency_core_area")
    )
    original_dependency_area = coerce_float(
        original_row.get("alphaq_dependency_core_area")
    )
    candidate_depth = coerce_float(candidate_row.get("alphaq_nc_core_depth"))
    candidate_width = coerce_float(candidate_row.get("alphaq_nc_core_width"))
    candidate_dependency_depth = coerce_float(
        candidate_row.get("alphaq_dependency_core_depth")
    )
    candidate_dependency_width = coerce_float(
        candidate_row.get("alphaq_dependency_core_width")
    )
    candidate_total_depth = coerce_float(candidate_row.get("alphaq_total_depth"))
    candidate_total_area = coerce_float(candidate_row.get("alphaq_total_area"))
    original_total_depth = coerce_float(original_row.get("alphaq_total_depth"))
    original_total_width = coerce_float(original_row.get("alphaq_total_width"))
    original_total_area = coerce_float(original_row.get("alphaq_total_area"))

    candidate_dependency_area = (
        candidate_area if candidate_dependency_area is None else candidate_dependency_area
    )
    original_dependency_area = (
        original_area if original_dependency_area is None else original_dependency_area
    )
    candidate_dependency_depth = (
        candidate_depth if candidate_dependency_depth is None else candidate_dependency_depth
    )
    candidate_dependency_width = (
        candidate_width if candidate_dependency_width is None else candidate_dependency_width
    )

    required = (
        candidate_area,
        original_area,
        candidate_dependency_area,
        original_dependency_area,
        candidate_depth,
        candidate_width,
        candidate_dependency_depth,
        candidate_dependency_width,
        candidate_total_depth,
        candidate_total_area,
        original_total_depth,
        original_total_width,
        original_total_area,
    )
    if any(value is None for value in required):
        return _target_status_metrics(
            "missing-alphaq-border",
            "Required AlphaQuantum border depth/area fields are missing.",
        )
    if any(value < 0 for value in required if value is not None):
        return _target_status_metrics(
            "invalid-alphaq-border",
            "AlphaQuantum border metrics must be non-negative.",
        )

    depth_denominator = max(original_total_depth or 0.0, 1.0)
    width_denominator = max(original_total_width or 0.0, 1.0)
    area_denominator = max(original_total_area or 0.0, 1.0)
    cost = (candidate_area or 0.0) / area_denominator
    original_cost = (original_area or 0.0) / area_denominator
    dependency_cost = (candidate_dependency_area or 0.0) / area_denominator
    original_dependency_cost = (original_dependency_area or 0.0) / area_denominator

    return {
        "alphaq_target_status": "ok",
        "alphaq_target_error": None,
        ALPHAQ_BORDER_COST_KEY: cost,
        ALPHAQ_DEPENDENCY_COST_KEY: dependency_cost,
        "alphaq_nc_core_area_delta_vs_original": cost - original_cost,
        "alphaq_dependency_core_area_delta_vs_original": (
            dependency_cost - original_dependency_cost
        ),
        "alphaq_nc_core_depth_ratio": (candidate_depth or 0.0) / depth_denominator,
        "alphaq_nc_core_width_ratio": (candidate_width or 0.0) / width_denominator,
        "alphaq_dependency_core_depth_ratio": (candidate_dependency_depth or 0.0)
        / depth_denominator,
        "alphaq_dependency_core_width_ratio": (candidate_dependency_width or 0.0)
        / width_denominator,
        "alphaq_total_depth_ratio": (candidate_total_depth or 0.0)
        / depth_denominator,
        "alphaq_total_area_ratio": (candidate_total_area or 0.0)
        / area_denominator,
    }


def _gate_records(circuit: Any) -> list[GateRecord]:
    records: list[GateRecord] = []
    for instruction in circuit.data:
        name = instruction.operation.name
        if name not in CLIFFORD_GATES and name not in NON_CLIFFORD_GATES:
            continue
        records.append(
            GateRecord(
                index=len(records),
                name=name,
                qubits=tuple(circuit.find_bit(qubit).index for qubit in instruction.qubits),
            )
        )
    return records


def _forward_nonclifford_closure(
    records: list[GateRecord],
    num_qubits: int,
) -> set[int]:
    first_core = [len(records) + 1 for _ in range(num_qubits)]
    for record in records:
        if record.is_nonclifford:
            for qubit in record.qubits:
                first_core[qubit] = min(first_core[qubit], record.index)

    changed = True
    while changed:
        changed = False
        for record in records:
            if not record.is_entangling_clifford:
                continue
            if any(first_core[qubit] <= record.index for qubit in record.qubits):
                for qubit in record.qubits:
                    if record.index < first_core[qubit]:
                        first_core[qubit] = record.index
                        changed = True

    return {
        record.index
        for record in records
        if any(record.index >= first_core[qubit] for qubit in record.qubits)
    }


def _backward_nonclifford_closure(
    records: list[GateRecord],
    num_qubits: int,
) -> set[int]:
    last_core = [-1 for _ in range(num_qubits)]
    for record in records:
        if record.is_nonclifford:
            for qubit in record.qubits:
                last_core[qubit] = max(last_core[qubit], record.index)

    changed = True
    while changed:
        changed = False
        for record in reversed(records):
            if not record.is_entangling_clifford:
                continue
            if any(last_core[qubit] >= record.index for qubit in record.qubits):
                for qubit in record.qubits:
                    if record.index > last_core[qubit]:
                        last_core[qubit] = record.index
                        changed = True

    return {
        record.index
        for record in records
        if any(record.index <= last_core[qubit] for qubit in record.qubits)
    }


def _subset_depth_width(
    records: list[GateRecord],
    selected_indices: set[int],
    num_qubits: int,
) -> tuple[int, int]:
    if not selected_indices:
        return 0, 0
    qubit_depths = [0 for _ in range(num_qubits)]
    touched_qubits: set[int] = set()
    for record in records:
        if record.index not in selected_indices:
            continue
        gate_depth = max(qubit_depths[qubit] for qubit in record.qubits) + 1
        for qubit in record.qubits:
            qubit_depths[qubit] = gate_depth
            touched_qubits.add(qubit)
    return max(qubit_depths, default=0), len(touched_qubits)


def _crossing_closure_count(
    records: list[GateRecord],
    selected_indices: set[int],
) -> int:
    return sum(
        record.index in selected_indices and record.is_entangling_clifford
        for record in records
    )


def _dependency_edges(records: list[GateRecord]) -> set[tuple[int, int]]:
    last_by_qubit: dict[int, int] = {}
    edges: set[tuple[int, int]] = set()
    for record in records:
        for qubit in record.qubits:
            previous = last_by_qubit.get(qubit)
            if previous is not None:
                edges.add((previous, record.index))
            last_by_qubit[qubit] = record.index
    return edges


def _nonclifford_dependency_closure(
    records: list[GateRecord],
    peeled_core: set[int],
) -> tuple[set[int], int]:
    selected = set(peeled_core)
    if not selected:
        return set(), 0

    adjacency: dict[int, set[int]] = {record.index: set() for record in records}
    for left, right in _dependency_edges(records):
        adjacency[left].add(right)
        adjacency[right].add(left)

    rounds = 0
    changed = True
    while changed:
        changed = False
        to_add: set[int] = set()
        for record in records:
            if record.index in selected or not record.is_entangling_clifford:
                continue
            if adjacency[record.index] & selected:
                to_add.add(record.index)
        if to_add:
            selected.update(to_add)
            rounds += 1
            changed = True
    return selected, rounds


def _dependency_profile(
    records: list[GateRecord],
    selected_indices: set[int],
) -> dict[str, Any]:
    if not selected_indices:
        return _empty_dependency_profile()

    edges = _dependency_edges(records)
    internal_edges = [
        edge
        for edge in edges
        if edge[0] in selected_indices and edge[1] in selected_indices
    ]
    boundary_edges = [
        edge
        for edge in edges
        if (edge[0] in selected_indices) != (edge[1] in selected_indices)
    ]
    components = _component_sizes(selected_indices, internal_edges)
    largest_component = max(components) if components else 0
    dependency_chain_depth = _selected_dependency_chain_depth(records, selected_indices)
    return {
        "alphaq_dependency_internal_edge_count": len(internal_edges),
        "alphaq_dependency_boundary_edge_count": len(boundary_edges),
        "alphaq_dependency_component_count": len(components),
        "alphaq_dependency_largest_component_size": largest_component,
        "alphaq_dependency_largest_component_fraction": _safe_ratio(
            largest_component,
            len(selected_indices),
        ),
        "alphaq_dependency_chain_depth": dependency_chain_depth,
        "alphaq_dependency_edge_density": _safe_ratio(
            len(internal_edges),
            max(len(selected_indices) - 1, 1),
        ),
    }


def _component_sizes(
    selected_indices: set[int],
    edges: list[tuple[int, int]],
) -> list[int]:
    adjacency: dict[int, set[int]] = {index: set() for index in selected_indices}
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)

    seen: set[int] = set()
    sizes: list[int] = []
    for start in selected_indices:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        size = 0
        while stack:
            current = stack.pop()
            size += 1
            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        sizes.append(size)
    return sizes


def _selected_dependency_chain_depth(
    records: list[GateRecord],
    selected_indices: set[int],
) -> int:
    gate_depths: dict[int, int] = {}
    qubit_depths: dict[int, int] = {}
    for record in records:
        if record.index not in selected_indices:
            for qubit in record.qubits:
                qubit_depths[qubit] = max(qubit_depths.get(qubit, 0), 0)
            continue
        depth = max((qubit_depths.get(qubit, 0) for qubit in record.qubits), default=0) + 1
        gate_depths[record.index] = depth
        for qubit in record.qubits:
            qubit_depths[qubit] = depth
    return max(gate_depths.values(), default=0)


def _empty_core_metrics() -> dict[str, Any]:
    return {
        "alphaq_prefix_clifford_depth": None,
        "alphaq_suffix_clifford_depth": None,
        "alphaq_clifford_shaved_depth": None,
        "alphaq_clifford_shaved_depth_fraction": None,
        "alphaq_left_nc_core_depth": 0,
        "alphaq_left_nc_core_width": 0,
        "alphaq_left_nc_core_area": 0,
        "alphaq_left_crossing_closure_count": 0,
        "alphaq_right_nc_core_depth": 0,
        "alphaq_right_nc_core_width": 0,
        "alphaq_right_nc_core_area": 0,
        "alphaq_right_crossing_closure_count": 0,
        "alphaq_nc_core_depth": 0,
        "alphaq_nc_core_width": 0,
        "alphaq_nc_core_area": 0,
        "alphaq_nc_core_size": 0,
        "alphaq_core_tcount": 0,
        "alphaq_crossing_closure_count": 0,
        "alphaq_dependency_core_depth": 0,
        "alphaq_dependency_core_width": 0,
        "alphaq_dependency_core_area": 0,
        "alphaq_dependency_core_size": 0,
        "alphaq_dependency_core_tcount": 0,
        "alphaq_dependency_entangling_count": 0,
        "alphaq_dependency_closure_rounds": 0,
        **_empty_dependency_profile(),
    }


def _empty_dependency_profile() -> dict[str, Any]:
    return {
        "alphaq_dependency_internal_edge_count": 0,
        "alphaq_dependency_boundary_edge_count": 0,
        "alphaq_dependency_component_count": 0,
        "alphaq_dependency_largest_component_size": 0,
        "alphaq_dependency_largest_component_fraction": 0.0,
        "alphaq_dependency_chain_depth": 0,
        "alphaq_dependency_edge_density": 0.0,
    }


def _border_status_metrics(status: str, error: str) -> dict[str, Any]:
    return {
        **_empty_core_metrics(),
        ALPHAQ_BORDER_STATUS_KEY: status,
        "alphaq_border_error": error,
        "alphaq_total_depth": None,
        "alphaq_total_width": None,
        "alphaq_total_area": None,
        "alphaq_total_gates": None,
        "alphaq_tcount": None,
        "alphaq_has_nonclifford": None,
    }


def _target_status_metrics(status: str, error: str) -> dict[str, Any]:
    return {
        "alphaq_target_status": status,
        "alphaq_target_error": error,
        ALPHAQ_BORDER_COST_KEY: None,
        ALPHAQ_DEPENDENCY_COST_KEY: None,
        "alphaq_nc_core_area_delta_vs_original": None,
        "alphaq_dependency_core_area_delta_vs_original": None,
        "alphaq_nc_core_depth_ratio": None,
        "alphaq_nc_core_width_ratio": None,
        "alphaq_dependency_core_depth_ratio": None,
        "alphaq_dependency_core_width_ratio": None,
        "alphaq_total_depth_ratio": None,
        "alphaq_total_area_ratio": None,
    }


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)
