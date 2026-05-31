from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pyzx as zx
from pyzx.utils import VertexType
from qiskit import qasm2

from scripts._analysis_common import load_qasm_circuit
from scripts._analysis_common import normalize_circuit_to_basis


Direction = Literal["left", "right"]


@dataclass(frozen=True)
class WireBounds:
    input_row: float
    output_row: float


@dataclass(frozen=True)
class BorderResult:
    direction: Direction
    clifford_depth: float
    nonclifford_depth: float
    clifford_fraction: float
    nonclifford_fraction: float
    core_qubits: int
    closure_iterations: int
    border_min_row: float
    border_max_row: float


def compute_zx_splitting_metrics_from_qasm(path: Path) -> dict[str, Any]:
    circuit = load_qasm_circuit(path)
    normalized, status, error = normalize_circuit_to_basis(circuit)
    if normalized is None:
        return {
            "zx_split_status": status,
            "zx_split_error": error,
            "zx_split_normalization_status": status,
            "zx_split_normalization_error": error,
        }
    metrics = compute_zx_splitting_metrics(normalized)
    metrics["zx_split_normalization_status"] = status
    metrics["zx_split_normalization_error"] = error
    return metrics


def compute_zx_splitting_metrics(circuit: Any) -> dict[str, Any]:
    try:
        graph = _circuit_to_graph(circuit)
        return _compute_graph_metrics(graph)
    except Exception as exc:  # pragma: no cover - defensive integration path
        return {
            "zx_split_status": "failed",
            "zx_split_error": str(exc),
        }


def _circuit_to_graph(circuit: Any) -> Any:
    qasm_text = qasm2.dumps(circuit)
    return zx.Circuit.from_qasm(qasm_text).to_graph()


def _compute_graph_metrics(graph: Any) -> dict[str, Any]:
    bounds = _wire_bounds(graph)
    if not bounds:
        return {
            "zx_split_status": "empty",
            "zx_split_error": None,
        }

    nonclifford_vertices = _nonclifford_vertices(graph, bounds)
    left = _detect_border(graph, bounds, nonclifford_vertices, "left")
    right = _detect_border(graph, bounds, nonclifford_vertices, "right")
    best_side, best = _best_border(left, right)
    total_depth = _total_internal_depth(graph, bounds)
    cross_edges = _count_cross_qubit_edges(graph, bounds)

    return {
        "zx_split_status": "ok",
        "zx_split_error": None,
        "zx_total_depth": _as_number(total_depth),
        "zx_num_qubits": len(bounds),
        "zx_num_nonclifford_spiders": len(nonclifford_vertices),
        "zx_num_cross_edges": cross_edges,
        **_prefix_metrics("zx_left", left),
        **_prefix_metrics("zx_right", right),
        "zx_best_side": best_side,
        "zx_best_clifford_depth": _as_number(best.clifford_depth),
        "zx_best_nonclifford_depth": _as_number(best.nonclifford_depth),
        "zx_best_clifford_fraction": best.clifford_fraction,
        "zx_best_nonclifford_fraction": best.nonclifford_fraction,
    }


def _wire_bounds(graph: Any) -> dict[int, WireBounds]:
    inputs = _boundary_rows_by_qubit(graph, graph.inputs())
    outputs = _boundary_rows_by_qubit(graph, graph.outputs())
    return {
        qubit: WireBounds(inputs[qubit], outputs[qubit])
        for qubit in sorted(inputs.keys() & outputs.keys())
    }


def _boundary_rows_by_qubit(graph: Any, vertices: list[int]) -> dict[int, float]:
    rows = {}
    for vertex in vertices:
        qubit = _integer_qubit(graph.qubit(vertex))
        if qubit is not None:
            rows[qubit] = float(graph.row(vertex))
    return rows


def _integer_qubit(value: Any) -> int | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    rounded = round(numeric)
    if abs(numeric - rounded) > 1e-9:
        return None
    return int(rounded)


def _vertex_qubit(graph: Any, bounds: dict[int, WireBounds], vertex: int) -> int | None:
    qubit = _integer_qubit(graph.qubit(vertex))
    return qubit if qubit in bounds else None


def _is_zx_spider(graph: Any, vertex: int) -> bool:
    return graph.type(vertex) in {VertexType.Z, VertexType.X}


def _is_nonclifford_vertex(graph: Any, vertex: int) -> bool:
    if not _is_zx_spider(graph, vertex):
        return False
    return not zx.simplify.phase_is_clifford(graph.phase(vertex))


def _nonclifford_vertices(graph: Any, bounds: dict[int, WireBounds]) -> list[int]:
    return [
        vertex
        for vertex in graph.vertices()
        if _vertex_qubit(graph, bounds, vertex) is not None
        and _is_nonclifford_vertex(graph, vertex)
    ]


def _internal_vertices(graph: Any, bounds: dict[int, WireBounds]) -> list[int]:
    return [
        vertex
        for vertex in graph.vertices()
        if graph.type(vertex) != VertexType.BOUNDARY
        and _vertex_qubit(graph, bounds, vertex) is not None
    ]


def _total_internal_depth(graph: Any, bounds: dict[int, WireBounds]) -> int:
    return len({float(graph.row(vertex)) for vertex in _internal_vertices(graph, bounds)})


def _detect_border(
    graph: Any,
    bounds: dict[int, WireBounds],
    nonclifford_vertices: list[int],
    direction: Direction,
) -> BorderResult:
    border = _initial_border(graph, bounds, nonclifford_vertices, direction)
    iterations = _close_crossing_edges(graph, bounds, border, direction)
    total_depth = _total_internal_depth(graph, bounds)
    clifford_depth, nonclifford_depth, core_qubits = _depths(
        graph, bounds, border, direction
    )
    clifford_fraction = 0.0 if total_depth == 0 else clifford_depth / total_depth
    nonclifford_fraction = 0.0 if total_depth == 0 else nonclifford_depth / total_depth

    return BorderResult(
        direction=direction,
        clifford_depth=clifford_depth,
        nonclifford_depth=nonclifford_depth,
        clifford_fraction=clifford_fraction,
        nonclifford_fraction=nonclifford_fraction,
        core_qubits=core_qubits,
        closure_iterations=iterations,
        border_min_row=min(border.values()),
        border_max_row=max(border.values()),
    )


def _initial_border(
    graph: Any,
    bounds: dict[int, WireBounds],
    nonclifford_vertices: list[int],
    direction: Direction,
) -> dict[int, float]:
    rows_by_qubit: dict[int, list[float]] = {qubit: [] for qubit in bounds}
    for vertex in nonclifford_vertices:
        qubit = _vertex_qubit(graph, bounds, vertex)
        if qubit is not None:
            rows_by_qubit[qubit].append(float(graph.row(vertex)))

    border = {}
    for qubit, wire_bounds in bounds.items():
        rows = rows_by_qubit[qubit]
        if direction == "left":
            border[qubit] = min(rows) if rows else wire_bounds.output_row
        else:
            border[qubit] = max(rows) if rows else wire_bounds.input_row
    return border


def _close_crossing_edges(
    graph: Any,
    bounds: dict[int, WireBounds],
    border: dict[int, float],
    direction: Direction,
) -> int:
    iterations = 0
    while True:
        changed = False
        for edge in graph.edges():
            source, target = graph.edge_st(edge)
            source_qubit = _vertex_qubit(graph, bounds, source)
            target_qubit = _vertex_qubit(graph, bounds, target)
            if source_qubit is None or target_qubit is None or source_qubit == target_qubit:
                continue

            source_core = _is_in_core(graph, border, source, source_qubit, direction)
            target_core = _is_in_core(graph, border, target, target_qubit, direction)
            if source_core == target_core:
                continue

            if source_core:
                changed |= _include_vertex_in_core(graph, border, target, target_qubit, direction)
            else:
                changed |= _include_vertex_in_core(graph, border, source, source_qubit, direction)

        if not changed:
            return iterations
        iterations += 1


def _is_in_core(
    graph: Any,
    border: dict[int, float],
    vertex: int,
    qubit: int,
    direction: Direction,
) -> bool:
    row = float(graph.row(vertex))
    if direction == "left":
        return row >= border[qubit]
    return row <= border[qubit]


def _include_vertex_in_core(
    graph: Any,
    border: dict[int, float],
    vertex: int,
    qubit: int,
    direction: Direction,
) -> bool:
    row = float(graph.row(vertex))
    previous = border[qubit]
    if direction == "left" and row < previous:
        border[qubit] = row
        return True
    if direction == "right" and row > previous:
        border[qubit] = row
        return True
    return False


def _depths(
    graph: Any,
    bounds: dict[int, WireBounds],
    border: dict[int, float],
    direction: Direction,
) -> tuple[float, float, int]:
    clifford_rows: set[float] = set()
    nonclifford_rows: set[float] = set()
    core_qubits: set[int] = set()

    for vertex in _internal_vertices(graph, bounds):
        qubit = _vertex_qubit(graph, bounds, vertex)
        if qubit is None:
            continue
        row = float(graph.row(vertex))
        if _is_in_core(graph, border, vertex, qubit, direction):
            nonclifford_rows.add(row)
            core_qubits.add(qubit)
        else:
            clifford_rows.add(row)

    return float(len(clifford_rows)), float(len(nonclifford_rows)), len(core_qubits)


def _best_border(left: BorderResult, right: BorderResult) -> tuple[str, BorderResult]:
    if left.clifford_depth > right.clifford_depth:
        return "left", left
    if right.clifford_depth > left.clifford_depth:
        return "right", right
    return "tie", left


def _count_cross_qubit_edges(graph: Any, bounds: dict[int, WireBounds]) -> int:
    count = 0
    for edge in graph.edges():
        source, target = graph.edge_st(edge)
        source_qubit = _vertex_qubit(graph, bounds, source)
        target_qubit = _vertex_qubit(graph, bounds, target)
        if source_qubit is not None and target_qubit is not None and source_qubit != target_qubit:
            count += 1
    return count


def _prefix_metrics(prefix: str, result: BorderResult) -> dict[str, Any]:
    return {
        f"{prefix}_clifford_depth": _as_number(result.clifford_depth),
        f"{prefix}_nonclifford_depth": _as_number(result.nonclifford_depth),
        f"{prefix}_clifford_fraction": result.clifford_fraction,
        f"{prefix}_nonclifford_fraction": result.nonclifford_fraction,
        f"{prefix}_core_qubits": result.core_qubits,
        f"{prefix}_closure_iters": result.closure_iterations,
        f"{prefix}_border_min_row": _as_number(result.border_min_row),
        f"{prefix}_border_max_row": _as_number(result.border_max_row),
    }


def _as_number(value: float) -> int | float:
    rounded = round(value)
    if abs(value - rounded) <= 1e-9:
        return int(rounded)
    return value
