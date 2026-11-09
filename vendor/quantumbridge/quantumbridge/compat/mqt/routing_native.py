# This file is independently implemented for QuantumBridge SDK.
# No source code from MQT, IBM, Qiskit, or QMAP was copied.
"""Educational topology and routing primitives for MQT QMAP-like compatibility."""

from __future__ import annotations

from collections import deque
from typing import Any, Iterable


def create_coupling_graph(edges: Iterable[tuple[int, int]], num_qubits: int | None = None) -> dict[str, Any]:
    canonical = sorted({_edge(edge[0], edge[1]) for edge in edges})
    if num_qubits is None:
        num_qubits = max((max(edge) for edge in canonical), default=-1) + 1
    graph = {
        "num_qubits": int(num_qubits),
        "edges": [{"source": int(a), "target": int(b)} for a, b in canonical],
        "directed": False,
        "topology": "custom",
    }
    return validate_coupling_graph(graph)


def create_line_topology(num_qubits: int) -> dict[str, Any]:
    graph = create_coupling_graph([(index, index + 1) for index in range(int(num_qubits) - 1)], num_qubits)
    graph["topology"] = "line"
    return graph


def create_ring_topology(num_qubits: int) -> dict[str, Any]:
    num_qubits = int(num_qubits)
    edges = [(index, (index + 1) % num_qubits) for index in range(num_qubits)]
    graph = create_coupling_graph(edges, num_qubits)
    graph["topology"] = "ring"
    return graph


def create_fully_connected_topology(num_qubits: int) -> dict[str, Any]:
    num_qubits = int(num_qubits)
    edges = [(left, right) for left in range(num_qubits) for right in range(left + 1, num_qubits)]
    graph = create_coupling_graph(edges, num_qubits)
    graph["topology"] = "fully_connected"
    return graph


def validate_coupling_graph(graph: dict[str, Any]) -> dict[str, Any]:
    payload = dict(graph)
    num_qubits = int(payload.get("num_qubits", 0))
    if num_qubits <= 0:
        raise ValueError("coupling graph requires a positive num_qubits")
    edges = []
    for item in payload.get("edges", ()):
        if isinstance(item, dict):
            source, target = int(item["source"]), int(item["target"])
        else:
            source, target = int(item[0]), int(item[1])
        if source == target:
            raise ValueError("coupling graph cannot contain self edges")
        if not 0 <= source < num_qubits or not 0 <= target < num_qubits:
            raise ValueError("coupling graph edge is outside topology")
        edges.append(_edge(source, target))
    payload["num_qubits"] = num_qubits
    payload["edges"] = [{"source": a, "target": b} for a, b in sorted(set(edges))]
    payload["directed"] = False
    payload.setdefault("topology", "custom")
    return payload


def initial_layout_trivial(num_logical_qubits: int, num_physical_qubits: int) -> dict[int, int]:
    num_logical_qubits = int(num_logical_qubits)
    num_physical_qubits = int(num_physical_qubits)
    if num_logical_qubits <= 0 or num_physical_qubits <= 0:
        raise ValueError("layout qubit counts must be positive")
    if num_logical_qubits > num_physical_qubits:
        raise ValueError("logical qubits cannot exceed physical qubits")
    return {logical: logical for logical in range(num_logical_qubits)}


def route_cnot_sequence_greedy(
    operations: list[dict[str, Any]],
    coupling_graph: dict[str, Any],
    initial_layout: dict[int, int] | None = None,
    num_logical_qubits: int | None = None,
) -> dict[str, Any]:
    graph = validate_coupling_graph(coupling_graph)
    if num_logical_qubits is None:
        num_logical_qubits = _infer_num_logical_qubits(operations)
    layout = dict(initial_layout or initial_layout_trivial(num_logical_qubits, graph["num_qubits"]))
    inverse = {physical: logical for logical, physical in layout.items()}
    routed: list[dict[str, Any]] = []
    swap_count = 0
    for operation in operations:
        name = str(operation["name"]).lower()
        if name in {"cx", "cnot", "cz"}:
            control = int(operation["controls"][0])
            target = int(operation["targets"][0])
            control_physical = layout[control]
            target_physical = layout[target]
            if not _has_edge(graph, control_physical, target_physical):
                path = _shortest_path(graph, control_physical, target_physical)
                for index in range(len(path) - 2):
                    _append_swap(routed, path[index], path[index + 1])
                    _swap_layout(layout, inverse, path[index], path[index + 1])
                    swap_count += 1
                control_physical = layout[control]
                target_physical = layout[target]
            routed.append(
                {
                    "name": "cx" if name == "cnot" else name,
                    "targets": [target_physical],
                    "controls": [control_physical],
                    "params": [],
                    "metadata": {"routed_from": operation},
                }
            )
            if not _has_edge(graph, control_physical, target_physical):
                raise ValueError("routing failed to create an adjacent two-qubit operation")
            if control_physical != layout[control]:
                raise ValueError("routing layout bookkeeping failed")
            # Restore the logical layout for transparent educational equivalence.
            if control_physical != int(operation["controls"][0]):
                path = _shortest_path(graph, int(operation["controls"][0]), control_physical)
                for index in range(len(path) - 1, 0, -1):
                    _append_swap(routed, path[index - 1], path[index])
                    _swap_layout(layout, inverse, path[index - 1], path[index])
                    swap_count += 1
        elif name == "swap":
            left = layout[int(operation["targets"][0])]
            right = layout[int(operation["targets"][1])]
            if not _has_edge(graph, left, right):
                raise ValueError("Stage 9J QMAP-like routing only supports adjacent logical SWAP operations")
            _append_swap(routed, left, right)
            _swap_layout(layout, inverse, left, right)
            swap_count += 1
        else:
            routed.append(_map_single_or_parameterized_operation(operation, layout))
    return {
        "routed_operations": routed,
        "initial_layout": dict(initial_layout or initial_layout_trivial(num_logical_qubits, graph["num_qubits"])),
        "final_layout": dict(layout),
        "swap_count": swap_count,
        "depth_estimate": len(routed),
        "topology": graph,
    }


def insert_swaps_for_unavailable_edges(
    operations: list[dict[str, Any]],
    coupling_graph: dict[str, Any],
    layout: dict[int, int],
) -> dict[str, Any]:
    return route_cnot_sequence_greedy(
        operations,
        coupling_graph,
        initial_layout=layout,
        num_logical_qubits=len(layout),
    )


def _edge(left: int, right: int) -> tuple[int, int]:
    a, b = int(left), int(right)
    return (a, b) if a < b else (b, a)


def _has_edge(graph: dict[str, Any], left: int, right: int) -> bool:
    return _edge(left, right) in {
        _edge(item["source"], item["target"]) for item in graph.get("edges", ())
    }


def _adjacency(graph: dict[str, Any]) -> dict[int, set[int]]:
    adjacent = {index: set() for index in range(int(graph["num_qubits"]))}
    for item in graph.get("edges", ()):
        left, right = int(item["source"]), int(item["target"])
        adjacent[left].add(right)
        adjacent[right].add(left)
    return adjacent


def _shortest_path(graph: dict[str, Any], source: int, target: int) -> list[int]:
    if source == target:
        return [source]
    adjacent = _adjacency(graph)
    queue: deque[tuple[int, list[int]]] = deque([(source, [source])])
    seen = {source}
    while queue:
        node, path = queue.popleft()
        for neighbor in sorted(adjacent[node]):
            if neighbor in seen:
                continue
            if neighbor == target:
                return path + [neighbor]
            seen.add(neighbor)
            queue.append((neighbor, path + [neighbor]))
    raise ValueError(f"no path between physical qubits {source} and {target}")


def _append_swap(routed: list[dict[str, Any]], left: int, right: int) -> None:
    routed.append(
        {
            "name": "swap",
            "targets": [int(left), int(right)],
            "controls": [],
            "params": [],
            "metadata": {"inserted_by": "quantumbridge_qmap_like_greedy_router"},
        }
    )


def _swap_layout(layout: dict[int, int], inverse: dict[int, int], left: int, right: int) -> None:
    logical_left = inverse.get(left)
    logical_right = inverse.get(right)
    if logical_left is not None:
        layout[logical_left] = right
    if logical_right is not None:
        layout[logical_right] = left
    inverse[left], inverse[right] = logical_right, logical_left


def _map_single_or_parameterized_operation(
    operation: dict[str, Any],
    layout: dict[int, int],
) -> dict[str, Any]:
    return {
        "name": str(operation["name"]).lower(),
        "targets": [layout[int(target)] for target in operation.get("targets", ())],
        "controls": [layout[int(control)] for control in operation.get("controls", ())],
        "params": [float(value) for value in operation.get("params", ())],
        "metadata": {"routed_from": operation},
    }


def _infer_num_logical_qubits(operations: list[dict[str, Any]]) -> int:
    max_wire = -1
    for operation in operations:
        wires = list(operation.get("targets", ())) + list(operation.get("controls", ()))
        for wire in wires:
            max_wire = max(max_wire, int(wire))
    return max_wire + 1
