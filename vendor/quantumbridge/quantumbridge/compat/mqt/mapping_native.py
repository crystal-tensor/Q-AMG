# This file is independently implemented for QuantumBridge SDK.
# No source code from MQT, IBM, Qiskit, or QMAP was copied.
"""Educational QMAP-like mapping workflows."""

from __future__ import annotations

from typing import Any

from quantumbridge.compat.qiskit_aer.simulator_native import run_qasm_simulator_native
from quantumbridge.schema.mqt_results import MQTComparisonResult, QMAPLikeMappingResult

from .core_adapter import (
    mqt_core_like_dict_to_quantumbridge_ir,
    quantumbridge_ir_to_mqt_core_like_dict,
    validate_mqt_core_like_circuit,
)
from .routing_native import (
    create_coupling_graph,
    create_fully_connected_topology,
    create_line_topology,
    create_ring_topology,
    initial_layout_trivial,
    insert_swaps_for_unavailable_edges,
    route_cnot_sequence_greedy,
    validate_coupling_graph,
)
from .warnings import QMAP_LIKE_WARNING, mqt_warnings, native_provenance


def map_quantumbridge_ir_to_topology(
    ir_or_circuit: Any,
    coupling_graph: dict[str, Any],
    initial_layout: dict[int, int] | None = None,
) -> QMAPLikeMappingResult:
    core_like = quantumbridge_ir_to_mqt_core_like_dict(ir_or_circuit)
    graph = validate_coupling_graph(coupling_graph)
    layout = initial_layout or initial_layout_trivial(core_like["num_qubits"], graph["num_qubits"])
    routed = route_cnot_sequence_greedy(
        list(core_like["operations"]),
        graph,
        initial_layout=layout,
        num_logical_qubits=core_like["num_qubits"],
    )
    mapped_core_like = {
        **core_like,
        "num_qubits": graph["num_qubits"],
        "operations": routed["routed_operations"],
        "measurements": _map_measurements(core_like.get("measurements", ()), routed["final_layout"]),
        "metadata": {
            **dict(core_like.get("metadata", {})),
            "mapped_by": "quantumbridge_qmap_like_greedy_router",
            "qmap_parity_claim": False,
            "optimal_mapper_claim": False,
        },
    }
    mapped_core_like = validate_mqt_core_like_circuit(mapped_core_like)
    return QMAPLikeMappingResult(
        workflow="qmap_like_mapping_native",
        project="mqt-qmap",
        mode="native_minimal",
        capability_level=3,
        production_ready=False,
        native_implementation=True,
        circuit_summary={
            "original_operation_count": len(core_like["operations"]),
            "mapped_operation_count": len(mapped_core_like["operations"]),
            "measurement_count": len(mapped_core_like["measurements"]),
        },
        num_qubits=graph["num_qubits"],
        operations=list(mapped_core_like["operations"]),
        measurements=list(mapped_core_like["measurements"]),
        topology=graph,
        initial_layout=dict(routed["initial_layout"]),
        final_layout=dict(routed["final_layout"]),
        swap_count=int(routed["swap_count"]),
        depth_estimate=int(routed["depth_estimate"]),
        mapped_ir=mapped_core_like,
        raw_type="QuantumBridgeQMAPLikeMapping",
        metadata={
            "cloud_access": False,
            "token_read": False,
            "hardware_access": False,
            "qmap_parity_claim": False,
            "production_compiler": False,
            "optimal_mapper_claim": False,
        },
        warnings=mqt_warnings(QMAP_LIKE_WARNING),
        provenance=native_provenance("qmap_like_mapping_native", "mqt-qmap"),
    )


def mapping_cost(mapped: QMAPLikeMappingResult | dict[str, Any]) -> dict[str, int]:
    if hasattr(mapped, "swap_count"):
        return {
            "swap_count": int(mapped.swap_count or 0),
            "depth_estimate": int(mapped.depth_estimate or 0),
            "operation_count": len(mapped.operations),
        }
    operations = list(mapped.get("operations", ()))
    return {
        "swap_count": sum(1 for op in operations if op.get("name") == "swap"),
        "depth_estimate": len(operations),
        "operation_count": len(operations),
    }


def compare_original_and_mapped_execution(
    original_ir_or_circuit: Any,
    mapped_ir_or_result: QMAPLikeMappingResult | dict[str, Any],
    shots: int = 1024,
    seed: int | None = 23,
) -> MQTComparisonResult:
    mapped_ir = mapped_ir_or_result.mapped_ir if hasattr(mapped_ir_or_result, "mapped_ir") else mapped_ir_or_result
    original = run_qasm_simulator_native(original_ir_or_circuit, shots=shots, seed=seed)
    mapped_program = mqt_core_like_dict_to_quantumbridge_ir(mapped_ir)
    mapped = run_qasm_simulator_native(mapped_program, shots=shots, seed=seed)
    distance = _l1_distance(original.probabilities, mapped.probabilities)
    return MQTComparisonResult(
        workflow="qmap_like_original_mapped_execution_comparison",
        project="mqt-qmap",
        mode="comparison",
        capability_level=3,
        production_ready=False,
        native_implementation=True,
        circuit_summary={
            "original_counts": dict(original.counts),
            "mapped_counts": dict(mapped.counts),
        },
        num_qubits=mapped.num_qubits,
        counts=dict(mapped.counts),
        probabilities=dict(mapped.probabilities),
        mapped_ir=dict(mapped_ir),
        comparison={
            "original_probabilities": dict(original.probabilities),
            "mapped_probabilities": dict(mapped.probabilities),
            "l1_probability_distance": distance,
            "comparable": distance <= 1e-12,
        },
        raw_type="QuantumBridgeQMAPLikeExecutionComparison",
        metadata={
            "shots": int(shots),
            "seed": seed,
            "cloud_access": False,
            "token_read": False,
            "hardware_access": False,
            "qmap_parity_claim": False,
            "production_compiler": False,
            "optimal_mapper_claim": False,
        },
        warnings=mqt_warnings(QMAP_LIKE_WARNING),
        provenance=native_provenance("qmap_like_original_mapped_execution_comparison", "mqt-qmap"),
    )


def _map_measurements(measurements: list[dict[str, Any]], final_layout: dict[int, int]) -> list[dict[str, Any]]:
    return [
        {
            "kind": str(item.get("kind", "measure")),
            "wire": int(final_layout[int(item["wire"])]),
            "bit": int(item["bit"]),
        }
        for item in measurements
    ]


def _l1_distance(left: dict[str, float], right: dict[str, float]) -> float:
    keys = set(left) | set(right)
    return float(sum(abs(float(left.get(key, 0.0)) - float(right.get(key, 0.0))) for key in keys))


__all__ = [
    "compare_original_and_mapped_execution",
    "create_coupling_graph",
    "create_fully_connected_topology",
    "create_line_topology",
    "create_ring_topology",
    "initial_layout_trivial",
    "insert_swaps_for_unavailable_edges",
    "map_quantumbridge_ir_to_topology",
    "mapping_cost",
    "mqt_core_like_dict_to_quantumbridge_ir",
    "route_cnot_sequence_greedy",
    "validate_coupling_graph",
]
