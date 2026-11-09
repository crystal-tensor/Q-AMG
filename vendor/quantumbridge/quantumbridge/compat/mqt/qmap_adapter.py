# This file is independently implemented for QuantumBridge SDK.
# No source code from MQT, IBM, Qiskit, or QMAP was copied.
"""QMAP-like adapter facade."""

from .mapping_native import (
    compare_original_and_mapped_execution,
    create_coupling_graph,
    create_fully_connected_topology,
    create_line_topology,
    create_ring_topology,
    initial_layout_trivial,
    insert_swaps_for_unavailable_edges,
    map_quantumbridge_ir_to_topology,
    mapping_cost,
    route_cnot_sequence_greedy,
    validate_coupling_graph,
)

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
    "route_cnot_sequence_greedy",
    "validate_coupling_graph",
]
