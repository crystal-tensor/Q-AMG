# This file is independently implemented for QuantumBridge SDK.
# No source code from MQT, IBM, or Qiskit was copied.
"""Clean-room MQT Core/DDSIM/QMAP compatibility executable slice."""

from quantumbridge.compat.mqt.core_adapter import (
    build_mqt_core_like_circuit_from_quantumbridge_ir,
    export_mqt_core_like_qasm_subset,
    import_mqt_core_like_qasm_subset,
    mqt_core_like_dict_to_quantumbridge_ir,
    quantumbridge_ir_to_mqt_core_like_dict,
    validate_mqt_core_like_circuit,
)
from quantumbridge.compat.mqt.ddsim_adapter import (
    compare_ddsim_like_with_stage9f_simulator,
    run_ddsim_like_counts_native,
    run_ddsim_like_statevector_native,
)
from quantumbridge.compat.mqt.decision_diagram_metadata import (
    build_decision_diagram_metadata,
    summarize_decision_diagram_metadata,
    wrap_decision_diagram_metadata,
)
from quantumbridge.compat.mqt.dependency import (
    dependency_available,
    get_upstream_version,
    validate_mqt_dependencies,
)
from quantumbridge.compat.mqt.examples import (
    mqt_bell_circuit,
    mqt_nonlocal_cnot_circuit,
    run_mqt_core_like_example,
    run_mqt_ddsim_like_example,
    run_mqt_qmap_like_example,
)
from quantumbridge.compat.mqt.mapping_native import (
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
from quantumbridge.compat.mqt.result_adapter import to_quantumbridge_schema, wrap_mqt_result
from quantumbridge.compat.mqt.upstream_adapter import (
    run_upstream_ddsim_if_available,
    run_upstream_mqt_core_if_available,
    run_upstream_qmap_if_available,
    wrap_upstream_mqt_result,
)
from quantumbridge.compat.mqt.warnings import (
    DDSIM_LIKE_WARNING,
    MQT_COMPAT_WARNING,
    QMAP_LIKE_WARNING,
    UPSTREAM_MQT_WARNING,
    mqt_warnings,
    native_provenance,
)

__all__ = [
    "DDSIM_LIKE_WARNING",
    "MQT_COMPAT_WARNING",
    "QMAP_LIKE_WARNING",
    "UPSTREAM_MQT_WARNING",
    "build_decision_diagram_metadata",
    "build_mqt_core_like_circuit_from_quantumbridge_ir",
    "compare_ddsim_like_with_stage9f_simulator",
    "compare_original_and_mapped_execution",
    "create_coupling_graph",
    "create_fully_connected_topology",
    "create_line_topology",
    "create_ring_topology",
    "dependency_available",
    "export_mqt_core_like_qasm_subset",
    "get_upstream_version",
    "import_mqt_core_like_qasm_subset",
    "initial_layout_trivial",
    "insert_swaps_for_unavailable_edges",
    "map_quantumbridge_ir_to_topology",
    "mapping_cost",
    "mqt_bell_circuit",
    "mqt_core_like_dict_to_quantumbridge_ir",
    "mqt_nonlocal_cnot_circuit",
    "mqt_warnings",
    "native_provenance",
    "quantumbridge_ir_to_mqt_core_like_dict",
    "route_cnot_sequence_greedy",
    "run_ddsim_like_counts_native",
    "run_ddsim_like_statevector_native",
    "run_mqt_core_like_example",
    "run_mqt_ddsim_like_example",
    "run_mqt_qmap_like_example",
    "run_upstream_ddsim_if_available",
    "run_upstream_mqt_core_if_available",
    "run_upstream_qmap_if_available",
    "summarize_decision_diagram_metadata",
    "to_quantumbridge_schema",
    "validate_coupling_graph",
    "validate_mqt_core_like_circuit",
    "validate_mqt_dependencies",
    "wrap_decision_diagram_metadata",
    "wrap_mqt_result",
    "wrap_upstream_mqt_result",
]
