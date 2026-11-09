# This file is independently implemented for QuantumBridge SDK.
# No source code from MQT, IBM, or Qiskit was copied.
"""Examples for Stage 9J MQT Core/DDSIM/QMAP compatibility."""

from __future__ import annotations

from quantumbridge.core import Circuit

from .core_adapter import (
    build_mqt_core_like_circuit_from_quantumbridge_ir,
    export_mqt_core_like_qasm_subset,
    import_mqt_core_like_qasm_subset,
    mqt_core_like_dict_to_quantumbridge_ir,
)
from .ddsim_adapter import (
    compare_ddsim_like_with_stage9f_simulator,
    run_ddsim_like_counts_native,
    run_ddsim_like_statevector_native,
)
from .mapping_native import (
    compare_original_and_mapped_execution,
    create_line_topology,
    map_quantumbridge_ir_to_topology,
    mapping_cost,
)
from .upstream_adapter import (
    run_upstream_ddsim_if_available,
    run_upstream_mqt_core_if_available,
    run_upstream_qmap_if_available,
)


def mqt_bell_circuit(measured: bool = True) -> Circuit:
    circuit = Circuit(2, 2 if measured else 0, name="qb_mqt_bell")
    circuit.h(0).cx(0, 1)
    if measured:
        circuit.measure(0, 0).measure(1, 1)
    return circuit


def mqt_nonlocal_cnot_circuit(measured: bool = True) -> Circuit:
    circuit = Circuit(3, 3 if measured else 0, name="qb_mqt_nonlocal_cnot")
    circuit.h(0).cx(0, 2)
    if measured:
        circuit.measure(0, 0).measure(1, 1).measure(2, 2)
    return circuit


def run_mqt_core_like_example() -> dict[str, object]:
    circuit = mqt_bell_circuit(measured=True)
    native = build_mqt_core_like_circuit_from_quantumbridge_ir(circuit.to_ir())
    qasm = export_mqt_core_like_qasm_subset(
        {
            "num_qubits": native.num_qubits,
            "num_bits": 2,
            "operations": native.operations,
            "measurements": native.measurements,
        }
    )
    imported = import_mqt_core_like_qasm_subset(qasm)
    roundtrip_ir = mqt_core_like_dict_to_quantumbridge_ir(imported)
    upstream = run_upstream_mqt_core_if_available(imported)
    return {
        "native": native,
        "qasm": qasm,
        "roundtrip_ir": roundtrip_ir,
        "upstream": upstream,
    }


def run_mqt_ddsim_like_example(shots: int = 128, seed: int = 11) -> dict[str, object]:
    circuit = mqt_bell_circuit(measured=True)
    statevector = run_ddsim_like_statevector_native(circuit)
    counts = run_ddsim_like_counts_native(circuit, shots=shots, seed=seed)
    comparison = compare_ddsim_like_with_stage9f_simulator(circuit, shots=shots, seed=seed)
    upstream = run_upstream_ddsim_if_available(circuit)
    return {
        "statevector": statevector,
        "counts": counts,
        "comparison": comparison,
        "upstream": upstream,
    }


def run_mqt_qmap_like_example(shots: int = 128, seed: int = 13) -> dict[str, object]:
    circuit = mqt_nonlocal_cnot_circuit(measured=True)
    topology = create_line_topology(3)
    mapping = map_quantumbridge_ir_to_topology(circuit, topology)
    comparison = compare_original_and_mapped_execution(circuit, mapping, shots=shots, seed=seed)
    upstream = run_upstream_qmap_if_available(mapping.mapped_ir)
    return {
        "mapping": mapping,
        "cost": mapping_cost(mapping),
        "comparison": comparison,
        "upstream": upstream,
    }
