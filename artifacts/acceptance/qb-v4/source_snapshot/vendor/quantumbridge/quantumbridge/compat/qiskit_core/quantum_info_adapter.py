# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
"""Qiskit quantum_info public API inventory and passthrough scaffold.

Design source: docs/compat/strategy/qiskit_ecosystem_coverage_strategy.md.
"""

from quantumbridge.ecosystem.registry import EcosystemAdapter

ADAPTER = EcosystemAdapter("qiskit", "qiskit", ("qiskit.quantum_info",), "qiskit-core", "qiskit_core_quantum_info")

dependency_available = ADAPTER.dependency_available
get_upstream_version = ADAPTER.get_upstream_version
list_public_api_inventory = ADAPTER.list_public_api_inventory
passthrough_class = ADAPTER.passthrough_class
passthrough_function = ADAPTER.passthrough_function
wrap_result = ADAPTER.wrap_result
to_quantumbridge_schema = ADAPTER.to_quantumbridge_schema
provenance_metadata = ADAPTER.provenance_metadata
warn_unsupported = ADAPTER.warn_unsupported


def get_native_quantum_info_symbols() -> dict[str, object]:
    from quantumbridge.information import (
        Chi,
        Choi,
        Clifford,
        DensityMatrix,
        Kraus,
        Operator,
        PTM,
        Statevector,
        StabilizerState,
        Stinespring,
        SuperOp,
        anti_commutator,
        average_gate_fidelity,
        commutator,
        concurrence,
        double_commutator,
        entanglement_of_formation,
        entropy,
        gate_error,
        partial_trace,
        mutual_information,
        negativity,
        pauli_basis,
        process_fidelity,
        purity,
        random_clifford,
        random_density_matrix,
        random_hermitian,
        random_pauli,
        random_pauli_list,
        random_quantum_channel,
        random_statevector,
        random_unitary,
        schmidt_decomposition,
        shannon_entropy,
        state_fidelity,
        state_fidelity_general,
        trace_distance,
    )
    from quantumbridge.operators import Pauli, PauliList, SparsePauliOp, SparsePauliOperator

    return {
        "Chi": Chi,
        "Choi": Choi,
        "Clifford": Clifford,
        "DensityMatrix": DensityMatrix,
        "Kraus": Kraus,
        "Operator": Operator,
        "PTM": PTM,
        "Pauli": Pauli,
        "PauliList": PauliList,
        "SparsePauliOp": SparsePauliOp,
        "SparsePauliOperator": SparsePauliOperator,
        "Statevector": Statevector,
        "StabilizerState": StabilizerState,
        "Stinespring": Stinespring,
        "SuperOp": SuperOp,
        "anti_commutator": anti_commutator,
        "average_gate_fidelity": average_gate_fidelity,
        "commutator": commutator,
        "concurrence": concurrence,
        "double_commutator": double_commutator,
        "entanglement_of_formation": entanglement_of_formation,
        "entropy": entropy,
        "gate_error": gate_error,
        "partial_trace": partial_trace,
        "mutual_information": mutual_information,
        "negativity": negativity,
        "pauli_basis": pauli_basis,
        "process_fidelity": process_fidelity,
        "purity": purity,
        "random_clifford": random_clifford,
        "random_density_matrix": random_density_matrix,
        "random_hermitian": random_hermitian,
        "random_pauli": random_pauli,
        "random_pauli_list": random_pauli_list,
        "random_quantum_channel": random_quantum_channel,
        "random_statevector": random_statevector,
        "random_unitary": random_unitary,
        "schmidt_decomposition": schmidt_decomposition,
        "shannon_entropy": shannon_entropy,
        "state_fidelity": state_fidelity,
        "state_fidelity_general": state_fidelity_general,
        "trace_distance": trace_distance,
    }


def native_class(name: str):
    try:
        return get_native_quantum_info_symbols()[name]
    except KeyError as exc:
        raise KeyError(f"QuantumBridge has no native qiskit.quantum_info symbol {name!r}.") from exc
