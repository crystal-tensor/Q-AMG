# This file is independently implemented for QuantumBridge SDK.
# No source code from MQT, IBM, Qiskit, or DDSIM was copied.
"""Clean-room DDSIM-like educational simulator compatibility helpers."""

from __future__ import annotations

from typing import Any

from quantumbridge.compat.qiskit_aer.simulator_native import (
    run_qasm_simulator_native,
    run_statevector_simulator_native,
)
from quantumbridge.schema.mqt_results import DDSIMLikeSimulationResult, MQTComparisonResult

from .core_adapter import quantumbridge_ir_to_mqt_core_like_dict
from .decision_diagram_metadata import build_decision_diagram_metadata
from .warnings import DDSIM_LIKE_WARNING, mqt_warnings, native_provenance


def run_ddsim_like_statevector_native(circuit_or_ir: Any) -> DDSIMLikeSimulationResult:
    aer_result = run_statevector_simulator_native(circuit_or_ir)
    metadata = build_decision_diagram_metadata(aer_result.final_statevector, aer_result.num_qubits)
    core_like = quantumbridge_ir_to_mqt_core_like_dict(circuit_or_ir)
    return DDSIMLikeSimulationResult(
        workflow="ddsim_like_statevector_native",
        project="mqt-ddsim",
        mode="native_minimal",
        capability_level=3,
        production_ready=False,
        native_implementation=True,
        circuit_summary=_summary_from_core_like(core_like),
        num_qubits=aer_result.num_qubits,
        operations=list(core_like["operations"]),
        measurements=list(core_like["measurements"]),
        statevector=list(aer_result.final_statevector),
        probabilities=dict(aer_result.probabilities),
        decision_diagram_metadata=metadata,
        raw_type="QuantumBridgeDDSIMLikeStatevector",
        metadata={
            "backend": "quantumbridge_native_statevector",
            "cloud_access": False,
            "token_read": False,
            "hardware_access": False,
            "decision_diagram_parity_claim": False,
            "production_simulator": False,
        },
        warnings=mqt_warnings(DDSIM_LIKE_WARNING),
        provenance=native_provenance("ddsim_like_statevector_native", "mqt-ddsim"),
    )


def run_ddsim_like_counts_native(
    circuit_or_ir: Any,
    shots: int = 1024,
    seed: int | None = None,
) -> DDSIMLikeSimulationResult:
    qasm_result = run_qasm_simulator_native(circuit_or_ir, shots=shots, seed=seed)
    state_result = run_ddsim_like_statevector_native(circuit_or_ir)
    return DDSIMLikeSimulationResult(
        workflow="ddsim_like_counts_native",
        project="mqt-ddsim",
        mode="native_minimal",
        capability_level=3,
        production_ready=False,
        native_implementation=True,
        circuit_summary=dict(state_result.circuit_summary),
        num_qubits=qasm_result.num_qubits,
        operations=list(state_result.operations),
        measurements=list(state_result.measurements),
        statevector=list(state_result.statevector),
        counts=dict(qasm_result.counts),
        probabilities=dict(qasm_result.probabilities),
        decision_diagram_metadata=dict(state_result.decision_diagram_metadata),
        raw_type="QuantumBridgeDDSIMLikeCounts",
        metadata={
            "backend": "quantumbridge_native_qasm",
            "shots": int(shots),
            "seed": seed,
            "cloud_access": False,
            "token_read": False,
            "hardware_access": False,
            "decision_diagram_parity_claim": False,
            "production_simulator": False,
        },
        warnings=mqt_warnings(DDSIM_LIKE_WARNING),
        provenance=native_provenance("ddsim_like_counts_native", "mqt-ddsim"),
    )


def compare_ddsim_like_with_stage9f_simulator(
    circuit_or_ir: Any,
    shots: int = 1024,
    seed: int | None = 17,
) -> MQTComparisonResult:
    ddsim_like = run_ddsim_like_counts_native(circuit_or_ir, shots=shots, seed=seed)
    stage9f = run_qasm_simulator_native(circuit_or_ir, shots=shots, seed=seed)
    distance = _l1_distance(ddsim_like.probabilities, stage9f.probabilities)
    return MQTComparisonResult(
        workflow="ddsim_like_stage9f_comparison",
        project="mqt-ddsim",
        mode="comparison",
        capability_level=3,
        production_ready=False,
        native_implementation=True,
        circuit_summary=dict(ddsim_like.circuit_summary),
        num_qubits=ddsim_like.num_qubits,
        operations=list(ddsim_like.operations),
        measurements=list(ddsim_like.measurements),
        counts=dict(ddsim_like.counts),
        probabilities=dict(ddsim_like.probabilities),
        decision_diagram_metadata=dict(ddsim_like.decision_diagram_metadata),
        comparison={
            "stage9f_counts": dict(stage9f.counts),
            "stage9f_probabilities": dict(stage9f.probabilities),
            "l1_probability_distance": distance,
            "comparable": distance <= 1e-12,
        },
        raw_type="QuantumBridgeDDSIMLikeComparison",
        metadata={
            "cloud_access": False,
            "token_read": False,
            "hardware_access": False,
            "decision_diagram_parity_claim": False,
            "production_simulator": False,
        },
        warnings=mqt_warnings(DDSIM_LIKE_WARNING),
        provenance=native_provenance("ddsim_like_stage9f_comparison", "mqt-ddsim"),
    )


def _summary_from_core_like(core_like: dict[str, Any]) -> dict[str, Any]:
    return {
        "num_qubits": int(core_like["num_qubits"]),
        "num_bits": int(core_like.get("num_bits", 0)),
        "operation_count": len(core_like.get("operations", ())),
        "measurement_count": len(core_like.get("measurements", ())),
    }


def _l1_distance(left: dict[str, float], right: dict[str, float]) -> float:
    keys = set(left) | set(right)
    return float(sum(abs(float(left.get(key, 0.0)) - float(right.get(key, 0.0))) for key in keys))
