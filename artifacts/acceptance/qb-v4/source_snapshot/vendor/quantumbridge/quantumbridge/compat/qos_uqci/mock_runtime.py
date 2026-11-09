# This file is independently implemented for QuantumBridge SDK.
"""Offline mock runtime for QOS-UQCI job specs."""

from __future__ import annotations

from typing import Any

from quantumbridge.compat.qiskit_aer.simulator_native import run_qasm_simulator_native
from quantumbridge.schema.backend_results import QOSUQCIBackendResult

from .job_spec import build_qos_uqci_job_spec, validate_qos_uqci_job_spec
from .uqci_ir_adapter import uqci_ir_to_quantumbridge_ir
from .warnings import qos_uqci_warnings, native_provenance


def run_qos_uqci_mock_runtime(job_spec_or_circuit: Any, shots: int = 1024, seed: int | None = None) -> QOSUQCIBackendResult:
    if isinstance(job_spec_or_circuit, dict) and job_spec_or_circuit.get("target") == "qos_uqci":
        job_spec = dict(job_spec_or_circuit)
    else:
        job_spec = build_qos_uqci_job_spec(job_spec_or_circuit, shots=shots, seed=seed)
    validate_qos_uqci_job_spec(job_spec)
    qb_ir = uqci_ir_to_quantumbridge_ir(job_spec["uqci_ir"])
    qasm_result = run_qasm_simulator_native(qb_ir, shots=int(job_spec["shots"]), seed=job_spec.get("seed"))
    return QOSUQCIBackendResult(
        workflow="qos_uqci_mock_runtime",
        project="qos-uqci",
        mode="native_mock",
        capability_level=3,
        production_ready=False,
        native_implementation=True,
        job_spec=job_spec,
        payload=job_spec["uqci_ir"],
        device_spec=job_spec["device_spec"],
        calset=job_spec["calset"],
        manifest=job_spec["manifest"],
        openqasm_artifact=job_spec["openqasm_compatibility_artifact"],
        counts=dict(qasm_result.counts),
        probabilities=dict(qasm_result.probabilities),
        shots=int(job_spec["shots"]),
        seed=job_spec.get("seed"),
        raw_type="QuantumBridgeQOSUQCIMockRuntime",
        metadata={
            "backend_name": job_spec["backend_name"],
            "cloud_access": False,
            "token_read": False,
            "hardware_access": False,
            "production_backend": False,
            "official_endorsement_claim": False,
        },
        warnings=qos_uqci_warnings(),
        provenance=native_provenance("qos_uqci_mock_runtime"),
    )
