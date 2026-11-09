# This file is independently implemented for QuantumBridge SDK.
"""Offline mock Quafu backend execution."""

from __future__ import annotations

from typing import Any

from quantumbridge.compat.qiskit_aer.simulator_native import run_qasm_simulator_native
from quantumbridge.schema.backend_results import QuafuBackendResult

from .job_adapter import build_quafu_job_spec, validate_quafu_job_spec
from .payload_adapter import quafu_payload_to_quantumbridge_ir
from .warnings import native_provenance, quafu_warnings


def run_quafu_mock_backend(job_spec_or_circuit: Any, shots: int = 1024, seed: int | None = None) -> QuafuBackendResult:
    if isinstance(job_spec_or_circuit, dict) and job_spec_or_circuit.get("target") == "quafu":
        job_spec = dict(job_spec_or_circuit)
    else:
        job_spec = build_quafu_job_spec(job_spec_or_circuit, shots=shots, seed=seed)
    validate_quafu_job_spec(job_spec)
    qb_ir = quafu_payload_to_quantumbridge_ir(job_spec["payload"])
    qasm_result = run_qasm_simulator_native(qb_ir, shots=int(job_spec["shots"]), seed=job_spec.get("seed"))
    return QuafuBackendResult(
        workflow="quafu_mock_backend",
        project="quafu",
        mode="native_mock",
        capability_level=3,
        production_ready=False,
        native_implementation=True,
        job_spec=job_spec,
        payload=job_spec["payload"],
        counts=dict(qasm_result.counts),
        probabilities=dict(qasm_result.probabilities),
        shots=int(job_spec["shots"]),
        seed=job_spec.get("seed"),
        raw_type="QuantumBridgeQuafuMockBackend",
        metadata={
            "backend_name": job_spec["backend_name"],
            "cloud_access": False,
            "token_read": False,
            "hardware_access": False,
            "production_backend": False,
            "official_endorsement_claim": False,
        },
        warnings=quafu_warnings(),
        provenance=native_provenance("quafu_mock_backend"),
    )
