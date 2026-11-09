# This file is independently implemented for QuantumBridge SDK.
"""Examples for Stage 10A Quafu backend compatibility."""

from __future__ import annotations

from quantumbridge.core import Circuit

from .job_adapter import build_quafu_job_spec
from .mock_backend import run_quafu_mock_backend
from .payload_adapter import quantumbridge_ir_to_quafu_payload
from .upstream_adapter import run_upstream_pyquafu_if_available


def quafu_bell_circuit() -> Circuit:
    circuit = Circuit(2, 2, name="quafu_bell")
    circuit.h(0).cx(0, 1).measure(0, 0).measure(1, 1)
    return circuit


def run_quafu_bell_payload_example(shots: int = 128, seed: int = 23) -> dict[str, object]:
    circuit = quafu_bell_circuit()
    payload = quantumbridge_ir_to_quafu_payload(circuit)
    job_spec = build_quafu_job_spec(circuit, shots=shots, seed=seed)
    upstream = run_upstream_pyquafu_if_available(payload)
    return {"circuit_ir": circuit.to_ir().to_dict(), "payload": payload, "job_spec": job_spec, "upstream": upstream}


def run_quafu_mock_backend_example(shots: int = 128, seed: int = 23) -> dict[str, object]:
    circuit = quafu_bell_circuit()
    result = run_quafu_mock_backend(circuit, shots=shots, seed=seed)
    upstream = run_upstream_pyquafu_if_available(result.payload)
    return {"result": result, "upstream": upstream}
