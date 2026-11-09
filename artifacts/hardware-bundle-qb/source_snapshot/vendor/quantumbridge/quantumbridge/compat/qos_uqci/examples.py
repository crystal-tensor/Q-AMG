# This file is independently implemented for QuantumBridge SDK.
"""Examples for Stage 10A QOS-UQCI backend compatibility."""

from __future__ import annotations

from quantumbridge.core import Circuit

from .job_spec import build_qos_uqci_job_spec
from .mock_runtime import run_qos_uqci_mock_runtime
from .upstream_adapter import run_upstream_qos_uqci_if_available


def qos_uqci_bell_circuit() -> Circuit:
    circuit = Circuit(2, 2, name="qos_uqci_bell")
    circuit.h(0).cx(0, 1).measure(0, 0).measure(1, 1)
    return circuit


def run_qos_uqci_bell_job_example(shots: int = 128, seed: int = 21) -> dict[str, object]:
    circuit = qos_uqci_bell_circuit()
    job_spec = build_qos_uqci_job_spec(circuit, shots=shots, seed=seed)
    upstream = run_upstream_qos_uqci_if_available(job_spec)
    return {"circuit_ir": circuit.to_ir().to_dict(), "job_spec": job_spec, "upstream": upstream}


def run_qos_uqci_mock_runtime_example(shots: int = 128, seed: int = 21) -> dict[str, object]:
    circuit = qos_uqci_bell_circuit()
    result = run_qos_uqci_mock_runtime(circuit, shots=shots, seed=seed)
    upstream = run_upstream_qos_uqci_if_available(result.job_spec)
    return {"result": result, "upstream": upstream}
