# This file is independently implemented for QuantumBridge SDK.
"""QOS-UQCI offline job-spec helpers."""

from __future__ import annotations

from typing import Any

from quantumbridge.compat.qiskit_aer.simulator_native import normalize_circuit_to_quantumbridge_ir

from .calset_adapter import build_calset, validate_calset
from .devicespec_adapter import build_devicespec, validate_devicespec
from .manifest_adapter import build_manifest, validate_manifest
from .openqasm_bridge import build_openqasm_compatibility_artifact
from .uqci_ir_adapter import quantumbridge_ir_to_uqci_ir, validate_uqci_ir
from .warnings import qos_uqci_warnings, native_provenance


def build_qos_uqci_job_spec(
    circuit_or_ir: Any,
    shots: int = 1024,
    seed: int | None = None,
    backend_name: str = "quantumbridge_mock_qos_backend",
) -> dict[str, Any]:
    circuit = normalize_circuit_to_quantumbridge_ir(circuit_or_ir)
    uqci_ir = quantumbridge_ir_to_uqci_ir(circuit)
    device_spec = build_devicespec(backend_name, circuit.num_qubits)
    calset = build_calset(circuit.num_qubits)
    job_spec = {
        "schema_version": "0.1",
        "job_id": f"qb-qos-uqci-{circuit.name or 'job'}",
        "target": "qos_uqci",
        "backend_name": backend_name,
        "execution_mode": "offline_mock",
        "shots": int(shots),
        "seed": seed,
        "uqci_ir": uqci_ir,
        "device_spec": device_spec,
        "calset": calset,
        "openqasm_compatibility_artifact": build_openqasm_compatibility_artifact(circuit),
        "cloud_access": False,
        "token_read": False,
        "hardware_access": False,
        "production_runtime": False,
        "official_endorsement_claim": False,
        "warnings": qos_uqci_warnings(),
        "provenance": native_provenance("build_qos_uqci_job_spec"),
    }
    job_spec["manifest"] = build_manifest(job_spec)
    validate_qos_uqci_job_spec(job_spec)
    return job_spec


def validate_qos_uqci_job_spec(job_spec: dict[str, Any]) -> bool:
    required = {"schema_version", "job_id", "target", "execution_mode", "shots", "uqci_ir"}
    missing = required - set(job_spec)
    if missing:
        raise ValueError(f"QOS-UQCI job spec missing fields: {', '.join(sorted(missing))}")
    if job_spec["target"] != "qos_uqci":
        raise ValueError("QOS-UQCI job spec target must be qos_uqci")
    if job_spec["execution_mode"] != "offline_mock":
        raise ValueError("QOS-UQCI job spec execution_mode must be offline_mock")
    if int(job_spec["shots"]) <= 0:
        raise ValueError("QOS-UQCI job spec shots must be positive")
    for key in ("cloud_access", "token_read", "hardware_access", "production_runtime", "official_endorsement_claim"):
        if job_spec.get(key) is True:
            raise ValueError(f"QOS-UQCI job spec must not set {key}=True")
    validate_uqci_ir(job_spec["uqci_ir"])
    validate_devicespec(job_spec["device_spec"])
    validate_calset(job_spec["calset"])
    validate_manifest(job_spec["manifest"])
    return True
