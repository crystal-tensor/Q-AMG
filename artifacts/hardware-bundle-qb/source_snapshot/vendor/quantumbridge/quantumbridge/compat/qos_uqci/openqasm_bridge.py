# This file is independently implemented for QuantumBridge SDK.
"""OpenQASM compatibility artifact helpers for QOS-UQCI."""

from __future__ import annotations

from typing import Any

from quantumbridge.compat.qiskit_aer.simulator_native import normalize_circuit_to_quantumbridge_ir
from quantumbridge.serialization import serialize_circuit,qasm2_loads


def build_openqasm_compatibility_artifact(circuit_or_ir: Any) -> dict[str, object]:
    circuit = normalize_circuit_to_quantumbridge_ir(circuit_or_ir)
    artifact = serialize_circuit(circuit,"qasm2",metadata_policy="external")
    return {
        "schema_version": "0.1",
        "format": "openqasm2_compatibility_artifact",
        "canonical_ir": "qos_uqci_clean_room_ir",
        "qasm": artifact.data,
        "external_native_metadata": artifact.sidecar,
        "generated": True,
        "cloud_access": False,
        "token_read": False,
        "hardware_access": False,
    }


def validate_openqasm_compatibility_artifact(artifact: dict[str, object]) -> bool:
    if artifact.get("format") != "openqasm2_compatibility_artifact":
        raise ValueError("OpenQASM artifact format must be openqasm2_compatibility_artifact")
    if "OPENQASM 2.0" not in str(artifact.get("qasm", "")):
        raise ValueError("OpenQASM artifact must contain an OPENQASM 2.0 program")
    qasm2_loads(str(artifact["qasm"]))
    for key in ("cloud_access", "token_read", "hardware_access"):
        if artifact.get(key) is True:
            raise ValueError(f"OpenQASM artifact must not set {key}=True")
    return True
