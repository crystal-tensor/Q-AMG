# This file is independently implemented for QuantumBridge SDK.
"""DeviceSpec helpers for QOS-UQCI offline compatibility."""

from __future__ import annotations


def build_devicespec(name: str = "quantumbridge_mock_qos_backend", num_qubits: int = 2) -> dict[str, object]:
    if int(num_qubits) <= 0:
        raise ValueError("DeviceSpec requires a positive qubit count")
    return {
        "schema_version": "0.1",
        "name": name,
        "backend_type": "offline_mock",
        "num_qubits": int(num_qubits),
        "basis_gates": ["x", "y", "z", "h", "rx", "ry", "rz", "phase", "cx", "cz", "swap"],
        "coupling_map": [[i, i + 1] for i in range(max(0, int(num_qubits) - 1))],
        "cloud_access": False,
        "token_read": False,
        "hardware_access": False,
        "production_backend": False,
    }


def validate_devicespec(spec: dict[str, object]) -> bool:
    if spec.get("backend_type") != "offline_mock":
        raise ValueError("DeviceSpec backend_type must be offline_mock")
    if int(spec.get("num_qubits", 0)) <= 0:
        raise ValueError("DeviceSpec num_qubits must be positive")
    for key in ("cloud_access", "token_read", "hardware_access", "production_backend"):
        if spec.get(key) is True:
            raise ValueError(f"DeviceSpec must not set {key}=True")
    return True
