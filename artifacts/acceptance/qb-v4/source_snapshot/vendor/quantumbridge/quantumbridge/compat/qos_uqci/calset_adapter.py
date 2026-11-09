# This file is independently implemented for QuantumBridge SDK.
"""Calibration-set helpers for QOS-UQCI offline compatibility."""

from __future__ import annotations


def build_calset(num_qubits: int = 2) -> dict[str, object]:
    if int(num_qubits) <= 0:
        raise ValueError("CalSet requires a positive qubit count")
    return {
        "schema_version": "0.1",
        "calibration_type": "offline_mock",
        "qubits": [{"id": i, "t1_us": None, "t2_us": None, "readout_error": 0.0} for i in range(int(num_qubits))],
        "generated_from_hardware": False,
        "cloud_access": False,
        "token_read": False,
        "hardware_access": False,
    }


def validate_calset(calset: dict[str, object]) -> bool:
    if calset.get("calibration_type") != "offline_mock":
        raise ValueError("CalSet calibration_type must be offline_mock")
    for key in ("generated_from_hardware", "cloud_access", "token_read", "hardware_access"):
        if calset.get(key) is True:
            raise ValueError(f"CalSet must not set {key}=True")
    return True
