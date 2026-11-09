# This file is independently implemented for QuantumBridge SDK.
"""Manifest helpers for QOS-UQCI offline compatibility."""

from __future__ import annotations


def build_manifest(job_spec: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": "0.1",
        "manifest_type": "qos_uqci_offline_mock",
        "job_id": job_spec.get("job_id"),
        "artifact_count": 3,
        "artifacts": ["uqci_ir", "openqasm_compatibility_artifact", "mock_result"],
        "cloud_access": False,
        "token_read": False,
        "hardware_access": False,
        "production_runtime": False,
    }


def validate_manifest(manifest: dict[str, object]) -> bool:
    if manifest.get("manifest_type") != "qos_uqci_offline_mock":
        raise ValueError("Manifest type must be qos_uqci_offline_mock")
    for key in ("cloud_access", "token_read", "hardware_access", "production_runtime"):
        if manifest.get(key) is True:
            raise ValueError(f"Manifest must not set {key}=True")
    return True
