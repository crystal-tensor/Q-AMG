# This file is independently implemented for QuantumBridge SDK.
# No source code from PennyLane-Qiskit was copied.
"""Optional dependency helpers for the PennyLane-Qiskit plugin."""

from __future__ import annotations

from importlib import metadata, util


def dependency_available() -> bool:
    return util.find_spec("pennylane_qiskit") is not None


def get_upstream_version() -> str | None:
    for package_name in ("pennylane-qiskit", "pennylane_qiskit"):
        try:
            return metadata.version(package_name)
        except metadata.PackageNotFoundError:
            continue
    return None


def validate_pennylane_qiskit_dependencies() -> dict[str, object]:
    available = dependency_available()
    return {
        "package": "pennylane-qiskit",
        "import_name": "pennylane_qiskit",
        "available": available,
        "version": get_upstream_version() if available else None,
        "required_by_default": False,
        "cloud_access": False,
        "token_read": False,
        "hardware_access": False,
    }
