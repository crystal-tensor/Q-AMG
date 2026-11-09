# This file is independently implemented for QuantumBridge SDK.
# No source code from IBM, Qiskit, or Qiskit Experiments was copied.
"""Optional dependency helpers for qiskit-experiments."""

from __future__ import annotations

import importlib
from importlib import metadata


def dependency_available() -> bool:
    return importlib.util.find_spec("qiskit_experiments") is not None


def get_upstream_version() -> str | None:
    if not dependency_available():
        return None
    try:
        return metadata.version("qiskit-experiments")
    except metadata.PackageNotFoundError:
        return "unknown"


def get_version() -> str | None:
    return get_upstream_version()


def validate_experiments_dependencies() -> dict[str, object]:
    available = dependency_available()
    return {
        "dependency": "qiskit-experiments",
        "available": available,
        "version": get_upstream_version() if available else None,
        "mode": "upstream_passthrough" if available else "unsupported",
        "cloud_access": False,
        "token_read": False,
        "hardware_access": False,
    }
