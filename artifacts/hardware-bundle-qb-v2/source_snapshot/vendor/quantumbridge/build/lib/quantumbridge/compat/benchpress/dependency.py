# This file is independently implemented for QuantumBridge SDK.
# No source code from IBM, Qiskit, or Benchpress was copied.
"""Optional upstream Benchpress dependency detection."""

from __future__ import annotations

from importlib import metadata, util


def dependency_available(package_name: str = "benchpress") -> bool:
    return util.find_spec(package_name) is not None


def get_upstream_version(package_name: str = "benchpress") -> str | None:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def validate_benchpress_dependencies(package_name: str = "benchpress") -> dict[str, object]:
    available = dependency_available(package_name)
    return {
        "package": package_name,
        "available": available,
        "version": get_upstream_version(package_name) if available else None,
        "network_required": False,
        "cloud_required": False,
        "credentials_required": False,
        "hardware_required": False,
        "production_ready": False,
        "missing": () if available else (package_name,),
    }
