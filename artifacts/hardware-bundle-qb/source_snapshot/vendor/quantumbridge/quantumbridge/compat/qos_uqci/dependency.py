# This file is independently implemented for QuantumBridge SDK.
# No source code from QOS-UQCI was copied.
"""Optional dependency checks for QOS-UQCI compatibility."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version


def dependency_available(package_name: str = "qos_uqci") -> bool:
    try:
        import_module(package_name)
        return True
    except Exception:
        return False


def get_upstream_version(package_name: str = "qos_uqci") -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def validate_qos_uqci_dependencies(package_name: str = "qos_uqci") -> dict[str, object]:
    return {
        "package": package_name,
        "available": dependency_available(package_name),
        "version": get_upstream_version(package_name),
        "required_by_default": False,
        "cloud_access": False,
        "token_read": False,
        "hardware_access": False,
    }
