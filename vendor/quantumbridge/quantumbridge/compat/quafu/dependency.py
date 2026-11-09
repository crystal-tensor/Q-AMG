# This file is independently implemented for QuantumBridge SDK.
# No source code from Quafu or pyquafu was copied.
"""Optional dependency checks for Quafu / pyquafu compatibility."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version


def dependency_available(package_name: str = "quafu") -> bool:
    try:
        import_module(package_name)
        return True
    except Exception:
        return False


def get_upstream_version(package_name: str = "quafu") -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def validate_quafu_dependencies(package_name: str = "quafu") -> dict[str, object]:
    return {
        "package": package_name,
        "available": dependency_available(package_name),
        "version": get_upstream_version(package_name),
        "required_by_default": False,
        "numpy_lane": "numpy<2 for optional pyquafu lane",
        "cloud_access": False,
        "token_read": False,
        "hardware_access": False,
    }


def validate_quarkstudio_dependencies() -> dict[str, object]:
    """Inspect the optional QuarkStudio client without reading credentials."""

    task_available = False
    try:
        module = import_module("quark")
        task_available = callable(getattr(module, "Task", None))
    except Exception:
        pass
    return {
        "package": "quarkstudio",
        "import_name": "quark",
        "available": task_available,
        "version": get_upstream_version("quarkstudio"),
        "circuit_package": "quarkcircuit",
        "circuit_version": get_upstream_version("quarkcircuit"),
        "required_by_default": False,
        "cloud_access": False,
        "token_read": False,
        "hardware_access": False,
    }
