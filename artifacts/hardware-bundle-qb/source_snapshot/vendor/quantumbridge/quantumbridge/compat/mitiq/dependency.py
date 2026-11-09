# This file is independently implemented for QuantumBridge SDK.
# No source code from Mitiq was copied.
"""Optional Mitiq dependency detection."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .warnings import UPSTREAM_MITIQ_WARNING, mitiq_warnings


def dependency_available() -> bool:
    try:
        import_module("mitiq")
    except Exception:
        return False
    return True


def get_upstream_version() -> str | None:
    try:
        module = import_module("mitiq")
    except Exception:
        return None
    return getattr(module, "__version__", None)


def validate_mitiq_dependencies() -> dict[str, Any]:
    return {
        "upstream_package": "mitiq",
        "available": dependency_available(),
        "upstream_version": get_upstream_version(),
        "cloud_access": False,
        "token_read": False,
        "hardware_access": False,
        "warnings": mitiq_warnings(UPSTREAM_MITIQ_WARNING),
    }
