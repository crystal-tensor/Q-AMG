# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
"""Optional PennyLane dependency checks for Stage 8B adapters."""

from __future__ import annotations

from importlib import import_module, metadata, util
from typing import Any


PACKAGE_NAME = "pennylane"
DEPENDENCY_EXTRA = "pennylane-full"


def dependency_available() -> bool:
    """Return whether the optional PennyLane package can be imported."""

    return util.find_spec(PACKAGE_NAME) is not None


def get_upstream_version() -> str | None:
    """Return the installed PennyLane distribution version, if available."""

    try:
        return metadata.version(PACKAGE_NAME)
    except metadata.PackageNotFoundError:
        return None


def require_pennylane() -> Any:
    """Import PennyLane or raise a clear optional-dependency error."""

    if not dependency_available():
        raise ImportError(
            "QuantumBridge PennyLane full adapter requires the optional "
            "'pennylane-full' extra. Install it explicitly to use upstream "
            "PennyLane passthrough paths."
        )
    return import_module(PACKAGE_NAME)


def get_dependency_report() -> dict[str, Any]:
    """Return a stable dependency report without installing or contacting networks."""

    return {
        "package": PACKAGE_NAME,
        "dependency_extra": DEPENDENCY_EXTRA,
        "available": dependency_available(),
        "upstream_version": get_upstream_version(),
        "optional_dependency": True,
        "auto_install": False,
        "network_access": False,
        "token_access": False,
    }
