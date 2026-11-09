# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
"""Provenance metadata for upstream passthrough and adapter results.

Design source: docs/compat/full_ecosystem_coverage_policy_v0.1.md.
"""

from __future__ import annotations

from datetime import datetime, timezone


def provenance_metadata(
    package: str,
    mode: str,
    level: int,
    dependency_extra: str,
    upstream_version: str | None = None,
) -> dict:
    return {
        "project": "QuantumBridge",
        "adapter_package": package,
        "integration_mode": mode,
        "coverage_level": int(level),
        "dependency_extra": dependency_extra,
        "upstream_version": upstream_version,
        "source_policy": "optional_dependency_no_vendored_source",
        "official_endorsement": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
