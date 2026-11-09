# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit, IBM, Aer, Nature, Finance, ML, Optimization, Experiments, Metal, or Addons was copied.
"""Shared Qiskit ecosystem adapter contract helpers."""

from __future__ import annotations

import inspect
from typing import Any, Iterable

from quantumbridge.compat.contracts import AdapterProvenance, AdapterWarning, CapabilityLevel, UnsupportedCapability

BASE_QISKIT_WARNING = (
    "This Qiskit ecosystem adapter is an optional passthrough/schema bridge, "
    "not a full native Qiskit replacement."
)
RUNTIME_OFFLINE_WARNING = (
    "Qiskit Runtime support is offline-only in QuantumBridge. "
    "It does not contact IBM Cloud or read credentials."
)
METAL_ADVISORY_WARNING = (
    "Qiskit Metal support is advisory only. It does not imply chip fabrication, "
    "EM solver validation, or production design readiness."
)


class QiskitAdapterFacade:
    """Small facade that aligns existing Qiskit adapters to the Stage 8 contract."""

    capability_level = CapabilityLevel.SCHEMA_ADAPTER
    production_ready = False
    native_implementation = False
    upstream_required = True

    def __init__(
        self,
        *,
        ecosystem: str,
        upstream_package: str,
        dependency_extra: str,
        adapters: Iterable[Any],
        advisory: bool = False,
        offline_only: bool = False,
        notes: str = "Optional Qiskit ecosystem adapter.",
    ) -> None:
        self.ecosystem = ecosystem
        self.upstream_package = upstream_package
        self.dependency_extra = dependency_extra
        self.adapters = tuple(adapters)
        self.advisory = advisory
        self.offline_only = offline_only
        self.notes = notes

    def dependency_available(self) -> bool:
        return any(adapter.dependency_available() for adapter in self.adapters)

    def get_upstream_version(self) -> str | None:
        for adapter in self.adapters:
            version = adapter.get_upstream_version()
            if version:
                return version
        return None

    def get_dependency_report(self) -> dict[str, Any]:
        return {
            "ecosystem": self.ecosystem,
            "upstream_package": self.upstream_package,
            "dependency_extra": self.dependency_extra,
            "available": self.dependency_available(),
            "upstream_version": self.get_upstream_version(),
            "advisory": self.advisory,
            "offline_only": self.offline_only,
            "cloud_access": False,
            "token_read": False,
            "token_storage": False,
            "warnings": [warning.message for warning in self.get_warnings()],
        }

    def list_public_api_inventory(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for adapter in self.adapters:
            for record in adapter.list_public_api_inventory():
                rows.append(normalize_qiskit_inventory_record(self, adapter, record))
        return rows

    def get_public_object(self, name: str) -> Any:
        last_error: Exception | None = None
        for adapter in self.adapters:
            try:
                return adapter.get_public_object(name)
            except Exception as exc:  # pragma: no cover - depends on optional package shape
                last_error = exc
        raise ImportError(f"{self.ecosystem} public object {name!r} is unavailable: {last_error}")

    def passthrough_call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        obj = self.get_public_object(name)
        if not callable(obj):
            raise TypeError(f"{name!r} is not callable.")
        return obj(*args, **kwargs)

    def passthrough_class(self, name: str) -> Any:
        obj = self.get_public_object(name)
        if not inspect.isclass(obj):
            raise TypeError(f"{name!r} is not a class.")
        return obj

    def wrap_result(self, obj: Any):
        from quantumbridge.schema.qiskit_results import result_class_for_ecosystem

        result_class = result_class_for_ecosystem(self.ecosystem)
        return result_class(
            upstream_package=self.upstream_package,
            upstream_version=self.get_upstream_version(),
            capability_level=int(self.capability_level),
            mode="schema-adapter",
            raw_type=type(obj).__name__,
            data=obj,
            metadata={"adapter": self.ecosystem, "notes": self.notes},
            warnings=[warning.message for warning in self.get_warnings()],
            provenance=self.get_provenance().to_dict(),
            advisory=self.advisory,
            offline_only=self.offline_only,
        )

    def to_quantumbridge_schema(self, obj: Any) -> dict[str, Any]:
        return self.wrap_result(obj).to_dict()

    def get_warnings(self) -> list[AdapterWarning]:
        warnings = [AdapterWarning("qiskit_optional_passthrough", BASE_QISKIT_WARNING)]
        if self.offline_only:
            warnings.append(AdapterWarning("qiskit_runtime_offline_only", RUNTIME_OFFLINE_WARNING))
        if self.ecosystem == "qiskit_metal":
            warnings.append(AdapterWarning("qiskit_metal_advisory", METAL_ADVISORY_WARNING))
        if self.advisory and self.ecosystem not in {"qiskit_metal", "qiskit_runtime"}:
            warnings.append(
                AdapterWarning(
                    "qiskit_advisory_adapter",
                    "This Qiskit ecosystem adapter is advisory and does not claim production readiness.",
                )
            )
        return warnings

    def get_provenance(self) -> AdapterProvenance:
        return AdapterProvenance(
            ecosystem=self.ecosystem,
            upstream_package=self.upstream_package,
            upstream_version=self.get_upstream_version(),
            adapter=f"quantumbridge.compat.{self.ecosystem}",
            capability_level=self.capability_level,
            mode="qiskit-ecosystem-adapter",
            dependency_extra=self.dependency_extra,
            official_endorsement=False,
            cloud_access=False,
            token_storage=False,
            source_code_copied=False,
        )

    def unsupported(self, reason: str) -> UnsupportedCapability:
        return UnsupportedCapability(
            reason=reason,
            capability_level=CapabilityLevel.INVENTORY,
            warnings=tuple(self.get_warnings()),
            provenance=self.get_provenance(),
        )

    def validate_environment(self) -> bool:
        report = self.get_dependency_report()
        if report["cloud_access"] or report["token_read"] or report["token_storage"]:
            raise RuntimeError("Qiskit adapter attempted cloud or credential access.")
        return True


def normalize_qiskit_inventory_record(facade: QiskitAdapterFacade, adapter: Any, record: Any) -> dict[str, Any]:
    payload = record.to_dict() if hasattr(record, "to_dict") else dict(record)
    api_name = payload.get("public_api") or payload.get("name") or "unknown"
    unavailable = api_name in {"dependency-not-installed", "import-failed", "module-imported"}
    return {
        "ecosystem": facade.ecosystem,
        "module": payload.get("module"),
        "name": api_name,
        "object_type": payload.get("api_type", "unknown"),
        "importable": not unavailable,
        "callable": payload.get("api_type") in {"class", "function"},
        "quantumbridge_level": int(payload.get("quantumbridge_level", 0)),
        "adapter": getattr(adapter, "package_key", facade.ecosystem),
        "upstream_required": True,
        "supported": not unavailable,
        "unsupported_reason": payload.get("notes") if unavailable else None,
        "advisory": facade.advisory,
        "risk": payload.get("risk", "MEDIUM"),
        "notes": payload.get("notes", "Runtime introspection only; no upstream source copied."),
    }


def summarize_inventory(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    level_counts = {str(level): 0 for level in range(4)}
    unsupported = 0
    for row in rows:
        level = str(row.get("quantumbridge_level", 0))
        if level in level_counts:
            level_counts[level] += 1
        if not row.get("supported", False):
            unsupported += 1
    return {
        "records": len(rows),
        "level_counts": level_counts,
        "unsupported": unsupported,
    }
