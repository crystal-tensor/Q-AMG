# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
"""Runtime registry for optional ecosystem adapter scaffolds.

Design source: docs/compat/full_ecosystem_coverage_policy_v0.1.md.
"""

from __future__ import annotations

import inspect
import json
import warnings
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Iterable

from quantumbridge.ecosystem.capability import CapabilityRecord, CoverageLevel
from quantumbridge.ecosystem.dependency import dependency_available, get_upstream_version, import_optional
from quantumbridge.ecosystem.provenance import provenance_metadata


MATRIX_HEADER = (
    "| Upstream Package | Module | Public API | Type | QuantumBridge Level | Mode | "
    "Dependency Extra | Test | Risk | Notes |\n"
    "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
)


@dataclass(frozen=True)
class EcosystemAdapter:
    package_name: str
    distribution_name: str
    root_modules: tuple[str, ...]
    dependency_extra: str
    package_key: str
    mode: str = "Inventory/Passthrough scaffold"
    default_level: int = int(CoverageLevel.INVENTORY)
    max_inventory_items_per_module: int = 80

    def dependency_available(self) -> bool:
        return any(dependency_available(module) for module in self.root_modules)

    def get_upstream_version(self) -> str | None:
        for module in self.root_modules:
            version = get_upstream_version(module, self.distribution_name)
            if version:
                return version
        return None

    def provenance_metadata(self) -> dict:
        return provenance_metadata(
            package=self.package_name,
            mode=self.mode,
            level=self.default_level,
            dependency_extra=self.dependency_extra,
            upstream_version=self.get_upstream_version(),
        )

    def list_public_api_inventory(self) -> list[CapabilityRecord]:
        records: list[CapabilityRecord] = []
        for module_name in self.root_modules:
            if not dependency_available(module_name):
                records.append(
                    CapabilityRecord(
                        upstream_package=self.package_name,
                        module=module_name,
                        public_api="dependency-not-installed",
                        api_type="dependency",
                        quantumbridge_level=int(CoverageLevel.INVENTORY),
                        mode="Inventory only",
                        dependency_extra=self.dependency_extra,
                        test="dependency availability smoke",
                        risk="LOW",
                        notes="Optional dependency is not installed in this environment.",
                    )
                )
                continue
            try:
                module = import_module(module_name)
            except Exception as exc:
                records.append(
                    CapabilityRecord(
                        upstream_package=self.package_name,
                        module=module_name,
                        public_api="import-failed",
                        api_type="dependency",
                        quantumbridge_level=int(CoverageLevel.INVENTORY),
                        mode="Inventory only",
                        dependency_extra=self.dependency_extra,
                        test="module import smoke",
                        risk="HIGH",
                        notes=f"Module discovery succeeded but import failed: {type(exc).__name__}: {exc}",
                    )
                )
                continue
            discovered = 0
            for name in sorted(dir(module)):
                if name.startswith("_"):
                    continue
                try:
                    obj = getattr(module, name)
                except Exception:
                    continue
                api_type = _api_type(obj)
                if api_type is None and not _is_public_constant(obj):
                    continue
                api_type = api_type or "constant"
                records.append(
                    CapabilityRecord(
                        upstream_package=self.package_name,
                        module=module_name,
                        public_api=name,
                        api_type=api_type,
                        quantumbridge_level=self.default_level,
                        mode=self.mode,
                        dependency_extra=self.dependency_extra,
                        test="inventory/introspection smoke",
                        risk="MEDIUM",
                        notes="Runtime public-name inventory; no upstream source or docs copied.",
                    )
                )
                discovered += 1
                if discovered >= self.max_inventory_items_per_module:
                    records.append(
                        CapabilityRecord(
                            upstream_package=self.package_name,
                            module=module_name,
                            public_api="inventory-truncated",
                            api_type="metadata",
                            quantumbridge_level=self.default_level,
                            mode=self.mode,
                            dependency_extra=self.dependency_extra,
                            test="inventory/introspection smoke",
                            risk="MEDIUM",
                            notes=f"Inventory capped at {self.max_inventory_items_per_module} public names for this module.",
                        )
                    )
                    break
            if discovered == 0:
                records.append(
                    CapabilityRecord(
                        upstream_package=self.package_name,
                        module=module_name,
                        public_api="module-imported",
                        api_type="module",
                        quantumbridge_level=self.default_level,
                        mode=self.mode,
                        dependency_extra=self.dependency_extra,
                        test="module import smoke",
                        risk="MEDIUM",
                        notes="Module imported but exposes no inventory-safe public names at its top level.",
                    )
                )
        return records

    def get_public_object(self, name: str):
        return self._resolve_public_object(name)

    def passthrough_class(self, name: str):
        obj = self._resolve_public_object(name)
        if not inspect.isclass(obj):
            raise TypeError(f"Upstream object {name!r} is available but is not a class.")
        return obj

    def passthrough_function(self, name: str):
        obj = self._resolve_public_object(name)
        if not callable(obj) or inspect.isclass(obj):
            raise TypeError(f"Upstream object {name!r} is available but is not a function-like callable.")
        return obj

    def wrap_result(self, obj) -> dict:
        return {
            "schema": "quantumbridge.ecosystem.result.v0.1",
            "upstream_type": type(obj).__name__,
            "value_repr": repr(obj),
            "provenance": self.provenance_metadata(),
        }

    def to_quantumbridge_schema(self, obj) -> dict:
        return {
            "schema": "quantumbridge.ecosystem.object.v0.1",
            "upstream_type": type(obj).__name__,
            "is_passthrough": True,
            "payload": obj if _json_safe(obj) else repr(obj),
            "provenance": self.provenance_metadata(),
        }

    def make_result(self, result_class, obj, capability_level: int = 2, metadata: dict | None = None):
        return result_class(
            ecosystem=getattr(result_class, "ECOSYSTEM", None) or self.package_key,
            upstream_package=self.package_name,
            upstream_version=self.get_upstream_version(),
            capability_level=capability_level,
            mode="Adapter",
            raw_type=type(obj).__name__,
            data=obj if _json_safe(obj) else repr(obj),
            metadata=dict(metadata or {}),
            provenance=self.provenance_metadata(),
        )

    def get_capability_level(self, name: str) -> int:
        for record in self.list_public_api_inventory():
            if record.public_api == name or f"{record.module}.{record.public_api}" == name:
                return int(record.quantumbridge_level)
        return int(CoverageLevel.INVENTORY)

    def get_provenance(self) -> dict:
        return self.provenance_metadata()

    def unsupported(self, name: str, reason: str) -> dict:
        self.warn_unsupported(name)
        return {
            "name": name,
            "supported": False,
            "reason": reason,
            "capability_level": int(CoverageLevel.INVENTORY),
            "provenance": self.provenance_metadata(),
        }

    def warn_unsupported(self, feature: str) -> None:
        warnings.warn(
            f"QuantumBridge {self.package_key} adapter does not implement {feature!r} natively; "
            "use the upstream optional dependency directly or wait for a reviewed adapter.",
            UserWarning,
            stacklevel=2,
        )

    def _resolve_public_object(self, name: str):
        if "." in name:
            module_name, attr = name.rsplit(".", 1)
            module = import_optional(module_name, self.dependency_extra)
            return getattr(module, attr)
        for module_name in self.root_modules:
            if not dependency_available(module_name):
                continue
            module = import_module(module_name)
            if hasattr(module, name):
                return getattr(module, name)
        raise ImportError(
            f"QuantumBridge optional adapter {self.package_key!r} cannot find upstream object {name!r}. "
            f"Install extra {self.dependency_extra!r} and confirm the upstream package version."
        )


def _api_type(obj) -> str | None:
    if inspect.isclass(obj):
        return "class"
    if inspect.isfunction(obj) or inspect.isbuiltin(obj):
        return "function"
    return None


def _is_public_constant(obj) -> bool:
    return isinstance(obj, (str, int, float, bool, complex, tuple, frozenset, type(None)))


def _json_safe(obj) -> bool:
    try:
        json.dumps(obj)
    except TypeError:
        return False
    return True


def write_inventory(
    adapter: EcosystemAdapter,
    inventory_dir: str | Path = "docs/compat/inventory",
    matrix_dir: str | Path = "docs/compat/matrix",
) -> tuple[Path, Path]:
    records = adapter.list_public_api_inventory()
    inventory_path = Path(inventory_dir) / f"{adapter.package_key}_inventory.json"
    matrix_path = Path(matrix_dir) / f"{adapter.package_key}_coverage_matrix.md"
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    installed = adapter.dependency_available()
    version = adapter.get_upstream_version()
    inventory_path.write_text(
        json.dumps(
            [
                {
                    **record.to_dict(),
                    "installed": installed,
                    "inventory_generated": True,
                    "upstream_version": version,
                    "unsupported_reason": (
                        record.notes if record.public_api in {"dependency-not-installed", "import-failed"} else None
                    ),
                }
                for record in records
            ],
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    matrix_path.write_text(_matrix_markdown(adapter, records), encoding="utf-8")
    return inventory_path, matrix_path


def _matrix_markdown(adapter: EcosystemAdapter, records: Iterable[CapabilityRecord]) -> str:
    records = list(records)
    installed = any(record.public_api != "dependency-not-installed" for record in records)
    public_count = sum(1 for record in records if record.public_api != "dependency-not-installed")
    lines = [
        f"# {adapter.package_name} Coverage Matrix\n\n",
        "Generated from runtime public-name introspection. No upstream source, tests, comments, or documentation text is copied.\n\n",
        f"- Installed: {installed}\n",
        f"- Upstream version: {adapter.get_upstream_version() or 'not installed'}\n",
        "- Inventory generated: true\n",
        f"- Public API count: {public_count}\n",
        f"- QuantumBridge coverage level: {adapter.default_level}\n\n",
        MATRIX_HEADER,
    ]
    for record in records:
        lines.append(
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    record.upstream_package,
                    record.module,
                    record.public_api,
                    record.api_type,
                    record.quantumbridge_level,
                    record.mode,
                    record.dependency_extra,
                    record.test,
                    record.risk,
                    record.notes,
                )
            )
            + " |\n"
        )
    return "".join(lines)


def _cell(value) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")
