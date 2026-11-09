# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
"""Runtime PennyLane public API inventory for Stage 8B."""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Iterable

from quantumbridge.compat.contracts import CapabilityLevel

from .dependency import dependency_available, get_upstream_version


PENNYLANE_MODULES: tuple[tuple[str, str], ...] = (
    ("qml top-level", "pennylane"),
    ("qml.ops", "pennylane.ops"),
    ("qml.measurements", "pennylane.measurements"),
    ("qml.gradients", "pennylane.gradients"),
    ("qml.transforms", "pennylane.transforms"),
    ("qml.templates", "pennylane.templates"),
    ("qml.qchem", "pennylane.qchem"),
    ("qml.fermi", "pennylane.fermi"),
    ("qml.bose", "pennylane.bose"),
    ("qml.qaoa", "pennylane.qaoa"),
    ("qml.qnn", "pennylane.qnn"),
    ("qml.kernels", "pennylane.kernels"),
    ("qml.devices", "pennylane.devices"),
    ("qml.tape", "pennylane.tape"),
    ("qml.workflow", "pennylane.workflow"),
    ("qml.resource", "pennylane.resource"),
    ("qml.qcut", "pennylane.qcut"),
    ("qml.shadows", "pennylane.shadows"),
    ("qml.data", "pennylane.data"),
    ("qml.math", "pennylane.math"),
    ("qml.numpy", "pennylane.numpy"),
    ("qml.pauli", "pennylane.pauli"),
    ("qml.spin", "pennylane.spin"),
    ("qml.fourier", "pennylane.fourier"),
    ("qml.liealg", "pennylane.liealg"),
    ("qml.pulse", "pennylane.pulse"),
    ("qml.noise", "pennylane.noise"),
    ("qml.io / converters", "pennylane.io"),
    ("interfaces", "pennylane.interfaces"),
    ("plugins / plugin-adjacent discovery", "pennylane.devices"),
)


@dataclass(frozen=True)
class PennyLaneInventoryRecord:
    module: str
    import_module: str
    name: str
    object_type: str
    importable: bool
    callable: bool
    quantumbridge_level: int
    adapter: str
    upstream_required: bool
    supported: bool
    unsupported_reason: str | None
    risk: str
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def list_public_api_inventory(max_items_per_module: int = 120) -> list[PennyLaneInventoryRecord]:
    """Return a runtime public-name inventory without copying upstream source."""

    if not dependency_available():
        return [
            PennyLaneInventoryRecord(
                module=label,
                import_module=module_name,
                name="dependency-not-installed",
                object_type="dependency",
                importable=False,
                callable=False,
                quantumbridge_level=int(CapabilityLevel.INVENTORY),
                adapter=_adapter_for_module(label),
                upstream_required=True,
                supported=False,
                unsupported_reason="Optional PennyLane dependency is not installed.",
                risk="LOW",
                notes="Inventory records unavailable dependency without failing main tests.",
            )
            for label, module_name in PENNYLANE_MODULES
        ]

    records: list[PennyLaneInventoryRecord] = []
    for label, module_name in PENNYLANE_MODULES:
        records.extend(_inventory_module(label, module_name, max_items_per_module))
    return records


def write_full_inventory(
    inventory_path: str | Path = "docs/compat/inventory/pennylane_full_public_api_inventory.json",
    matrix_path: str | Path = "docs/compat/matrix/pennylane_full_feature_coverage_matrix.md",
) -> tuple[Path, Path, list[PennyLaneInventoryRecord]]:
    records = list_public_api_inventory()
    inventory_path = Path(inventory_path)
    matrix_path = Path(matrix_path)
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(
        json.dumps(
            {
                "package": "pennylane",
                "installed": dependency_available(),
                "upstream_version": get_upstream_version(),
                "inventory_generated": True,
                "source_policy": "runtime_public_api_introspection_no_vendored_source",
                "records": [record.to_dict() for record in records],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    matrix_path.write_text(_matrix_markdown(records), encoding="utf-8")
    return inventory_path, matrix_path, records


def inventory_summary(records: Iterable[PennyLaneInventoryRecord]) -> dict[str, int]:
    summary = {"api_count": 0, "level_0": 0, "level_1": 0, "level_2": 0, "level_3": 0, "unsupported": 0}
    for record in records:
        if record.name != "dependency-not-installed":
            summary["api_count"] += 1
        key = f"level_{record.quantumbridge_level}"
        if key in summary:
            summary[key] += 1
        if not record.supported:
            summary["unsupported"] += 1
    return summary


def _inventory_module(label: str, module_name: str, max_items: int) -> list[PennyLaneInventoryRecord]:
    try:
        module = import_module(module_name)
    except Exception as exc:
        return [
            PennyLaneInventoryRecord(
                module=label,
                import_module=module_name,
                name="import-failed",
                object_type="dependency",
                importable=False,
                callable=False,
                quantumbridge_level=int(CapabilityLevel.INVENTORY),
                adapter=_adapter_for_module(label),
                upstream_required=True,
                supported=False,
                unsupported_reason=f"Module import failed: {type(exc).__name__}: {exc}",
                risk="MEDIUM",
                notes="Optional module may be version-specific or unavailable in this environment.",
            )
        ]

    records: list[PennyLaneInventoryRecord] = []
    for name in sorted(dir(module)):
        if name.startswith("_"):
            continue
        try:
            obj = getattr(module, name)
        except Exception:
            continue
        object_type = _object_type(obj)
        if object_type is None:
            continue
        records.append(
            PennyLaneInventoryRecord(
                module=label,
                import_module=module_name,
                name=name,
                object_type=object_type,
                importable=True,
                callable=callable(obj),
                quantumbridge_level=_level_for(label, name),
                adapter=_adapter_for_module(label),
                upstream_required=True,
                supported=True,
                unsupported_reason=None,
                risk=_risk_for(label),
                notes="Runtime public API inventory; no upstream source or docs copied.",
            )
        )
        if len(records) >= max_items:
            records.append(
                PennyLaneInventoryRecord(
                    module=label,
                    import_module=module_name,
                    name="inventory-truncated",
                    object_type="metadata",
                    importable=True,
                    callable=False,
                    quantumbridge_level=int(CapabilityLevel.INVENTORY),
                    adapter=_adapter_for_module(label),
                    upstream_required=True,
                    supported=False,
                    unsupported_reason=f"Inventory capped at {max_items} public names for this module.",
                    risk="MEDIUM",
                    notes="Increase max_items_per_module for deeper local analysis.",
                )
            )
            break
    if not records:
        records.append(
            PennyLaneInventoryRecord(
                module=label,
                import_module=module_name,
                name="module-imported",
                object_type="module",
                importable=True,
                callable=False,
                quantumbridge_level=int(CapabilityLevel.INVENTORY),
                adapter=_adapter_for_module(label),
                upstream_required=True,
                supported=True,
                unsupported_reason=None,
                risk="LOW",
                notes="Module imported but no inventory-safe public names were found.",
            )
        )
    return records


def _object_type(obj: Any) -> str | None:
    if inspect.isclass(obj):
        return "class"
    if inspect.isfunction(obj) or inspect.isbuiltin(obj):
        return "function"
    if inspect.ismodule(obj):
        return "module"
    if isinstance(obj, (str, int, float, bool, complex, tuple, frozenset, type(None))):
        return "constant"
    if callable(obj):
        return "callable"
    return None


def _level_for(label: str, name: str) -> int:
    if label in {"qml top-level", "qml.ops", "qml.measurements", "qml.tape"}:
        if name in {"QNode", "Hadamard", "PauliX", "PauliY", "PauliZ", "RX", "RY", "RZ", "PhaseShift", "CNOT", "CZ", "SWAP"}:
            return int(CapabilityLevel.SCHEMA_ADAPTER)
        return int(CapabilityLevel.PASSTHROUGH)
    return int(CapabilityLevel.INVENTORY)


def _adapter_for_module(label: str) -> str:
    cleaned = (
        label.replace("qml.", "")
        .replace(" / ", "_")
        .replace(" ", "_")
        .replace("-", "_")
        .replace("top_level", "registry")
    )
    return f"quantumbridge.compat.pennylane_full.{cleaned}_adapter"


def _risk_for(label: str) -> str:
    if label in {"qml.qchem", "qml.qnn", "plugins / plugin-adjacent discovery", "qml.pulse", "qml.noise"}:
        return "HIGH"
    if label in {"qml.ops", "qml.measurements", "qml.tape", "qml.devices"}:
        return "MEDIUM"
    return "LOW"


def _matrix_markdown(records: list[PennyLaneInventoryRecord]) -> str:
    summary = inventory_summary(records)
    lines = [
        "# PennyLane Full API Feature Coverage Matrix\n\n",
        "Generated from runtime public API introspection. No upstream source, tests, comments, or documentation text is copied.\n\n",
        f"- Installed: {dependency_available()}\n",
        f"- Upstream version: {get_upstream_version() or 'not installed'}\n",
        f"- Public API count: {summary['api_count']}\n",
        f"- Level 0 count: {summary['level_0']}\n",
        f"- Level 1 count: {summary['level_1']}\n",
        f"- Level 2 count: {summary['level_2']}\n",
        f"- Level 3 count: {summary['level_3']}\n",
        f"- Unsupported count: {summary['unsupported']}\n\n",
        "| Module | API | Type | Importable | Callable | Level | Adapter | Upstream Required | Supported | Unsupported Reason | Risk | Notes |\n",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n",
    ]
    for record in records:
        lines.append(
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    record.module,
                    record.name,
                    record.object_type,
                    record.importable,
                    record.callable,
                    record.quantumbridge_level,
                    record.adapter,
                    record.upstream_required,
                    record.supported,
                    record.unsupported_reason or "",
                    record.risk,
                    record.notes,
                )
            )
            + " |\n"
        )
    return "".join(lines)


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")
