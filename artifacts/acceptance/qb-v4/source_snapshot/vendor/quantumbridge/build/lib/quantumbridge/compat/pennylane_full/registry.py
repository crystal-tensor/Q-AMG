# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
"""PennyLane full public API registry."""

from __future__ import annotations

from .inventory import PennyLaneInventoryRecord, list_public_api_inventory


def list_public_api_registry() -> list[PennyLaneInventoryRecord]:
    return list_public_api_inventory()


def get_registry_entry(module: str, name: str) -> PennyLaneInventoryRecord | None:
    for record in list_public_api_registry():
        if record.import_module == module and record.name == name:
            return record
    return None


def registry_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in list_public_api_registry():
        counts[record.module] = counts.get(record.module, 0) + 1
    return counts
