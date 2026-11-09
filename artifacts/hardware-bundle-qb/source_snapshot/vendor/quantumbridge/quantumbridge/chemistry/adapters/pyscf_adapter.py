# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
# Design source: docs/roadmap/p2_upstream_integration_strategy_v0.1.md.

from __future__ import annotations

from quantumbridge.chemistry.result import DriverResult


class PySCFDriverAdapter:
    mode = "Upstream Passthrough"

    def __init__(self, molecule):
        self.molecule = molecule

    def run(self) -> DriverResult:
        try:
            import pyscf  # noqa: F401
        except Exception as exc:
            raise ImportError("QuantumBridge PySCF adapter requires optional dependency 'pyscf'.") from exc
        return DriverResult(
            metadata={"status": "adapter-available", "molecule": self.molecule.to_xyz() if hasattr(self.molecule, "to_xyz") else self.molecule},
            provenance={"mode": "Upstream Passthrough", "dependency": "pyscf"},
        )
