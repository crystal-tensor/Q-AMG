# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
# Design source: docs/roadmap/p2_upstream_integration_strategy_v0.1.md.

from __future__ import annotations

from quantumbridge.chemistry.fermion import FermionicOp


class OpenFermionAdapter:
    mode = "Adapter Integration"

    def fermion_operator_to_quantumbridge(self, operator) -> FermionicOp:
        try:
            import openfermion  # noqa: F401
        except Exception as exc:
            raise ImportError("QuantumBridge OpenFermion adapter requires optional dependency 'openfermion'.") from exc
        terms = {}
        for term, coeff in getattr(operator, "terms", {}).items():
            label_parts = []
            for index, action in term:
                label_parts.append(f"{'+' if action else '-'}_{index}")
            terms[" ".join(label_parts)] = complex(coeff)
        return FermionicOp(terms)
