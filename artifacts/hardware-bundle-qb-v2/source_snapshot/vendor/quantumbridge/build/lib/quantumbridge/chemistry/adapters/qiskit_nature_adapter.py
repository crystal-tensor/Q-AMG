# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
# Design source: docs/roadmap/p2_upstream_integration_strategy_v0.1.md.

from __future__ import annotations

from quantumbridge.chemistry.result import DriverResult


class QiskitNatureDriverAdapter:
    mode = "Adapter Integration"

    def __init__(self, driver=None):
        self.driver = driver

    def run(self) -> DriverResult:
        try:
            import qiskit_nature  # noqa: F401
        except Exception as exc:
            raise ImportError("QuantumBridge Qiskit Nature adapter requires optional dependency 'qiskit-nature'.") from exc
        if self.driver is not None and hasattr(self.driver, "run"):
            upstream_result = self.driver.run()
        else:
            upstream_result = None
        return DriverResult(
            metadata={"upstream_result_type": type(upstream_result).__name__ if upstream_result is not None else None},
            provenance={"mode": "Adapter Integration", "dependency": "qiskit-nature"},
        )


def hamiltonian_from_qiskit_nature(operator):
    from quantumbridge.chemistry.qubit_hamiltonian import QubitHamiltonian
    from quantumbridge.utils.math import PauliString

    if hasattr(operator, "to_list"):
        terms = []
        for label, coeff in operator.to_list():
            paulis = [(idx, char) for idx, char in enumerate(label) if char != "I"]
            terms.append((float(complex(coeff).real), PauliString(paulis)))
        return QubitHamiltonian(terms)
    raise ValueError("QuantumBridge Qiskit Nature Hamiltonian adapter requires an operator with to_list().")
