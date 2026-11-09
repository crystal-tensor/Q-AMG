# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
# Design source: docs/roadmap/p2_execution_plan_v0.1.md.


class QASMVisitor:
    """Small visitor base for QuantumBridge-owned QASM ASTs."""

    def visit(self, node):
        method = getattr(self, f"visit_{type(node).__name__}", None)
        if method is None:
            return self.generic_visit(node)
        return method(node)

    def generic_visit(self, node):
        return node
