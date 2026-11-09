# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
"""Public circuit-contract failures, distinct from execution resource limits."""


class QiskitError(ValueError):
    """Native public SDK-contract error; no upstream exception dependency."""


class CircuitError(QiskitError):
    """A circuit object's public arguments do not satisfy its contract."""
