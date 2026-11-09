# Copyright 2026 QuantumBridge Contributors. Apache-2.0.
"""Explicit measurement-only QASM export, not a Quafu submission or layout."""
from quantumbridge.compiler import native_transpile
from .execution import circuit_artifact


def measurement_artifact(circuit):
    """Lower relative phases first; then record removal of top-level global phase.

    QASM2 cannot represent a global phase. Removing only that phase is valid for
    terminal probabilities, not for coherent controlled reuse of this artifact.
    """
    lowered = native_transpile(circuit, basis_gates=["x", "y", "z", "h", "rx", "ry", "rz", "cx"])
    phase = float(lowered.global_phase)
    lowered.global_phase = 0.
    artifact = circuit_artifact(lowered)
    return dict(artifact, removed_global_phase=phase,
                semantics="Terminal measurement probabilities only; not a phase-preserving controlled subroutine",
                hardware_status="NOT_SUBMITTED; physical layout, backend basis and bit order require acceptance",
                hardware_jobs=0)
