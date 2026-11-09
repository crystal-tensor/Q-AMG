# This file is independently implemented for QuantumBridge SDK.
# No source code from MQT, IBM, or Qiskit was copied.
"""Decision-diagram-inspired metadata for educational DDSIM-like workflows."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from quantumbridge.schema.mqt_results import DecisionDiagramMetadataResult

from .warnings import DDSIM_LIKE_WARNING, mqt_warnings, native_provenance


def build_decision_diagram_metadata(
    statevector: Iterable[complex],
    num_qubits: int | None = None,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    state = np.asarray([complex(value) for value in statevector], dtype=complex)
    if state.ndim != 1 or state.size == 0:
        raise ValueError("statevector must be a non-empty one-dimensional vector")
    if num_qubits is None:
        num_qubits = int(round(np.log2(state.size)))
    if 1 << int(num_qubits) != state.size:
        raise ValueError("statevector length must be 2**num_qubits")
    tolerance = float(tolerance)
    nonzero_indices = [index for index, value in enumerate(state) if abs(value) > tolerance]
    rounded = {
        (round(float(value.real), 12), round(float(value.imag), 12))
        for value in state
        if abs(value) > tolerance
    }
    norm = float(np.sum(np.abs(state) ** 2))
    support = [format(index, f"0{num_qubits}b") for index in nonzero_indices]
    ratio = len(nonzero_indices) / float(state.size)
    if len(rounded) <= max(2, state.size // 4):
        hint = "repeated_amplitudes"
    elif ratio <= 0.5:
        hint = "sparse_support"
    else:
        hint = "low_compression_expected"
    return {
        "num_qubits": int(num_qubits),
        "statevector_length": int(state.size),
        "nonzero_amplitudes": len(nonzero_indices),
        "unique_amplitude_count": len(rounded),
        "support_bitstrings": support,
        "compression_hint": hint,
        "normalized": bool(abs(norm - 1.0) <= max(tolerance * 10, 1e-10)),
        "norm": norm,
        "tolerance": tolerance,
        "decision_diagram_parity_claim": False,
    }


def summarize_decision_diagram_metadata(metadata: dict[str, Any]) -> str:
    return (
        f"{metadata.get('num_qubits')} qubits, "
        f"{metadata.get('nonzero_amplitudes')} nonzero amplitudes, "
        f"{metadata.get('unique_amplitude_count')} unique amplitudes, "
        f"hint={metadata.get('compression_hint')}"
    )


def wrap_decision_diagram_metadata(metadata: dict[str, Any]) -> DecisionDiagramMetadataResult:
    return DecisionDiagramMetadataResult(
        workflow="decision_diagram_metadata_native",
        project="mqt-ddsim",
        mode="native_minimal",
        capability_level=3,
        production_ready=False,
        native_implementation=True,
        num_qubits=int(metadata["num_qubits"]),
        decision_diagram_metadata=dict(metadata),
        raw_type="DecisionDiagramInspiredMetadata",
        metadata={
            "cloud_access": False,
            "token_read": False,
            "hardware_access": False,
            "decision_diagram_parity_claim": False,
        },
        warnings=mqt_warnings(DDSIM_LIKE_WARNING),
        provenance=native_provenance("decision_diagram_metadata_native", "mqt-ddsim"),
    )
