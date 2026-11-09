# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
# Design source: docs/roadmap/p2_upstream_integration_strategy_v0.1.md.

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from math import pi
from numbers import Real
from typing import Any, Optional

import numpy as np

from quantumbridge.core import Circuit
from quantumbridge.devices import StatevectorDevice
from quantumbridge.results import Result


@dataclass(frozen=True)
class AmplitudeEstimationResult:
    """Native amplitude-estimation result for small statevector workflows."""

    amplitude: float
    probability: float
    good_probability: float
    good_states: tuple[str, ...]
    num_qubits: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_result(self) -> Result:
        return Result(
            probabilities_data={"good": self.good_probability, "bad": 1.0 - self.good_probability},
            expectation_data=self.amplitude,
            metadata_data={
                "algorithm": "AmplitudeEstimation",
                "amplitude": self.amplitude,
                "probability": self.probability,
                "good_probability": self.good_probability,
                "good_states": list(self.good_states),
                "num_qubits": self.num_qubits,
                **dict(self.metadata),
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return self.to_result().to_dict()


@dataclass(frozen=True)
class PhaseEstimationResult:
    """Native phase-estimation result from exact eigenvector overlap."""

    phase: float
    eigenvalue: complex
    overlap: float
    precision_bits: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_result(self) -> Result:
        return Result(
            expectation_data=self.phase,
            metadata_data={
                "algorithm": "PhaseEstimation",
                "phase": self.phase,
                "eigenvalue": {"real": float(np.real(self.eigenvalue)), "imag": float(np.imag(self.eigenvalue))},
                "overlap": self.overlap,
                "precision_bits": self.precision_bits,
                **dict(self.metadata),
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return self.to_result().to_dict()


def estimate_amplitude(
    circuit: Circuit,
    good_states: Optional[Iterable[str] | Callable[[str], bool]] = None,
    wires: Optional[Sequence[int]] = None,
    device: Optional[StatevectorDevice] = None,
    parameters: Optional[Mapping[Any, Real]] = None,
) -> AmplitudeEstimationResult:
    """Estimate the amplitude for marked computational-basis states.

    This is an exact local statevector estimator. It provides the native
    primitive needed by amplitude-estimation-style workflows without requiring
    upstream Qiskit Algorithms.
    """

    device = device or StatevectorDevice()
    probabilities = device.probabilities(circuit, wires=wires, parameters=parameters)
    predicate, labels = _good_state_predicate(good_states, probabilities)
    good_probability = float(sum(prob for label, prob in probabilities.items() if predicate(label)))
    good_probability = min(max(good_probability, 0.0), 1.0)
    return AmplitudeEstimationResult(
        amplitude=float(np.sqrt(good_probability)),
        probability=good_probability,
        good_probability=good_probability,
        good_states=tuple(labels),
        num_qubits=circuit.num_qubits if wires is None else len(tuple(wires)),
        metadata={"method": "exact_statevector_probability", "wires": None if wires is None else list(wires)},
    )


def estimate_phase(
    unitary,
    state: Optional[Sequence[complex]] = None,
    precision_bits: int = 8,
) -> PhaseEstimationResult:
    """Estimate the eigenphase with largest overlap against ``state``.

    ``phase`` is returned in cycles in ``[0, 1)`` so that eigenvalue
    ``exp(2j*pi*phase)`` matches common phase-estimation conventions.
    """

    matrix = np.asarray(unitary, dtype=complex)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("QuantumBridge phase estimation requires a square unitary matrix.")
    if not np.allclose(matrix.conj().T @ matrix, np.eye(matrix.shape[0]), atol=1e-8):
        raise ValueError("QuantumBridge phase estimation requires a unitary matrix.")
    values, vectors = np.linalg.eig(matrix)
    if state is None:
        index = 0
        overlap = 1.0
    else:
        ket = np.asarray(state, dtype=complex).reshape(-1)
        if ket.shape[0] != matrix.shape[0]:
            raise ValueError("QuantumBridge phase estimation state dimension must match unitary dimension.")
        norm = np.linalg.norm(ket)
        if norm == 0:
            raise ValueError("QuantumBridge phase estimation state cannot be the zero vector.")
        ket = ket / norm
        overlaps = np.abs(vectors.conj().T @ ket) ** 2
        index = int(np.argmax(overlaps))
        overlap = float(overlaps[index])
    raw_phase = float((np.angle(values[index]) / (2 * pi)) % 1.0)
    scale = 2**int(precision_bits)
    phase = round(raw_phase * scale) / scale
    return PhaseEstimationResult(
        phase=float(phase % 1.0),
        eigenvalue=complex(values[index]),
        overlap=overlap,
        precision_bits=int(precision_bits),
        metadata={"method": "exact_eigendecomposition", "dimension": matrix.shape[0]},
    )


def _good_state_predicate(good_states, probabilities: Mapping[str, float]):
    if good_states is None:
        labels = tuple(label for label in probabilities if label.endswith("1"))
        return lambda label: label in labels, labels
    if callable(good_states):
        labels = tuple(label for label in probabilities if bool(good_states(label)))
        return lambda label: label in labels, labels
    labels = tuple(str(label) for label in good_states)
    return lambda label: label in labels, labels
