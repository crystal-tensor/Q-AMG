# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is part of QuantumBridge SDK.
# This implementation is developed for the QuantumBridge native architecture.

import numpy as np


def _kraus_arrays(values):
    from quantumbridge.execution.resource_context import current_resources
    if current_resources() is not None:
        from quantumbridge.execution.array_reservations import numeric_array_copies
        return list(numeric_array_copies(values, category="noise_kraus_matrix"))
    return [np.asarray(value, dtype=complex) for value in values]


def _validate_probability(p: float) -> float:
    p = float(p)
    if not 0.0 <= p <= 1.0:
        raise ValueError("QuantumBridge noise probabilities must be between 0 and 1.")
    return p


def _is_trace_preserving(kraus_ops, atol: float = 1e-10) -> bool:
    from quantumbridge.execution.resource_context import current_resources
    if current_resources() is not None:
        from quantumbridge.utils.math import _validate_isometry_sum
        shape = kraus_ops[0].shape
        if len(shape)!=2 or any(op.shape!=shape for op in kraus_ops):return False
        return _validate_isometry_sum(kraus_ops, shape[1], atol=atol, rtol=1e-5)
    total = sum(op.conj().T @ op for op in kraus_ops)
    return bool(np.allclose(total, np.eye(total.shape[0]), atol=atol))


class BitFlipChannel:
    name = "bit_flip"

    def __init__(self, probability: float):
        self.probability = _validate_probability(probability)

    def apply_probabilities(self, probabilities):
        p = self.probability
        return {"0": (1 - p) * probabilities.get("0", 0) + p * probabilities.get("1", 0), "1": p * probabilities.get("0", 0) + (1 - p) * probabilities.get("1", 0)}

    def kraus(self):
        p = self.probability
        a,b=np.sqrt(1-p),np.sqrt(p)
        return _kraus_arrays([[[a,0],[0,a]],[[0,b],[b,0]]])

    def is_trace_preserving(self, atol: float = 1e-10) -> bool:
        return _is_trace_preserving(self.kraus(), atol)


class PhaseFlipChannel:
    name = "phase_flip"

    def __init__(self, probability: float):
        self.probability = _validate_probability(probability)

    def kraus(self):
        p = self.probability
        a,b=np.sqrt(1-p),np.sqrt(p)
        return _kraus_arrays([[[a,0],[0,a]],[[b,0],[0,-b]]])

    def is_trace_preserving(self, atol: float = 1e-10) -> bool:
        return _is_trace_preserving(self.kraus(), atol)


class DepolarizingChannel:
    name = "depolarizing"

    def __init__(self, probability: float):
        self.probability = _validate_probability(probability)

    def kraus(self):
        p = self.probability
        a,b=np.sqrt(1-p),np.sqrt(p/3)
        return _kraus_arrays([[[a,0],[0,a]],[[0,b],[b,0]],[[0,-1j*b],[1j*b,0]],[[b,0],[0,-b]]])

    def is_trace_preserving(self, atol: float = 1e-10) -> bool:
        return _is_trace_preserving(self.kraus(), atol)


class QuantumChannel:
    name = "quantum_channel"

    def kraus(self):
        raise NotImplementedError("QuantumBridge quantum channels must define Kraus matrices.")

    def is_trace_preserving(self, atol: float = 1e-10) -> bool:
        return _is_trace_preserving(self.kraus(), atol)


class KrausChannel(QuantumChannel):
    name = "kraus"

    def __init__(self, kraus_ops, name: str = "kraus"):
        from quantumbridge.execution.resource_context import current_resources
        self._kraus = _kraus_arrays(kraus_ops) if current_resources() is not None else [np.asarray(op, dtype=complex) for op in kraus_ops]
        if not self._kraus:
            raise ValueError("QuantumBridge KrausChannel requires at least one operator.")
        self.name = name

    def kraus(self):
        from quantumbridge.execution.resource_context import current_resources
        if current_resources() is not None:return _kraus_arrays(self._kraus)
        return [op.copy() for op in self._kraus]


class AmplitudeDampingChannel:
    name = "amplitude_damping"

    def __init__(self, probability: float):
        self.probability = _validate_probability(probability)

    def kraus(self):
        p = self.probability
        return _kraus_arrays([[[1,0],[0,np.sqrt(1-p)]],[[0,np.sqrt(p)],[0,0]]])

    def is_trace_preserving(self, atol: float = 1e-10) -> bool:
        return _is_trace_preserving(self.kraus(), atol)


class PhaseDampingChannel:
    name = "phase_damping"

    def __init__(self, probability: float):
        self.probability = _validate_probability(probability)

    def kraus(self):
        p = self.probability
        return _kraus_arrays([[[1,0],[0,np.sqrt(1-p)]],[[0,0],[0,np.sqrt(p)]]])

    def is_trace_preserving(self, atol: float = 1e-10) -> bool:
        return _is_trace_preserving(self.kraus(), atol)
