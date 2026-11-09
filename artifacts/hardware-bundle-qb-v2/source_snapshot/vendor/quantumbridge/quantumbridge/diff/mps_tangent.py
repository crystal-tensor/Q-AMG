# Copyright 2026 QuantumBridge Contributors. Apache-2.0.
"""Local two-direction product rule on the native scaled MPS components.

This is a numerical propagation kernel, not yet a public QNode method. It
accepts already differentiated, bounded local matrices from native gate jets.
"""

from dataclasses import dataclass

from quantumbridge.core.gate_limits import current_gate_limits
from quantumbridge.execution import mps_arrays as arrays
from quantumbridge.information.mps_tangent import MPSTangent


@dataclass(frozen=True)
class MPSTangentJet:
    primal: MPSTangent
    left: MPSTangent
    right: MPSTangent
    mixed: MPSTangent

    def __post_init__(self):
        parts = (self.primal, self.left, self.right, self.mixed)
        if any(type(part) is not MPSTangent for part in parts):
            raise TypeError("MPS jets require four native tangent vectors.")
        if any(part.num_qubits != self.primal.num_qubits for part in parts):
            raise ValueError("MPS jet component widths differ.")

    @classmethod
    def from_state(cls, state, *, max_components=256):
        primal = MPSTangent(state, max_components=max_components)
        zero = MPSTangent.zero(primal.num_qubits, max_components=max_components)
        return cls(primal, zero, zero, zero)

    def applied(self, matrix, wires, *, left=None, right=None, mixed=None):
        """Propagate U_ab psi + U_a psi_b + U_b psi_a + U psi_ab."""
        parts = (self.primal, self.left, self.right, self.mixed)
        maximum = min(part._max_components for part in parts)
        p, a, b, ab = (part.num_components for part in parts)
        planned = (p, a + p * (left is not None), b + p * (right is not None),
                   ab + b * (left is not None) + a * (right is not None) + p * (mixed is not None))
        if max(planned) > maximum:
            raise MemoryError("MPS jet product rule exceeds max_components before expansion.")
        current_gate_limits().check("max_decomposition_work", sum(planned) * self.primal.num_qubits)

        def updated(base, contributions):
            terms = [(1.0, base.applied(matrix, wires))]
            for derivative, tangent in contributions:
                if derivative is not None:
                    terms.append((1.0, tangent.applied(derivative, wires)))
            return MPSTangent.linear_combination(terms, max_components=maximum)

        primal = self.primal.applied(matrix, wires)
        first_left = updated(self.left, ((left, self.primal),))
        first_right = updated(self.right, ((right, self.primal),))
        second = updated(self.mixed, ((left, self.right), (right, self.left), (mixed, self.primal)))
        return type(self)(primal, first_left, first_right, second)

    def moment(self, pauli=None):
        """Fixed Hermitian Pauli moment: value, first pair, mixed second."""
        result = arrays.owned("mps_jet_moment", (4,), float)
        result[0] = self.primal.inner(self.primal, pauli).real
        result[1] = 2 * self.left.inner(self.primal, pauli).real
        result[2] = 2 * self.right.inner(self.primal, pauli).real
        result[3] = 2 * (self.mixed.inner(self.primal, pauli)
                         + self.left.inner(self.right, pauli)).real
        return result
