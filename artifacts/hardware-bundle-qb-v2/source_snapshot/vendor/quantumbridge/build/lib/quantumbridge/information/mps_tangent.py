# Copyright 2026 QuantumBridge Contributors. Apache-2.0.
"""Coherent, scaled tangent vectors using the existing native MPS engine.

This internal numerical type is not a normalized quantum state or a mixture.
Each component is a normalized tensor chain times a finite complex scalar.
Sums retain interference through bilinear contractions, never dense vectors.
"""

import math

import numpy as np

from quantumbridge.core.gate_limits import current_gate_limits
from quantumbridge.execution import mps_arrays as arrays
from quantumbridge.utils.math import pauli_matrix
from .matrix_product_state import MatrixProductState


_SCALARS = (int, float, complex, np.int32, np.int64, np.float32, np.float64,
            np.complex64, np.complex128)


def _scalar(value):
    if type(value) not in _SCALARS:
        raise TypeError("MPS tangent scales require native numeric scalars.")
    value = complex(value)
    if not (math.isfinite(value.real) and math.isfinite(value.imag)):
        raise ValueError("MPS tangent scales must be finite.")
    return value


def _product(*values):
    """Accumulate exponents separately so balanced tiny/large factors survive."""
    result, exponent = 1.0 + 0j, 0
    for value in values:
        value = _scalar(value)
        scale = max(abs(value.real), abs(value.imag))
        if scale == 0:
            return 0j
        mantissa, shift = math.frexp(scale)
        result *= complex(value.real / scale, value.imag / scale) * mantissa
        largest = max(abs(result.real), abs(result.imag))
        mantissa, renormalize = math.frexp(largest)
        result = complex(result.real / largest, result.imag / largest) * mantissa
        exponent += shift + renormalize
    try:
        real = math.ldexp(result.real, exponent)
        imag = math.ldexp(result.imag, exponent)
    except OverflowError as error:
        raise FloatingPointError("MPS tangent scale overflows complex128.") from error
    if ((result.real != 0 and real == 0) or (result.imag != 0 and imag == 0)):
        raise FloatingPointError("MPS tangent scale underflows complex128.")
    return complex(real, imag)


def _limit(value):
    if type(value) is not int or not 1 <= value <= 4096:
        raise ValueError("MPS tangent max_components must be an integer in [1, 4096].")
    return value


class MPSTangent:
    """Bounded coherent sum with immutable ownership of its MPS components.

    Applied operators may be nonunitary. User simulation cutoffs are ignored:
    the existing SVD retains numerical rank (8 eps * matrix width threshold).
    A bond limit below that rank raises instead of truncating the derivative.
    Roundoff-level SVD loss is reported, not an exact-arithmetic guarantee.
    """

    def __init__(self, state, scale=1.0, *, max_components=256):
        if type(state) is not MatrixProductState:
            raise TypeError("MPS tangent requires a native MatrixProductState.")
        current_gate_limits().check("max_circuit_bits", state.num_qubits)
        self._max_components = _limit(max_components)
        scale = _scalar(scale)
        if state.discarded_weight:
            raise ValueError("MPS tangent cannot certify a previously truncated source state.")
        self._width = state.num_qubits
        self._components = () if scale == 0 else ((state.copy(), scale),)

    @classmethod
    def _owned(cls, width, components, maximum):
        result = object.__new__(cls)
        result._width, result._components = width, tuple(components)
        result._max_components = maximum
        return result

    @classmethod
    def zero(cls, num_qubits, *, max_components=256):
        if type(num_qubits) is not int or num_qubits < 1:
            raise ValueError("MPS tangent width must be a positive native integer.")
        current_gate_limits().check("max_circuit_bits", num_qubits)
        return cls._owned(num_qubits, (), _limit(max_components))

    @property
    def num_qubits(self):
        return self._width

    @property
    def num_components(self):
        return len(self._components)

    @property
    def component_scales(self):
        return tuple(scale for _, scale in self._components)

    @property
    def component_states(self):
        return tuple(state.copy() for state, _ in self._components)

    @property
    def discarded_weight(self):
        """Sum of relative, roundoff-level per-component SVD losses."""
        return sum(state.discarded_weight for state, _ in self._components)

    @property
    def tensor_bytes(self):
        return sum(state.tensor_bytes for state, _ in self._components)

    def copy(self):
        return self._owned(self._width,
            ((state.copy(), scale) for state, scale in self._components), self._max_components)

    @classmethod
    def linear_combination(cls, terms, *, max_components=256):
        maximum = _limit(max_components)
        if type(terms) not in (tuple, list) or not 1 <= len(terms) <= 4096:
            raise ValueError("MPS tangent sum requires a bounded native sequence of terms.")
        width, count, admitted = None, 0, []
        for term in terms:
            if type(term) not in (tuple, list) or len(term) != 2 or type(term[1]) is not cls:
                raise TypeError("MPS tangent terms must be (native scale, tangent) pairs.")
            coefficient, tangent = _scalar(term[0]), term[1]
            width = tangent.num_qubits if width is None else width
            if tangent.num_qubits != width:
                raise ValueError("MPS tangent sum widths differ.")
            maximum = min(maximum, tangent._max_components)
            count += tangent.num_components if coefficient != 0 else 0
            if count > maximum:
                raise MemoryError("MPS tangent coherent sum exceeds max_components.")
            admitted.append((coefficient, tangent))
        # Components are immutable internally; sharing a chain is safe, and
        # applied() always makes a private copy before changing any gauge.
        components = []
        for coefficient, tangent in admitted:
            if coefficient != 0:
                components.extend((state, _product(scale, coefficient))
                                  for state, scale in tangent._components)
        return cls._owned(width, components, maximum)

    def applied(self, matrix, wires):
        limits = current_gate_limits()
        if type(wires) not in (tuple, list) or not wires:
            raise TypeError("MPS tangent wires require a nonempty bounded native sequence.")
        if len(wires) > limits.max_matrix_dimension.bit_length() - 1:
            raise MemoryError("MPS tangent operator exceeds max_matrix_dimension.")
        if (any(type(wire) is not int or not 0 <= wire < self._width for wire in wires)
                or len(set(wires)) != len(wires)):
            raise ValueError("MPS tangent wires must be distinct valid native integers.")
        size = 1 << len(wires)
        if not issubclass(type(matrix), np.ndarray):
            raise TypeError("MPS tangent operators require an explicit numeric ndarray.")
        matrix = np.ndarray.view(matrix, np.ndarray)
        if matrix.shape != (size, size) or matrix.dtype.kind not in "biufc":
            raise ValueError("MPS tangent matrix dimensions/dtype must match its wires.")
        matrix = arrays.numeric(matrix)
        if not arrays.finite(matrix):
            raise ValueError("MPS tangent matrix must be finite.")
        scale = arrays.component_scale(matrix)
        if scale == 0 or not self._components:
            return self._owned(self._width, (), self._max_components)
        matrix = arrays.rescaled(matrix, scale)
        components = []
        for state, coefficient in self._components:
            chain, factors = state.copy(), []
            chain._apply(matrix, wires, _tangent_factors=factors)
            if 0.0 not in factors:
                components.append((chain, _product(coefficient, scale, *factors)))
        return self._owned(self._width, components, self._max_components)

    def inner(self, other, pauli=None):
        """Return <self|P|other>, including every coherent cross term."""
        if type(other) is not type(self) or other.num_qubits != self._width:
            raise ValueError("MPS tangent contraction requires equal native widths.")
        if pauli is None:
            pauli = "I" * self._width
        if type(pauli) is not str or len(pauli) != self._width or any(char not in "IXYZ" for char in pauli):
            raise ValueError("MPS tangent Pauli label must match its width in q0-first order.")
        current_gate_limits().check("max_decomposition_work",
                                    self.num_components * other.num_components * self._width)
        values = []
        for bra, left_scale in self._components:
            for ket, right_scale in other._components:
                env = arrays.filled("mps_tangent_overlap", (1, 1), complex, 1)
                factors = []
                for left, right, char in zip(bra._tensors, ket._tensors, pauli):
                    bra._guard((left.size + right.size + left.shape[2] * right.shape[2]) * 8)
                    ket._guard((left.size + right.size + left.shape[2] * right.shape[2]) * 8)
                    env = arrays.einsum("ab,asr,st,btv->rv", (left.shape[2], right.shape[2]),
                        env, arrays.conjugated(left), pauli_matrix(char), right)
                    scale = arrays.component_scale(env)
                    if scale == 0:
                        break
                    factors.append(scale)
                    env = arrays.rescaled(env, scale)
                values.append(_product(left_scale.conjugate(), right_scale,
                                       complex(env[0, 0]), *factors))
        try:
            result = complex(math.fsum(value.real for value in values),
                             math.fsum(value.imag for value in values))
        except OverflowError as error:
            raise FloatingPointError("MPS tangent contraction overflows complex128.") from error
        return _scalar(result)
