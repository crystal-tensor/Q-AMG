# Copyright 2026 QuantumBridge Contributors. Apache-2.0.
"""Native primitive unified blocs; no simulator or alternate parameter store."""

from collections.abc import Mapping
from math import prod
from numbers import Integral, Real

import numpy as np

from quantumbridge.core import Circuit

from .bindings_array import BindingsArray
from .observables_array import ObservablesArray


class SamplerPub:
    """A native circuit, shaped binding array and optional per-PUB shots."""

    def __init__(self, circuit, parameter_values=None, shots=None, validate=True):
        self._circuit = circuit
        self._parameter_values = BindingsArray() if parameter_values is None else parameter_values
        self._shots = shots
        if validate:
            self.validate()

    @property
    def circuit(self):
        return self._circuit

    @property
    def parameter_values(self):
        return self._parameter_values

    @property
    def shots(self):
        return self._shots

    @property
    def shape(self):
        return self.parameter_values.shape

    @property
    def ndim(self):
        return self.parameter_values.ndim

    @property
    def size(self):
        return self.parameter_values.size

    def validate(self):
        if not isinstance(self.circuit, Circuit):
            raise TypeError("A native SamplerPub requires a native Circuit.")
        if not isinstance(self.parameter_values, BindingsArray):
            raise TypeError("SamplerPub parameter_values must be a BindingsArray; use coerce for numeric inputs.")
        self.parameter_values.validate()
        if self.parameter_values.num_parameters != self.circuit.num_parameters:
            raise ValueError("SamplerPub binding count must match the circuit parameter count.")
        if self.shots is not None:
            if isinstance(self.shots, bool) or not isinstance(self.shots, Integral):
                raise TypeError("Sampler shots must be an integer.")
            if self.shots <= 0:
                raise ValueError("Sampler shots must be positive.")

    @classmethod
    def coerce(cls, pub, shots=None):
        if isinstance(pub, cls):
            if pub.shots is not None or shots is None:
                pub.validate()
                return pub
            return cls(pub.circuit, pub.parameter_values, shots)
        if isinstance(pub, Circuit):
            return cls(pub, shots=shots)
        if not isinstance(pub, (tuple, list)) or not 1 <= len(pub) <= 3:
            raise TypeError("A sampler PUB must be a native Circuit or a tuple of one to three items.")
        circuit = pub[0]
        if not isinstance(circuit, Circuit):
            raise TypeError("A sampler PUB must start with a native Circuit.")
        values = pub[1] if len(pub) > 1 else None
        if isinstance(values, BindingsArray):
            bindings = values
        elif values is None or isinstance(values, Mapping):
            bindings = BindingsArray(values)
        else:
            bindings = BindingsArray({tuple(circuit.parameters): values})
        return cls(circuit, bindings, pub[2] if len(pub) == 3 and pub[2] is not None else shots)


def _precision(value):
    if value is not None and (isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value < 0):
        raise ValueError("Estimator precision must be finite and nonnegative.")
    return None if value is None else float(value)


class EstimatorPub:
    """A circuit, shaped observables and bindings with broadcast result axes."""

    def __init__(self, circuit, observables, parameter_values=None, precision=None, validate=True):
        self._circuit, self._observables = circuit, observables
        self._parameter_values = BindingsArray() if parameter_values is None else parameter_values
        self._precision = _precision(precision)
        try:
            self._shape = np.broadcast_shapes(self.observables.shape, self.parameter_values.shape)
        except ValueError as error:
            raise ValueError("Observable and parameter shapes must broadcast.") from error
        if validate:
            self.validate()

    @property
    def circuit(self):
        return self._circuit

    @property
    def observables(self):
        return self._observables

    @property
    def parameter_values(self):
        return self._parameter_values

    @property
    def precision(self):
        return self._precision

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return prod(self.shape)

    def validate(self):
        if not isinstance(self.circuit, Circuit) or not isinstance(self.observables, ObservablesArray) or not isinstance(self.parameter_values, BindingsArray):
            raise TypeError("Native EstimatorPub requires a Circuit, ObservablesArray and BindingsArray.")
        self.observables.validate()
        self.parameter_values.validate()
        _precision(self.precision)
        if self.parameter_values.num_parameters != self.circuit.num_parameters:
            raise ValueError("Estimator binding count must match the circuit.")
        if self.observables.num_qubits != self.circuit.num_qubits:
            raise ValueError("Observable width must match the circuit.")

    @classmethod
    def coerce(cls, pub, precision=None):
        if isinstance(pub, cls):
            pub.validate()
            return pub if pub.precision is not None or precision is None else cls(pub.circuit, pub.observables, pub.parameter_values, precision)
        if not isinstance(pub, (tuple, list)) or not 2 <= len(pub) <= 4:
            raise TypeError("An estimator PUB must contain two to four items.")
        circuit = pub[0]
        if not isinstance(circuit, Circuit):
            raise TypeError("An estimator PUB must start with a native Circuit.")
        observables = ObservablesArray._for_circuit(pub[1], circuit.num_qubits)
        values = pub[2] if len(pub) > 2 else None
        if isinstance(values, BindingsArray):
            bindings = values
        elif values is None or isinstance(values, Mapping):
            bindings = BindingsArray(values)
        else:
            bindings = BindingsArray({tuple(circuit.parameters): values})
        return cls(circuit, observables, bindings, pub[3] if len(pub) == 4 and pub[3] is not None else precision)
