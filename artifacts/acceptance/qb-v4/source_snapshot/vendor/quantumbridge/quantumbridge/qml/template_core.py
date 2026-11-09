# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
"""Shared validation and native-operation emission for executable templates."""
from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral, Real
from math import prod
from itertools import islice
import numpy as np
from quantumbridge.core import Circuit, Parameter, ParameterExpression
from quantumbridge.core.classical import UnsupportedFeature
from quantumbridge.diff.trace import Dual


@dataclass(frozen=True)
class TemplateLimits:
    max_wires: int = 128
    max_layers: int = 256
    max_operations: int = 100000
    max_parameter_elements: int = 4096
    max_amplitude_dimension: int = 64
    max_qft_wires: int = 12
    max_trotter_steps: int = 1024

    def __post_init__(self):
        if any(isinstance(v, bool) or not isinstance(v, Integral) or v < 1 for v in self.__dict__.values()):
            raise ValueError("Template limits must be positive integers.")

    def check(self, name, value):
        if value > getattr(self, name):
            raise MemoryError("Native template exceeded " + name)


def resolve_limits(limits):
    if limits is None:
        return TemplateLimits()
    if not isinstance(limits, TemplateLimits):
        raise TypeError("Template limits require TemplateLimits.")
    return limits


def native_circuit(target):
    circuit = getattr(target, "circuit", target)
    if not isinstance(circuit, Circuit):
        raise TypeError("Native templates require a Circuit or Tape target.")
    return circuit


def selected_wires(target, wires, limits):
    circuit = native_circuit(target)
    if wires is None:
        limits.check("max_wires", circuit.num_qubits)
        wires = tuple(range(circuit.num_qubits))
    else:
        wires = wire_labels(wires, limits)
    if not wires or any(isinstance(q, bool) or not isinstance(q, Integral) or not 0 <= q < circuit.num_qubits for q in wires) or len(set(wires)) != len(wires):
        raise ValueError("Template wires must be nonempty, distinct in-register integers.")
    limits.check("max_wires", len(wires))
    return tuple(map(int, wires))


def wire_labels(wires, limits):
    labels = (wires,) if isinstance(wires, (str, Integral)) else tuple(islice(wires,limits.max_wires+1))
    limits.check("max_wires",len(labels))
    return labels


def bounded_pairs(pattern, limits):
    pairs = []
    for pair in pattern:
        if len(pairs) >= limits.max_operations//3:
            raise MemoryError("Native template pattern exceeds the operation budget.")
        value = tuple(islice(pair,3))
        if len(value) != 2:
            raise ValueError("Template patterns require pairs.")
        pairs.append(value)
    return tuple(pairs)


def shape_of(value):
    if hasattr(value, "shape"):
        return tuple(value.shape)
    if isinstance(value, (list, tuple)):
        shapes = [shape_of(child) for child in value]
        if shapes and any(shape != shapes[0] for shape in shapes):
            raise ValueError("Template parameters must have a rectangular shape.")
        return (len(value),) + (shapes[0] if shapes else ())
    return ()


def entry(value, indices):
    for index in indices:
        value = value[index]
    return value


def angle_value(value):
    if isinstance(value, (Parameter, ParameterExpression)):
        return
    if isinstance(value, Dual):
        value = value.value
    elif type(value).__module__.startswith("tensorflow"):
        if value.shape.rank != 0 or not value.dtype.is_floating:
            raise TypeError("Template gate angles require real scalar parameters.")
        return
    elif type(value).__module__.startswith(("jax", "jaxlib")):
        if shape_of(value) or value.dtype.kind != "f":
            raise TypeError("Template gate angles require real scalar parameters.")
        return
    elif type(value).__module__.startswith(("torch", "autograd")):
        dtype = getattr(value, "dtype", None)
        if shape_of(value) or getattr(dtype, "is_complex", False) or getattr(dtype, "kind", "f") == "c" or not getattr(dtype,"is_floating_point",True):
            raise TypeError("Template gate angles require real scalar parameters.")
        return
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("Template gate angles require real scalar parameters.")
    if not np.isfinite(value):
        raise ValueError("Template gate angles require finite real scalar parameters.")


def angle_array(value, rank, limits):
    shape = shape_of(value)
    if len(shape) != rank:
        raise ValueError(f"Template parameters require rank {rank}, received {shape}.")
    limits.check("max_parameter_elements", prod(shape))
    for indices in np.ndindex(shape):
        angle_value(entry(value, indices))
    return shape


def operation_budget(target, limits, count):
    limits.check("max_operations", len(native_circuit(target).operations) + count)


def rotation_axis(rotation):
    if not isinstance(rotation, str) or rotation.lower() not in {"x", "y", "z", "rx", "ry", "rz", "phase"}:
        raise ValueError("Template rotation must be X/Y/Z, RX/RY/RZ or phase.")
    return "r" + rotation.lower() if len(rotation) == 1 else rotation.lower()


def static_array(value, *, complex_values=False):
    shape = shape_of(value)
    for indices in np.ndindex(shape):
        item = entry(value, indices)
        if isinstance(item, (Dual, Parameter, ParameterExpression)) or type(item).__module__.startswith(("torch", "autograd", "tensorflow")):
            raise UnsupportedFeature("state_preparation_input_gradient", "Native basis/amplitude features are static; input differentiation is not implemented.")
    if not complex_values and np.iscomplexobj(value):
        raise ValueError("Basis features must be real bits.")
    array = np.asarray(value, dtype=complex if complex_values else None)
    if not np.all(np.isfinite(array)):
        raise ValueError("State preparation features must be finite.")
    return array


class NativeTemplate:
    """Concrete native decomposition; apply to a Circuit/Tape to record it."""
    differentiable = True
    requires_untouched_wires = False

    def _build(self, builder, wires, *args, limits=None, **options):
        self.limits = resolve_limits(limits)
        self.wires = wire_labels(wires,self.limits)
        if not self.wires or len(set(self.wires)) != len(self.wires):
            raise ValueError("Template wire labels must be nonempty and unique.")
        self.limits.check("max_wires", len(self.wires))
        self._circuit = Circuit(len(self.wires))
        builder(self._circuit, *args, wires=tuple(range(len(self.wires))), limits=self.limits, **options)

    def decomposition(self):
        from dataclasses import replace
        return tuple(replace(op, metadata=deepcopy(op.metadata)) for op in self._circuit.operations)

    def to_circuit(self):
        circuit = Circuit(len(self.wires), metadata=deepcopy(self._circuit.metadata))
        circuit.operations = list(self.decomposition())
        return circuit

    def apply(self, target, wire_map=None):
        circuit = native_circuit(target)
        mapping = tuple(self.wires if wire_map is None else (wire_map[q] for q in self.wires))
        wires = selected_wires(target, mapping, self.limits)
        operation_budget(target, self.limits, len(self._circuit.operations))
        if self.requires_untouched_wires and (any(set(op.wires) & set(wires) for op in circuit.operations) or any(m.wire in wires for m in circuit.measurements)):
            raise UnsupportedFeature("amplitude_preparation_position", "Amplitude preparation requires untouched target wires.")
        parameter = getattr(target, "parameter_transform", lambda value: value)
        for op in self.decomposition():
            circuit.append(op.name, tuple(wires[q] for q in op.targets), tuple(wires[q] for q in op.controls),
                           tuple(parameter(value) for value in op.params), op.metadata)
        return target

    __call__ = apply

    @property
    def provenance(self):
        return {"implementation": "quantumbridge_native", "template": type(self).__name__,
                "operation_count": len(self._circuit.operations), "input_differentiable": self.differentiable,
                "scope": "supported native subset; not full PennyLane template parity"}
