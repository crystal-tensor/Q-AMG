# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is part of QuantumBridge SDK.
# This implementation is developed for the QuantumBridge native architecture.


from numbers import Integral
import numpy as np
from quantumbridge.core.classical import UnsupportedFeature
from quantumbridge.circuit_library.data_preparation import amplitude_encoding
from .template_core import (resolve_limits, selected_wires, angle_array, operation_budget,
                            rotation_axis, native_circuit, static_array, shape_of)


def angle_embedding(circuit, features, wires=None, rotation="ry", *, limits=None):
    """The legacy functional entry point retains its default Y rotation."""
    limits = resolve_limits(limits)
    wires = selected_wires(circuit, wires, limits)
    shape = angle_array(features, 1, limits)
    axis = rotation_axis(rotation)
    if axis == "phase" or shape[0] > len(wires):
        raise ValueError("AngleEmbedding needs X/Y/Z rotations and at most one feature per wire.")
    operation_budget(circuit, limits, shape[0])
    for index in range(shape[0]):
        getattr(circuit, axis)(features[index], wires[index])
    return circuit


def basis_embedding(circuit, features, wires=None, *, limits=None):
    limits = resolve_limits(limits)
    wires = selected_wires(circuit, wires, limits)
    if isinstance(features, Integral) and not isinstance(features, bool):
        if not 0 <= features < 1 << len(wires):
            raise ValueError("Basis integer does not fit the selected wires.")
        values = [(features >> (len(wires)-1-q)) & 1 for q in range(len(wires))]
    else:
        if isinstance(features, str):
            if any(bit not in {"0", "1"} for bit in features):
                raise ValueError("Basis bitstring contains a non-binary character.")
            features = [int(bit) for bit in features]
        values = static_array(features)
        if values.shape != (len(wires),) or any(bit not in (0, 1) for bit in values):
            raise ValueError("BasisEmbedding requires one zero/one bit per selected wire.")
    operation_budget(circuit, limits, sum(values))
    for wire, bit in zip(wires, values):
        if bit:
            circuit.x(wire)
    return circuit


def amplitude_embedding(circuit, features, wires=None, normalize=True, *, pad_with=None, limits=None):
    limits = resolve_limits(limits)
    wires = selected_wires(circuit, wires, limits)
    dimension = 1 << len(wires)
    limits.check("max_amplitude_dimension", dimension)
    shape = shape_of(features)
    if len(shape) != 1 or not 1 <= shape[0] <= dimension:
        raise ValueError("AmplitudeEmbedding requires a nonempty bounded vector.")
    values = static_array(features, complex_values=True)
    if values.ndim != 1 or values.size > dimension or values.size < 1:
        raise ValueError("AmplitudeEmbedding requires a nonempty bounded vector.")
    if not isinstance(normalize, bool):
        raise TypeError("Amplitude normalization must be boolean.")
    if values.size != dimension:
        if pad_with is None:
            raise ValueError("Amplitude vector dimension requires padding for these wires.")
        padding = static_array(pad_with, complex_values=True)
        if padding.shape:
            raise ValueError("Amplitude padding must be a finite scalar.")
        values = np.pad(values, (0, dimension-values.size), constant_values=padding.item())
    operation_budget(circuit, limits, 1)
    native = native_circuit(circuit)
    if any(set(op.wires) & set(wires) for op in native.operations) or any(m.wire in wires for m in native.measurements):
        raise UnsupportedFeature("amplitude_preparation_position", "Amplitude preparation requires untouched target wires.")
    scale = np.max(np.abs(values))
    if not np.isfinite(scale) or scale == 0:
        raise ValueError("AmplitudeEmbedding requires a nonzero finite norm.")
    scaled_norm = np.linalg.norm(values/scale)
    if normalize:
        values = (values/scale)/scaled_norm
    elif not np.isclose(scale*scaled_norm, 1., atol=1e-10, rtol=0):
        raise ValueError("Amplitude features must be normalized unless normalize=True.")
    prepared = amplitude_encoding(values, normalize=False)
    op = prepared.operations[0]
    native.append(op.name, wires, metadata={**op.metadata, "requires_zero_input": True})
    native.metadata["amplitude_embedding"] = {"wires": list(wires), "mode": "native-unitary-preparation",
                                             "status": "native-executed", "input_differentiable": False}
    return circuit
