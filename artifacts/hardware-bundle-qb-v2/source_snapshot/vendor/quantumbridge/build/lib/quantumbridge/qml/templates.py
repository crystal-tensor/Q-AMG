# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is part of QuantumBridge SDK.
# This implementation is developed for the QuantumBridge native architecture.


from numbers import Integral
from itertools import islice
import warnings
from .template_core import (resolve_limits, selected_wires, angle_array, operation_budget,
                            rotation_axis, entry)


def basic_entangler(circuit, parameters, *, wires=None, limits=None):
    """Legacy RY/open-chain helper, distinct from BasicEntanglerLayers."""
    return hardware_efficient_ansatz(circuit, [parameters], wires=wires, limits=limits)


def basic_entangler_layers(circuit, parameters, wires=None, rotation="X", *, limits=None):
    warnings.warn("basic_entangler_layers now uses RX/ring semantics. For the previous RY/open-chain behavior use hardware_efficient_ansatz; new code may use templates.BasicEntanglerLayers.", FutureWarning, stacklevel=2)
    return _basic_entangler_layers(circuit, parameters, wires, rotation, limits=limits)


def _basic_entangler_layers(circuit, parameters, wires=None, rotation="X", *, limits=None):
    limits = resolve_limits(limits)
    wires = selected_wires(circuit, wires, limits)
    layers, width = angle_array(parameters, 2, limits)
    axis = rotation_axis(rotation)
    if width != len(wires):
        raise ValueError("BasicEntanglerLayers shape must be (layers, wires).")
    limits.check("max_layers", layers)
    entanglers = 0 if width == 1 else 1 if width == 2 else width
    operation_budget(circuit, limits, layers*(width+entanglers))
    for layer in parameters:
        for wire, value in zip(wires, layer):
            getattr(circuit, axis)(value, wire)
        for q in range(entanglers):
            circuit.cx(wires[q], wires[(q+1) % width])
    return circuit


def strongly_entangling_layers(circuit, parameters, wires=None, ranges=None, imprimitive="cx", *, limits=None):
    limits = resolve_limits(limits)
    wires = selected_wires(circuit, wires, limits)
    layers, width, angles = angle_array(parameters, 3, limits)
    if (width, angles) != (len(wires), 3):
        raise ValueError("StronglyEntanglingLayers shape must be (layers, wires, 3).")
    limits.check("max_layers", layers)
    if imprimitive not in {"cx", "cz"}:
        raise NotImplementedError("Native StronglyEntanglingLayers supports cx or cz imprimitive.")
    ranges = tuple((layer % (width-1)+1) if width > 1 else 0 for layer in range(layers)) if ranges is None else tuple(islice(ranges,layers+1))
    if len(ranges) != layers or any(isinstance(r, bool) or not isinstance(r, Integral) or r < 0 or (width > 1 and r % width == 0) for r in ranges):
        raise ValueError("Entangler ranges need one nonnegative nonzero-mod-width integer per layer.")
    operation_budget(circuit, limits, layers*(3*width+(width if width > 1 else 0)))
    for layer in range(layers):
        for q, wire in enumerate(wires):
            phi, theta, omega = (entry(parameters, (layer,q,i)) for i in range(3))
            circuit.rz(phi, wire)
            circuit.ry(theta, wire)
            circuit.rz(omega, wire)
        if width > 1:
            for q, wire in enumerate(wires):
                getattr(circuit, imprimitive)(wire, wires[(q+ranges[layer]) % width])
    return circuit


def hardware_efficient_ansatz(circuit, parameters, wires=None, *, limits=None):
    limits = resolve_limits(limits)
    wires = selected_wires(circuit, wires, limits)
    layers, width = angle_array(parameters, 2, limits)
    if width != len(wires):
        raise ValueError("Hardware-efficient ansatz shape must be (layers, wires).")
    limits.check("max_layers", layers)
    operation_budget(circuit, limits, layers*(2*width-1))
    for layer in parameters:
        for wire, value in zip(wires, layer):
            circuit.ry(value, wire)
        for a,b in zip(wires,wires[1:]):
            circuit.cx(a,b)
    return circuit
