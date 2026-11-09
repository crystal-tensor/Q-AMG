# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
"""Retain classical scalar graphs; inspect primals only for native planning."""

from dataclasses import dataclass

import numpy as np

from quantumbridge.core.classical import UnsupportedFeature


def is_live_scalar(value):
    return type(value).__module__.startswith(("torch", "autograd", "jax", "jaxlib", "tensorflow")) or hasattr(value, "derivative")


def scalar_primal(value):
    module = type(value).__module__
    if module.startswith(("jax", "jaxlib", "tensorflow")):
        raise UnsupportedFeature("coefficient_interface", "Use the framework QNode bridge; live JAX/TF primals cannot be read by native host planning.")
    if hasattr(value, "derivative"):
        value = value.value
    elif module.startswith("torch"):
        if value.device.type != "cpu":
            raise UnsupportedFeature("torch_device", "Native coefficient planning requires CPU scalar tensors.")
        if value.ndim or not value.is_floating_point():
            raise TypeError("Hamiltonian coefficient tensors must be real floating scalars.")
        value = value.detach().item()
    elif module.startswith("autograd"):
        from autograd.tracer import getval
        value = getval(value)
    array = np.asarray(value)
    if array.shape or array.dtype.kind not in "fiu" or not np.isfinite(array.item()):
        raise ValueError("Native coefficients must be finite real scalar values.")
    return float(array.item())


def retain_coefficient(value):
    if is_live_scalar(value):
        if type(value).__module__.startswith("tensorflow"):
            if value.shape.rank != 0 or not value.dtype.is_floating:
                raise TypeError("TensorFlow coefficients require real floating scalars.")
            return value
        if type(value).__module__.startswith(("jax", "jaxlib")):
            if value.shape or value.dtype.kind != "f":
                raise TypeError("JAX coefficients require real floating scalars.")
            # Preserve the graph; the JAX bridge validates concrete host values.
            return value
        scalar_primal(value)
        return value
    return float(value)


@dataclass(frozen=True)
class CoefficientSlot:
    index: int
    primal: float


class CoefficientTable:
    def __init__(self):
        self.parameters = []

    def add(self, value):
        primal = scalar_primal(value)
        if not is_live_scalar(value):
            return primal
        slot = CoefficientSlot(len(self.parameters), primal)
        self.parameters.append(value)
        return slot

    @staticmethod
    def resolve(value, parameters=None):
        if not isinstance(value, CoefficientSlot):
            return value
        return value.primal if parameters is None else parameters[value.index]
