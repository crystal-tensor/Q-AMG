# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is part of QuantumBridge SDK.
# This implementation is developed for the QuantumBridge native architecture.

import numpy as np
from math import prod
from numbers import Integral

from .state_dimensions import state_dimensions, qubit_count, subsystem_positions
from quantumbridge.execution.resource_context import current_resources


def partial_trace(density_matrix, keep: tuple[int, ...], num_qubits: int | None = None, *, dims=None):
    from quantumbridge.execution import mps_arrays as arrays
    from quantumbridge.execution.array_reservations import numeric_array, numeric_shape
    scope=current_resources()
    if scope is None:
        rho=np.asarray(density_matrix,dtype=complex)
        shape=rho.shape
    else:shape=numeric_shape(density_matrix)
    if len(shape)!=2 or shape[0]!=shape[1]:raise ValueError("Partial trace requires a square density matrix.")
    dimensions=state_dimensions(dims,size=shape[0])
    if num_qubits is not None:
        if isinstance(num_qubits,bool) or not isinstance(num_qubits,Integral) or num_qubits!=qubit_count(dimensions):
            raise ValueError("Partial trace qubit count must match binary subsystem dimensions.")
    selected=subsystem_positions(keep,len(dimensions))
    if scope is not None:
        rho=numeric_array(density_matrix,category="partial_trace_input")
        scope.ledger.unavailable("partial_trace_numpy_workspace","Trace outputs are admitted; hidden NumPy reduction/iterator workspace and Python containers are not measured RSS.")
    rho=arrays.reshaped(rho,dimensions+dimensions)
    remaining=list(dimensions)
    trace_out = [wire for wire in range(len(dimensions)) if wire not in selected]
    for wire in reversed(trace_out):
        column=wire+len(remaining)
        shape=tuple(value for axis,value in enumerate(rho.shape) if axis not in (wire,column))
        output=arrays.owned("partial_trace_reduction",shape,complex)
        np.trace(rho,axis1=wire,axis2=column,out=output)
        rho=output
        del remaining[wire]
    dim=prod(remaining)
    return arrays.reshaped(rho,(dim,dim))

