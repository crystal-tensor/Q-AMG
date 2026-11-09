# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
"""Ordered dimensions and native q0-first mixed-radix basis indexing."""

from math import prod
from numbers import Integral

from quantumbridge.execution.resource_context import current_resources


def state_dimensions(dims, *, size=None):
    if dims is None:
        if size is None or size<=0 or size&(size-1):
            raise ValueError("Native state dimension must be a power of two unless explicit dims are supplied.")
        return (2,)*(size.bit_length()-1)
    if isinstance(dims,bool):
        raise ValueError("State dimensions must be positive integers, not booleans.")
    if isinstance(dims,Integral):
        total=int(dims)
        if total<=0:raise ValueError("State dimensions must be positive integers.")
        dimensions=(2,)*(total.bit_length()-1) if not total&(total-1) else (total,)
    else:
        if current_resources() is not None and type(dims) not in (tuple,list):
            raise ValueError("Managed state dimensions require a builtin integer sequence.")
        try:dimensions=tuple(dims)
        except TypeError as error:raise ValueError("State dimensions require an integer or integer sequence.") from error
        if any(isinstance(dim,bool) or not isinstance(dim,Integral) or dim<=0 for dim in dimensions):
            raise ValueError("State dimensions must be positive integers.")
        dimensions=tuple(int(dim) for dim in dimensions)
        total=prod(dimensions)
    if size is not None and total!=size:
        raise ValueError("State dimension product must match the numeric array dimension.")
    return dimensions


def qubit_count(dims):
    return len(dims) if all(dim==2 for dim in dims) else None


def require_qubits(dims,feature):
    if qubit_count(dims) is None:
        from quantumbridge.core.classical import UnsupportedFeature
        raise UnsupportedFeature(feature,"This path requires binary subsystems; mixed-radix data cannot be reinterpreted as qubits.")


def subsystem_positions(qargs,count):
    positions=tuple(range(count) if qargs is None else qargs)
    if any(isinstance(position,bool) or not isinstance(position,Integral) for position in positions):
        raise ValueError("Native state qargs must be integer subsystem positions.")
    if len(set(positions))!=len(positions):raise ValueError("Native state qargs must be distinct.")
    if any(position<0 or position>=count for position in positions):
        raise ValueError("Native state qargs are outside this state.")
    return positions


def basis_digits(index,dims):
    digits=[0]*len(dims)
    for position in range(len(dims)-1,-1,-1):
        index,digits[position]=divmod(index,dims[position])
    return tuple(digits)


def basis_index(digits,dims):
    result=0
    for digit,dim in zip(digits,dims):result=result*dim+digit
    return result


def basis_label(index,dims,selected):
    digits=basis_digits(index,dims)
    separator="|" if any(dims[position]>10 for position in selected) else ""
    return separator.join(str(digits[position]) for position in selected)
