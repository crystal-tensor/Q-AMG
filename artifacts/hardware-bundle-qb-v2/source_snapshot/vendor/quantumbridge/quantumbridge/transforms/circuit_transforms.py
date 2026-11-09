# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is part of QuantumBridge SDK.
# This implementation is developed for the QuantumBridge native architecture.


def identity_circuit_transform(circuit):
    return circuit.copy()


from .dispatch import transform


def _safe_segments(tape, simplify, limits=None):
    from dataclasses import replace
    from quantumbridge.core import Circuit, Measurement
    from .program import resolve_limits, check_batch

    limits = resolve_limits(limits)
    check_batch((tape,), limits)
    out = tape.copy()
    ordered = tuple(out.circuit.iter_instructions())
    out.circuit.operations, out.circuit.measurements = [], []
    segment = Circuit(out.circuit.num_qubits, out.circuit.num_bits)

    def flush():
        nonlocal segment
        if segment.operations:
            merged = simplify(segment)
            out.circuit.operations.extend(merged.operations)
            segment = Circuit(out.circuit.num_qubits, out.circuit.num_bits)

    for item in ordered:
        if isinstance(item, Measurement):
            flush()
            out.circuit.measurements.append(replace(item, metadata={**item.metadata, "operation_index": len(out.circuit.operations)}))
        elif item.name in {"reset", "kraus", "store"} or item.metadata:
            flush()
            out.circuit.operations.append(item)
        else:
            segment.operations.append(item)
    flush()
    check_batch((out,), limits)
    return out


@transform
def merge_rotations(tape, *, limits=None):
    """Merge adjacent same-axis native rotations within safe dynamic segments."""
    from quantumbridge.compiler.cancellation import MergeAdjacentRotationPass
    from .program import resolve_limits
    limits = resolve_limits(limits)
    def simplify(segment):
        work = 0
        while True:
            work += len(segment.operations)
            limits.check("max_expansion_steps", work)
            merged = MergeAdjacentRotationPass(tolerance=0).run(segment).circuit
            if len(merged.operations) == len(segment.operations):
                return merged
            segment = merged
    return (_safe_segments(tape, simplify, limits),), lambda results: results[0]


def _opposite(left, right):
    from numbers import Real
    from quantumbridge.core import Parameter, ParameterExpression
    if isinstance(left, Real) and isinstance(right, Real):
        return left + right == 0
    left = left.expression() if isinstance(left, Parameter) else left
    right = right.expression() if isinstance(right, Parameter) else right
    if not isinstance(left, ParameterExpression) or not isinstance(right, ParameterExpression):
        return False
    return left.op == "neg" and left.args[0] == right or right.op == "neg" and right.args[0] == left


@transform
def cancel_inverses(tape, recursive=True, *, limits=None):
    """Cancel proved adjacent inverse pairs, never through runtime boundaries."""
    from quantumbridge.compiler.optimization import CancelAdjacentInversePass
    if not isinstance(recursive, bool):
        raise TypeError("cancel_inverses recursive must be boolean.")
    self_inverse = CancelAdjacentInversePass.self_inverse | {"cy", "ch", "ccx", "cswap"}
    pairs = CancelAdjacentInversePass.inverse_pairs | {("sx","sxdg"),("sxdg","sx")}
    rotations = {"rx","ry","rz","phase","p","u1","rxx","ryy","rzz","crx","cry","crz","cp","cphase"}
    def simplify(segment):
        stack = []
        for index, operation in enumerate(segment.operations):
            if stack and (recursive or stack[-1][0] == index-1):
                previous = stack[-1][1]
                equal_wires = previous.targets == operation.targets and previous.controls == operation.controls
                fixed = previous.name == operation.name and operation.name in self_inverse or (previous.name,operation.name) in pairs
                inverse_angle = previous.name == operation.name and operation.name in rotations and len(previous.params) == len(operation.params) == 1 and _opposite(previous.params[0],operation.params[0])
                if equal_wires and (fixed or inverse_angle):
                    stack.pop()
                    continue
            stack.append((index,operation))
        segment.operations = [operation for _,operation in stack]
        return segment
    return (_safe_segments(tape,simplify,limits),), lambda results:results[0]

