# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# Algebraic native basis translation, including exact global phase.

from copy import deepcopy
from dataclasses import replace
from math import pi
from numbers import Real, Integral

from quantumbridge.core import Circuit, Operation, Measurement
from quantumbridge.core.classical import UnsupportedFeature
from .pass_base import PassResult, TransformationPass
from .traversal import append_item, map_control_bodies
from .limits import CompilationLimits


def needs_decomposition(circuit, basis_gates):
    return any(op.name not in set(basis_gates) for op in circuit.operations)


class BasisGateConversionPass(TransformationPass):
    handles_dynamic = True
    def __init__(self, basis_gates): self.basis_gates = set(basis_gates)

    def run(self, circuit):
        unsupported = sorted({op.name for op in circuit.operations if op.name not in self.basis_gates})
        if unsupported: raise ValueError(f"Operations outside basis: {unsupported}.")
        return PassResult(circuit, {"basis_gates": sorted(self.basis_gates)}, False)


class GateDecompositionPass(TransformationPass):
    handles_dynamic = True

    def __init__(self, basis_gates=None, *, limits=None):
        self.basis_gates = set(basis_gates or {"x", "y", "z", "h", "rx", "ry", "rz", "cx", "cz", "swap"})
        self.limits = limits or CompilationLimits()

    def run(self, circuit, *, max_operations=None, max_expansion_steps=None, max_depth=None, _budget=None, _nesting=0):
        max_operations = self.limits.max_dag_nodes if max_operations is None else max_operations
        max_expansion_steps = 4*self.limits.max_dag_nodes if max_expansion_steps is None else max_expansion_steps
        for value in (max_operations,max_expansion_steps,24 if max_depth is None else max_depth):
            if isinstance(value,bool) or not isinstance(value,Integral) or value < 1:
                raise ValueError("Decomposition limits must be positive integers.")
        if _nesting > (self.limits.max_control_nesting if max_depth is None else max_depth):
            raise MemoryError("Decomposition exceeds its control nesting budget.")
        budget = {"steps":0,"operations":0} if _budget is None else _budget
        def charge(output=False):
            budget["steps"] += 1
            if budget["steps"] > max_expansion_steps:
                raise MemoryError("Decomposition exceeds max_expansion_steps.")
            if output:
                budget["operations"] += 1
                if budget["operations"] > max_operations:
                    raise MemoryError("Decomposition exceeds max_operations.")
        from quantumbridge.core.circuit_metadata import empty_execution_circuit
        out = empty_execution_circuit(circuit)
        decomposed = []
        def translate(op, depth=0):
            charge()
            if depth > (24 if max_depth is None else max_depth): raise UnsupportedFeature("basis_translation", f"No bounded algebraic route to basis for {op.name}.")
            if op.name in self.basis_gates or op.metadata.get("directive") or op.name in {"reset", "kraus"}:
                charge(output=True)
                out.operations.append(deepcopy(op)); return
            if op.metadata.get("control_node") is not None:
                charge(output=True)
                out.operations.append(map_control_bodies(op, lambda body: self.run(body,max_operations=max_operations,
                    max_expansion_steps=max_expansion_steps,max_depth=max_depth,_budget=budget,_nesting=_nesting+1).circuit)); return
            name, wires, params = op.name, op.wires, op.params
            phase, recipe = 0., []
            a = wires[0] if wires else None
            b = wires[1] if len(wires) > 1 else None
            def one(gate, wire, *angles): recipe.append(Operation(gate, (wire,), params=angles))
            def cx(left, right): recipe.append(Operation("cx", (right,), (left,)))
            hardware = {"rz", "sx"} <= self.basis_gates
            if name in {"id", "identity"}: pass
            elif name in {"phase", "p", "u1", "s", "sdg", "t", "tdg", "z"}:
                angle = params[0] if params else {"s": pi/2, "sdg": -pi/2, "t": pi/4, "tdg": -pi/4, "z": pi}[name]
                if name == "sdg" and "s" in self.basis_gates:
                    for _ in range(3): one("s", a)
                elif name == "tdg" and "t" in self.basis_gates:
                    for _ in range(7): one("t", a)
                else: phase = angle / 2; one("rz", a, angle)
            elif name == "h":
                if hardware:
                    phase = pi/4; one("rz", a, pi/2); one("sx", a); one("rz", a, pi/2)
                else:
                    phase = pi/2; one("ry", a, pi/2); one("rx", a, pi)
            elif name == "x": phase = pi/2; one("rx", a, pi)
            elif name == "y": phase = pi/2; one("ry", a, pi)
            elif name in {"sx", "sxdg"}:
                phase = pi/4 if name == "sx" else -pi/4; one("rx", a, pi/2 if name == "sx" else -pi/2)
            elif name == "rx" and hardware:
                phase = pi/2
                one("rz", a, pi/2); one("sx", a); one("rz", a, params[0] + pi); one("sx", a); one("rz", a, pi/2)
            elif name == "ry":
                if hardware:
                    phase = pi/2
                    one("sx", a); one("rz", a, params[0] + pi); one("sx", a); one("rz", a, pi)
                else:
                    one("rz", a, -pi/2); one("rx", a, params[0]); one("rz", a, pi/2)
            elif name in {"u", "u3", "u2"}:
                theta, phi, lam = (pi/2, *params) if name == "u2" else params
                phase = (phi + lam) / 2
                one("rz", a, lam); one("ry", a, theta); one("rz", a, phi)
            elif name == "cz": one("h", b); cx(a, b); one("h", b)
            elif name == "cx" and "cz" in self.basis_gates:
                one("h", b); recipe.append(Operation("cz", (b,), (a,))); one("h", b)
            elif name == "swap": cx(a,b); cx(b,a); cx(a,b)
            elif name in {"rzz", "rxx", "ryy"}:
                if name == "rxx": one("h", a); one("h", b)
                if name == "ryy": one("rx", a, pi/2); one("rx", b, pi/2)
                cx(a,b); one("rz", b, params[0]); cx(a,b)
                if name == "rxx": one("h", a); one("h", b)
                if name == "ryy": one("rx", a, -pi/2); one("rx", b, -pi/2)
            elif name in {"cry", "crz"}:
                axis = name[1:]; one(axis, b, params[0]/2); cx(a,b); one(axis, b, -params[0]/2); cx(a,b)
            elif name == "crx":
                one("h", b); recipe.append(Operation("crz", (b,), (a,), params)); one("h", b)
            elif name in {"cp", "cphase"}:
                one("phase", a, params[0]/2); cx(a,b); one("phase", b, -params[0]/2); cx(a,b); one("phase", b, params[0]/2)
            else:
                raise UnsupportedFeature("basis_translation", f"No native algebraic decomposition for {name!r} into {sorted(self.basis_gates)}.")
            if (not isinstance(phase, Real) or phase != 0) and any(key in op.metadata for key in ("condition", "gate_condition", "control_flow_stack", "control_flow")):
                raise UnsupportedFeature("conditional_basis_phase", "Use a structured if body to preserve decomposition phase in its lexical scope.")
            out.global_phase = out.global_phase + phase
            decomposed.append(name)
            for child in recipe:
                translate(replace(child, metadata=deepcopy(op.metadata)), depth + 1)
        for item in circuit.iter_instructions():
            if isinstance(item, Measurement):
                charge(output=True)
                append_item(out, item)
            else: translate(item)
        return PassResult(out, {"decomposed": decomposed, "global_phase_preserved": True}, bool(decomposed))
