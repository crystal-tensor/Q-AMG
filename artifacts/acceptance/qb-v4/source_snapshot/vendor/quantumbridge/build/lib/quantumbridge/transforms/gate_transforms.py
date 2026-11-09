# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
"""Bounded native definitions, basis decomposition and compiler Tape transforms."""

from collections.abc import Mapping
from itertools import islice
import numpy as np

from quantumbridge.core import Circuit, Instruction, Measurement
from quantumbridge.core.classical import UnsupportedFeature
from quantumbridge.compiler import native_transpile
from quantumbridge.compiler.decomposition import GateDecompositionPass
from quantumbridge.compiler.limits import CompilationLimits
from quantumbridge.compiler.traversal import append_item, map_control_bodies
from quantumbridge.compiler.transpile import _DEFAULT_BASIS
from .dispatch import transform
from .program import resolve_limits, check_batch, operation_count, _reject_live


def _gate_names(names):
    if names is None:
        return None
    aliases = {"cnot":"cx","hadamard":"h","paulix":"x","pauliy":"y","pauliz":"z","identity":"id"}
    sequence = (names,) if isinstance(names,str) else tuple(islice(names,257))
    if len(sequence) > 256:
        raise MemoryError("Gate set exceeds the native name budget.")
    result = set()
    for name in sequence:
        if not isinstance(name,str):
            raise TypeError("Native transform gate sets use string names.")
        result.add(aliases.get(name.lower(),name.lower()))
    if not result:
        raise ValueError("A gate set cannot be empty.")
    return result


def _compiler_input(circuit):
    pending = [circuit]
    while pending:
        body = pending.pop()
        _reject_live((body.global_phase,) + tuple(value for op in body.operations for value in op.params))
        for operation in body.operations:
            node = operation.metadata.get("control_node")
            if node is not None:
                pending.extend(node.bodies)


def _expand_definitions(circuit, definitions, limits):
    from quantumbridge.qml.template_core import NativeTemplate
    from quantumbridge.information import Operator
    if definitions is not None and not isinstance(definitions,Mapping):
        raise TypeError("Native definitions require a finite mapping.")
    limits.check("max_postprocessing_nodes",len(definitions or {}))
    definitions = dict(definitions or {})
    reserved = set(_DEFAULT_BASIS) | {"i","identity","sx","sxdg","crx","cry","crz","cp","cphase","rxx","ryy","rzz","iswap","ecr","cswap","csx","csxdg","dcx","global_phase","reset","kraus","store","barrier","delay","measure","control_flow"}
    if any(not isinstance(name,str) or name in reserved for name in definitions):
        raise ValueError("Custom definitions need fresh native names, not overrides of standard instructions.")
    limits.check("max_postprocessing_nodes",len(definitions))
    work = 0

    def expand(body, path=(), nesting=0):
        nonlocal work
        limits.check("max_structure_depth",nesting)
        from quantumbridge.core.circuit_metadata import empty_execution_circuit
        out = empty_execution_circuit(body)
        out.global_phase = body.global_phase
        size = 0
        for operation in body.iter_instructions():
            work += 1
            limits.check("max_expansion_steps",work)
            if isinstance(operation,Measurement):
                size += 1
                limits.check("max_operations_per_tape",size)
                append_item(out,operation)
            elif operation.metadata.get("control_node") is not None:
                mapped = map_control_bodies(operation,lambda child:expand(child,path,nesting+1))
                size += 1 + sum(operation_count(child,limits) for child in mapped.metadata["control_node"].bodies)
                limits.check("max_operations_per_tape",size)
                append_item(out,mapped)
            elif operation.name not in definitions:
                size += 1
                limits.check("max_operations_per_tape",size)
                append_item(out,operation)
            else:
                if operation.name in path:
                    raise UnsupportedFeature("decomposition_cycle", "A native definition recursively references itself.")
                if set(operation.metadata) - {"matrix","label"}:
                    raise UnsupportedFeature("definition_metadata", "Custom definition side effects/conditions cannot be silently discarded.")
                value = definitions[operation.name]
                if isinstance(value,NativeTemplate):
                    value = value.to_circuit()
                if isinstance(value,Instruction):
                    if value.definition is None:
                        raise UnsupportedFeature("missing_definition", "Registered Instructions require a native circuit definition.")
                    value = value.definition
                if not isinstance(value,Circuit):
                    raise TypeError("Definitions must be native Circuit, Instruction or NativeTemplate objects.")
                operation_count(value,limits)
                _compiler_input(value)
                if value.num_qubits != len(operation.wires) or value.num_bits or value.measurements or value.has_dynamic_operations or value.has_simulator_instructions:
                    raise UnsupportedFeature("definition_shape", "Custom definitions require a matching-width, unitary, bit-free native circuit.")
                if len(value.parameters) != len(operation.params):
                    raise ValueError("Custom definition parameter count does not match the operation.")
                bound = value.assign_parameters(dict(zip(value.parameters,operation.params)),strict=False)
                expanded = expand(bound,path+(operation.name,),nesting+1)
                if "matrix" in operation.metadata:
                    if expanded.parameters or expanded.num_qubits > 10:
                        raise UnsupportedFeature("definition_phase_proof", "An opaque matrix definition needs a bounded fully bound complex-unitary proof.")
                    proof = Operator.from_circuit(expanded)
                    if not np.allclose(proof.data.conj().T@proof.data,np.eye(len(proof.data)),atol=1e-12,rtol=0):
                        raise ValueError("A custom opaque definition must be unitary.")
                    if not np.allclose(proof.data,np.asarray(operation.metadata["matrix"]),atol=1e-12,rtol=0):
                        raise ValueError("Custom definition does not preserve the complete matrix/global phase.")
                size += operation_count(expanded,limits)
                limits.check("max_operations_per_tape",size)
                out.compose(expanded,qubits=operation.wires,inplace=True)
        return out

    result = expand(circuit)
    return result, work


@transform
def decompose(tape, *, gate_set=None, definitions=None, limits=None):
    """Expand declared native definitions/templates and algebraic compound gates."""
    limits = resolve_limits(limits)
    check_batch((tape,),limits)
    _compiler_input(tape.circuit)
    expanded, work = _expand_definitions(tape.circuit,definitions,limits)
    if work >= limits.max_expansion_steps:
        raise MemoryError("Decomposition has no remaining expansion budget.")
    converted = GateDecompositionPass(_gate_names(gate_set)).run(expanded,
        max_operations=limits.max_operations_per_tape,
        max_expansion_steps=limits.max_expansion_steps-work,max_depth=limits.max_structure_depth)
    out = tape.copy()
    out.circuit = converted.circuit
    check_batch((out,),limits)
    return (out,),lambda results:results[0]


@transform
def compile(tape, *, basis_set=None, target=None, coupling_map=None, optimization_level=1,
            initial_layout=None, restore_layout=True, scheduling_method=None,
            seed_transpiler=0, limits=None, compilation_limits=None):
    """Compile a Tape with the existing native compiler and proved identity maps."""
    limits = resolve_limits(limits)
    check_batch((tape,),limits)
    _compiler_input(tape.circuit)
    if compilation_limits is None:
        compilation_limits = CompilationLimits(max_dag_nodes=limits.max_operations_per_tape,
            max_control_nesting=min(32,limits.max_structure_depth))
    if not isinstance(compilation_limits,CompilationLimits):
        raise TypeError("compilation_limits requires native CompilationLimits.")
    if compilation_limits.max_dag_nodes > limits.max_operations_per_tape:
        raise ValueError("Compiler node budget cannot exceed the transform instruction budget.")
    compiled = native_transpile(tape.circuit,basis_gates=_gate_names(basis_set),target=target,
        coupling_map=coupling_map,optimization_level=optimization_level,initial_layout=initial_layout,
        restore_layout=restore_layout,scheduling_method=scheduling_method,
        seed_transpiler=seed_transpiler,limits=compilation_limits)
    identity = dict(enumerate(range(tape.circuit.num_qubits)))
    from quantumbridge.core.circuit_metadata import execution_metadata
    if compiled.num_qubits != tape.circuit.num_qubits or execution_metadata(compiled).get("initial_layout") != identity or execution_metadata(compiled).get("routing",{}).get("final_layout",identity) != identity:
        raise UnsupportedFeature("compiled_transform_layout", "Transform measurements require equal width and restored identity layout; unproved remapping is rejected.")
    out = tape.copy()
    out.circuit = compiled
    check_batch((out,),limits)
    return (out,),lambda results:results[0]
