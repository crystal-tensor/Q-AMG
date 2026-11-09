# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
"""Transactional native composition over existing operation and block maps."""

from .classical_scope import composition_scope, entries, set_entries
from .gate_limits import current_gate_limits
from .operations import Instruction, Operation, instruction_definition_work
from .parameters import validate_parameter_names


def _admit_graph(value):
    from quantumbridge.execution.resource_context import current_resources
    if current_resources() is None:return
    from quantumbridge.execution.array_ownership import _exact_type, _defined_type, _immutable_callback_data
    from .classical import UnsupportedFeature
    from types import FunctionType, BuiltinFunctionType
    import numpy as np
    pending,seen=[value],set()
    while pending:
        item=pending.pop()
        if id(item) in seen:continue
        seen.add(id(item));kind=type(item)
        current_gate_limits().check("max_decomposition_work",len(seen)+len(pending))
        if _exact_type(kind,(type(None),str,int,float,complex,bool,bytes,type,range)):continue
        if kind is np.ndarray:
            if item.dtype.hasobject:raise UnsupportedFeature("managed_composition_object_array")
            continue
        if _exact_type(kind,(FunctionType,BuiltinFunctionType)):
            if not _immutable_callback_data(item):raise UnsupportedFeature("managed_composition_callback")
            continue
        if _defined_type(kind,"numpy"):
            if any(base is np.generic for base in type.__dict__["__mro__"].__get__(kind)) and item.dtype.kind in "biufc":continue
            if isinstance(item,np.dtype) and item.metadata is None:continue
        if _exact_type(kind,(tuple,list,set,frozenset)):
            children=item
        elif kind is slice:children=(item.start,item.stop,item.step)
        elif kind is dict:
            current_gate_limits().check("max_decomposition_work",len(seen)+len(pending)+2*len(item))
            pending.extend(item.keys());children=item.values()
        elif _defined_type(kind,"quantumbridge"):
            fields=object.__getattribute__(item,"__dict__")
            if type(fields) is not dict:raise UnsupportedFeature("managed_composition_fields")
            from .circuit import is_native_circuit
            children=tuple(child for key,child in fields.items() if not (
                is_native_circuit(item) and "_execution_metadata" in fields and key=="metadata"))
        else:raise UnsupportedFeature("managed_composition_foreign_graph")
        current_gate_limits().check("max_decomposition_work",len(seen)+len(pending)+len(children))
        pending.extend(children)


def compose(owner, other, qubits, clbits, front, inplace, wrap, *, copy, var_remap, inline_captures):
    from .circuit import Circuit,is_native_circuit, _index_map, _remap_operation, _remap_measurement, _replace_self
    from .gates import Gate, is_native_instruction
    from .annotated_operation import AnnotatedOperation
    from .instruction_parameters import admit_owner
    from .errors import CircuitError
    from quantumbridge.execution.array_ownership import native_configuration_copy
    from quantumbridge.execution.resource_context import current_resources
    limits = current_gate_limits()
    _admit_graph((owner,other,var_remap,qubits,clbits,front,inplace,wrap,copy,inline_captures))
    direct = not is_native_circuit(other)
    if direct:
        if not is_native_instruction(other) and type(other) is not AnnotatedOperation:
            raise TypeError("Circuit.compose needs a native Circuit or Instruction.")
        if type(other) is not AnnotatedOperation: admit_owner(other)
        source = Circuit(other.num_qubits,other.num_clbits)
        from .standard_gates import invocation_params
        from .control_flow_operations import is_control_owner
        if is_control_owner(other):
            source._append_instruction_reference(other, range(other.num_qubits), range(other.num_clbits))
            other = source
        elif type(other) is Instruction:
            metadata={"instruction_object":other,"instruction_clbits":tuple(range(other.num_clbits))}
        else:
            metadata={"annotated_operation" if type(other) is AnnotatedOperation else "gate":other}
        if other is not source:
            label=getattr(other,"label",None)
            if label is not None:metadata["label"]=label
            source.operations.append(Operation(other.name,tuple(range(other.num_qubits)),params=invocation_params(other),metadata=metadata))
            other = source
    left_work, right_work = instruction_definition_work(owner), instruction_definition_work(other)
    limits.check("max_total_operations", left_work+right_work)
    limits.check("max_decomposition_work", left_work+right_work)
    qubit_map = _index_map(other.num_qubits,owner.num_qubits,qubits,"qubits",owner=owner)
    clbit_map = _index_map(other.num_bits,owner.num_bits,clbits,"clbits",owner=owner)
    validate_parameter_names((*owner.parameters,*other.parameters))
    records, identifiers = composition_scope(owner,other,var_remap,inline_captures)
    if wrap and not direct:
        if entries(other): raise CircuitError("Circuits with classical identifiers must be composed inline, not wrapped.")
        try: operation = other.to_gate()
        except (TypeError,ValueError): operation = other.to_instruction()
        source = Circuit(other.num_qubits,other.num_bits)
        if type(operation) is Instruction:
            source._append_instruction_reference(operation,range(other.num_qubits),range(other.num_bits))
        else: source.append(operation,range(other.num_qubits))
        other=source
    if inplace:
        from .circuit_metadata import empty_execution_circuit
        target=empty_execution_circuit(owner)
        from .bits import copy_bits
        copy_bits(owner,target)
        target.operations,target.measurements=list(owner.operations),list(owner.measurements)
    else: target=owner.copy()
    offset=0 if front else len(target.operations)
    mapped_ops=[_remap_operation(op,qubit_map,clbit_map,target.num_qubits,target.num_bits,
        identifiers=identifiers,copy_operation=copy and not direct,target_circuit=target) for op in other.operations]
    mapped_measurements=[_remap_measurement(item,qubit_map,clbit_map,offset,len(other.operations)) for item in other.measurements]
    if front:
        identity_q=dict(enumerate(range(target.num_qubits)))
        identity_c=dict(enumerate(range(target.num_bits)))
        target.measurements=[_remap_measurement(item,identity_q,identity_c,len(mapped_ops),len(target.operations)) for item in target.measurements]
        target.operations=mapped_ops+target.operations
        target.measurements=mapped_measurements+target.measurements
    else:
        target.operations.extend(mapped_ops)
        target.measurements.extend(mapped_measurements)
    target.global_phase=target.global_phase+other.global_phase
    set_entries(target,records)
    entries(target)
    if inplace:
        _replace_self(owner,target)
        return None
    return target
