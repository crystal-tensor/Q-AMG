# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
"""Public instruction contexts and checked handles over native Circuit storage."""
from collections.abc import MutableSequence
from itertools import islice
import operator

from .bits import bounded_items,operand_index,is_bit,REGISTER_TYPES
from .errors import CircuitError
from .gate_limits import current_gate_limits
from .operations import Instruction,Operation,instruction_definition_work,has_public_frame,capture_public_frame
from .parameters import parameter_set,validate_parameter_names


def admit(value):
    from .composition import _admit_graph
    _admit_graph(value)


def operation_admission(operation):
    from .gates import is_native_instruction
    from .annotated_operation import AnnotatedOperation,validate_annotated_structure
    from .instruction_parameters import admit_owner
    if type(operation) is AnnotatedOperation:
        validate_annotated_structure(operation)
    elif is_native_instruction(operation):admit_owner(operation)
    else:raise CircuitError("A public instruction context requires an admitted native operation.")
    return operation


class CircuitInstruction:
    def __init__(self,operation,qubits=None,clbits=None):
        admit((operation,qubits,clbits))
        operation_admission(operation)
        object.__setattr__(self,"_operation",operation)
        object.__setattr__(self,"_name",operation.name)
        from .standard_gates import UnitaryGate
        object.__setattr__(self,"_params",() if type(operation) is UnitaryGate else tuple(operation.params))
        object.__setattr__(self,"_qubits",() if qubits is None else bounded_items(qubits))
        object.__setattr__(self,"_clbits",() if clbits is None else bounded_items(clbits))

    def __setattr__(self,name,value):raise AttributeError("CircuitInstruction fields are read-only; use replace.")
    @property
    def operation(self):return self._operation
    @property
    def qubits(self):return self._qubits
    @property
    def clbits(self):return self._clbits
    @property
    def name(self):return self._name
    @property
    def params(self):return list(self._params)
    def is_control_flow(self):
        from .control_flow_operations import ControlFlowOp
        return isinstance(self.operation, ControlFlowOp)
    def is_standard_gate(self):
        from .standard_gates import standard_type_name
        return standard_type_name(self.operation) not in (None,"UnitaryGate")
    def copy(self):
        admit(self)
        result=object.__new__(CircuitInstruction)
        result.__dict__.update(self.__dict__)
        return result
    def replace(self,/,operation=None,qubits=None,clbits=None,params=None):
        admit((self,operation,qubits,clbits,params))
        chosen=self.operation if operation is None else operation
        if params is not None:
            operation_admission(chosen)
            chosen=chosen.to_mutable() if hasattr(chosen,"to_mutable") else chosen.copy()
            chosen.params=params
        result=CircuitInstruction(chosen,self.qubits if qubits is None else qubits,self.clbits if clbits is None else clbits)
        if operation is None:
            object.__setattr__(result,"_name",self._name)
            if params is None:object.__setattr__(result,"_params",self._params)
        return result
    def __iter__(self):return iter((self.operation,list(self.qubits),list(self.clbits)))
    def __len__(self):return 3
    def __getitem__(self,index):return tuple(self)[index]
    def __eq__(self,other):
        if type(other) is not CircuitInstruction:return False
        admit((self,other))
        if self.qubits!=other.qubits or self.clbits!=other.clbits:return False
        from .instruction_parameters import same_values
        from .standard_gates import standard_type_name
        if (standard_type_name(self.operation) not in (None,"UnitaryGate")
                or standard_type_name(other.operation) not in (None,"UnitaryGate")):
            if self.name!=other.name or not same_values(self._params,other._params):return False
        if self.operation is other.operation:return True
        a,b=self.operation,other.operation
        if type(a) is not type(b) or (a.name,a.num_qubits,a.num_clbits)!=(b.name,b.num_qubits,b.num_clbits):return False
        import numpy as np
        return len(a.params)==len(b.params) and all(np.array_equal(x,y) for x,y in zip(a.params,b.params))


def context_from_entry(circuit,entry):
    from .measurements import Measurement
    if type(entry) is Measurement:
        return CircuitInstruction(Instruction("measure",1,1,[]),[circuit._qubits[entry.wire]],[circuit._clbits[entry.bit]])
    from .control_flow_operations import operation_owner
    operation=operation_owner(entry)
    if operation is None:
        operation=next((entry.metadata[key] for key in ("gate","instruction_object","annotated_operation") if key in entry.metadata),None)
    if operation is None:
        from .classical import UnsupportedFeature
        raise UnsupportedFeature("public_context_owner","This legacy primitive has not been admitted as a public operation owner.")
    bits=entry.metadata.get("instruction_clbits",())
    result=CircuitInstruction(operation,[circuit._qubits[i] for i in entry.wires],[circuit._clbits[i] for i in bits])
    if has_public_frame(entry):
        object.__setattr__(result,"_name",object.__getattribute__(entry,"name"))
        object.__setattr__(result,"_params",object.__getattribute__(entry,"params"))
    return result


def entry_from_context(circuit,context,*,check_width=True,refresh=False):
    from .gates import is_native_gate,Gate,ControlledGate
    from .annotated_operation import AnnotatedOperation
    from .standard_gates import invocation_params
    admit((context,))
    if type(context) is not CircuitInstruction:
        values=bounded_items(context,"instruction context")
        context=CircuitInstruction(*values)
    operation=operation_admission(context.operation)
    qubits=tuple(operand_index(circuit,bit,True) for bit in context.qubits)
    clbits=tuple(operand_index(circuit,bit,False) for bit in context.clbits)
    from .control_flow_operations import is_control_owner, control_entry
    if is_control_owner(operation):
        return control_entry(circuit, operation, qubits, clbits)
    if check_width and (len(qubits)!=operation.num_qubits or len(clbits)!=operation.num_clbits):
        raise CircuitError("Instruction operand count differs from its operation width.")
    if len(set(qubits))!=len(qubits) or len(set(clbits))!=len(clbits):raise CircuitError("Duplicate instruction operands.")
    if type(operation) is AnnotatedOperation:key="annotated_operation"
    else:key="gate" if is_native_gate(operation) else "instruction_object"
    metadata={key:operation}
    if key=="instruction_object" or type(operation) is AnnotatedOperation and clbits:metadata["instruction_clbits"]=clbits
    if getattr(operation,"label",None) is not None:metadata["label"]=operation.label
    entry=Operation(operation.name,qubits,params=invocation_params(operation),metadata=metadata)
    from .standard_gates import standard_type_name
    from .operations import _matrix_owner_family
    if (type(operation) in (Gate,ControlledGate,Instruction,AnnotatedOperation)
            or standard_type_name(operation) is not None or _matrix_owner_family(operation)):
        capture_public_frame(entry,operation.name if refresh else context.name,
            invocation_params(operation) if refresh else context.params)
    return entry


def validate_entries(circuit,entries):
    limits=current_gate_limits()
    limits.check("max_total_operations",len(entries))
    symbols=set(parameter_set((circuit.global_phase,)))
    work=0
    for entry in entries:
        symbols.update(parameter_set(entry.params))
        node = entry.metadata.get("control_node")
        if node is not None:
            for body in node.bodies:
                work += instruction_definition_work(body)
        owner=next((entry.metadata[key] for key in ("gate","instruction_object","annotated_operation") if key in entry.metadata),None)
        if owner is not None:
            from .annotated_operation import AnnotatedOperation
            base=owner.base_op if type(owner) is AnnotatedOperation else owner
            if base.definition is not None:work+=instruction_definition_work(base.definition,owner=base)
        work+=1;limits.check("max_decomposition_work",work)
        limits.check("max_total_operations",work)
    validate_parameter_names(symbols)


def commit_entries(circuit,entries):
    from .measurements import Measurement
    validate_entries(circuit,entries)
    operations,measurements=[],[]
    for entry in entries:
        if type(entry) is Measurement:
            measurements.append(Measurement(entry.wire,entry.bit,entry.kind,{**entry.metadata,"operation_index":len(operations)}))
        else:operations.append(entry)
    circuit.operations,circuit.measurements=operations,measurements


class CircuitData(MutableSequence):
    def __init__(self,circuit):self._circuit=circuit
    def _entries(self):return list(self._circuit.iter_instructions())
    def __len__(self):return len(self._circuit.operations)+len(self._circuit.measurements)
    def __getitem__(self,index):
        entries=self._entries()
        if type(index) is slice:return [context_from_entry(self._circuit,item) for item in entries[index]]
        return context_from_entry(self._circuit,entries[operator.index(index)])
    def __setitem__(self,index,value):
        admit((self._circuit,index,value))
        entries=self._entries()
        # Public slice assignment interprets one instruction triple, not a list.
        if type(value) is not CircuitInstruction and type(value) in (list,tuple) and len(value)!=3:
            raise ValueError("A data assignment requires one instruction triple.")
        entry=entry_from_context(self._circuit,value,refresh=True)
        if type(index) is slice:
            positions=list(range(len(entries)))[index]
            if not positions:raise ValueError("Instruction slice has no selected entries.")
            for position in positions:entries[position]=entry
        else:entries[operator.index(index)]=entry
        commit_entries(self._circuit,entries)
    def __delitem__(self,index):
        admit((self._circuit,index))
        entries=self._entries();del entries[index];commit_entries(self._circuit,entries)
    def insert(self,index,value):
        admit((self._circuit,index,value))
        entries=self._entries();entries.insert(operator.index(index),entry_from_context(self._circuit,value,refresh=True))
        commit_entries(self._circuit,entries)
    def clear(self):commit_entries(self._circuit,[])
    def assign(self,values):
        admit((self._circuit,values))
        entries=[entry_from_context(self._circuit,value,refresh=True) for value in bounded_contexts(values)]
        commit_entries(self._circuit,entries)


def bounded_contexts(values):
    limit=current_gate_limits().max_total_operations
    if type(values) in (tuple,list):current_gate_limits().check("max_total_operations",len(values))
    result=tuple(islice(iter(values),limit+1))
    current_gate_limits().check("max_total_operations",len(result))
    return result


def public_factory_entries(target,source):
    """Admit one explicit native factory output into real public operation owners."""
    from .circuit import Circuit,_remap_operation
    from .measurements import Measurement
    from .standard_gates import STANDARD_GATE_NAMES,UnitaryGate
    entries=[]
    for item in source.iter_instructions():
        if type(item) is Measurement:
            context=CircuitInstruction(Instruction("measure",1,1,[]),[target._qubits[item.wire]],[target._clbits[item.bit]])
        elif any(key in item.metadata for key in ("gate","instruction_object","annotated_operation","control_node")):
            entries.append(item);continue
        elif item.name in {"reset","barrier","delay"}:
            metadata={key:value for key,value in item.metadata.items() if key!="label"}
            owner=Instruction(item.name,len(item.wires),0,item.params,item.metadata.get("label"),metadata=metadata)
            context=CircuitInstruction(owner,[target._qubits[i] for i in item.wires],[])
        else:
            kind=STANDARD_GATE_NAMES.get(item.name)
            if item.name == "unitary" and not item.controls and "matrix" in item.metadata and not set(item.metadata)-{"matrix","label","operation_index"}:
                matrix=item.metadata["matrix"]
                converted=getattr(item,"_public_matrix_basis",None)=="native"
                if converted:
                    from quantumbridge.execution.array_reservations import numeric_shape
                    from quantumbridge.information.operator import _reverse_qubit_order
                    dimension=1<<len(item.wires)
                    current_gate_limits().check("max_matrix_dimension",dimension)
                    if numeric_shape(matrix)!=(dimension,dimension):
                        raise CircuitError("Factory unitary matrix differs from its wire width.")
                    matrix=_reverse_qubit_order(matrix,len(item.wires))
                owner=UnitaryGate(matrix,label=item.metadata.get("label"))
                object.__setattr__(owner,"_public_matrix_basis","little" if converted else "native")
            elif kind is not None and not set(item.metadata)-{"label","operation_index"}:
                owner=kind(*item.params,label=item.metadata.get("label"))
                if owner.num_qubits!=len(item.wires):raise CircuitError("Factory primitive width differs from its standard class.")
            else:
                # A native-only primitive keeps its actual executable definition,
                # not a public class name or a delegated matrix factory.
                body=Circuit(len(item.wires))
                body.operations=[_remap_operation(item,{wire:i for i,wire in enumerate(item.wires)},{},body.num_qubits,0)]
                owner=body.to_gate(_name=item.name)
            context=CircuitInstruction(owner,[target._qubits[i] for i in item.wires],[])
        entries.append(entry_from_context(target,context))
    return entries


class InstructionSet:
    def __init__(self,*,resource_requester=None):
        admit(resource_requester)
        self._references=[]
        self._resource_requester=resource_requester
    @classmethod
    def bound(cls,circuit,indices):
        result=cls();result._references=[(circuit,index) for index in indices];return result
    def __len__(self):return len(self._references)
    def __getitem__(self,index):
        if type(index) is slice:return [self[i] for i in range(len(self))[index]]
        circuit,value=self._references[operator.index(index)]
        return value if circuit is None else CircuitData(circuit)[value]
    def __iter__(self):return (self[i] for i in range(len(self)))
    @property
    def instructions(self):return [context.operation for context in self]
    @property
    def qargs(self):return [list(context.qubits) for context in self]
    @property
    def cargs(self):return [list(context.clbits) for context in self]
    def add(self,instruction,qargs=None,cargs=None):
        admit((instruction,qargs,cargs))
        current_gate_limits().check("max_total_operations",len(self._references)+1)
        if type(instruction) is not CircuitInstruction:
            if qargs is None or cargs is None:raise CircuitError("InstructionSet.add requires its operands.")
            instruction=CircuitInstruction(instruction,qargs,cargs)
        self._references.append((None,instruction))
    def inverse(self,annotated=False):
        admit((self,annotated))
        plans={};standalone=[]
        for position,(circuit,index) in enumerate(self._references):
            context=self[position]
            if circuit is None:
                operation=context.operation.inverse(annotated=annotated)
                standalone.append((position,context.replace(operation=operation)))
            else:
                key=id(circuit)
                if key not in plans:plans[key]=(circuit,list(circuit.iter_instructions()))
                from .circuit import _inverse_operation
                plans[key][1][index]=_inverse_operation(plans[key][1][index],annotated=annotated)
        for circuit,entries in plans.values():validate_entries(circuit,entries)
        for circuit,entries in plans.values():commit_entries(circuit,entries)
        for position,value in standalone:self._references[position]=(None,value)
        return self


def operand_group(circuit,value,quantum):
    width=circuit.num_qubits if quantum else circuit.num_bits
    if type(value) is slice:return tuple(range(width))[value]
    if is_bit(value):return (operand_index(circuit,value,quantum),)
    if type(value) in (int,bool):return (operand_index(circuit,value,quantum),)
    try:return (operand_index(circuit,value,quantum),)
    except (TypeError,CircuitError):
        return tuple(operand_index(circuit,item,quantum) for item in bounded_items(value))


def public_append(circuit,instruction,qargs,cargs,copy=True):
    from .gates import is_native_gate
    from .annotated_operation import AnnotatedOperation
    admit((circuit,instruction,qargs,cargs,copy))
    if type(instruction) is CircuitInstruction:
        if qargs is None:qargs=instruction.qubits
        if cargs is None:cargs=instruction.clbits
        instruction=instruction.operation
    operation=operation_admission(instruction)
    if qargs is None or type(qargs) in (int,bool) and qargs==0:qargs=()
    if cargs is None or type(cargs) in (int,bool) and cargs==0:cargs=()
    qgroups=[operand_group(circuit,item,True) for item in bounded_items(qargs)]
    cgroups=[operand_group(circuit,item,False) for item in bounded_items(cargs)]
    if len(qgroups)!=operation.num_qubits or len(cgroups)!=operation.num_clbits:
        raise CircuitError("Instruction argument groups do not match its width.")
    if operation.name=="measure" and type(operation) is Instruction:
        qs,cs=qgroups[0],cgroups[0]
        if not qs or not cs or len(qs) not in (1,len(cs)):raise CircuitError("Measurement register size error.")
        arguments=[((qs[0] if len(qs)==1 else q,), (c,)) for q,c in zip(qs*len(cs) if len(qs)==1 else qs,cs)]
    elif type(operation) is Instruction and operation.name in {"reset","delay"} and operation.num_qubits==1 and not operation.num_clbits:
        if not qgroups[0]:raise CircuitError("One or more instruction arguments are empty.")
        current_gate_limits().check("max_total_operations",len(circuit.operations)+len(circuit.measurements)+len(qgroups[0]))
        arguments=[((q,),()) for q in qgroups[0]]
    elif is_native_gate(operation) or type(operation) is AnnotatedOperation and not operation.num_clbits:
        if cgroups:raise CircuitError("Gates cannot have classical operands.")
        if any(not group for group in qgroups):raise CircuitError("One or more instruction arguments are empty.")
        lengths=[len(group) for group in qgroups]
        count=max(lengths,default=1)
        if len(qgroups)==2:
            if any(length not in (1,count) for length in lengths):raise CircuitError("Cannot broadcast gate operands.")
        elif len(qgroups)>2 and any(length!=count for length in lengths):raise CircuitError("Cannot broadcast gate operands.")
        current_gate_limits().check("max_total_operations",len(circuit.operations)+len(circuit.measurements)+count)
        arguments=[(tuple(group[0] if len(group)==1 else group[i] for group in qgroups),()) for i in range(count)]
    else:
        arguments=[(tuple(bit for group in qgroups for bit in group),tuple(bit for group in cgroups for bit in group))]
    limits=current_gate_limits()
    limits.check("max_total_operations",len(circuit.operations)+len(circuit.measurements)+len(arguments))
    # Direct numeric operations remain borrowed, as in both measured SDKs.
    from .control_flow_operations import is_control_owner
    symbols = operation.parameters if is_control_owner(operation) else parameter_set(operation.params)
    owner=operation.copy() if copy and symbols else operation
    entries=[]
    for qubits,clbits in arguments:
        context=CircuitInstruction(owner,[circuit._qubits[i] for i in qubits],[circuit._clbits[i] for i in clbits])
        entries.append(entry_from_context(circuit,context,check_width=is_native_gate(owner) or type(owner) is AnnotatedOperation or owner.name=="measure"))
    before=len(circuit.operations)+len(circuit.measurements)
    commit_entries(circuit,list(circuit.iter_instructions())+entries)
    return InstructionSet.bound(circuit,range(before,before+len(entries)))
