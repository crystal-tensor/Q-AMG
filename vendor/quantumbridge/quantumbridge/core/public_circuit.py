# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
"""Native public Circuit ABI with the existing authoritative operation store."""
from itertools import count,islice
from collections.abc import Mapping
from numbers import Integral,Real
import math

from .circuit import Circuit
from .circuit_data import CircuitInstruction,InstructionSet,CircuitData,public_append,operand_group,admit,bounded_contexts
from .bits import REGISTER_TYPES,is_bit,bounded_items,copy_bits,integer_index
from .errors import CircuitError
from .operations import Instruction


class QuantumCircuit(Circuit):
    _counter=count()

    def __init__(self,*regs,name=None,global_phase=0,metadata=None,inputs=(),captures=(),declarations=()):
        admit((regs,name,global_phase,inputs,captures,declarations))
        if name is not None and type(name) is not str:raise CircuitError("Circuit name must be a string or None.")
        if metadata is not None and not isinstance(metadata,dict):raise TypeError("Only a dictionary is accepted for metadata.")
        self._execution_metadata={}
        counts=[]
        for value in regs:
            if isinstance(value,Integral):counts.append(integer_index(value))
            elif isinstance(value,Real) and math.isfinite(value) and float(value).is_integer():counts.append(int(value))
            else:break
        if len(counts)==len(regs):
            if len(counts)>2 or any(value<0 for value in counts):raise CircuitError("Expected up to two nonnegative Circuit widths.")
            nq,nc=(counts+[0,0])[:2]
            super().__init__(nq,nc,name,metadata)
        else:
            super().__init__(0,0,name,metadata)
            for value in regs:
                if any(type(value) is kind for kind in REGISTER_TYPES):self.add_register(value)
                elif type(value) in (list,tuple):self.add_bits(value)
                else:raise CircuitError("Circuit arguments must be widths, native Registers or bit sequences.")
        if self.name is None:self.name="circuit-"+str(next(self._counter))
        self.global_phase=global_phase
        for value in bounded_items(inputs):self.add_input(value)
        for value in bounded_items(captures):self.add_capture(value)
        declarations=tuple(declarations.items()) if type(declarations) is dict else declarations
        for variable,initial in bounded_items(declarations):self.add_var(variable,initial)

    @property
    def metadata(self):return self.__dict__["metadata"]
    @metadata.setter
    def metadata(self,value):
        if not isinstance(value,dict):raise TypeError("Only a dictionary is accepted for metadata.")
        self.__dict__["metadata"]=value

    @property
    def global_phase(self):return Circuit.global_phase.fget(self)
    @global_phase.setter
    def global_phase(self,value):
        if isinstance(value,Real) and math.isfinite(value):value=float(value)%(2*math.pi)
        Circuit.global_phase.fset(self,value)
    @property
    def data(self):return CircuitData(self)
    @data.setter
    def data(self,value):CircuitData(self).assign(value)

    @property
    def parameters(self):
        from .parameter_view import ParameterView
        admit(self)
        return ParameterView(Circuit.parameters.fget(self))

    def append(self,instruction,qargs=None,cargs=None,*,copy=True):
        result = public_append(self,instruction,qargs,cargs,copy)
        from .control_builders import handle_owner
        owner = handle_owner(self)
        if owner is not self:
            result._references = [(owner, index) for _, index in result._references]
        return result

    def add_register(self, *registers):
        from .control_builders import check_bit_mutation
        check_bit_mutation(self)
        return Circuit.add_register(self, *registers)

    def add_bits(self, bits):
        from .control_builders import check_bit_mutation
        check_bit_mutation(self)
        return Circuit.add_bits(self, bits)

    def unitary(self,obj,qubits,label=None):
        from .standard_gates import UnitaryGate
        from .gates import is_native_gate,Gate,ControlledGate
        from quantumbridge.information.operator import Operator
        if is_native_gate(obj):
            admit((self,obj,qubits,label))
            if type(obj) in (Gate,ControlledGate):
                raise CircuitError("to_matrix not defined for this public generic gate input")
            owner=obj
            if type(owner) is UnitaryGate:
                from .operations import _matrix_owner_family
                _matrix_owner_family(owner)
            obj=owner.to_matrix()
            if type(owner) is not UnitaryGate or getattr(owner,"_public_matrix_basis",None)=="native":
                from quantumbridge.information.operator import _reverse_qubit_order
                obj=_reverse_qubit_order(obj,owner.num_qubits)
        elif type(obj) is Operator:
            admit((self,obj,qubits,label))
            owner=obj
            from quantumbridge.information.operator import _representation_admission, _admit_public_arithmetic_owner
            if _representation_admission(owner):_admit_public_arithmetic_owner(owner)
            basis=getattr(owner,"_public_matrix_basis",None)
            if basis is not None and (type(basis) is not str or basis not in ("native","unresolved","hybrid")):
                raise ValueError("Unknown Operator matrix origin.")
            if basis=="unresolved":
                from .classical import UnsupportedFeature
                raise UnsupportedFeature("public_operator_matrix_origin")
            if basis=="hybrid":
                obj=owner.as_qiskit_operator().data
            else:obj=owner.data
            if basis=="native":
                from quantumbridge.execution.array_reservations import numeric_shape
                from .gate_limits import current_gate_limits
                from quantumbridge.information.operator import _reverse_qubit_order
                if type(owner.num_qubits) is not int or numeric_shape(obj)!=(1<<owner.num_qubits,)*2:
                    raise ValueError("Operator origin width differs from its matrix.")
                current_gate_limits().check("max_matrix_dimension",1<<owner.num_qubits)
                obj=_reverse_qubit_order(obj,owner.num_qubits)
        owner=UnitaryGate(obj,label=label)
        object.__setattr__(owner,"_public_matrix_basis","little")
        return self.append(owner,qubits)

    def _gate(self,name,params,qargs,label=None,ctrl_state=None):
        from .standard_gates import STANDARD_GATE_NAMES
        kind=STANDARD_GATE_NAMES[name]
        if name in {"cx","cy","cz","ch","cswap","crx","cry","crz","cp","ccx","ccz"}:
            gate=kind(*params,label=label,ctrl_state=ctrl_state)
        else:gate=kind(*params,label=label)
        return self.append(gate,qargs)

    def id(self,qubit):return self._gate("id",(),[qubit])
    def x(self,qubit,label=None):return self._gate("x",(),[qubit],label)
    def y(self,qubit,label=None):return self._gate("y",(),[qubit],label)
    def z(self,qubit,label=None):return self._gate("z",(),[qubit],label)
    def h(self,qubit):return self._gate("h",(),[qubit])
    def s(self,qubit):return self._gate("s",(),[qubit])
    def sdg(self,qubit):return self._gate("sdg",(),[qubit])
    def t(self,qubit):return self._gate("t",(),[qubit])
    def tdg(self,qubit):return self._gate("tdg",(),[qubit])
    def sx(self,qubit):return self._gate("sx",(),[qubit])
    def sxdg(self,qubit):return self._gate("sxdg",(),[qubit])
    def rx(self,theta,qubit,label=None):return self._gate("rx",(theta,),[qubit],label)
    def ry(self,theta,qubit,label=None):return self._gate("ry",(theta,),[qubit],label)
    def rz(self,phi,qubit):return self._gate("rz",(phi,),[qubit])
    def p(self,theta,qubit):return self._gate("p",(theta,),[qubit])
    def u(self,theta,phi,lam,qubit):return self._gate("u",(theta,phi,lam),[qubit])
    def cx(self,control_qubit,target_qubit,label=None,ctrl_state=None):return self._gate("cx",(),[control_qubit,target_qubit],label,ctrl_state)
    def cy(self,control_qubit,target_qubit,ctrl_state=None):return self._gate("cy",(),[control_qubit,target_qubit],ctrl_state=ctrl_state)
    def cz(self,control_qubit,target_qubit,ctrl_state=None):return self._gate("cz",(),[control_qubit,target_qubit],ctrl_state=ctrl_state)
    def ch(self,control_qubit,target_qubit,label=None,ctrl_state=None):return self._gate("ch",(),[control_qubit,target_qubit],label,ctrl_state)
    def cswap(self,control_qubit,target_qubit1,target_qubit2,label=None,ctrl_state=None):return self._gate("cswap",(),[control_qubit,target_qubit1,target_qubit2],label,ctrl_state)
    def swap(self,qubit1,qubit2):return self._gate("swap",(),[qubit1,qubit2])
    def crx(self,theta,control_qubit,target_qubit,label=None,ctrl_state=None):return self._gate("crx",(theta,),[control_qubit,target_qubit],label,ctrl_state)
    def cry(self,theta,control_qubit,target_qubit,label=None,ctrl_state=None):return self._gate("cry",(theta,),[control_qubit,target_qubit],label,ctrl_state)
    def crz(self,theta,control_qubit,target_qubit,label=None,ctrl_state=None):return self._gate("crz",(theta,),[control_qubit,target_qubit],label,ctrl_state)
    def cp(self,theta,control_qubit,target_qubit,ctrl_state=None):return self._gate("cp",(theta,),[control_qubit,target_qubit],ctrl_state=ctrl_state)
    def ccx(self,control_qubit1,control_qubit2,target_qubit,ctrl_state=None):return self._gate("ccx",(),[control_qubit1,control_qubit2,target_qubit],ctrl_state=ctrl_state)
    def ccz(self,control_qubit1,control_qubit2,target_qubit,ctrl_state=None):return self._gate("ccz",(),[control_qubit1,control_qubit2,target_qubit],ctrl_state=ctrl_state)
    def rxx(self,theta,qubit1,qubit2):return self._gate("rxx",(theta,),[qubit1,qubit2])
    def ryy(self,theta,qubit1,qubit2):return self._gate("ryy",(theta,),[qubit1,qubit2])
    def rzz(self,theta,qubit1,qubit2):return self._gate("rzz",(theta,),[qubit1,qubit2])
    def measure(self,qubit,cbit):return self.append(Instruction("measure",1,1,[]),[qubit],[cbit])
    def reset(self,qubit):return self.append(Instruction("reset",1,0,[]),[qubit])

    def _control(self, node, inplace=True):
        from .control_builders import append_block
        target = self if inplace else self.copy()
        append_block(target, node)
        return target

    def if_test(self, condition, true_body=None, qubits=None, clbits=None, *, label=None):
        from .control_builders import condition as expression, if_context
        value = expression(self, condition)
        if true_body is None:
            if qubits is not None or clbits is not None:
                raise CircuitError("Builder operands are inferred from its body.")
            return if_context(self, value, label, source=condition)
        return self.if_else(condition, true_body, None, qubits, clbits, label=label)

    def if_else(self, condition, true_body, false_body, qubits, clbits, *, label=None):
        from .control_builders import condition as expression, explicit_operands, _public_resource
        from .control_flow_operations import IfElseOp
        expression(self, condition)
        explicit_operands(qubits, clbits)
        return self.append(IfElseOp(_public_resource(self, condition), true_body, false_body, label), qubits, clbits)

    def for_loop(self, indexset, loop_parameter=None, body=None, qubits=None, clbits=None, *, label=None):
        from .control_builders import explicit_operands, for_context, loop_values
        values = loop_values(indexset)
        if body is None:
            if qubits is not None or clbits is not None:
                raise CircuitError("Builder operands are inferred from its body.")
            return for_context(self, values, loop_parameter, label, indexset=indexset if type(indexset) is range else values)
        explicit_operands(qubits, clbits)
        from .control_flow_operations import ForLoopOp
        return self.append(ForLoopOp(indexset if type(indexset) is range else values, loop_parameter, body, label), qubits, clbits)

    def while_loop(self, condition, body=None, qubits=None, clbits=None, *, label=None):
        from .control_builders import condition as expression, explicit_operands, while_context
        value = expression(self, condition)
        if body is None:
            if qubits is not None or clbits is not None:
                raise CircuitError("Builder operands are inferred from its body.")
            return while_context(self, value, label, source=condition)
        explicit_operands(qubits, clbits)
        from .control_flow_operations import WhileLoopOp
        from .control_builders import _public_resource
        return self.append(WhileLoopOp(_public_resource(self, condition), body, label), qubits, clbits)

    def switch(self, target, cases=None, qubits=None, clbits=None, *, label=None):
        from .control_builders import explicit_operands, selector, switch_context
        value = selector(self, target)
        if cases is None:
            if qubits is not None or clbits is not None:
                raise CircuitError("Builder operands are inferred from its body.")
            return switch_context(self, value, label, source=target)
        explicit_operands(qubits, clbits)
        from .control_flow_operations import SwitchCaseOp
        from .control_builders import _public_resource
        return self.append(SwitchCaseOp(_public_resource(self, target), cases, label=label), qubits, clbits)

    def break_loop(self):
        from .control_builders import jump
        return jump(self, "break")

    def continue_loop(self):
        from .control_builders import jump
        return jump(self, "continue")
    def barrier(self,*qargs,label=None):
        admit((self,qargs,label))
        wires=tuple(range(self.num_qubits)) if not qargs else tuple(i for value in qargs for i in operand_group(self,value,True))
        operation=Instruction("barrier",len(wires),0,[],label,metadata={"directive":True})
        result=self.append(operation,[[self._qubits[i]] for i in wires])
        return result[0] if not qargs else result
    def delay(self,duration,qarg=None,unit="dt"):
        admit((duration,qarg,unit))
        if unit not in ("dt","s","ms","us","ns","ps"):raise CircuitError("Unknown duration unit.")
        from .parameters import parameter_set
        if not parameter_set((duration,)):
            if not isinstance(duration,Real) or not math.isfinite(duration) or duration<0:raise CircuitError("Delay duration must be finite and nonnegative.")
            if unit=="dt":
                if not float(duration).is_integer():raise CircuitError("Integer duration is required for dt.")
                duration=int(duration)
        selected=list(range(self.num_qubits)) if qarg is None else qarg
        return self.append(Instruction("delay",1,0,[duration],metadata={"directive":True,"unit":unit,"duration":duration}),[selected])

    def copy_empty_like(self,name=None,*,vars_mode="alike"):
        from .classical_scope import copy_scope
        from .circuit_metadata import copy_user_metadata,inherit_execution
        metadata=copy_user_metadata(self)
        result=QuantumCircuit(name=self.name if name is None else name,metadata=metadata)
        inherit_execution(self,result)
        copy_bits(self,result);copy_scope(self,result,vars_mode)
        result.global_phase=self.global_phase
        return result

    def copy(self,name=None):
        from .operations import execution_operation,instruction_definition_work
        from .bits import bit_graph
        from .standard_gates import standard_type_name
        admit(self)
        bit_graph(self)
        instruction_definition_work(self,_resolve_frames=False)
        result=Circuit.copy(self,name)
        result.operations=[execution_operation(op) if standard_type_name(op.metadata.get("gate")) not in (None,"UnitaryGate")
            else op for op in result.operations]
        return result

    def to_instruction(self,parameter_map=None,label=None,*,name=None):
        from .errors import QiskitError
        admit((self,parameter_map,label,name))
        try:return Circuit.to_instruction(self,name=name,label=label,parameter_map=parameter_map)
        except QiskitError:raise
        except ValueError as error:raise QiskitError("Circuit-to-Instruction conversion has invalid public arguments.") from error

    def to_gate(self,parameter_map=None,label=None,*,_name=None):
        from .errors import QiskitError
        admit((self,parameter_map,label,_name))
        try:return Circuit.to_gate(self,parameter_map=parameter_map,label=label,_name=_name)
        except QiskitError:raise
        except ValueError as error:raise QiskitError("Circuit-to-Gate conversion requires a legal unitary definition and complete parameter map.") from error

    def assign_parameters(self,parameters,inplace=False,*,flat_input=False,strict=True,_copy_memo=None):
        from .parameters import Parameter,ParameterVector
        from .circuit import _parameter_mapping_from_values
        from .gate_limits import current_gate_limits
        admit((self,parameters,inplace,flat_input,strict))
        ordered=self.parameters;known=set(ordered);names={p.name:p for p in ordered}
        limits=current_gate_limits();limits.check("max_gate_parameters",len(ordered))
        def sequence(values,expected):
            if type(values) in (list,tuple):limits.check("max_gate_parameters",len(values))
            result=tuple(islice(iter(values),min(expected,limits.max_gate_parameters)+1))
            limits.check("max_gate_parameters",len(result))
            return result
        if isinstance(parameters,Mapping):
            limits.check("max_gate_parameters",len(parameters))
            mapping={};work=0
            for key,value in parameters.items():
                if flat_input:
                    if isinstance(key,ParameterVector) and strict:raise CircuitError("A flat binding key must be a circuit Parameter.")
                    if not isinstance(key,Parameter):raise TypeError("A flat binding requires Parameter keys.")
                    pairs=((key,value),)
                elif isinstance(key,ParameterVector):
                    limits.check("max_gate_parameters",len(key))
                    values=sequence(value,len(key))
                    if len(values)!=len(key):raise CircuitError("ParameterVector binding length must match its length.")
                    pairs=zip(key,values)
                else:pairs=((key,value),)
                for parameter,item in pairs:
                    work+=1;limits.check("max_gate_parameters",work)
                    if isinstance(parameter,str):
                        if parameter not in names:
                            if strict:raise KeyError(parameter)
                            continue
                        parameter=names[parameter]
                    if parameter not in known:
                        if strict:raise CircuitError("Cannot bind a Parameter absent from this circuit.")
                        continue
                    mapping[parameter]=item
        else:mapping=_parameter_mapping_from_values(ordered,sequence(parameters,len(ordered)),strict)
        # The public name policy is applied only at this API boundary; internal
        # definition binding must retain deterministic structural signatures.
        result=Circuit.assign_parameters(self,mapping,inplace=inplace,strict=False,_copy_memo=_copy_memo)
        if not inplace:result.name=self.name+"-"+str(next(self._counter))
        return result

    def get_parameter(self,name,default=...):
        from .parameters import Parameter,ParameterVectorElement
        admit((self,name))
        if default is not ...:admit(default)
        if type(name) is not str:raise TypeError("A parameter name must be a string.")
        for parameter in self.parameters:
            if parameter.name==name:
                if type(parameter) not in (Parameter,ParameterVectorElement):raise TypeError("Expected a native Parameter.")
                view=object.__new__(type(parameter));view.__dict__.update(parameter.__dict__)
                return view
        if default is not ...:return default
        raise KeyError(name)

    def has_parameter(self,name_or_parameter,/):
        from .parameters import Parameter
        admit((self,name_or_parameter))
        if isinstance(name_or_parameter,str):return any(p.name==name_or_parameter for p in self.parameters)
        return isinstance(name_or_parameter,Parameter) and name_or_parameter in self.parameters

    def repeat(self,reps):
        from .bits import integer_index
        from .gate_limits import current_gate_limits
        from .operations import instruction_definition_work
        admit((self,reps))
        count=integer_index(reps);limits=current_gate_limits()
        limits.check("max_repeat_count",max(0,count))
        work=0 if count<=0 else instruction_definition_work(self)
        limits.check("max_total_operations",max(0,count)*max(1,work))
        limits.check("max_decomposition_work",max(0,count)*max(1,work))
        from .circuit_metadata import inherit_execution
        from .classical_scope import copy_scope
        result=QuantumCircuit(name=self.name+"**"+str(reps))
        copy_bits(self,result);inherit_execution(self,result);copy_scope(self,result,"drop")
        result.global_phase=0.
        if count<=0:return result
        try:owner=self.to_gate()
        except (TypeError,ValueError):owner=self.to_instruction()
        for _ in range(count):result.append(owner,list(range(self.num_qubits)),list(range(self.num_bits)))
        return result

    def power(self,power,matrix_power=False,annotated=False):
        admit((self,power,matrix_power,annotated))
        if not isinstance(power,Real) or not math.isfinite(power):raise CircuitError("Circuit power must be finite and real.")
        if not matrix_power and power>=0 and float(power).is_integer():return self.repeat(int(power))
        if self.parameters and not annotated:raise CircuitError("Bind free Circuit parameters before a matrix power.")
        try:gate=self.to_gate().power(power,annotated=annotated)
        except (TypeError,ValueError) as error:raise CircuitError("This Circuit cannot be raised to a matrix power.") from error
        result=QuantumCircuit();copy_bits(self,result)
        result.append(gate,list(range(self.num_qubits)))
        return result

    def tensor(self,other,inplace=False):
        from .circuit import is_native_circuit,_replace_self
        from .bits import QuantumRegister,ClassicalRegister
        from .gate_limits import current_gate_limits
        admit((self,other,inplace))
        if not is_native_circuit(other):raise TypeError("Circuit.tensor requires a native Circuit.")
        limits=current_gate_limits()
        limits.check("max_circuit_bits",self.num_qubits+self.num_bits+other.num_qubits+other.num_bits)
        limits.check("max_total_operations",len(self.operations)+len(self.measurements)+len(other.operations)+len(other.measurements))
        def tensor_bits(quantum):
            left,right=(other.qregs,self.qregs) if quantum else (other.cregs,self.cregs)
            first,last=(other.qubits,self.qubits) if quantum else (other.clbits,self.clbits)
            name="q" if quantum else "c";kind=QuantumRegister if quantum else ClassicalRegister
            if len(left)==len(right)==1 and left[0].name==right[0].name==name:
                register=kind(len(first)+len(last),name)
                return list(register),[register]
            return first+last,left+right
        qubits,qregs=tensor_bits(True);clbits,cregs=tensor_bits(False)
        result=QuantumCircuit(qubits,clbits)
        for register in qregs+cregs:result.add_register(register)
        result.compose(other,range(other.num_qubits),range(other.num_bits),inplace=True)
        result.compose(self,range(other.num_qubits,result.num_qubits),range(other.num_bits,result.num_bits),inplace=True)
        return _replace_self(self,result) if inplace else result

    def decompose(self,gates_to_decompose=None,reps=1):
        from .gates import decompose
        from .circuit_data import public_factory_entries,commit_entries
        admit((self,gates_to_decompose,reps))
        lowered=decompose(self,gates_to_decompose,reps)
        from .circuit_metadata import inherit_execution
        result=QuantumCircuit(name=lowered.name,metadata=self.metadata)
        copy_bits(self,result);inherit_execution(lowered,result)
        result.global_phase=lowered.global_phase
        commit_entries(result,public_factory_entries(result,lowered))
        return result

    @classmethod
    def from_instructions(cls,instructions,*,qubits=(),clbits=(),name=None,global_phase=0,metadata=None):
        admit((instructions,qubits,clbits,name,global_phase))
        contexts=[]
        for value in bounded_contexts(instructions):
            contexts.append(value if type(value) is CircuitInstruction else CircuitInstruction(*value))
        circuit=cls(name=name,global_phase=global_phase,metadata=metadata)
        circuit.add_bits(bounded_items(qubits)+bounded_items(clbits))
        for context in contexts:
            fresh=[bit for bit in context.qubits+context.clbits if is_bit(bit) and bit not in circuit._qubits+circuit._clbits]
            if fresh:circuit.add_bits(fresh)
        from .circuit_data import entry_from_context,commit_entries
        commit_entries(circuit,[entry_from_context(circuit,context,check_width=False) for context in contexts])
        return circuit

    @property
    def has_measurements(self):
        return bool(self.measurements) or any(op.name=="measure" or type(op.metadata.get("instruction_object")) is Instruction
            and op.metadata["instruction_object"].definition is not None and op.metadata["instruction_object"].definition.has_measurements for op in self.operations)
    @property
    def has_dynamic_operations(self):
        return Circuit.has_dynamic_operations.fget(self) or any(type(op.metadata.get("instruction_object")) is Instruction
            and op.metadata["instruction_object"].definition is not None
            and (op.metadata["instruction_object"].definition.has_dynamic_operations or op.metadata["instruction_object"].definition.has_measurements) for op in self.operations)
