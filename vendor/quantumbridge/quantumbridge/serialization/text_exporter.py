# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# Lossless supported-subset source generation with explicit unsupported semantics.
from copy import deepcopy
from dataclasses import replace
from numbers import Real
import math
import warnings

from quantumbridge.core import Circuit, Measurement, Operation, Parameter, ParameterExpression
from quantumbridge.core.classical import ClassicalExpression
from quantumbridge.core.control_flow import validate_control_flow
from quantumbridge.compiler.traversal import append_item, map_control_bodies
from quantumbridge.compiler.decomposition import GateDecompositionPass
from .errors import UnsupportedFeature, FormatValidationError
from .common import register_specs
from .limits import SerializationLimits
from quantumbridge.core.circuit_metadata import execution_metadata, empty_execution_circuit, serialized_user_metadata

STANDARD = {"id","h","x","y","z","s","sdg","t","tdg","rx","ry","rz","phase","u","u1","u2","u3","cx","cy","cz","ch","swap","ccx","cswap","cp"}
CLASSICAL_OPS = {"eq":"==","ne":"!=","lt":"<","le":"<=","gt":">","ge":">=","and":"&","or":"|","xor":"^","add":"+","sub":"-","mul":"*","shl":"<<","shr":">>","logical_and":"&&","logical_or":"||"}


def identifier(name):
    return isinstance(name,str) and name and (name[0].isalpha() or name[0] == "_") and all(char.isalnum() or char == "_" for char in name)


def _definition_preflight(circuit,limits):
    from quantumbridge.core.gates import validate_definition
    pending,definitions,operations = [(circuit,0,frozenset())],0,0
    while pending:
        body,depth,ancestors = pending.pop()
        register_specs(body)
        if "classical_scope" in execution_metadata(body):
            raise UnsupportedFeature("OpenQASM export does not yet preserve typed classical identifier scopes; these are not discardable metadata.",format="qasm")
        limits.check("max_nesting_depth",depth,format="qasm")
        if id(body) in ancestors: raise FormatValidationError("Cyclic source definition graph.",format="qasm")
        operations += len(body.operations)+len(body.measurements)
        limits.check("max_instructions",operations,format="qasm")
        ancestors = ancestors | {id(body)}
        for op in body.operations:
            if "instruction_object" in op.metadata:
                raise UnsupportedFeature("OpenQASM does not yet preserve native Instruction reference objects; explicitly lower a supported executable definition first.", format="qasm")
            if "annotated_operation" in op.metadata:
                raise UnsupportedFeature("Native OpenQASM export does not preserve AnnotatedOperation objects; explicitly resolve to a Gate first.", format="qasm")
            gate = op.metadata.get("gate")
            if gate is not None and gate.definition is not None:
                definitions += 1
                limits.check("max_custom_definitions",definitions,format="qasm")
                validate_definition(gate.definition,owner=gate)
                pending.append((gate.definition,depth+1,ancestors))
            node = op.metadata.get("control_node")
            if node is not None:
                pending.extend((child,depth+1,ancestors) for child in node.bodies)


def _expand_sx(circuit):
    out = empty_execution_circuit(circuit)
    for item in circuit.iter_instructions():
        if isinstance(item,Measurement): append_item(out,item)
        elif item.metadata.get("control_node") is not None: append_item(out,map_control_bodies(item,_expand_sx))
        elif "gate" in item.metadata: append_item(out,item)
        elif item.name in {"sx","sxdg"}:
            for name in ("h","s" if item.name == "sx" else "sdg","h"):
                out.operations.append(Operation(name,item.targets,metadata=deepcopy(item.metadata)))
        else: append_item(out,item)
    return out


class SourceExporter:
    def __init__(self,circuit,version,limits=None,allow_metadata_loss=False):
        self.version,self.format = str(version),"qasm2" if str(version).startswith("2") else "qasm3"
        self.limits = limits or SerializationLimits()
        self.allow_metadata_loss = allow_metadata_loss
        _definition_preflight(circuit,self.limits)
        self.specs = register_specs(circuit)
        from .control_normalization import normalize_source_control
        circuit = normalize_source_control(circuit,self.limits)
        validate_control_flow(circuit)
        self.wires = {"q":{},"c":{}}
        for spec in self.specs:
            if not identifier(spec["name"]): self.fail("Register name is not a source identifier.")
            for offset,wire in enumerate(spec["bits"]): self.wires[spec["kind"]][wire] = f'{spec["name"]}[{offset}]'
        self.variables,self.parameters,self.used = {},{},set(spec["name"] for spec in self.specs)
        from .text_lowering import RESERVED, STANDARD_NAMES3, QELIB_NAMES
        reserved = RESERVED | {"U","gphase"} | (QELIB_NAMES if self.format == "qasm2" else STANDARD_NAMES3)
        if self.used & reserved: self.fail("Register identifier conflicts with a keyword or standard-library gate.")
        self.used.update(reserved)
        for parameter in circuit.parameters:
            if not identifier(parameter.name) or parameter.name in self.used: self.fail("Unrepresentable or conflicting parameter identifier.")
            if parameter._uuid_explicit: self.metadata_loss("Parameter UUID cannot be carried by OpenQASM source.")
            self.parameters[parameter] = parameter.name; self.used.add(parameter.name)
        if self.format == "qasm2" and self.parameters: self.fail("OpenQASM 2 requires numeric bound parameters.")
        self.circuit = self.primitive_source(circuit)
        self.emitted,self.byte_count = 0,0
        self.gate_names,self.gate_declarations,self.active_gates = {},[],set()
        self.gate_objects = {}

    def primitive_source(self,circuit):
        names = set(STANDARD)
        def collect(body):
            for op in body.operations:
                if "gate" in op.metadata: names.add(op.name)
                node = op.metadata.get("control_node")
                if node is not None:
                    for child in node.bodies: collect(child)
        collect(circuit)
        return GateDecompositionPass(names).run(_expand_sx(circuit)).circuit

    def gate_reference(self,gate):
        from quantumbridge.core.gates import Gate, ControlledGate, validate_definition
        if id(gate) in self.active_gates: self.fail("Recursive custom Gate definition.")
        if id(gate) in self.gate_names: return self.gate_names[id(gate)]
        from quantumbridge.core.gates import is_native_gate
        if not is_native_gate(gate) or gate.definition is None:
            self.fail("Source custom Gate declarations require a native executable definition.")
        from quantumbridge.core.standard_gates import standard_type_name, standard_from_record
        from quantumbridge.core.instruction_parameters import definition_signature
        from quantumbridge.core.gates import ControlledGate
        standard = standard_type_name(gate)
        if standard is not None and gate.name in STANDARD | {"p"}:
            expected = standard_from_record(standard, tuple(gate.params), gate.label,
                gate.ctrl_state if isinstance(gate, ControlledGate) else None)
            if (gate.name == expected.name and not gate.metadata
                    and definition_signature(gate.definition) == definition_signature(expected.definition)):
                name = "u3" if gate.name == "u" else ("u1" if self.format == "qasm2" else "p") if gate.name in {"p", "phase"} else gate.name
                return name, tuple(gate.params)
        validate_definition(gate.definition,owner=gate)
        self.limits.check("max_nesting_depth",len(self.active_gates)+1,format=self.format)
        self.limits.check("max_custom_definitions",len(self.gate_names)+1,format=self.format)
        name = "qb_gate_"+str(len(self.gate_names))
        while name in self.used: name += "_"
        self.used.add(name)
        parameters = gate.definition.parameters
        self.gate_names[id(gate)] = (name,parameters)
        self.gate_objects[id(gate)] = gate
        self.active_gates.add(id(gate))
        if gate.metadata: self.metadata_loss("OpenQASM cannot carry custom Gate object metadata.")
        previous = self.wires
        self.wires = {"q":dict(enumerate("w"+str(w) for w in range(gate.num_qubits))),"c":{}}
        formal = {parameter:"theta"+str(i) for i,parameter in enumerate(parameters)}
        try: body = self.block(self.primitive_source(gate.definition),1,formal,_definition=True)
        finally:
            self.wires = previous
            self.active_gates.remove(id(gate))
        header = "gate "+name+("("+",".join(formal.values())+")" if formal else "")
        header += " "+",".join("w"+str(w) for w in range(gate.num_qubits))+" {"
        self.gate_declarations.extend([header,*body,"}"])
        return name,parameters

    def fail(self,reason): raise UnsupportedFeature(reason,format=self.format,version=self.version)
    def metadata_loss(self,reason):
        if not self.allow_metadata_loss: self.fail(reason+" Use an external sidecar or explicitly allow metadata loss.")
        warnings.warn(reason,UserWarning,stacklevel=3)

    def metadata(self,circuit):
        from quantumbridge.core.circuit_metadata import is_separated
        metadata=serialized_user_metadata(circuit)
        if not is_separated(circuit):metadata.pop("control_flow",None)
        if metadata: self.metadata_loss("OpenQASM has no standard fields for circuit user metadata.")
        if circuit.name is not None: self.metadata_loss("OpenQASM has no circuit-name object field.")

    def angle(self,value,loop_parameters=None,depth=0):
        self.limits.check("max_expression_depth",depth,format=self.format)
        loop_parameters = loop_parameters or {}
        if isinstance(value,Real):
            if not math.isfinite(value): self.fail("Nonfinite gate parameter.")
            return format(float(value),".17g")
        if isinstance(value,Parameter):
            if value in loop_parameters: return loop_parameters[value]
            if value not in self.parameters: self.fail("Unbound/unscoped source parameter.")
            return self.parameters[value]
        if isinstance(value,ParameterExpression):
            if value.op in {"const","parameter"}: return self.angle(value.args[0],loop_parameters,depth+1)
            if value.op == "neg": return "(-"+self.angle(value.args[0],loop_parameters,depth+1)+")"
            if value.op in {"sin","cos","tan","exp","log","arcsin","arccos","arctan","abs"}:
                name = "ln" if value.op == "log" else value.op
                return name+"("+self.angle(value.args[0],loop_parameters,depth+1)+")"
            if value.op not in {"+","-","*","/","**"}: self.fail("Unsupported parameter expression.")
            result = "("+self.angle(value.args[0],loop_parameters,depth+1)+value.op+self.angle(value.args[1],loop_parameters,depth+1)+")"
            self.limits.check("max_input_bytes",len(result.encode()),format=self.format)
            return result
        self.fail("Unsupported gate parameter type.")

    def dtype(self,dtype):
        return dtype.kind if dtype.kind in {"bit","bool"} else f"{dtype.kind}[{dtype.width}]"

    def variable_name(self,expr):
        key,name = expr.value
        if key not in self.variables:
            name = name if identifier(name) and name not in self.used else "qb_var_"+str(len(self.variables))
            while name in self.used: name += "_"
            self.variables[key] = name; self.used.add(name)
        return self.variables[key]

    def classical(self,expr):
        if not isinstance(expr,ClassicalExpression): self.fail("Expected executable typed classical expression.")
        if expr.op == "literal": return str(expr.value).lower()
        if expr.op == "var": return self.variable_name(expr)
        if expr.op == "bit": return self.wires["c"][expr.value]
        if expr.op == "register":
            for spec in self.specs:
                if spec["kind"] == "c" and tuple(spec["bits"]) == tuple(expr.value): return spec["name"]
            if self.format == "qasm3" and len(expr.value) == 1: return "uint[1]("+self.wires["c"][expr.value[0]]+")"
            self.fail("Noncontiguous/reordered classical register views cannot be emitted losslessly yet.")
        if expr.op == "cast": return self.dtype(expr.dtype)+"("+self.classical(expr.args[0])+")"
        if expr.op in {"not","invert"}: return "("+("!" if expr.op == "not" else "~")+self.classical(expr.args[0])+")"
        if expr.op in CLASSICAL_OPS: return "("+self.classical(expr.args[0])+CLASSICAL_OPS[expr.op]+self.classical(expr.args[1])+")"
        self.fail("Unsupported typed classical source expression.")

    def block(self,circuit,indent=0,loop_parameters=None,*,_definition=False):
        if not _definition: self.metadata(circuit)
        elif serialized_user_metadata(circuit):
            self.metadata_loss("OpenQASM cannot carry custom definition circuit metadata.")
        exporter = self
        class Lines(list):
            def append(self,value):
                exporter.byte_count += len(value.encode())+1
                exporter.limits.check("max_input_bytes",exporter.byte_count,format=exporter.format)
                super().append(value)
        prefix,lines = "  "*indent,Lines()
        phase = circuit.global_phase
        if not isinstance(phase,Real) or phase != 0:
            if self.format == "qasm2":
                if not isinstance(phase,Real) or math.remainder(phase,math.tau) != 0: self.fail("Nonzero global_phase is not representable losslessly in OpenQASM 2.")
            else: lines.append(prefix+f"gphase({self.angle(phase,loop_parameters)});")
        for item in circuit.iter_instructions():
            self.emitted += 1; self.limits.check("max_instructions",self.emitted,format=self.format)
            if "annotated_operation" in item.metadata:
                self.fail("OpenQASM cannot preserve an AnnotatedOperation object; explicitly resolve it to a Gate before export.")
            if any(key in item.metadata for key in ("control_flow_stack","control_flow")):
                self.fail("Legacy flattened conditional blocks must be normalized to structured ControlBlock before source export.")
            extra = set(item.metadata)-{"operation_index","control_node","condition","gate_condition","directive","duration","unit","gate"}
            if extra: self.metadata_loss("OpenQASM cannot carry instruction metadata: "+", ".join(sorted(extra)))
            condition = item.metadata.get("gate_condition",item.metadata.get("condition"))
            if condition is not None:
                from quantumbridge.core.control_flow import condition_expression
                body = Circuit(circuit.num_qubits,circuit.num_bits)
                append_item(body,replace(item,metadata={key:value for key,value in item.metadata.items() if key not in {"condition","gate_condition"}}))
                node = ControlBlock("if",{"condition":condition_expression(condition),"body":body,"alternate":None})
            else: node = item.metadata.get("control_node")
            if node is not None:
                data = node.data
                if self.format == "qasm2":
                    if node.kind != "if" or data["alternate"] is not None: self.fail("OpenQASM 2 conditional export requires an if(creg==integer) subset with no else.")
                    expr = data["condition"]
                    if expr.op != "eq" or expr.args[0].op not in {"register","bit"} or expr.args[1].op != "literal": self.fail("QASM2 conditional must test a complete classical register.")
                    bits = tuple(expr.args[0].value) if expr.args[0].op == "register" else (expr.args[0].value,)
                    name = next((spec["name"] for spec in self.specs if spec["kind"] == "c" and tuple(spec["bits"]) == bits),None)
                    if name is None: self.fail("QASM2 conditional needs a declared complete creg.")
                    if data["body"].has_dynamic_operations: self.fail("QASM2 cannot preserve a block snapshot if its body writes classical state.")
                    for line in self.block(data["body"],0,loop_parameters): lines.append(prefix+f"if({name}=={expr.args[1].value}) "+line)
                    continue
                if node.kind in {"declare","store"}:
                    target = self.classical(data["target"])
                    value = data.get("initial") if node.kind == "declare" else data["value"]
                    lines.append(prefix+(self.dtype(data["target"].dtype)+" " if node.kind == "declare" else "")+target+(" = "+self.classical(value) if value is not None else "")+";")
                elif node.kind in {"break","continue"}: lines.append(prefix+node.kind+";")
                elif node.kind in {"if","while"}:
                    lines.append(prefix+node.kind+" ("+self.classical(data["condition"])+") {")
                    lines.extend(self.block(data["body"],indent+1,loop_parameters)); lines.append(prefix+"}")
                    if node.kind == "if" and data["alternate"] is not None:
                        lines.append(prefix+"else {"); lines.extend(self.block(data["alternate"],indent+1,loop_parameters)); lines.append(prefix+"}")
                elif node.kind == "for":
                    from quantumbridge.core.classical import variable,Int
                    var = data["variable"] if data["variable"] is not None else variable("qb_loop",Int())
                    name = self.variable_name(var)
                    mapping = {**(loop_parameters or {})}
                    if data["parameter"] is not None: mapping[data["parameter"]] = name
                    lines.append(prefix+f"for {self.dtype(var.dtype)} {name} in "+"{"+", ".join(map(str,data["values"]))+"} {")
                    lines.extend(self.block(data["body"],indent+1,mapping)); lines.append(prefix+"}")
                elif node.kind == "switch":
                    if self.version != "3.1": self.fail("Switch export requires OpenQASM 3.1.")
                    lines.append(prefix+"switch ("+self.classical(data["selector"])+") {")
                    for values,body in data["cases"]:
                        lines.append(prefix+"  case "+", ".join(map(str,values))+" {"); lines.extend(self.block(body,indent+2,loop_parameters)); lines.append(prefix+"  }")
                    if data["default"] is not None:
                        lines.append(prefix+"  default {"); lines.extend(self.block(data["default"],indent+2,loop_parameters)); lines.append(prefix+"  }")
                    lines.append(prefix+"}")
                else: self.fail("Unsupported control node.")
                continue
            if "gate" in item.metadata:
                gate = item.metadata["gate"]
                from quantumbridge.core.standard_gates import invocation_params
                if item.controls or len(item.targets) != gate.num_qubits or item.params != invocation_params(gate):
                    self.fail("Custom Gate invocation differs from its definition.")
                name,parameters = self.gate_reference(gate)
                line = name+("("+",".join(self.angle(value,loop_parameters) for value in parameters)+")" if parameters else "")
                line += " "+",".join(self.wires["q"][wire] for wire in item.wires)+";"
            elif isinstance(item,Measurement):
                if item.kind != "computational": self.fail("Unsupported measurement basis cannot be represented as ordinary measure.")
                q,c = self.wires["q"][item.wire],self.wires["c"][item.bit]
                line = f"measure {q} -> {c};" if self.format == "qasm2" else f"{c} = measure {q};"
            elif item.name == "delay":
                if self.format == "qasm2": self.fail("OpenQASM 2 cannot preserve delay semantics.")
                line = "delay["+self.angle(item.params[0],loop_parameters)+item.metadata.get("unit","dt")+"] "+",".join(self.wires["q"][wire] for wire in item.wires)+";"
            elif item.name == "reset": line = "reset "+self.wires["q"][item.targets[0]]+";"
            elif item.name == "barrier": line = "barrier "+",".join(self.wires["q"][wire] for wire in item.wires)+";"
            else:
                if item.name not in STANDARD: self.fail("Unsupported source instruction "+item.name)
                name,params = item.name,item.params
                if name in {"u","u3"}: name = "u3"
                if name in {"phase","u1"}: name = "u1" if self.format == "qasm2" else "p"
                if name == "cp" and self.format == "qasm2": name = "cu1"
                line = name+("("+",".join(self.angle(value,loop_parameters) for value in params)+")" if params else "")+" "+",".join(self.wires["q"][wire] for wire in item.wires)+";"
            lines.append(prefix+line)
        return lines

    def dumps(self):
        lines = [f"OPENQASM {self.version};",'include "qelib1.inc";' if self.format == "qasm2" else 'include "stdgates.inc";']
        if self.format == "qasm2":
            used = set()
            def collect(body):
                for op in body.operations:
                    used.add(op.name)
                    if "gate" in op.metadata and op.metadata["gate"].definition is not None:
                        collect(op.metadata["gate"].definition)
                    node = op.metadata.get("control_node")
                    if node is not None:
                        for child in node.bodies: collect(child)
            collect(self.circuit)
            if "swap" in used: lines.append("gate swap a,b { cx a,b; cx b,a; cx a,b; }")
            if "cswap" in used: lines.append("gate cswap c,a,b { cx b,a; ccx c,a,b; cx b,a; }")
        body = self.block(self.circuit)
        lines.extend(self.gate_declarations)
        if self.format == "qasm3": lines.extend(f"input float[64] {name};" for name in self.parameters.values())
        for spec in self.specs:
            kind = ("qreg" if spec["kind"] == "q" else "creg") if self.format == "qasm2" else ("qubit" if spec["kind"] == "q" else "bit")
            lines.append(f'{kind} {spec["name"]}[{len(spec["bits"])}];' if self.format == "qasm2" else f'{kind}[{len(spec["bits"])}] {spec["name"]};')
        lines.extend(body)
        result = "\n".join(lines)+"\n"
        self.limits.check("max_input_bytes",len(result.encode()),format=self.format)
        return result


from quantumbridge.core.control_flow import ControlBlock
