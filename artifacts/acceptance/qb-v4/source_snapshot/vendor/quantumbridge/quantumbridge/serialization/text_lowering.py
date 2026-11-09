# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# Typed source AST lowering into the existing Circuit/ControlBlock interpreter.
from dataclasses import dataclass
import math
import operator
from numbers import Real, Integral

from quantumbridge.core import Circuit, Operation, Parameter, ParameterExpression
from quantumbridge.core.classical import (ClassicalExpression, Bool, Bit, UInt, Int, bit, register,
    lift, cast, binary, logical_not, logical_and, logical_or, variable)
from quantumbridge.core.control_flow import ControlBlock, condition_expression, validate_control_flow
from .errors import ParseError, UnsupportedFeature
from .qpy import OPS, CONTROLS

ARITHMETIC = {"+":operator.add,"-":operator.sub,"*":operator.mul,"/":operator.truediv,"%":operator.mod,"**":operator.pow,
              "<<":operator.lshift,">>":operator.rshift,"&":operator.and_,"|":operator.or_,"^":operator.xor,
              "==":operator.eq,"!=":operator.ne,"<":operator.lt,">":operator.gt,"<=":operator.le,">=":operator.ge}
CLASSICAL = {"+":"add","-":"sub","*":"mul","<<":"shl",">>":"shr","&":"and","|":"or","^":"xor",
             "==":"eq","!=":"ne","<":"lt",">":"gt","<=":"le",">=":"ge"}
ALIASES = {"U":"u","CX":"cx","p":"phase","cphase":"cp","cu1":"cp","cnot":"cx"}
QELIB_NAMES = {"u3","u2","u1","cx","id","u0","x","y","z","h","s","sdg","t","tdg","rx","ry","rz","cz","cy","ch","ccx","crz","cu1","cu3"}
STANDARD_NAMES3 = frozenset("p x y z h s sdg t tdg sx rx ry rz cx cy cz cp crx cry crz ch cu swap ccx cswap CX phase cphase id u1 u2 u3".split())
RESERVED = frozenset("OPENQASM include gate opaque qreg creg qubit bit bool uint int float angle const input output if else while for in switch case default break continue measure reset barrier delay true false pi tau euler sin cos tan exp ln log sqrt arcsin arccos arctan abs defcal cal extern def box stretch array let pragma".split())


@dataclass(frozen=True)
class Wires:
    kind: str
    indices: tuple


class Lowerer:
    def __init__(self,program,limits,strict=True,dialect="standard"):
        self.program,self.limits = program,limits
        self.format = "qasm2" if program.version.startswith("2") else "qasm3"
        self.registers,self.definitions,self.expansion,self.loop_count = {},{},0,0
        self.strict,self.standard_library = strict,False
        self.dialect = dialect

    def error(self,reason,node,cls=ParseError):
        raise cls(reason,format=self.format,version=self.program.version,line=node.line,column=node.column)

    def declaration_name(self,name,node,*,local=False):
        standard = name in QELIB_NAMES if self.format == "qasm2" else name in STANDARD_NAMES3
        if name in RESERVED or name in {"U","CX","gphase"} or not local and self.standard_library and standard:
            self.error("Declaration conflicts with a reserved or standard-library name.",node)

    def declaration_type(self,data,env,node):
        name = data["type"]
        width = self.integer(self.expr(data["width"],env),node) if data["width"] is not None else 32
        if name in {"float","angle"}:
            if name != "float" or width != 64: self.error("Only explicitly sized float[64] constants/inputs are implemented; fixed-point angles are unsupported.",node,UnsupportedFeature)
            return None
        if name == "bool" and data["width"] is not None: self.error("Bool has no width parameter.",node)
        return Bool() if name == "bool" else Bit() if name == "bit" and data["width"] is None else UInt(width) if name in {"uint","bit"} else Int(width) if name == "int" else self.error("Unsupported declaration type.",node,UnsupportedFeature)

    def constant(self,data,env,node):
        dtype = self.declaration_type(data,env,node)
        if data["value"] is None: self.error("Const requires an initializer.",node)
        value = self.expr(data["value"],env)
        if isinstance(value,(ClassicalExpression,Wires,Parameter)) or hasattr(value,"parameters"): self.error("Const initializer is not compile-time.",node)
        if dtype is None:
            if not isinstance(value,Real) or isinstance(value,bool) or not math.isfinite(value): self.error("Invalid float[64] constant.",node)
            return float(value)
        return dtype.normalize(value)

    def expr(self,node,env,*,angle_env=None,depth=0):
        self.limits.check("max_expression_depth",depth,format=self.format,line=node.line,column=node.column)
        def child(value): return self.expr(value,env,angle_env=angle_env,depth=depth+1)
        if node.op == "literal":
            if isinstance(node.value,float) and not math.isfinite(node.value): self.error("Nonfinite source literal.",node)
            return node.value
        if node.op == "string":
            if not node.value or any(c not in "01_" for c in node.value): self.error("Only binary bit-string literals are supported here.",node,UnsupportedFeature)
            return int(node.value.replace("_",""),2)
        if node.op == "name":
            if angle_env is not None and node.value in angle_env: return angle_env[node.value]
            if node.value in {"pi","tau","euler"}: return {"pi":math.pi,"tau":math.tau,"euler":math.e}[node.value]
            if node.value not in env: self.error(f"Undefined name {node.value!r}.",node)
            return env[node.value]
        if node.op == "index":
            value,index = map(child,node.args)
            if not isinstance(value,Wires) or isinstance(index,bool) or not isinstance(index,Integral) or not 0 <= index < len(value.indices):
                self.error("Invalid or out-of-range register index.",node)
            return Wires(value.kind,(value.indices[index],))
        if node.op == "call":
            (name,width_node),args = node.value,[child(arg) for arg in node.args]
            if len(args) != 1: self.error("Function/cast requires one supported argument.",node)
            value = self.runtime_value(args[0],node)
            if name in {"bool","bit","uint","int"}:
                width = self.integer(child(width_node),node) if width_node is not None else 32
                dtype = Bool() if name == "bool" else Bit() if name == "bit" and width_node is None else UInt(width) if name in {"bit","uint"} else Int(width)
                if isinstance(value,ClassicalExpression): return cast(value,dtype)
                return dtype.normalize(bool(value) if name == "bool" else int(value),wrap=True)
            functions = {"sin":math.sin,"cos":math.cos,"tan":math.tan,"exp":math.exp,"ln":math.log,"log":math.log,"sqrt":math.sqrt,
                         "arcsin":math.asin,"arccos":math.acos,"arctan":math.atan,"abs":abs}
            if name not in functions: self.error("Unsupported symbolic function or cast.",node,UnsupportedFeature)
            if isinstance(value,(Parameter,ParameterExpression)):
                return value**.5 if name == "sqrt" else getattr(value,"log" if name == "ln" else name)()
            if not isinstance(value,Real): self.error("Unsupported symbolic function or cast.",node,UnsupportedFeature)
            return functions[name](value)
        if node.op == "unary":
            value = self.runtime_value(child(node.args[0]),node)
            if node.value == "+": return value
            if node.value == "!": return logical_not(value) if isinstance(value,ClassicalExpression) else not value
            if node.value == "~": return ~value
            if isinstance(value,ClassicalExpression): return binary("sub",lift(0,value.dtype),value)
            return -value
        if node.op == "binary":
            left,right = (self.runtime_value(child(arg),node) for arg in node.args)
            op = node.value
            symbolic = isinstance(left,ClassicalExpression) or isinstance(right,ClassicalExpression)
            if op in {"&&","||"}:
                if symbolic: return (logical_and if op == "&&" else logical_or)(left,right)
                return bool(left) and bool(right) if op == "&&" else bool(left) or bool(right)
            if symbolic:
                if op not in CLASSICAL: self.error("This runtime integer operator is not implemented.",node,UnsupportedFeature)
                if not isinstance(left,ClassicalExpression): left = lift(left,right.dtype)
                if not isinstance(right,ClassicalExpression): right = lift(right,left.dtype)
                return binary(CLASSICAL[op],left,right)
            if op in {"**","<<",">>"} and isinstance(right,Real) and abs(right) > 4096: self.error("Arithmetic exponent/shift exceeds the numeric budget.",node)
            if op == "**" and isinstance(left,Integral) and isinstance(right,Integral) and right > 0 and max(1,left.bit_length())*right > 4096:
                self.error("Integer power exceeds the numeric budget.",node)
            try: result = ARITHMETIC[op](left,right)
            except (ArithmeticError,ValueError,TypeError): self.error("Invalid source arithmetic.",node)
            if isinstance(result,complex) or isinstance(result,float) and not math.isfinite(result): self.error("Non-real or nonfinite expression result.",node)
            if isinstance(result,Integral) and result.bit_length() > 4096: self.error("Integer result exceeds the numeric budget.",node)
            return result
        self.error("Unsupported expression node.",node,UnsupportedFeature)

    def runtime_value(self,value,node):
        if isinstance(value,Wires):
            if value.kind != "c": self.error("Quantum wires cannot be used as a classical value.",node)
            return bit(value.indices[0]) if len(value.indices) == 1 else register(value.indices)
        return value

    def integer(self,value,node):
        if isinstance(value,bool) or not isinstance(value,Integral): self.error("Expected a compile-time integer.",node)
        return int(value)

    def gate_signature(self,literal,node):
        if literal in self.definitions:
            definition = self.definitions[literal]
            return literal,len(definition["wires"]),len(definition["parameters"])
        name = ALIASES.get(literal,literal)
        if self.strict and not self.standard_library and literal not in {"U","CX"}: self.error(f"Source does not support gate {name!r} without its standard include.",node,UnsupportedFeature)
        if self.strict and self.format == "qasm2" and literal not in QELIB_NAMES|{"U","CX"}: self.error(f"Gate {literal!r} needs an explicit QASM2 definition.",node,UnsupportedFeature)
        if self.strict and self.format == "qasm3" and literal not in STANDARD_NAMES3|{"U"}: self.error(f"Gate {literal!r} needs an explicit QASM3 definition.",node,UnsupportedFeature)
        if name not in OPS or name in {"measure","reset","barrier","delay"}: self.error(f"Source does not support gate {name!r}.",node,UnsupportedFeature)
        return name,OPS[name][1],OPS[name][2]

    def validate_definition_body(self,data,node):
        # Validate each direct call once, without expanding previously admitted bodies.
        from quantumbridge.core.gate_limits import current_gate_limits
        gate_limits = current_gate_limits()
        gate_limits.check("max_gate_parameters",len(data["parameters"]))
        gate_limits.check("max_circuit_bits",len(data["wires"]))
        gate_limits.check("max_custom_definitions",len(self.definitions)+1)
        self.limits.check("max_qubits",len(data["wires"]),format=self.format)
        self.limits.check("max_parameter_nodes",len(data["parameters"]),format=self.format)
        scope = {name:Parameter(name) for name in data["parameters"]}
        for stmt in data["body"] or ():
            self.expansion += 1
            self.limits.check("max_gate_expansion",self.expansion,format=self.format,line=stmt.line,column=stmt.column)
            for arg in stmt.data["args"]:
                if arg.op != "name" or arg.value not in data["wires"]:
                    self.error("Gate bodies require unindexed local formal qubits.",arg)
            if stmt.kind == "barrier": continue
            literal = stmt.data["name"]
            if literal == data["name"]: self.error("Recursive gate definition.",stmt)
            _,width,arity = self.gate_signature(literal,stmt)
            if len(stmt.data["params"]) != arity or len(stmt.data["args"]) != width:
                self.error("Gate signature/resource mismatch inside definition.",stmt)
            if len({arg.value for arg in stmt.data["args"]}) != width:
                self.error("Duplicate gate qubit operand inside definition.",stmt)
            for value in stmt.data["params"]:
                angle = self.expr(value,scope)
                if not isinstance(angle,(Real,Parameter,ParameterExpression)):
                    self.error("Gate parameters must be real expressions of local formals.",value)
        memo = {}
        def footprint(definition,depth):
            self.limits.check("max_nesting_depth",depth,format=self.format)
            gate_limits.check("max_definition_depth",depth+1)
            name = definition["name"]
            if name not in memo:
                height,work = 1,0
                for stmt in definition["body"] or ():
                    self.expansion += 1
                    self.limits.check("max_gate_expansion",self.expansion,format=self.format)
                    work += 1
                    if stmt.kind == "gate" and stmt.data["name"] in self.definitions:
                        child_height,child_work = footprint(self.definitions[stmt.data["name"]],depth+1)
                        height,work = max(height,child_height+1),work+child_work
                    self.limits.check("max_gate_expansion",work,format=self.format)
                    gate_limits.check("max_total_operations",work)
                    gate_limits.check("max_decomposition_work",work)
                memo[name] = height,work
            height,work = memo[name]
            self.limits.check("max_nesting_depth",depth+height-1,format=self.format)
            gate_limits.check("max_definition_depth",depth+height)
            return height,work
        footprint(data,0)

    def collect_registers(self):
        sizes,constants,declared = {"q":0,"c":0},{},set()
        specs = []
        for stmt in self.program.statements:
            data = stmt.data
            if stmt.kind == "declare" and data["qualifier"] == "const":
                self.declaration_name(data["name"],stmt)
                if data["name"] in declared: self.error("Duplicate source declaration.",stmt)
                constants[data["name"]] = self.constant(data,constants,stmt); declared.add(data["name"])
            if stmt.kind == "register" or stmt.kind == "declare" and data["type"] in {"qubit","bit"} and data["qualifier"] is None:
                if data["name"] in declared: self.error("Duplicate source declaration.",stmt)
                declared.add(data["name"])
                kind = data.get("kind", "q" if data.get("type") == "qubit" else "c")
                expr = data.get("size",data.get("width"))
                size = 1 if expr is None else self.integer(self.expr(expr,constants),stmt)
                if size < 1: self.error("Register size must be positive.",stmt)
                self.limits.check("max_qubits" if kind == "q" else "max_clbits",sizes[kind]+size,format=self.format,line=stmt.line,column=stmt.column)
                wires = tuple(range(sizes[kind],sizes[kind]+size)); sizes[kind] += size
                self.registers[data["name"]] = Wires(kind,wires)
                specs.append({"kind":kind,"name":data["name"],"bits":list(wires)})
        self.limits.check("max_registers",len(specs),format=self.format)
        if not sizes["q"]: self.error("A qreg/qubit declaration is required.",self.program.statements[0] if self.program.statements else type("Location",(),{"line":1,"column":1})())
        return sizes,specs

    def coerce(self,value,dtype,node):
        value = self.runtime_value(value,node)
        if isinstance(value,ClassicalExpression):
            return value if value.dtype == dtype else cast(value,dtype)
        if dtype.kind == "bool" and isinstance(value,Integral): value = bool(value)
        return lift(value,dtype)

    def emit(self,statements,circuit,env,constants=None,angle_env=None,call_stack=(),depth=0):
        constants,angle_env = set(constants or ()),dict(angle_env or {})
        local_names = set()
        def expression(node,angle=False): return self.expr(node,env,angle_env=angle_env if angle else None)
        def block(body,scope=None,angles=None):
            result = Circuit(circuit.num_qubits,circuit.num_bits)
            self.emit(body,result,dict(env) if scope is None else scope,constants,angle_env if angles is None else angles,call_stack,depth+1)
            return result
        for stmt in statements:
            self.limits.check("max_nesting_depth",depth,format=self.format,line=stmt.line,column=stmt.column)
            self.expansion += 1
            self.limits.check("max_gate_expansion",self.expansion,format=self.format,line=stmt.line,column=stmt.column)
            self.limits.check("max_instructions",len(circuit.operations)+len(circuit.measurements),format=self.format)
            kind,data = stmt.kind,stmt.data
            if kind == "empty": continue
            if kind == "include":
                if data["name"] != ("qelib1.inc" if self.format == "qasm2" else "stdgates.inc"): self.error("Incorrect standard include for this version.",stmt,UnsupportedFeature)
                if self.format == "qasm2" and (self.standard_library or (set(env)|set(self.definitions)) & QELIB_NAMES):
                    self.error("Duplicate standard include or conflicting earlier declaration.",stmt)
                self.standard_library = True; continue
            if self.format == "qasm2" and kind in {"declare","for","while","switch","break","continue","assignment","delay"}:
                self.error("This source feature requires OpenQASM 3.",stmt,UnsupportedFeature)
            if kind == "definition":
                if depth: self.error("Gate definitions require global scope.",stmt)
                if data["body"] is not None and any(child.kind not in {"gate","barrier"} for child in data["body"]): self.error("Gate definitions may contain gate calls and barriers only.",stmt,UnsupportedFeature)
                name = data["name"]
                self.declaration_name(name,stmt,local=True)
                reserved = name in {"U","CX","gphase"} or self.standard_library and (name in QELIB_NAMES if self.format == "qasm2" else name in STANDARD_NAMES3)
                if name in self.definitions or reserved or name in env: self.error("Duplicate/reserved gate definition.",stmt)
                if len(set(data["parameters"]+data["wires"])) != len(data["parameters"]+data["wires"]): self.error("Duplicate gate formal argument.",stmt)
                for formal in data["parameters"]+data["wires"]: self.declaration_name(formal,stmt,local=True)
                self.limits.check("max_custom_definitions",len(self.definitions)+1,format=self.format)
                if self.format == "qasm2": self.validate_definition_body(data,stmt)
                self.definitions[name] = data; continue
            if kind == "register" or kind == "declare" and data["type"] in {"qubit","bit"} and data["qualifier"] is None and depth == 0:
                name = data["name"]
                self.declaration_name(name,stmt)
                if name in env or name in self.definitions: self.error("Duplicate register name.",stmt)
                env[name] = self.registers[name]
                if kind == "declare" and data["value"] is not None:
                    if data["measure"]: self.measure(circuit,expression(data["value"]),env[name],stmt)
                    elif env[name].kind == "q": self.error("Qubit initializers are unsupported.",stmt,UnsupportedFeature)
                    else:
                        target = self.runtime_value(env[name],stmt)
                        circuit.store(target,self.coerce(expression(data["value"]),target.dtype,stmt))
                continue
            if kind == "declare":
                name,dtype = data["name"],data["type"]
                self.declaration_name(name,stmt,local=depth>0)
                if name in local_names or depth == 0 and name in env: self.error("Duplicate variable declaration.",stmt)
                local_names.add(name)
                if data["qualifier"] == "const":
                    value = self.constant(data,env,stmt)
                    angle_env.pop(name,None)
                    env[name] = value; constants.add(name); continue
                if data["qualifier"] == "input":
                    if depth or dtype != "float" or data["value"] is not None: self.error("Only global float[64] gate-parameter inputs without initializers are implemented.",stmt,UnsupportedFeature)
                    self.declaration_type(data,env,stmt)
                    env[name] = Parameter(name); constants.add(name); continue
                if data["qualifier"] is not None: self.error("Output qualifiers are not implemented.",stmt,UnsupportedFeature)
                native_type = self.declaration_type(data,env,stmt)
                if native_type is None: self.error("Unsupported runtime variable type.",stmt,UnsupportedFeature)
                target = variable(name,native_type)
                initial = None if data["value"] is None else self.coerce(expression(data["value"]),native_type,stmt)
                angle_env.pop(name,None); constants.discard(name)
                env[name] = target
                circuit._control(ControlBlock("declare",{"target":target,"initial":initial}))
                continue
            if kind == "assignment":
                if data["target"].op == "name" and data["target"].value in constants: self.error("Cannot assign to const/input.",stmt)
                target = self.runtime_value(expression(data["target"]),stmt)
                if data["measure"]:
                    self.measure(circuit,expression(data["value"]),expression(data["target"]),stmt); continue
                if not isinstance(target,ClassicalExpression) or target.op not in {"bit","register","var"}: self.error("Invalid assignment target.",stmt)
                value = 1 if data["operator"] in {"++","--"} else self.runtime_value(expression(data["value"]),stmt)
                if data["operator"] != "=":
                    op = "+" if data["operator"] == "++" else "-" if data["operator"] == "--" else data["operator"][0]
                    if op not in CLASSICAL: self.error("Unsupported runtime compound assignment.",stmt,UnsupportedFeature)
                    value = binary(CLASSICAL[op],target,self.coerce(value,target.dtype,stmt))
                circuit.store(target,self.coerce(value,target.dtype,stmt)); continue
            if kind == "if":
                condition = self.runtime_value(expression(data["condition"]),stmt)
                if self.format == "qasm2":
                    node = data["condition"]
                    if node.op != "binary" or node.value != "==" or node.args[0].op != "name" or not isinstance(env.get(node.args[0].value),Wires) or env[node.args[0].value].kind != "c" or data["alternate"] is not None:
                        self.error("QASM2 condition requires if(creg == integer) and no else.",stmt,UnsupportedFeature)
                circuit._control(ControlBlock("if",{"condition":condition_expression(condition),"body":block(data["body"]),"alternate":None if data["alternate"] is None else block(data["alternate"])})); continue
            if kind == "while": circuit._control(ControlBlock("while",{"condition":condition_expression(self.runtime_value(expression(data["condition"]),stmt)),"body":block(data["body"])})); continue
            if kind == "for":
                self.declaration_name(data["name"],stmt,local=True)
                values = tuple(self.integer(expression(value),stmt) for value in data["sequence"][1])
                if data["sequence"][0] == "range":
                    start,stop = values[0],values[-1]; step = values[1] if len(values) == 3 else 1
                    if not step: self.error("Loop range step cannot be zero.",stmt)
                    sequence = range(start,stop+(1 if step>0 else -1),step)
                    self.limits.check("max_instructions",len(sequence),format=self.format)
                    values = tuple(sequence)
                width = self.integer(expression(data["width"]),stmt) if data["width"] is not None else 32
                if data["type"] not in {"uint","int"}: self.error("For-loop variables require int/uint.",stmt,UnsupportedFeature)
                var = variable(data["name"],UInt(width) if data["type"] == "uint" else Int(width))
                self.loop_count += 1; parameter = Parameter(f"__qasm_loop_{self.loop_count}")
                scope,angles = {**env,data["name"]:var},{**angle_env,data["name"]:parameter}
                child = Circuit(circuit.num_qubits,circuit.num_bits)
                self.emit(data["body"],child,scope,constants-{data["name"]},angles,call_stack,depth+1)
                circuit._control(ControlBlock("for",{"values":values,"variable":var,"parameter":parameter,"body":child})); continue
            if kind == "switch":
                if self.program.version != "3.1": self.error("Switch requires the declared OpenQASM 3.1 dialect.",stmt,UnsupportedFeature)
                if not data["cases"]: self.error("Switch requires at least one case.",stmt)
                selector = self.runtime_value(expression(data["selector"]),stmt)
                cases = tuple((tuple(self.integer(expression(value),stmt) for value in values),block(body)) for values,body in data["cases"])
                circuit._control(ControlBlock("switch",{"selector":lift(selector),"cases":cases,"default":None if data["default"] is None else block(data["default"])})); continue
            if kind in {"break","continue"}: circuit._control(ControlBlock(kind,{})); continue
            if kind == "measure": self.measure(circuit,expression(data["source"]),None if data["target"] is None else expression(data["target"]),stmt); continue
            if kind in {"reset","barrier","delay"}:
                wires = []
                for arg in data["args"]:
                    item = expression(arg)
                    if not isinstance(item,Wires) or item.kind != "q": self.error("Quantum instruction needs qubit operands.",stmt)
                    wires.extend(item.indices)
                if kind == "barrier": circuit.barrier(*wires)
                elif kind == "reset":
                    for wire in wires: circuit.reset(wire)
                else:
                    duration = expression(data["value"])
                    if not isinstance(duration,Real) or duration < 0 or not math.isfinite(duration) or data["unit"] not in {"dt","s","ms","us","ns","ps"}: self.error("Unsupported/invalid duration literal.",stmt)
                    for wire in wires: circuit.delay(duration,wire,unit=data["unit"])
                continue
            if kind == "gate":
                name = data["name"] if data["name"] in self.definitions else ALIASES.get(data["name"],data["name"])
                params = tuple(expression(value,True) for value in data["params"])
                if any(isinstance(value,(ClassicalExpression,Wires)) for value in params): self.error("Runtime gate-angle arithmetic is not implemented outside bound for-loop parameters.",stmt,UnsupportedFeature)
                if name == "gphase":
                    if self.format == "qasm2" or len(params) != 1 or data["args"]: self.error("Invalid global-phase instruction.",stmt,UnsupportedFeature)
                    circuit.global_phase = circuit.global_phase+params[0]; continue
                operands = [expression(value) for value in data["args"]]
                if any(not isinstance(value,Wires) or value.kind != "q" for value in operands): self.error("Gate operands must be quantum wires.",stmt)
                name,width,arity = self.gate_signature(data["name"],stmt)
                if len(params) != arity or len(operands) != width: self.error("Gate signature/resource mismatch.",stmt)
                count = max((len(value.indices) for value in operands),default=0)
                if not count or any(len(value.indices) not in (1,count) for value in operands): self.error("Mismatched gate register broadcast.",stmt)
                for index in range(count):
                    self.expansion += 1
                    self.limits.check("max_gate_expansion",self.expansion,format=self.format,line=stmt.line,column=stmt.column)
                    self.limits.check("max_instructions",len(circuit.operations)+len(circuit.measurements)+1,format=self.format)
                    wires = tuple(value.indices[0 if len(value.indices)==1 else index] for value in operands)
                    if len(set(wires)) != len(wires): self.error("Duplicate gate qubit operand.",stmt)
                    if name in self.definitions:
                        definition = self.definitions[name]
                        if definition["body"] is None: self.error("Opaque gate calls have no executable definition.",stmt,UnsupportedFeature)
                        if name in call_stack: self.error("Recursive gate definition.",stmt)
                        if len(params) != len(definition["parameters"]) or len(wires) != len(definition["wires"]): self.error("Custom gate argument count mismatch.",stmt)
                        scope = {**dict(zip(definition["parameters"],params)),**{name:Wires("q",(wire,)) for wire,name in enumerate(definition["wires"])}}
                        body = Circuit(len(wires))
                        self.emit(definition["body"],body,scope,set(scope),{},call_stack+(name,),depth+1)
                        if any(op.metadata.get("directive") for op in body.operations):
                            # Source barriers are legal, but a unitary Gate
                            # object must not acquire directive semantics.
                            circuit.compose(body,qubits=wires,inplace=True)
                        else:
                            from quantumbridge.core.gates import Gate
                            circuit.append(Gate(name,len(wires),params,definition=body),wires)
                    else:
                        ctrl = CONTROLS.get(name,0)
                        circuit.append(name,wires[ctrl:],controls=wires[:ctrl],params=params)
                        if self.format == "qasm3" and data["name"] == "U" and self.dialect == "standard":
                            circuit.global_phase = circuit.global_phase+sum(params)/2
                continue
            self.error(f"Unhandled source statement {kind!r}.",stmt,UnsupportedFeature)

    def measure(self,circuit,source,target,stmt):
        if not isinstance(source,Wires) or source.kind != "q" or not isinstance(target,Wires) or target.kind != "c" or len(source.indices) != len(target.indices):
            self.error("Measurement requires matching qubit/classical register operands.",stmt,UnsupportedFeature)
        for wire,bit_index in zip(source.indices,target.indices): circuit.measure(wire,bit_index)

    def lower(self):
        sizes,specs = self.collect_registers()
        circuit = Circuit(sizes["q"],sizes["c"],metadata={"registers":specs})
        from quantumbridge.core.bits import QuantumRegister,ClassicalRegister,install_bits
        qregs=tuple(QuantumRegister(len(spec["bits"]),spec["name"]) for spec in specs if spec["kind"]=="q")
        cregs=tuple(ClassicalRegister(len(spec["bits"]),spec["name"]) for spec in specs if spec["kind"]=="c")
        install_bits(circuit,tuple(bit for reg in qregs for bit in reg),tuple(bit for reg in cregs for bit in reg),qregs,cregs)
        self.emit(self.program.statements,circuit,{})
        # Validate even unused definitions: dead source must not hide cycles/typos.
        visited,active = set(),set()
        def definition(name,depth=0):
            self.limits.check("max_nesting_depth",depth,format=self.format)
            if name in active: raise ParseError("Recursive gate definition.",format=self.format)
            if name in visited: return
            active.add(name)
            for stmt in self.definitions[name]["body"] or ():
                if stmt.kind != "gate": continue
                literal = stmt.data["name"]
                child = literal if literal in self.definitions else ALIASES.get(literal,literal)
                if child in self.definitions: definition(child,depth+1)
                elif child == "gphase" and self.format == "qasm3":
                    if len(stmt.data["params"]) != 1 or stmt.data["args"]:
                        self.error("Invalid global-phase instruction in definition.",stmt)
                elif child not in OPS: self.error("Unknown gate inside definition.",stmt,UnsupportedFeature)
            active.remove(name); visited.add(name)
        for name in self.definitions: definition(name)
        count = 0
        def budget(body):
            nonlocal count
            for item in body.iter_instructions():
                count += 1
                self.limits.check("max_instructions",count,format=self.format)
                node = item.metadata.get("control_node")
                if node is not None:
                    for child in node.bodies: budget(child)
                gate = item.metadata.get("gate")
                if gate is not None and gate.definition is not None: budget(gate.definition)
        budget(circuit)
        validate_control_flow(circuit)
        return circuit
