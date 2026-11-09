# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# Independent public QPY v13 binary protocol implementation, not an SDK wrapper.

import io
import json
import math
import struct
from dataclasses import dataclass
from uuid import UUID, uuid5, NAMESPACE_URL
from numbers import Real, Integral
import numpy as np

from quantumbridge.core import Circuit, Instruction, Operation, Measurement, Parameter, ParameterExpression, ParameterVector, ParameterVectorElement
from quantumbridge.core.control_flow import validate_control_flow
from .limits import SerializationLimits
from .errors import SerializationError, FormatValidationError, UnsupportedFeature, UnsupportedVersion, ResourceLimitError
from .common import read_input, write_output, register_specs, BoundedParts

QPY_VERSION = 13
FILE_HEADER = struct.Struct("!6sBBBBQc")
CIRCUIT_HEADER = struct.Struct("!HcHIIQIQI")
REGISTER = struct.Struct("!c?IH?")
INSTRUCTION = struct.Struct("!HHHIIBHqII")
VALUE = struct.Struct("!cQ")
LAYOUT = struct.Struct("!ciiiIi")
CUSTOM_INSTRUCTION = struct.Struct("!HcII?QIIQ")
MODIFIER = struct.Struct("!cIId")
OPS = {"id":("IGate",1,0), "x":("XGate",1,0), "y":("YGate",1,0), "z":("ZGate",1,0),
       "h":("HGate",1,0), "s":("SGate",1,0), "sdg":("SdgGate",1,0), "t":("TGate",1,0),
       "tdg":("TdgGate",1,0), "sx":("SXGate",1,0), "sxdg":("SXdgGate",1,0),
       "rx":("RXGate",1,1), "ry":("RYGate",1,1), "rz":("RZGate",1,1),
       "phase":("PhaseGate",1,1), "u":("UGate",1,3), "u1":("U1Gate",1,1),
       "u2":("U2Gate",1,2), "u3":("U3Gate",1,3),
       "cx":("CXGate",2,0), "cy":("CYGate",2,0), "cz":("CZGate",2,0), "ch":("CHGate",2,0), "csx":("CSXGate",2,0),
       "crx":("CRXGate",2,1), "cry":("CRYGate",2,1), "crz":("CRZGate",2,1), "cp":("CPhaseGate",2,1),
       "swap":("SwapGate",2,0), "ccx":("CCXGate",3,0), "cswap":("CSwapGate",3,0),
       "ccz":("CCZGate",3,0), "cu":("CUGate",2,4),
       "rxx":("RXXGate",2,1), "ryy":("RYYGate",2,1), "rzz":("RZZGate",2,1),
       "unitary":("UnitaryGate",None,1),
       "reset":("Reset",1,0), "measure":("Measure",1,0), "barrier":("Barrier",None,0), "delay":("Delay",1,1)}
NAMES = {value[0]:name for name,value in OPS.items()}
DYNAMIC_CONTROLS = {"MCPhaseGate": ("phase", "mcphase", 1), "MCXGate": ("x", "mcx", 0)}
CONTROLS = {"cx":1,"cy":1,"cz":1,"ch":1,"csx":1,"crx":1,"cry":1,"crz":1,"cp":1,"ccx":2,"ccz":2,"cu":1,"cswap":1}
CONTROL_BASES = {"cx":"x", "cy":"y", "cz":"z", "ch":"h", "csx":"sx", "crx":"rx", "cry":"ry", "crz":"rz", "cp":"phase", "ccx":"x", "ccz":"z", "cu":"u", "cswap":"swap"}


@dataclass(frozen=True)
class _CustomRecord:
    name: str
    width: int
    kind: bytes
    definition: object
    controls: int
    state: int
    base: object
    payload_depth: int
    clbits: int = 0


class Reader:
    def __init__(self,data,limits,base=0,depth=0,budget=None):
        self.data, self.pos, self.limits, self.base = memoryview(data), 0, limits, base
        self.instructions = 0
        self.parameters = {}
        self.parameter_identities = {}
        self.vectors = {}
        self.depth,self.budget = depth,({"values":0} if budget is None else budget)
        self.max_depth = depth
        self.circuit_owner = None
        self.loop_depth = self.parameter_loop_depth = 0
        self.classical_vars = []
        self.variable_identities = {}
        self.enclosing_variables = {}
        self.nested_control_block = False
        self.bounded("max_nesting_depth",depth)

    def error(self,reason,cls=FormatValidationError):
        raise cls(reason,format="qpy",version=13,offset=self.base+self.pos)

    def canonical_parameter_graph(self, circuit):
        from quantumbridge.core.parameter_graph import canonical_parameter_graph
        return canonical_parameter_graph(circuit,self.parameters)

    def take(self,size):
        if not isinstance(size,int) or size < 0 or size > len(self.data)-self.pos: self.error("Truncated or invalid binary payload length.")
        start = self.pos; self.pos += size
        return bytes(self.data[start:self.pos])

    def unpack(self,fmt):
        fmt = fmt if isinstance(fmt,struct.Struct) else struct.Struct(fmt)
        return fmt.unpack(self.take(fmt.size))

    def text(self,size):
        try: return self.take(size).decode("utf-8")
        except UnicodeDecodeError: self.error("Invalid UTF-8 payload.")

    def bounded(self,field,value): self.limits.check(field,value,format="qpy",version=13,offset=self.base+self.pos)

    def admit_standard_object(self):
        # The object and its owned executable definition both require work;
        # admit this minimum before invoking any formal-binding constructor.
        self.budget["expanded"] = self.budget.get("expanded",0)+2
        self.bounded("max_gate_expansion",self.budget["expanded"])

    def dynamic_controlled_operation(self,key,label,params,width,nctrl,state,clbits):
        base_name,display,nparams = DYNAMIC_CONTROLS[key]
        if (nctrl < 1 or width != nctrl+1 or clbits or len(params) != nparams
                or not 0 <= state < 1<<nctrl):
            self.error("Dynamic controlled Gate signature differs from its header.")
        if any(not isinstance(value,(Real,Parameter,ParameterExpression)) or isinstance(value,bool) for value in params):
            self.error("Non-numeric dynamic controlled Gate parameter.")
        from quantumbridge.core.gate_limits import current_gate_limits
        limits = current_gate_limits()
        limits.check("max_control_qubits",nctrl)
        if nctrl+1 >= limits.max_matrix_dimension.bit_length():
            raise MemoryError("Dynamic controlled Gate exceeds max_matrix_dimension before construction.")
        self.budget["expanded"] = self.budget.get("expanded",0)+2*nctrl+2
        self.admit_standard_object()
        from quantumbridge.core.standard_gates import PhaseGate,XGate
        from quantumbridge.core.gates import ControlledGate
        base = PhaseGate(*params) if base_name == "phase" else XGate()
        result = ControlledGate(base,nctrl,state,label or None)
        result._restore_display_name(display+("_o"+str(state) if state != (1<<nctrl)-1 else ""))
        return result

    def base_instruction(self):
        name_size,label_size,nparams,nq,nc,condition,condition_size,condition_value,nctrl,state = self.unpack(INSTRUCTION)
        self.bounded("max_parameter_nodes",nparams)
        self.bounded("max_qubits",nq)
        self.bounded("max_clbits",nc)
        if condition or condition_size: self.error("Unsupported custom base instruction condition fields.",UnsupportedFeature)
        name,label = self.text(name_size),self.text(label_size)
        # A base-gate record has the usual header and parameters, but no wire
        # argument records: it is not an invocation on a containing circuit.
        params = tuple(self.value(*self.unpack(VALUE)) for _ in range(nparams))
        if self.pos != len(self.data): self.error("Trailing bytes in custom base instruction.")
        return name,label,params,nq,nctrl,state,nc

    def base_operation(self,description,table,ancestors):
        from quantumbridge.core.gates import Gate, ControlledGate
        key,label,params,width,nctrl,state,clbits = description
        if key in table:
            record = table[key]
            if width != record.width or clbits != record.clbits or (nctrl,state) != (record.controls,record.state):
                self.error("Custom base operation header differs from its referenced definition.")
            return self.custom_operation(key,params,label,table,ancestors)
        if key in DYNAMIC_CONTROLS:
            return self.dynamic_controlled_operation(key,label,params,width,nctrl,state,clbits)
        if key not in NAMES: self.error("Unresolved custom base operation reference.",UnsupportedFeature)
        name = NAMES[key]
        if (OPS[name][1] is not None and OPS[name][1] != width or OPS[name][2] != len(params)
                or clbits != int(name=="measure") or nctrl != CONTROLS.get(name,0) or state >= 1<<nctrl):
            self.error("Standard base operation signature differs from its header.")
        if name in {"reset","measure","barrier","delay"}:
            metadata={"directive":True} if name in {"barrier","delay"} else {}
            if name=="delay":metadata.update(unit="dt",duration=params[0])
            return Instruction(name,width,clbits,params,label or None,metadata=metadata)
        from quantumbridge.core.standard_gates import STANDARD_GATE_TYPES, standard_from_record
        if key in {kind.__name__ for kind in STANDARD_GATE_TYPES}:
            self.admit_standard_object()
            result = standard_from_record(key, params, label or None, state if nctrl else None)
            if result.num_qubits != width: self.error("Standard base width differs from its parameters.")
            return result
        if nctrl:
            base_name = CONTROL_BASES[name]
            base_width = width-nctrl
            body = Circuit(base_width).append(base_name,range(base_width),params=params)
            result = ControlledGate(Gate(base_name,base_width,params,definition=body),nctrl,state,label or None)
            result._restore_display_name(name+("_o"+str(state) if state != (1<<nctrl)-1 else ""))
            return result
        body = Circuit(width).append(name,range(width),params=params)
        return Gate(name,width,params,label or None,definition=body)

    def custom_operation(self,key,params,label,table,ancestors=()):
        from quantumbridge.core.gates import Gate, ControlledGate, validate_definition
        if key in ancestors: self.error("Cyclic custom base-gate reference.")
        record = table[key]
        self.bounded("max_nesting_depth",self.depth+len(ancestors)+record.payload_depth)
        display,width,definition,nctrl,state,base = (record.name,record.width,record.definition,record.controls,record.state,record.base)
        if record.kind == b"a":
            from quantumbridge.core.annotated_operation import AnnotatedOperation, InverseModifier, ControlModifier, PowerModifier
            if label: self.error("Annotation labels are not represented by the native operation protocol.",UnsupportedFeature)
            if any(type(value) not in (InverseModifier,ControlModifier,PowerModifier) for value in params):
                self.error("An annotated invocation requires typed Modifier parameters.")
            self.budget["expanded"] = self.budget.get("expanded",0)+1+len(params)
            self.bounded("max_gate_expansion",self.budget["expanded"])
            result = AnnotatedOperation(self.base_operation(base,table,(*ancestors,key)),list(params))
            if result.num_qubits != width or result.num_clbits != record.clbits:self.error("Annotation modifier widths disagree with its definition header.")
            return result
        from quantumbridge.core.operations import instruction_definition_work
        work = (1 if definition is None else instruction_definition_work(definition)) if record.kind == b"i" else validate_definition(definition)
        self.budget["expanded"] = self.budget.get("expanded",0)+work
        self.bounded("max_gate_expansion",self.budget["expanded"])
        if record.kind == b"i" and definition is None:
            return Instruction(display, width, record.clbits,params=params, label=label or None)
        formal = definition.parameters
        from quantumbridge.core.parameters import parameter_set
        if not set(formal) <= set(parameter_set(params)):
            if len(formal) != len(params): self.error("Custom Gate formal parameter mapping is ambiguous.",UnsupportedFeature)
            definition = definition.assign_parameters(dict(zip(formal,params)))
        if record.kind == b"i":
            metadata = dict(definition.metadata)
            metadata.setdefault("definition_name",definition.name)
            return Instruction(display,width,record.clbits,params=params,label=label or None,definition=definition,metadata=metadata)
        if base is None: return Gate(display,width,params,label or None,definition=definition)
        base_gate = self.base_operation(base,table,(*ancestors,key))
        if not isinstance(base_gate,Gate): self.error("A ControlledGate base record must represent a Gate.")
        if base_gate.num_qubits+nctrl != width or not _same_parameters(base_gate.params,params):
            self.error("Controlled custom Gate base width/parameters differ from its header.")
        result = ControlledGate(base_gate,nctrl,state,label or None)
        # v13 stores the closed-control definition; ctrl_state is applied at
        # the object boundary, not already baked into that definition.
        actual = Circuit(width)
        opened = [wire for wire in range(nctrl) if not (state>>wire)&1]
        for wire in opened: actual.x(wire)
        actual = actual.compose(definition)
        for wire in reversed(opened): actual.x(wire)
        result._restore_display_name(display)
        result.definition = actual
        return result

    def parameter_identity(self, name, identity, kind, vector_identity=None):
        previous = self.parameters.get(name)
        if previous is not None and (type(previous) is not kind or previous.uuid != identity):
            self.error("Conflicting parameter identities sharing a name.",UnsupportedFeature)
        descriptor = (name,kind,vector_identity)
        if identity in self.parameter_identities and self.parameter_identities[identity] != descriptor:
            self.error("Conflicting names or parameter kinds for one UUID identity.")
        self.parameter_identities[identity] = descriptor
        return previous

    def parameter(self):
        length,identity = self.unpack("!H16s")
        name = self.text(length)
        if not name: self.error("Empty parameter name.")
        identity = str(UUID(bytes=identity))
        value = self.parameter_identity(name,identity,Parameter)
        if value is None:
            value = Parameter(name,identity)
            self.parameters[name] = value
        return value

    def parameter_vector(self):
        name_size,length,identity,index=self.unpack("!HQ16sQ")
        name=self.text(name_size)
        if not length or index>=length:self.error("Invalid parameter vector size or index.")
        self.bounded("max_parameter_nodes",length)
        identity=UUID(bytes=identity)
        root=UUID(int=(identity.int-index)%(1<<128))
        previous = self.parameter_identity(f"{name}[{index}]",str(identity),ParameterVectorElement,
                                           (str(root),length,index))
        key=(name,str(root),length)
        if key not in self.vectors:
            count=self.budget.get("vector_elements",0)+length
            self.bounded("max_parameter_nodes",count)
            vector=ParameterVector(name)
            vector._root_uuid=str(root)
            vector.resize(length)
            self.vectors[key]=vector
            self.budget["vector_elements"]=count
        value=self.vectors[key][index]
        if previous is not None:return previous
        object.__setattr__(value,"_uuid_explicit",True)
        self.parameters[value.name]=value
        return value

    def value(self,kind,size):
        self.budget["values"] += 1
        self.bounded("max_parameter_nodes",self.budget["values"])
        start = self.base+self.pos
        sub = Reader(self.take(size),self.limits,start,self.depth+1,self.budget)
        sub.parameters = self.parameters
        sub.parameter_identities = self.parameter_identities
        sub.vectors = self.vectors
        sub.circuit_owner = self.circuit_owner
        sub.loop_depth = self.loop_depth
        sub.parameter_loop_depth = self.parameter_loop_depth
        sub.classical_vars = self.classical_vars
        sub.variable_identities = self.variable_identities
        sub.enclosing_variables = self.enclosing_variables
        if kind == b"f":
            value, = sub.unpack("<d")
            if not math.isfinite(value): sub.error("Nonfinite numeric parameter.")
        elif kind == b"i": value, = sub.unpack("<q")
        elif kind == b"p": value = sub.parameter()
        elif kind == b"v": value=sub.parameter_vector()
        elif kind == b"s": value = sub.text(size)
        elif kind == b"z": value = None
        elif kind == b"d":
            from quantumbridge.core.control_flow_operations import CASE_DEFAULT
            value = CASE_DEFAULT
        elif kind == b"R":
            from .qpy_control_records import read_resource
            value = read_resource(sub, sub.text(size))
        elif kind == b"r":
            from .qpy_control_records import admit_range
            start_, stop_, step = sub.unpack("!qqq")
            if not step:
                sub.error("A QPY range step cannot be zero.")
            value = range(start_, stop_, step)
            admit_range(value, self.limits, self.budget)
        elif kind == b"t":
            count, = sub.unpack("!Q")
            sub.bounded("max_parameter_nodes", count+self.budget["values"])
            if count*VALUE.size > len(sub.data)-sub.pos:
                sub.error("QPY sequence count exceeds its parameter payload.")
            value = tuple(sub.value(*sub.unpack(VALUE)) for _ in range(count))
        elif kind == b"q":
            from .qpy_classical import lexical_variables
            sub.enclosing_variables = lexical_variables(self.circuit_owner, self.enclosing_variables)
            value = sub.circuit(_control_block=True, _loop_depth=self.parameter_loop_depth)
            for symbol in sub.parameters.values():
                previous = self.parameter_identity(symbol.name, symbol.uuid, type(symbol),
                    (symbol.vector._root_uuid, len(symbol.vector), symbol.index) if type(symbol) is ParameterVectorElement else None)
                if previous is None:
                    self.parameters[symbol.name] = symbol
        elif kind == b"x":
            from .qpy_classical import read_expression
            value = read_expression(sub)
        elif kind == b"e":
            from .qpy_expressions import read_expression
            value = read_expression(sub)
        elif kind == b"n":
            stream = io.BytesIO(bytes(sub.data))
            try:
                version = np.lib.format.read_magic(stream)
                if version not in ((1,0),(2,0)): sub.error("Unsupported NumPy value format.",UnsupportedFeature)
                shape, order, dtype = (np.lib.format.read_array_header_1_0(stream) if version == (1,0)
                    else np.lib.format.read_array_header_2_0(stream))
            except (ValueError, EOFError) as exc: sub.error("Malformed NumPy value header: "+str(exc))
            if (len(shape) != 2 or any(type(x) is not int or x < 1 for x in shape)
                    or dtype.hasobject or dtype.kind not in "biufc"):
                sub.error("Only explicit numeric matrix parameter values are supported.",UnsupportedFeature)
            count = shape[0]*shape[1]
            total = self.budget.get("numeric_elements",0)+count
            sub.bounded("max_numeric_elements",total)
            self.budget["numeric_elements"] = total
            if stream.tell()+count*dtype.itemsize != len(sub.data): sub.error("NumPy value length differs from its matrix shape.")
            from quantumbridge.core.gate_limits import current_gate_limits
            current_gate_limits().check("max_matrix_dimension",max(shape))
            matrix = np.frombuffer(sub.data, dtype=dtype, offset=stream.tell()).reshape(shape, order="F" if order else "C")
            value = _unitary_wire_order(matrix)
            sub.pos = len(sub.data)
        elif kind == b"m":
            from quantumbridge.core.annotated_operation import InverseModifier, ControlModifier, PowerModifier
            modifier,nctrl,state,power = sub.unpack(MODIFIER)
            self.budget["modifiers"] = self.budget.get("modifiers",0)+1
            self.bounded("max_gate_expansion",self.budget["modifiers"])
            if modifier == b"i": value = InverseModifier()
            elif modifier == b"c":
                sub.bounded("max_qubits",nctrl)
                value = ControlModifier(nctrl,state)
            elif modifier == b"p":
                if not math.isfinite(power): sub.error("A modifier power must be finite.")
                value = PowerModifier(power)
            else: sub.error("Unknown QPY modifier type.",UnsupportedFeature)
        else: sub.error(f"Unsupported parameter type {kind!r}.",UnsupportedFeature)
        if sub.pos != len(sub.data): sub.error("Invalid parameter payload size.")
        self.max_depth = max(self.max_depth,sub.max_depth)
        return value

    def circuit(self, *, _control_block=False, _loop_depth=0):
        self.loop_depth = self.parameter_loop_depth = _loop_depth
        self.nested_control_block = _control_block
        self.parameters = {}
        self.parameter_identities = {}
        self.vectors = {}
        name_size,phase_type,phase_size,nq,nc,metadata_size,nreg,ninst,nvars = self.unpack(CIRCUIT_HEADER)
        for field,value in (("max_qubits",nq),("max_clbits",nc),("max_metadata_bytes",metadata_size),("max_registers",nreg),("max_instructions",ninst)):
            self.bounded(field,value)
        self.bounded("max_parameter_nodes", nvars)
        name = self.text(name_size) if name_size else None
        self.budget["instructions"] = self.budget.get("instructions",0)+ninst
        self.bounded("max_instructions",self.budget["instructions"])
        # Global-phase scalars use network order, unlike instruction scalars in v13.
        phase = self.unpack("!d")[0] if phase_type == b"f" and phase_size == 8 else self.value(phase_type,phase_size)
        def nonfinite(value): raise ValueError("Nonfinite metadata literal.")
        try: metadata = json.loads(self.text(metadata_size),parse_constant=nonfinite)
        except (ValueError,RecursionError): self.error("Invalid JSON circuit metadata.")
        if not isinstance(metadata,dict): self.error("Circuit metadata must be an object.")
        stack = [(metadata,0)]
        while stack:
            value,depth = stack.pop(); self.bounded("max_nesting_depth",depth)
            if isinstance(value,(dict,list)):
                stack.extend((child,depth+1) for child in (value.values() if isinstance(value,dict) else value))
        specs = [];register_work=0
        for _ in range(nreg):
            raw = self.take(REGISTER.size)
            if raw[1] not in (0,1) or raw[-1] not in (0,1): self.error("Invalid canonical register boolean flags.")
            kind,standalone,size,length,in_circuit = REGISTER.unpack(raw)
            if kind not in {b"q",b"c"}: self.error("Invalid register type.")
            self.bounded("max_qubits" if kind == b"q" else "max_clbits",size)
            register_work+=size
            self.bounded("max_gate_expansion",self.budget.get("register_bits",0)+register_work)
            from quantumbridge.core.gate_limits import current_gate_limits
            current_gate_limits().check("max_decomposition_work",register_work)
            reg_name = self.text(length)
            bits = list(self.unpack("!"+"q"*size))
            specs.append({"kind":kind.decode(),"name":reg_name,"bits":bits,
                "standalone":standalone,"in_circuit":in_circuit})
        circuit = Circuit(nq,nc,name,metadata)
        circuit._execution_metadata={}
        circuit.global_phase = phase
        from .qpy_registers import restore_register_records
        restore_register_records(circuit,specs,self.limits,self.budget)
        self.circuit_owner = circuit
        from .qpy_classical import read_declarations
        read_declarations(self, circuit, nvars)
        from .qpy_control_records import CONTROL_NAMES
        custom_count, = self.unpack("!Q")
        self.budget["definitions"] = self.budget.get("definitions",0)+custom_count
        self.bounded("max_custom_definitions",self.budget["definitions"])
        custom = {}
        for _ in range(custom_count):
            raw = self.take(CUSTOM_INSTRUCTION.size)
            if raw[11] not in (0,1): self.error("Invalid custom definition flag.")
            size,kind,width,clbits,has_definition,payload_size,nctrl,state,base_size = CUSTOM_INSTRUCTION.unpack(raw)
            self.bounded("max_qubits",width)
            self.bounded("max_clbits",clbits)
            key = self.text(size)
            if not key or key in custom or key in NAMES or key in DYNAMIC_CONTROLS or key in CONTROL_NAMES: self.error("Conflicting QPY custom definition identifier.")
            if kind not in {b"g",b"c",b"a",b"i"} or clbits and kind not in {b"a",b"i"}:
                self.error("This QPY custom record type/control payload is not implemented.",UnsupportedFeature)
            if kind in {b"g",b"i"} and (nctrl or state or base_size):
                self.error("A non-controlled definition contains coherent control fields.")
            if kind == b"c" and (not 0 < nctrl < width or state >= 1<<nctrl or not base_size):
                self.error("Invalid custom ControlledGate control fields.")
            definition,payload_depth = None,0
            if kind == b"a":
                if has_definition or payload_size or nctrl or state or not base_size:
                    self.error("An annotation requires a base-operation payload, not a synthesized Gate definition.")
            elif kind == b"i" and not has_definition and not payload_size:
                pass
            else:
                if not has_definition or not payload_size:
                    self.error("Opaque custom QPY definitions are not executable.",UnsupportedFeature)
                offset = self.base+self.pos
                sub = Reader(self.take(payload_size),self.limits,offset,self.depth+1,self.budget)
                definition = sub.circuit()
                if sub.pos != len(sub.data): sub.error("Trailing bytes in QPY custom definition.")
                if definition.num_qubits != width or definition.num_bits != clbits:
                    self.error("Custom Gate definition width differs from its header.")
                from quantumbridge.core.gates import validate_definition
                from quantumbridge.core.operations import instruction_definition_work
                instruction_definition_work(definition) if kind == b"i" else validate_definition(definition)
                payload_depth = sub.max_depth-self.depth
                self.max_depth = max(self.max_depth,sub.max_depth)
            # Public v13 uses a UUID suffix for the serialization-local key;
            # it is not part of the user's Gate name.
            display, separator, suffix = key.rpartition("_")
            try: is_uuid = separator and len(suffix) in (32,36) and UUID(suffix)
            except ValueError: is_uuid = False
            display = display if is_uuid else key
            if kind == b"a" and display != "annotated": self.error("Nonstandard annotation display name.")
            base = None
            if base_size:
                offset = self.base+self.pos
                sub = Reader(self.take(base_size),self.limits,offset,self.depth+1,self.budget)
                sub.parameters,sub.vectors = self.parameters,self.vectors
                sub.parameter_identities = self.parameter_identities
                base = sub.base_instruction()
                payload_depth = max(payload_depth,sub.max_depth-self.depth)
                self.max_depth = max(self.max_depth,sub.max_depth)
            custom[key] = _CustomRecord(display,width,kind,definition,nctrl,state,base,payload_depth,clbits)
        for key in custom:
            seen = set()
            current = key
            while current in custom and custom[current].base is not None:
                if current in seen: self.error("Cyclic custom base-gate reference.")
                seen.add(current)
                self.budget["references"] = self.budget.get("references",0)+1
                self.bounded("max_gate_expansion",self.budget["references"])
                self.bounded("max_nesting_depth",self.depth+len(seen))
                current = custom[current].base[0]
            if current not in custom and current not in NAMES and current not in DYNAMIC_CONTROLS:
                self.error("Unresolved custom base-gate reference.",UnsupportedFeature)
        for _ in range(ninst):
            name_size,label_size,nparams,nqargs,ncargs,condition,condition_length,condition_value,nctrl,ctrl_state = self.unpack(INSTRUCTION)
            if nqargs > nq or ncargs > nc: self.error("Instruction arity exceeds circuit resources.")
            self.bounded("max_parameter_nodes",nparams)
            class_name, label = self.text(name_size), self.text(label_size)
            classical_condition = None
            if condition == 1:
                from .qpy_control_records import read_resource
                classical_condition = (read_resource(self, self.text(condition_length)), condition_value)
            elif condition == 2 and not condition_length:
                kind, size = self.unpack(VALUE)
                if kind != b"x":
                    self.error("QPY conditional expression requires an expression parameter record.")
                classical_condition = self.value(kind, size)
            elif condition or condition_length:
                self.error("Unsupported QPY13 conditional flag.",UnsupportedFeature)
            if classical_condition is not None and class_name not in {"IfElseOp", "WhileLoopOp"}:
                self.error("Conditional non-control QPY instructions are not implemented.", UnsupportedFeature)
            if class_name not in NAMES and class_name not in custom and class_name not in DYNAMIC_CONTROLS and class_name not in CONTROL_NAMES: self.error(f"Unknown instruction class {class_name!r}.",UnsupportedFeature)
            op_name = NAMES.get(class_name)
            qargs,cargs = [],[]
            for index in range(nqargs+ncargs):
                kind,wire = self.unpack("!cI")
                if kind != (b"q" if index < nqargs else b"c") or wire >= (nq if index < nqargs else nc): self.error("Invalid instruction argument mapping.")
                (qargs if index < nqargs else cargs).append(wire)
            if len(set(qargs)) != len(qargs) or len(set(cargs))!=len(cargs): self.error("Repeated instruction argument.")
            self.parameter_loop_depth = self.loop_depth+int(class_name in {"ForLoopOp", "WhileLoopOp"})
            try:
                params = [self.value(*self.unpack(VALUE)) for _ in range(nparams)]
            finally:
                self.parameter_loop_depth = self.loop_depth
            if class_name in CONTROL_NAMES:
                from .qpy_control_records import decode_operation
                circuit.operations.append(decode_operation(self, class_name, label or None, params,
                    qargs, cargs, classical_condition, nctrl, ctrl_state))
                continue
            if class_name in custom:
                record = custom[class_name]
                if nqargs != record.width or ncargs != record.clbits or (nctrl,ctrl_state) != (record.controls,record.state):
                    self.error("QPY custom Gate invocation differs from its definition.")
                if record.kind == b"i":
                    circuit._append_instruction_reference(self.custom_operation(class_name,tuple(params),label,custom), qargs,cargs)
                else:circuit.append(self.custom_operation(class_name,tuple(params),label,custom),qargs,clbits=cargs)
                continue
            if class_name in DYNAMIC_CONTROLS:
                gate = self.dynamic_controlled_operation(class_name,label,params,nqargs,nctrl,ctrl_state,ncargs)
                circuit.append(gate,qargs)
                continue
            expected = OPS[op_name]
            if (expected[1] is not None and nqargs != expected[1]) or nparams != expected[2] or ncargs != int(op_name == "measure"):
                self.error("Instruction signature differs from its standard gate.")
            expected_controls = CONTROLS.get(op_name,0)
            if nctrl not in (0,expected_controls) or not 0 <= ctrl_state < 1<<nctrl: self.error("Invalid coherent control fields.")
            if op_name != "unitary" and any(not isinstance(value,(Real,Parameter,ParameterExpression)) or isinstance(value,bool) for value in params):
                self.error("Non-numeric standard instruction parameter.")
            from quantumbridge.core.standard_gates import STANDARD_GATE_TYPES, standard_from_record
            if class_name in {kind.__name__ for kind in STANDARD_GATE_TYPES}:
                self.admit_standard_object()
                gate = standard_from_record(class_name, params, label or None, ctrl_state if nctrl else None)
                if gate.num_qubits != nqargs: self.error("Standard Gate width differs from its matrix or constructor.")
                circuit.append(gate, qargs)
                continue
            if ctrl_state != (1<<nctrl)-1: self.error("Unsupported open control standard type.",UnsupportedFeature)
            if op_name == "measure": circuit.measure(qargs[0],cargs[0])
            elif op_name == "delay": circuit.delay(params[0],qargs[0],unit="dt")
            elif op_name == "barrier": circuit.barrier(*qargs)
            else:
                operation = Operation(op_name,tuple(qargs[expected_controls:]),tuple(qargs[:expected_controls]),tuple(params),{"label":label} if label else {})
                circuit.operations.append(operation)
            if label and op_name in {"measure","delay","barrier"}:
                item = circuit.measurements[-1] if op_name == "measure" else circuit.operations[-1]
                item.metadata["label"] = label
        calibrations, = self.unpack("!H")
        if calibrations: self.error("Pulse calibration payload is unsupported.",UnsupportedFeature)
        exists,initial,mapping,final,extra,input_count = self.unpack(LAYOUT)
        if exists != b"\x00" or extra or any(value != -1 for value in (initial,mapping,final)) or input_count not in (-1,0):
            self.error("Native QPY TranspileLayout fields are not implemented.",UnsupportedFeature)
        from .qpy_classical import validate_circuit
        validate_circuit(circuit, self.limits, self.loop_depth,
                         self.enclosing_variables if _control_block else None)
        from quantumbridge.core.control_flow_operations import operation_owner
        if _control_block or any(operation_owner(op) is not None for op in circuit.operations):
            from .qpy_control_records import public_block
            circuit = public_block(circuit)
        return self.canonical_parameter_graph(circuit)


def _uid(parameter): return UUID(parameter.uuid) if parameter.uuid else uuid5(NAMESPACE_URL,"quantumbridge:parameter:"+parameter.name)


def _same_parameters(left, right):
    if len(left) != len(right): return False
    for first, second in zip(left, right):
        if type(first) is np.ndarray or type(second) is np.ndarray:
            if type(first) is not np.ndarray or type(second) is not np.ndarray or first.shape != second.shape:
                return False
            if any(a != b for a, b in zip(first.flat, second.flat)): return False
        elif first != second: return False
    return True


def _unitary_wire_order(matrix):
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 1 or matrix.shape[0] & (matrix.shape[0]-1):
        raise FormatValidationError("A NumPy Gate parameter requires a square qubit matrix.",format="qpy",version=13)
    from quantumbridge.execution.array_ownership import retained_array
    from quantumbridge.execution.resource_context import current_resources
    from quantumbridge.core.gate_limits import current_gate_limits
    current_gate_limits().check("max_matrix_dimension",matrix.shape[0])
    width = matrix.shape[0].bit_length()-1
    scope = current_resources()
    result = retained_array(None if scope is None else scope.ledger,"qpy_matrix_wire_order",matrix.shape,complex)
    order = [int(format(index,"0"+str(width)+"b")[::-1],2) for index in range(matrix.shape[0])]
    for row, source_row in enumerate(order):
        for col, source_col in enumerate(order): result[row,col] = matrix[source_row,source_col]
    return result


def _parameter_value(value,limits=None,*,budget=None):
    if type(value) is np.ndarray:
        limits = limits or SerializationLimits()
        total = (0 if budget is None else budget.get("numeric_elements",0))+value.size
        limits.check("max_numeric_elements",total,format="qpy",version=13)
        limits.check("max_input_bytes",value.size*16,format="qpy",version=13)
        if budget is not None: budget["numeric_elements"] = total
        if value.dtype.hasobject or value.dtype.kind not in "biufc":
            raise UnsupportedFeature("Only numeric NumPy matrix parameters are supported.",format="qpy",version=13)
        payload = io.BytesIO()
        np.save(payload, _unitary_wire_order(value), allow_pickle=False)
        limits.check("max_input_bytes",payload.tell(),format="qpy",version=13)
        return b"n",payload.getvalue()
    if isinstance(value,ParameterVectorElement):
        vector=value.vector
        if value.index>=len(vector):raise UnsupportedFeature("A QPY vector element index is outside its current vector.",format="qpy",version=13)
        (limits or SerializationLimits()).check("max_parameter_nodes",len(vector),format="qpy",version=13)
        name=vector.name.encode("utf-8")
        return b"v",struct.pack("!HQ16sQ",len(name),len(vector),_uid(value).bytes,value.index)+name
    if isinstance(value,Parameter):
        name = value.name.encode("utf-8")
        return b"p",struct.pack("!H16s",len(name),_uid(value).bytes)+name
    if isinstance(value,ParameterExpression):
        if value.op in {"const","parameter"} and not value._retained_parameters:
            return _parameter_value(value.args[0],limits,budget=budget)
        from .qpy_expressions import write_expression
        return b"e",write_expression(value,limits)
    if isinstance(value,bool): raise UnsupportedFeature("Boolean gate parameters are unsupported.",format="qpy",version=13)
    if isinstance(value,Integral): return b"i",struct.pack("<q",int(value))
    if isinstance(value,Real) and math.isfinite(value): return b"f",struct.pack("<d",float(value))
    if isinstance(value,str): return b"s",value.encode("utf-8")
    raise UnsupportedFeature("Unsupported QPY parameter value.",format="qpy",version=13)


def _modifier_value(value,limits,budget):
    from quantumbridge.core.annotated_operation import InverseModifier, ControlModifier, PowerModifier
    from quantumbridge.core.parameters import parameter_set, resolve_parameter
    budget["modifiers"] = budget.get("modifiers",0)+1
    limits.check("max_gate_expansion",budget["modifiers"],format="qpy",version=13)
    if type(value) is InverseModifier: payload = MODIFIER.pack(b"i",0,0,0.)
    elif type(value) is ControlModifier:
        checked = ControlModifier(value.num_ctrl_qubits,value.ctrl_state)
        payload = MODIFIER.pack(b"c",checked.num_ctrl_qubits,checked.ctrl_state,0.)
    elif type(value) is PowerModifier:
        PowerModifier(value.power)
        if parameter_set((value.power,)):
            raise UnsupportedFeature("QPY v13 Modifier power is a float64 field, not a symbolic expression record.",format="qpy",version=13)
        power = resolve_parameter(value.power,{})
        if float(power) != power:
            raise UnsupportedFeature("This modifier power is not exactly representable as QPY float64.",format="qpy",version=13)
        payload = MODIFIER.pack(b"p",0,0,float(power))
    else: raise UnsupportedFeature("An annotation requires native typed modifiers.",format="qpy",version=13)
    return b"m",payload


def _object_parameters(operation):
    from quantumbridge.core.annotated_operation import AnnotatedOperation
    return operation.modifiers if type(operation) is AnnotatedOperation else operation.params


def _object_control_fields(operation,key):
    from quantumbridge.core.gates import ControlledGate
    if isinstance(operation, ControlledGate): return operation.num_ctrl_qubits,operation.ctrl_state
    if key.decode() in NAMES:
        count = CONTROLS.get(NAMES[key.decode()],0)
        return count,(1<<count)-1
    return 0,0


def _base_instruction_payload(operation,key,limits,budget):
    from quantumbridge.core.annotated_operation import AnnotatedOperation
    params = _object_parameters(operation)
    label = (getattr(operation,"label",None) or "").encode()
    nctrl,state = _object_control_fields(operation,key)
    data = BoundedParts(limits,"qpy")
    data.extend((INSTRUCTION.pack(len(key),len(label),len(params),operation.num_qubits,operation.num_clbits,0,0,0,nctrl,state),key,label))
    for value in params:
        kind,payload = _modifier_value(value,limits,budget) if type(operation) is AnnotatedOperation else _parameter_value(value,limits,budget=budget)
        data.extend((VALUE.pack(kind,len(payload)),payload))
    return b"".join(data)


def _flatten_controlled_entry(item,limits):
    """Flatten only audited coherent-control descriptors, never opaque metadata."""
    def descriptor(value):
        return (type(value) is dict and all(type(key) is str for key in value)
            and set(value)=={"name","metadata","controls"} and type(value["name"]) is str
            and type(value["metadata"]) is dict and all(type(key) is str for key in value["metadata"])
            and type(value["controls"]) is int and value["controls"]>0)
    base=item.metadata.get("controlled_base");controls,targets=item.controls,item.targets
    segment=len(controls);seen=set()
    while descriptor(base) and base["name"]=="native_controlled":
        if id(base) in seen:raise FormatValidationError("Cyclic coherent control descriptor.",format="qpy",version=13)
        seen.add(id(base));limits.check("max_nesting_depth",len(seen),format="qpy",version=13)
        if base["controls"]!=segment or set(base["metadata"])!={"controlled_base"}:
            raise UnsupportedFeature("Nested coherent control has opaque or inconsistent metadata.",format="qpy",version=13)
        child=base["metadata"]["controlled_base"]
        if not descriptor(child):raise UnsupportedFeature("Unsupported nested coherent control descriptor.",format="qpy",version=13)
        segment=child["controls"]
        if segment>=len(targets):raise FormatValidationError("Nested controls exceed base operands.",format="qpy",version=13)
        from quantumbridge.core.gate_limits import current_gate_limits
        current_gate_limits().check("max_control_qubits",len(controls)+segment)
        controls,targets=controls+targets[:segment],targets[segment:]
        base=child
    if not seen:return item
    if not descriptor(base) or base["metadata"]:
        raise UnsupportedFeature("Nested coherent control cannot discard base metadata.",format="qpy",version=13)
    metadata={**item.metadata,"controlled_base":{**base,"controls":len(controls)}}
    return Operation(item.name,targets,controls,item.params,metadata)


def _standard_instruction_entry(item,limits=None,budget=None,planned=None):
    if type(item) is not Operation:return item
    if type(item.metadata) is not dict or any(type(key) is not str for key in item.metadata):
        raise UnsupportedFeature("QPY operation metadata requires a native dictionary.",format="qpy",version=13)
    if (item.name=="unitary" and not item.controls and not item.params
            and "matrix" in item.metadata
            and not set(item.metadata)-{"matrix","label","operation_index"}):
        from quantumbridge.core.standard_gates import UnitaryGate
        from quantumbridge.execution.array_reservations import numeric_shape
        from quantumbridge.core.gate_limits import current_gate_limits
        limits=limits or SerializationLimits()
        shape=numeric_shape(item.metadata["matrix"])
        if shape!=(1<<len(item.targets),)*2:
            raise FormatValidationError("Unitary matrix width differs from its operands.",format="qpy",version=13)
        current_gate_limits().check("max_matrix_dimension",shape[0])
        limits.check("max_numeric_elements",shape[0]*shape[1],format="qpy",version=13)
        limits.check("max_input_bytes",shape[0]*shape[1]*16,format="qpy",version=13)
        owner=UnitaryGate(item.metadata["matrix"],label=item.metadata.get("label"))
        metadata={key:value for key,value in item.metadata.items() if key!="matrix"}
        return Operation(owner.name,item.targets,metadata={**metadata,"gate":owner})
    if item.name=="native_controlled":
        item=_flatten_controlled_entry(item,limits or SerializationLimits())
        base=item.metadata.get("controlled_base",{})
        canonical={"cx":"ccx","cz":"ccz","swap":"cswap"}
        rotations={"crx":"crx","cry":"cry","crz":"crz","cp":"cp","cphase":"cp"}
        axes={"rx":"crx","ry":"cry","rz":"crz","phase":"cp","p":"cp","u1":"cp"}
        admitted=(type(base) is dict and all(type(key) is str for key in base)
            and set(base)=={"name","metadata","controls"} and type(base.get("name")) is str
            and type(base.get("metadata")) is dict and not base["metadata"]
            and type(base.get("controls")) is int and base["controls"]>0
            and type(item.controls) is tuple and len(item.controls)==base["controls"]
            and type(item.targets) is tuple and type(item.params) is tuple
            and all(type(key) is str for key in item.metadata)
            and not set(item.metadata)-{"controlled_base","label","operation_index"})
        wrapped=admitted and len(item.targets)==2
        flat=admitted and len(item.targets)==1
        if wrapped and len(item.controls)==1 and base["name"] in canonical and not item.params:
            controls=item.controls if base["name"]=="swap" else item.controls+item.targets[:1]
            targets=item.targets if base["name"]=="swap" else item.targets[1:]
            metadata={key:value for key,value in item.metadata.items() if key!="controlled_base"}
            return Operation(canonical[base["name"]],targets,controls,metadata=metadata)
        if flat and len(item.controls)==2 and base["name"] in {"x","z"} and not item.params:
            metadata={key:value for key,value in item.metadata.items() if key!="controlled_base"}
            return Operation("cc"+base["name"],item.targets,item.controls,metadata=metadata)
        name=rotations.get(base["name"]) if wrapped else axes.get(base["name"]) if flat else None
        constant={"sx":(math.pi/2,(("h",()),),(("h",()),)),
            "sxdg":(-math.pi/2,(("h",()),),(("h",()),)),
            "x":(math.pi,(("h",()),),(("h",()),)),"z":(math.pi,(),()),
            "y":(math.pi,(("sdg",()),("h",())),(("h",()),("s",()))),
            "h":(math.pi,(("ry",(-math.pi/4,)),),(("ry",(math.pi/4,)),))}
        constant_name=base["name"] if flat else {"csx":"sx","csxdg":"sxdg","cy":"y","ch":"h","cx":"x","cz":"z"}.get(base["name"]) if wrapped else None
        fixed=constant.get(constant_name) if not item.params else None
        if name is not None and len(item.params)==1 or fixed is not None:
            from quantumbridge.core.gates import Gate
            from quantumbridge.core.gate_limits import current_gate_limits
            from quantumbridge.core.parameters import _coerce_expression,resolve_parameter
            from quantumbridge.execution.array_reservations import numeric_shape
            limits=limits or SerializationLimits();budget={} if budget is None else budget
            planned=[0,0] if planned is None else planned
            count=len(item.controls)+int(wrapped)
            current_gate_limits().check("max_control_qubits",count)
            limits.check("max_qubits",count+1,format="qpy",version=13)
            prefix,suffix=((),()) if fixed is None else fixed[1:]
            work=(count-1)*(1<<count)+1+len(prefix)+len(suffix)
            limits.check("max_gate_expansion",budget.get("expanded",0)+planned[0]+work,format="qpy",version=13)
            limits.check("max_instructions",budget.get("instructions",0)+planned[0]+work,format="qpy",version=13)
            limits.check("max_custom_definitions",budget.get("definitions",0)+planned[1]+1,format="qpy",version=13)
            current_gate_limits().check("max_total_operations",work)
            current_gate_limits().check("max_decomposition_work",work)
            value=item.params[0] if fixed is None else fixed[0]
            if type(value) not in (Parameter,ParameterVectorElement,ParameterExpression):
                if numeric_shape(value,real_only=True)!=():
                    raise UnsupportedFeature("Controlled rotation angle must be a native scalar.",format="qpy",version=13)
            angle=_coerce_expression(value)
            if angle.parameters:
                limits.check("max_symbolic_expression_depth",angle._depth+2,format="qpy",version=13)
                unit=angle/(1<<(count-1))
            else:unit=resolve_parameter(angle,{})/(1<<(count-1))
            label=item.metadata.get("label")
            if label is not None and type(label) is not str:
                raise UnsupportedFeature("QPY labels require native strings.",format="qpy",version=13)
            if fixed is not None:name="cp"
            body=Circuit(count+1,name="native_c"+(str(count) if count!=2 else "")+name)
            for primitive,params in prefix:body.append(primitive,(count,),params=params)
            if count==2:
                body.append(name,(2,),controls=(1,),params=(unit,))
                body.cx(0,1)
                body.append(name,(2,),controls=(1,),params=(-unit,))
                body.cx(0,1)
                body.append(name,(2,),controls=(0,),params=(unit,))
            else:
                # The alternating sum of all nonempty control parities is
                # 2**(count-1) times their AND. Same-axis rotations commute.
                for mask in range(1,1<<count):
                    members=tuple(wire for wire in range(count) if mask>>wire&1)
                    pivot=members[-1]
                    for wire in members[:-1]:body.cx(wire,pivot)
                    body.append(name,(count,),controls=(pivot,),params=(unit if len(members)%2 else -unit,))
                    for wire in reversed(members[:-1]):body.cx(wire,pivot)
            for primitive,params in suffix:body.append(primitive,(count,),params=params)
            owner=Gate(body.name,count+1,body.parameters,label,definition=body)
            planned[0]+=work;planned[1]+=1
            metadata={"gate":owner}
            if label is not None:metadata["label"]=label
            return Operation(owner.name,item.wires,params=tuple(owner.params),metadata=metadata)
    owner=item.metadata.get("instruction_object")
    if type(owner) is not Instruction or owner.definition is not None or owner.name not in {"measure","reset","barrier","delay"}:return item
    bits=item.metadata.get("instruction_clbits",())
    if (item.name!=owner.name or item.controls or len(item.targets)!=owner.num_qubits
            or len(bits)!=owner.num_clbits or tuple(item.params)!=tuple(owner.params)):
        raise FormatValidationError("Builtin instruction invocation differs from its native owner.",format="qpy",version=13)
    metadata={**owner.metadata,**{key:value for key,value in item.metadata.items()
        if key not in {"instruction_object","instruction_clbits"}}}
    if owner.name=="measure":
        if owner.num_qubits!=1 or owner.num_clbits!=1 or owner.params:
            raise FormatValidationError("Measurement signature differs from the standard record.",format="qpy",version=13)
        return Measurement(item.targets[0],bits[0],metadata=metadata)
    if bits:raise FormatValidationError("Builtin directive/reset cannot take classical operands.",format="qpy",version=13)
    return Operation(owner.name,item.targets,params=tuple(owner.params),metadata=metadata)


def _circuit_bytes(circuit,limits,*,_budget=None,_depth=0,_path="root",_loop_depth=0,_ancestors=(),_enclosing_variables=None):
    budget = {} if _budget is None else _budget
    limits.check("max_nesting_depth",_depth,format="qpy",version=13)
    if id(circuit) in _ancestors:
        raise FormatValidationError("Cyclic QPY control block graph.", format="qpy", version=13)
    _ancestors = (*_ancestors, id(circuit))
    from quantumbridge.core.circuit_metadata import execution_metadata, serialized_user_metadata
    from .qpy_classical import validate_circuit, declaration_bytes
    variable_count, variable_payload = declaration_bytes(circuit, limits, budget, _enclosing_variables)
    validate_circuit(circuit, limits, _loop_depth, _enclosing_variables)
    from .qpy_registers import register_records
    specs = register_records(circuit,limits,budget)
    for field,value in (("max_qubits",circuit.num_qubits),("max_clbits",circuit.num_bits),("max_registers",len(specs))): limits.check(field,value,format="qpy",version=13)
    from quantumbridge.core.operations import Operation,execution_operation
    budget["instructions"] = budget.get("instructions",0)+len(circuit.operations)+len(circuit.measurements)
    limits.check("max_instructions",budget["instructions"],format="qpy",version=13)
    planned=[0,0]
    from quantumbridge.core.control_flow_operations import operation_owner
    items = [item if type(item) is Operation and operation_owner(item) is not None else
        _standard_instruction_entry(execution_operation(item,standardize_inverse=True,serialization_limits=limits),limits,budget,planned)
        if type(item) is Operation else item for item in circuit.iter_instructions()]
    names,identities,nodes = {},{},0
    def parameter_identity(value,depth=0):
        nonlocal nodes
        nodes += 1
        budget["parameter_nodes"] = budget.get("parameter_nodes",0)+1
        limits.check("max_parameter_nodes",budget["parameter_nodes"],format="qpy",version=13)
        limits.check("max_symbolic_expression_depth",depth,format="qpy",version=13)
        if isinstance(value,Parameter):
            uid = _uid(value)
            kind = "vector" if isinstance(value,ParameterVectorElement) else "scalar"
            if value.name in names and names[value.name] != (uid,kind) or uid in identities and identities[uid] != (value.name,kind):
                raise UnsupportedFeature("Parameter names and UUIDs must identify one another uniquely.",format="qpy",version=13)
            names[value.name],identities[uid] = (uid,kind),(value.name,kind)
        elif isinstance(value,ParameterExpression):
            for child in value.args: parameter_identity(child,depth+1)
            for parameter in value._retained_parameters: parameter_identity(parameter,depth+1)
    parameter_identity(circuit.global_phase)
    for item in items:
        for value in getattr(item,"params",()): parameter_identity(value)
    if circuit.name == "" or circuit.name is not None and not isinstance(circuit.name,str):
        raise UnsupportedFeature("Circuit names must be nonempty strings or absent.",format="qpy",version=13)
    name = (circuit.name or "").encode("utf-8")
    phase_type,phase = _parameter_value(circuit.global_phase,limits,budget=budget)
    if phase_type in {b"f",b"i"}: phase_type,phase = b"f",struct.pack("!d",float(circuit.global_phase))
    metadata = serialized_user_metadata(circuit)
    from quantumbridge.core.circuit_metadata import is_separated
    if not is_separated(circuit):
        metadata = {key: value for key, value in metadata.items() if key != "classical_scope"}
    from .common import checked_json_metadata
    checked_json_metadata(metadata,limits,format="qpy")
    try: encoded_metadata = json.dumps(metadata,ensure_ascii=True,allow_nan=False,separators=(",",":")).encode()
    except (TypeError,ValueError,RecursionError) as exc: raise UnsupportedFeature("Metadata must be finite JSON data.",format="qpy",version=13) from exc
    limits.check("max_metadata_bytes",len(encoded_metadata),format="qpy",version=13)
    parts = BoundedParts(limits,"qpy")
    parts.extend([CIRCUIT_HEADER.pack(len(name),phase_type,len(phase),circuit.num_qubits,circuit.num_bits,len(encoded_metadata),len(specs),len(items),variable_count),name,phase,encoded_metadata])
    for spec in specs:
        reg_name = spec["name"].encode()
        parts.extend([REGISTER.pack(spec["kind"].encode(),spec["standalone"],len(spec["bits"]),len(reg_name),spec["in_circuit"]),reg_name,struct.pack("!"+"q"*len(spec["bits"]),*spec["bits"])])
    parts.append(variable_payload)
    custom,records = {},[]
    from quantumbridge.core.gates import Gate, ControlledGate, validate_definition
    from quantumbridge.core.annotated_operation import AnnotatedOperation, validate_annotated_structure
    def register_operation(operation,path,depth,ancestors=(),primitive_base=False):
        limits.check("max_nesting_depth",depth,format="qpy",version=13)
        if id(operation) in ancestors: raise FormatValidationError("Cyclic custom operation base graph.",format="qpy",version=13)
        from quantumbridge.core.standard_gates import standard_type_name, standard_from_record
        from quantumbridge.core.instruction_parameters import definition_signature
        standard = standard_type_name(operation)
        if standard is not None:
            expected = standard_from_record(standard, tuple(operation.params), operation.label,
                operation.ctrl_state if isinstance(operation, ControlledGate) else None)
            if (operation.metadata or operation.name != expected.name
                    or definition_signature(operation.definition) != definition_signature(expected.definition)):
                raise UnsupportedFeature("A changed standard object cannot masquerade as its canonical QPY class.",format="qpy",version=13)
            return standard.encode()
        if type(operation) is AnnotatedOperation:
            validate_annotated_structure(operation)
            work = 1+len(operation.modifiers)
            kind = b"a"
        elif type(operation) in (Gate,ControlledGate,Instruction):
            metadata = dict(operation.metadata)
            if type(operation) is Instruction and operation.definition is not None:
                # to_instruction duplicates circuit metadata and its name.
                # The definition record already transports those fields;
                # unrelated per-object metadata still cannot be discarded.
                expected = dict(operation.definition.metadata)
                expected.setdefault("definition_name",operation.definition.name)
                metadata = {key:value for key,value in metadata.items()
                    if key not in expected or value != expected[key]}
            if metadata:
                raise UnsupportedFeature("QPY base records cannot carry opaque object metadata.",format="qpy",version=13)
            if operation.definition is None and type(operation) is not Instruction:
                if primitive_base and type(operation) is Gate and operation.name in OPS:
                    if OPS[operation.name][1:] != (operation.num_qubits,len(operation.params)):
                        raise FormatValidationError("Primitive base signature differs from its operation.",format="qpy",version=13)
                    from quantumbridge.circuit_library import standard_gate
                    validate_definition(standard_gate(operation.name,*operation.params))
                    return OPS[operation.name][0].encode()
                raise UnsupportedFeature("A custom Gate/Instruction QPY base needs an executable definition.",format="qpy",version=13)
            from quantumbridge.core.operations import instruction_definition_work
            work = (1 if operation.definition is None else instruction_definition_work(operation.definition, owner=operation)) if type(operation) is Instruction else validate_definition(operation.definition,owner=operation)
            kind = b"c" if type(operation) is ControlledGate else b"i" if type(operation) is Instruction else b"g"
        else: raise UnsupportedFeature("Only native unitary operation objects have QPY custom records.",format="qpy",version=13)
        budget["expanded"] = budget.get("expanded",0)+work
        limits.check("max_gate_expansion",budget["expanded"],format="qpy",version=13)
        budget["definitions"] = budget.get("definitions",0)+1
        limits.check("max_custom_definitions",budget["definitions"],format="qpy",version=13)
        local_name = (operation.name+"_"+uuid5(NAMESPACE_URL,"quantumbridge:qpy13:"+path).hex).encode()
        base_key = None
        if kind in {b"a",b"c"}:
            base = operation.base_op if kind == b"a" else operation.base_gate
            base_key = register_operation(base,path+"/base",depth+1,(*ancestors,id(operation)),primitive_base=True)
        records.append((local_name,operation,base_key,kind,depth))
        return local_name
    for index,item in enumerate(items):
        if type(item) is Operation and operation_owner(item) is not None:
            continue
        gate,annotation = item.metadata.get("gate"),item.metadata.get("annotated_operation")
        instruction = item.metadata.get("instruction_object")
        if sum(value is not None for value in (gate, annotation, instruction)) > 1:
            raise FormatValidationError("One invocation cannot have multiple operation owners.",format="qpy",version=13)
        operation = instruction if instruction is not None else annotation if annotation is not None else gate
        if operation is None and not isinstance(item, Measurement) and item.name == "csxdg":
            if (len(item.controls) != 1 or len(item.targets) != 1 or item.params
                    or set(item.metadata)-{"label", "operation_index"}):
                raise UnsupportedFeature("Only the canonical controlled SX inverse has this QPY definition.",format="qpy",version=13)
            limits.check("max_gate_expansion",budget.get("expanded",0)+3,format="qpy",version=13)
            limits.check("max_custom_definitions",budget.get("definitions",0)+1,format="qpy",version=13)
            from quantumbridge.core.gate_limits import current_gate_limits
            current_gate_limits().check("max_total_operations",3)
            # SX has eigenvalues 1/i, so three controlled SX operations give
            # its exact inverse without inventing an unsupported class field.
            body = Circuit(2)
            for _ in range(3): body.append("csx", [1], controls=[0])
            operation = Gate("csxdg",2,label=item.metadata.get("label"),definition=body)
            item = Operation(operation.name,item.wires,params=(),metadata={**item.metadata,"gate":operation})
            items[index] = item
        if operation is None: continue
        custom[index] = (register_operation(operation,_path+":"+str(index),_depth),operation)
    parts.append(struct.pack("!Q",len(records)))
    for local_name,operation,base_key,kind,depth in records:
        controlled,annotated = kind == b"c",kind == b"a"
        body = None if annotated else operation.definition
        nctrl,state,base_payload = 0,0,b""
        if controlled:
            nctrl,state = operation.num_ctrl_qubits,operation.ctrl_state
            closed = Circuit(operation.num_qubits)
            opened = [wire for wire in range(nctrl) if not (state>>wire)&1]
            for wire in opened: closed.x(wire)
            body = closed.compose(body)
            for wire in reversed(opened): body.x(wire)
            if any(item.name == "native_controlled" and "matrix" in item.metadata.get("controlled_base",{}).get("metadata",{}) for item in body.operations):
                # A numeric matrix-controlled definition has no standard
                # primitive opcode. Bound its dense export before allocation;
                # execution itself still uses the structural controlled body.
                from quantumbridge.core.standard_gates import UnitaryGate
                limits.check("max_numeric_elements",budget.get("numeric_elements",0)+(1<<body.num_qubits)**2,format="qpy",version=13)
                body = Circuit(body.num_qubits).append(UnitaryGate(body.to_gate().to_matrix()),range(body.num_qubits))
        if controlled or annotated:
            base = operation.base_op if annotated else operation.base_gate
            base_payload = _base_instruction_payload(base,base_key,limits,budget)
        definition = b"" if body is None else _circuit_bytes(body,limits,_budget=budget,_depth=depth+1,_path=local_name.decode())
        parts.extend((CUSTOM_INSTRUCTION.pack(len(local_name),kind,operation.num_qubits,operation.num_clbits,body is not None,len(definition),nctrl,state,len(base_payload)),local_name,definition,base_payload))
    for index,item in enumerate(items):
        if type(item) is Operation and operation_owner(item) is not None:
            from .qpy_control_records import encode_operation
            parts.append(encode_operation(item, circuit, limits, budget, _depth,
                _path+"/"+str(index), _loop_depth, _ancestors))
            continue
        if isinstance(item,Measurement) and item.kind != "computational":
            raise UnsupportedFeature("Only computational-basis measurement has a supported QPY instruction.",format="qpy",version=13)
        op_name = "measure" if isinstance(item,Measurement) else item.name
        if index in custom:
            class_name,operation = custom[index]
            annotated = type(operation) is AnnotatedOperation
            from quantumbridge.core.standard_gates import invocation_params
            if item.name != operation.name or item.controls or len(item.targets) != operation.num_qubits or item.params != invocation_params(operation):
                raise FormatValidationError("Custom Gate invocation and definition disagree.",format="qpy",version=13)
            allowed = {"annotated_operation","instruction_clbits","operation_index"} if annotated else {"gate","label","operation_index"}
            if type(operation) is Instruction:
                allowed = {"instruction_object", "instruction_clbits", "label", "operation_index"}
            clbits=item.metadata.get("instruction_clbits",())
            if len(clbits)!=operation.num_clbits:raise FormatValidationError("Custom classical invocation width differs from its operation.",format="qpy",version=13)
            if set(item.metadata)-allowed:
                raise UnsupportedFeature("Custom Gate instruction metadata is not representable in QPY v13.",format="qpy",version=13)
            label = item.metadata.get("label","").encode()
            ctrl = operation.num_ctrl_qubits if isinstance(operation, ControlledGate) else 0
            state = operation.ctrl_state if isinstance(operation, ControlledGate) else 0
            params = _object_parameters(operation)
            parts.extend((INSTRUCTION.pack(len(class_name),len(label),len(params),operation.num_qubits,len(clbits),0,0,0,ctrl,state),class_name,label))
            parts.extend(struct.pack("!cI",b"q",wire) for wire in item.targets)
            parts.extend(struct.pack("!cI",b"c",bit) for bit in clbits)
            for value in params:
                kind,payload = _modifier_value(value,limits,budget) if annotated else _parameter_value(value,limits,budget=budget)
                parts.extend((VALUE.pack(kind,len(payload)),payload))
            continue
        if op_name not in OPS or any(key in item.metadata for key in ("condition","gate_condition","control_flow_stack","control_flow","control_node","matrix","kraus")):
            raise UnsupportedFeature(f"QPY does not yet support instruction {op_name!r} or its control/opaque metadata.",format="qpy",version=13)
        class_name = OPS[op_name][0].encode()
        label = item.metadata.get("label","").encode()
        if set(item.metadata)-{"label","operation_index","directive","duration","unit"}:
            raise UnsupportedFeature("Instruction metadata cannot be represented by this QPY subset.",format="qpy",version=13)
        qargs,cargs = ((item.wire,),(item.bit,)) if isinstance(item,Measurement) else (item.wires,())
        params = () if isinstance(item,Measurement) else item.params
        if op_name == "delay" and item.metadata.get("unit","dt") != "dt":
            raise UnsupportedFeature("The tested QPY v13 Delay payload cannot preserve non-dt units; use QASM3.",format="qpy",version=13)
        ctrl = CONTROLS.get(op_name,0)
        expected = OPS[op_name]
        if (expected[1] is not None and len(qargs) != expected[1]) or len(params) != expected[2] or not isinstance(item,Measurement) and len(item.controls) != ctrl:
            raise FormatValidationError("Instruction signature/control count differs from its standard gate.",format="qpy",version=13)
        parts.append(INSTRUCTION.pack(len(class_name),len(label),len(params),len(qargs),len(cargs),0,0,0,ctrl,(1<<ctrl)-1))
        parts.extend((class_name,label))
        parts.extend(struct.pack("!cI",b"q",wire) for wire in qargs)
        parts.extend(struct.pack("!cI",b"c",wire) for wire in cargs)
        for value in params:
            kind,payload = _parameter_value(value,limits,budget=budget)
            parts.extend((VALUE.pack(kind,len(payload)),payload))
    parts.extend((struct.pack("!H",0),LAYOUT.pack(b"\x00",-1,-1,-1,0,-1)))
    result = b"".join(parts)
    limits.check("max_input_bytes",len(result),format="qpy",version=13)
    return result


def qpy_dumps(circuits, *, version=13, limits=None):
    if version != 13: raise UnsupportedVersion("Only the tested QPY version 13 is supported.",format="qpy",version=version)
    from quantumbridge.core.symbolic_limits import symbolic_serialization_limits
    limits = symbolic_serialization_limits(limits)
    from itertools import islice
    circuits = [circuits] if isinstance(circuits,Circuit) else list(islice(iter(circuits),limits.max_circuits+1))
    limits.check("max_circuits",len(circuits),format="qpy",version=13)
    parts = BoundedParts(limits,"qpy")
    parts.extend((FILE_HEADER.pack(b"QISKIT",13,0,0,0,len(circuits),b"p"),b"q"))
    try:
        budget = {}
        for index,circuit in enumerate(circuits):
            parts.append(_circuit_bytes(circuit,limits,_budget=budget,_path=str(index)))
    except (struct.error,OverflowError,RecursionError) as exc:
        raise FormatValidationError("Object fields exceed supported binary lengths or nesting.",format="qpy",version=13) from exc
    result = b"".join(parts)
    limits.check("max_input_bytes",len(result),format="qpy",version=13)
    return result


def qpy_loads(data, *, limits=None):
    from quantumbridge.core.symbolic_limits import symbolic_serialization_limits
    limits = symbolic_serialization_limits(limits)
    if not isinstance(data,(bytes,bytearray,memoryview)): raise TypeError("QPY input must be binary data.")
    limits.check("max_input_bytes",len(data),format="qpy")
    reader = Reader(data,limits)
    magic,version,major,minor,patch,count,encoding = reader.unpack(FILE_HEADER)
    if magic != b"QISKIT": reader.error("Invalid QPY magic.")
    if version != 13: raise UnsupportedVersion("Only tested QPY version 13 is supported.",format="qpy",version=version,offset=6)
    if encoding != b"p": reader.error("Unsupported symbolic encoding.",UnsupportedFeature)
    reader.bounded("max_circuits",count)
    if reader.take(1) != b"q": reader.error("Only circuit program payloads are supported.",UnsupportedFeature)
    try: circuits = [reader.circuit() for _ in range(count)]
    except SerializationError: raise
    except (ValueError,TypeError,KeyError,OverflowError,RecursionError) as exc:
        reader.error(f"Invalid circuit payload: {type(exc).__name__}.")
    if reader.pos != len(reader.data): reader.error("Unexpected trailing binary data.")
    return circuits


def qpy_dump(circuits,stream,**options): write_output(stream,qpy_dumps(circuits,**options))
def qpy_load(stream,**options): return qpy_loads(read_input(stream,binary=True,limits=options.get("limits"),format="qpy"),**options)


def qpy_get_version(stream):
    if hasattr(stream,"read"):
        position = stream.tell()
        try: data = stream.read(FILE_HEADER.size)
        finally: stream.seek(position)
    else: data = read_input(stream,binary=True,format="qpy")[:FILE_HEADER.size]
    reader = Reader(data,SerializationLimits())
    magic,version,*_ = reader.unpack(FILE_HEADER)
    if magic != b"QISKIT": reader.error("Invalid QPY magic.")
    return version
