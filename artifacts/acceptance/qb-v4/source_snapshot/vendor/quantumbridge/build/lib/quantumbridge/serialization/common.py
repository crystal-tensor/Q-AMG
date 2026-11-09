# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
from pathlib import Path
from .limits import SerializationLimits
from .errors import FormatValidationError, ResourceLimitError, UnsupportedFeature


class BoundedParts(list):
    def __init__(self,limits,format):
        super().__init__()
        self.limits,self.format,self.size = limits,format,0
    def append(self,value):
        self.size += len(value)
        self.limits.check("max_input_bytes",self.size,format=self.format)
        super().append(value)
    def extend(self,values):
        for value in values: self.append(value)


def read_input(source, *, binary=False, limits=None, format=None):
    limits = limits or SerializationLimits()
    if hasattr(source,"read"):
        data = source.read(limits.max_input_bytes+1)
    else:
        with Path(source).open("rb" if binary else "r", **({} if binary else {"encoding":"utf-8"})) as stream:
            data = stream.read(limits.max_input_bytes+1)
    expected = bytes if binary else str
    if not isinstance(data,expected): raise FormatValidationError("Incorrect stream mode.",format=format)
    limits.check("max_input_bytes",len(data) if binary else len(data.encode("utf-8")),format=format)
    return data


def write_output(destination, data):
    if hasattr(destination,"write"): destination.write(data)
    else:
        with Path(destination).open("wb" if isinstance(data,bytes) else "w",**({} if isinstance(data,bytes) else {"encoding":"utf-8"})) as stream:
            stream.write(data)


def checked_json_metadata(value, limits, *, format):
    """Refuse opaque/callback-backed data before a transport encoder touches it."""
    from math import isfinite
    pending, active, count = [(value,0,False)], set(), 0
    if type(value) is not dict:raise UnsupportedFeature("Metadata requires an exact JSON object.",format=format)
    while pending:
        item, depth, leaving = pending.pop()
        if leaving:active.remove(id(item));continue
        count += 1
        limits.check("max_ast_nodes",count+len(pending),format=format)
        limits.check("max_nesting_depth",depth,format=format)
        kind = type(item)
        if kind is dict or kind is list:
            if id(item) in active:raise UnsupportedFeature("Metadata must be acyclic JSON data.",format=format)
            active.add(id(item));pending.append((item,depth,True))
            limits.check("max_ast_nodes",count+len(pending)+len(item),format=format)
            if kind is dict and any(type(key) is not str for key in item):raise UnsupportedFeature("Metadata keys must be strings.",format=format)
            children = item.values() if kind is dict else item
            pending.extend((child,depth+1,False) for child in children)
        elif kind is float:
            if not isfinite(item):raise UnsupportedFeature("Metadata must be finite JSON data.",format=format)
        elif not any(kind is allowed for allowed in (str,int,bool,type(None))):
            raise UnsupportedFeature("Metadata must use exact JSON types.",format=format)
    return value


def register_specs(circuit):
    from quantumbridge.core.bits import bit_graph
    graph=bit_graph(circuit)
    from quantumbridge.core.circuit_metadata import execution_metadata
    specs = execution_metadata(circuit).get("registers")
    if graph is not None:
        from quantumbridge.core.bits import QuantumRegister,ClassicalRegister
        specs=[];order={"q":[],"c":[]}
        for register in circuit._qregs+circuit._cregs:
            kind="c" if type(register) is ClassicalRegister else "q"
            if type(register) not in (QuantumRegister,ClassicalRegister) or not len(register) or any(
                    bit._key!=(register._bit_type.__name__,type(register).__name__,register.name,len(register),i)
                    for i,bit in enumerate(register._bits)):
                raise UnsupportedFeature("This source subset requires normal register arrays, not aliases or ancillas.",feature="native_source_bit_graph")
            positions=circuit._clbit_indices if kind=="c" else circuit._qubit_indices
            indices=[positions[bit._key] for bit in register._bits]
            order[kind].extend(indices)
            specs.append({"kind":kind,"name":register.name,"bits":indices})
        if order!={"q":list(range(circuit.num_qubits)),"c":list(range(circuit.num_bits))}:
            raise UnsupportedFeature("Source declarations cannot preserve this loose or reordered bit graph.",feature="native_source_bit_graph")
    if specs is None:
        specs = [] if not circuit.num_qubits else [{"kind":"q","name":"q","bits":list(range(circuit.num_qubits))}]
        if circuit.num_bits: specs.append({"kind":"c","name":"c","bits":list(range(circuit.num_bits))})
    used, names = {"q":set(),"c":set()}, set()
    for spec in specs:
        kind, name, bits = spec["kind"], spec["name"], spec["bits"]
        if kind not in used or not isinstance(name,str) or not name or name in names or not bits:
            raise FormatValidationError("Invalid/duplicate register declaration.")
        width = circuit.num_qubits if kind == "q" else circuit.num_bits
        if any(isinstance(bit,bool) or not isinstance(bit,int) or not 0 <= bit < width or bit in used[kind] for bit in bits) or len(set(bits)) != len(bits):
            raise FormatValidationError("Invalid/aliased register mapping.")
        used[kind].update(bits); names.add(name)
    if used != {"q":set(range(circuit.num_qubits)),"c":set(range(circuit.num_bits))}:
        raise FormatValidationError("Registers must cover each wire once; alias/loose-bit export is unsupported.")
    return specs
