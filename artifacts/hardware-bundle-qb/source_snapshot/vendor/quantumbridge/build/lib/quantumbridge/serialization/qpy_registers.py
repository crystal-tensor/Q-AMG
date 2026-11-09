# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
"""Standard QPY v13 register records over the native Bit/Register graph."""
from quantumbridge.core.bits import (Qubit,Clbit,QuantumRegister,ClassicalRegister,AncillaRegister,
    bit_graph,install_bits)
from quantumbridge.core.gate_limits import current_gate_limits
from .common import register_specs
from .errors import FormatValidationError,UnsupportedFeature


def _check_memberships(records,limits,budget):
    work=sum(len(record["bits"]) for record in records)
    total=budget.get("register_bits",0)+work
    limits.check("max_gate_expansion",total,format="qpy",version=13)
    current_gate_limits().check("max_decomposition_work",total)
    budget["register_bits"]=total


def register_records(circuit,limits,budget):
    graph=bit_graph(circuit)
    if graph is None:
        records=[{**spec,"standalone":True,"in_circuit":True} for spec in register_specs(circuit)]
        _check_memberships(records,limits,budget)
        return records
    records=[];origins={}
    for register in circuit._qregs+circuit._cregs:
        if type(register) is AncillaRegister:
            raise UnsupportedFeature("Ancilla Register records are not supported by both tested QPY v13 public loaders.",format="qpy",version=13)
        kind="c" if type(register) is ClassicalRegister else "q"
        positions=circuit._clbit_indices if kind=="c" else circuit._qubit_indices
        normal=all(bit._key==(register._bit_type.__name__,type(register).__name__,register.name,len(register),i)
            for i,bit in enumerate(register._bits))
        records.append({"kind":kind,"name":register.name,"bits":[positions[bit._key] for bit in register._bits],
            "standalone":normal,"in_circuit":True})
        origins[kind,register.name]=(normal,len(register),records[-1])
    for kind,bits in (("q",circuit._qubits),("c",circuit._clbits)):
        for wire,bit in enumerate(bits):
            if bit._key[0]=="AncillaQubit":
                raise UnsupportedFeature("Ancilla bit records are not portable across both tested public loaders.",format="qpy",version=13)
            if len(bit._key)==2:continue
            _,register_type,name,size,index=bit._key
            limits.check("max_qubits" if kind=="q" else "max_clbits",size,format="qpy",version=13)
            key=(kind,name)
            if key in origins:
                normal,width,record=origins[key]
                if not normal or width!=size:
                    raise UnsupportedFeature("QPY cannot disambiguate an alias and a different origin with the same register name.",format="qpy",version=13)
            else:
                limits.check("max_registers",len(records)+1,format="qpy",version=13)
                limits.check("max_gate_expansion",budget.get("register_bits",0)+sum(len(r["bits"]) for r in records)+size,format="qpy",version=13)
                record={"kind":kind,"name":name,"bits":[-1]*size,"standalone":True,"in_circuit":False}
                records.append(record);origins[key]=(True,size,record)
            if record["bits"][index] not in (-1,wire):
                raise FormatValidationError("Conflicting register origin bit references.",format="qpy",version=13)
            record["bits"][index]=wire
    _check_memberships(records,limits,budget)
    return records


def restore_register_records(circuit,records,limits,budget):
    """Construct all owners first, then aliases, independent of record order."""
    _check_memberships(records,limits,budget)
    slots={"q":[None]*circuit.num_qubits,"c":[None]*circuit.num_bits}
    names=set();owners={}
    for record in records:
        kind,name,mapping=record["kind"],record["name"],record["bits"]
        if name in names:raise FormatValidationError("Duplicate QPY register name.",format="qpy",version=13)
        names.add(name)
        width=len(slots[kind]);present=[i for i in mapping if i>=0]
        if (any(i < -1 or i>=width for i in mapping) or len(set(present))!=len(present)
                or record["in_circuit"] and len(present)!=len(mapping)):
            raise FormatValidationError("Invalid QPY register circuit-bit mapping.",format="qpy",version=13)
        if not record["in_circuit"] and not record["standalone"]:
            raise UnsupportedFeature("An external alias register has no portable native owner record.",format="qpy",version=13)
        if record["standalone"]:
            register=(QuantumRegister if kind=="q" else ClassicalRegister)(len(mapping),name)
            owners[kind,name]=register
            for bit,wire in zip(register._bits,mapping):
                if wire<0:continue
                if slots[kind][wire] is not None:
                    raise FormatValidationError("Two standalone register owners claim the same circuit bit.",format="qpy",version=13)
                slots[kind][wire]=bit
    for kind,constructor in (("q",Qubit),("c",Clbit)):
        for i,bit in enumerate(slots[kind]):
            if bit is None:slots[kind][i]=constructor()
    groups={"q":[],"c":[]}
    for record in records:
        if not record["in_circuit"]:continue
        kind,name=record["kind"],record["name"]
        register=owners[kind,name] if record["standalone"] else (QuantumRegister if kind=="q" else ClassicalRegister)(
            name=name,bits=[slots[kind][i] for i in record["bits"]])
        groups[kind].append(register)
    install_bits(circuit,tuple(slots["q"]),tuple(slots["c"]),tuple(groups["q"]),tuple(groups["c"]))
    # Retain the legacy structural view for existing QASM/caller consumers.
    from quantumbridge.core.circuit_metadata import execution_metadata
    execution_metadata(circuit)["registers"]=[{key:record[key] for key in ("kind","name","bits")}
        for record in records if record["in_circuit"]]
