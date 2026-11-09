# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
"""Immutable logical bit identities and bounded native register aliases."""
from itertools import count, islice
import operator
from uuid import uuid4

from .errors import CircuitError
from .gate_limits import current_gate_limits


def integer_index(value):
    from quantumbridge.execution.resource_context import current_resources
    if current_resources() is not None and type(value) is not int and type(value) is not bool:
        from quantumbridge.execution.array_ownership import _defined_type
        if not _defined_type(type(value),"numpy"):
            from .classical import UnsupportedFeature
            raise UnsupportedFeature("managed_bit_index")
    return operator.index(value)


def bounded_items(values, label="bit operands"):
    from quantumbridge.execution.resource_context import current_resources
    from .classical import UnsupportedFeature
    kind=type(values)
    known=any(kind is candidate for candidate in (tuple,list,range,*REGISTER_TYPES))
    if current_resources() is not None and not known:
        raise UnsupportedFeature("managed_bit_iterable",label+" requires an admitted native sequence.")
    if any(kind is candidate for candidate in REGISTER_TYPES):validate_register(values)
    limit=current_gate_limits().max_circuit_bits
    if known:
        current_gate_limits().check("max_circuit_bits",len(values))
    result=tuple(islice(iter(values),limit+1))
    current_gate_limits().check("max_circuit_bits",len(result))
    return result


class Bit:
    def __new__(cls,*args,**kwargs):
        if cls is Bit:raise TypeError("Bit is abstract; construct a Qubit or Clbit.")
        return object.__new__(cls)

    def __init__(self,register=None,index=None):
        if register is None and index is None:
            key=(type(self).__name__,uuid4().hex)
        elif register is not None and index is not None:
            expected=ClassicalRegister if type(self) is Clbit else QuantumRegister
            if not any(type(register) is kind for kind in REGISTER_TYPES) or not isinstance(register,expected):
                raise TypeError("Bit register has the wrong native kind.")
            position=integer_index(index)
            if not 0<=position<len(register):raise CircuitError("Bit index is outside its register.")
            if type(register._bits[position]) is not type(self):raise TypeError("Bit register member has the wrong type.")
            key=register._bits[position]._key
        else:raise CircuitError("Bit requires both register and index, or neither.")
        object.__setattr__(self,"_key",key)

    def __setattr__(self,name,value):raise AttributeError("Native bit identity is immutable.")

    def __hash__(self):
        validate_bit(self)
        return hash(self._key)

    def __eq__(self,other):
        if not is_bit(other) or type(self) is not type(other):return False
        validate_bit(self);validate_bit(other)
        return self._key==other._key

    def __repr__(self):return "<"+type(self).__name__+" "+repr(self._key)+">"


class Qubit(Bit):pass
class AncillaQubit(Qubit):pass
class Clbit(Bit):pass

BIT_TYPES=(Qubit,AncillaQubit,Clbit)


def is_bit(value):return any(type(value) is kind for kind in BIT_TYPES)


def validate_bit(value,expected=None):
    if not is_bit(value):raise CircuitError("Expected a native Qubit, Clbit or AncillaQubit.")
    fields=object.__getattribute__(value,"__dict__")
    if (type(fields) is not dict or any(type(key) is not str for key in fields)
            or set(fields)!={"_key"} or type(fields["_key"]) is not tuple):
        raise CircuitError("Malformed native bit identity.")
    key=fields["_key"]
    if (len(key) not in (2,5) or type(key[0]) is not str or key[0]!=type(value).__name__
            or any(type(item) is not str and type(item) is not int for item in key)):
        raise CircuitError("Malformed native bit key.")
    if len(key)==2:
        if type(key[1]) is not str or len(key[1])!=32 or any(c not in "0123456789abcdef" for c in key[1]):
            raise CircuitError("Malformed standalone bit identity.")
    elif (type(key[1]) is not str or type(key[2]) is not str or type(key[3]) is not int
            or type(key[4]) is not int or not 0<=key[4]<key[3]
            or key[1]!={Qubit:"QuantumRegister",Clbit:"ClassicalRegister",AncillaQubit:"AncillaRegister"}[type(value)]):
        raise CircuitError("Malformed registered bit identity.")
    if expected is not None and not isinstance(value,expected):
        raise TypeError("Native bit has the wrong quantum/classical kind.")
    return value


def bit_view(bit):
    validate_bit(bit)
    result=object.__new__(type(bit))
    object.__setattr__(result,"_key",bit._key)
    return result


class _Register:
    _counter=count()
    _prefix="r"
    _bit_type=Bit

    def __init__(self,size=None,name=None,bits=None):
        if (size is None)==(bits is None):raise CircuitError("Provide exactly one of register size or bits.")
        if name is None:name=self._prefix+str(next(self._counter))
        if type(name) is not str:raise TypeError("Register name must be a string.")
        if bits is None:
            # Admit the graph size before allocating any bit objects.
            size=integer_index(size)
            if size<0:raise CircuitError("Register size must be nonnegative.")
            current_gate_limits().check("max_circuit_bits",size)
            items=[]
            for position in range(size):
                item=object.__new__(self._bit_type)
                key=(self._bit_type.__name__,type(self).__name__,name,size,position)
                object.__setattr__(item,"_key",key)
                items.append(item)
            items=tuple(items)
        else:
            items=bounded_items(bits,"register bits")
            for bit in items:validate_bit(bit,self._bit_type)
            if len(set(items))!=len(items):raise CircuitError("Register bits must not be duplicated.")
            size=len(items)
        object.__setattr__(self,"_bits",items)
        object.__setattr__(self,"_name",name)
        object.__setattr__(self,"_size",size)

    def __setattr__(self,name,value):raise AttributeError("Native register membership is immutable.")
    @property
    def name(self):return self._name
    @property
    def size(self):return self._size
    def __len__(self):return self._size
    def __iter__(self):return (bit_view(bit) for bit in self._bits)
    def __getitem__(self,index):
        if type(index) is slice:return [bit_view(bit) for bit in self._bits[index]]
        return bit_view(self._bits[integer_index(index)])
    def __contains__(self,bit):return is_bit(bit) and bit in self._bits
    def __eq__(self,other):
        return type(self) is type(other) and self._name==other._name and self._bits==other._bits
    def __hash__(self):return hash((type(self).__name__,self._name,self._bits))
    def __repr__(self):return type(self).__name__+"("+str(self._size)+", "+repr(self._name)+")"


class QuantumRegister(_Register):
    _prefix="q"
    _bit_type=Qubit


class ClassicalRegister(_Register):
    _prefix="c"
    _bit_type=Clbit


class AncillaRegister(QuantumRegister):
    _prefix="a"
    _bit_type=AncillaQubit


REGISTER_TYPES=(QuantumRegister,ClassicalRegister,AncillaRegister)


def validate_register(register):
    if not any(type(register) is kind for kind in REGISTER_TYPES):raise CircuitError("Expected a native Register.")
    fields=object.__getattribute__(register,"__dict__")
    if (type(fields) is not dict or any(type(key) is not str for key in fields)
            or set(fields)!={"_bits","_name","_size"} or type(fields["_bits"]) is not tuple
            or type(fields["_name"]) is not str or type(fields["_size"]) is not int
            or fields["_size"]!=len(fields["_bits"])):
        raise CircuitError("Malformed native register fields.")
    current_gate_limits().check("max_circuit_bits",fields["_size"])
    for bit in fields["_bits"]:validate_bit(bit,register._bit_type)
    if len(set(fields["_bits"]))!=fields["_size"]:raise CircuitError("Register bits must be unique.")
    return register


class BitLocations:
    def __init__(self,index,registers):self.index,self.registers=index,registers
    def __iter__(self):return iter((self.index,self.registers))
    def __eq__(self,other):return isinstance(other,BitLocations) and tuple(self)==tuple(other)


def initialize_bits(circuit,quantum,classical):
    if any(type(quantum) is kind for kind in REGISTER_TYPES):
        registers=(quantum,) if type(classical) is int and classical==0 else (quantum,classical)
        install_bits(circuit,(),(),(),())
        circuit.num_qubits,circuit.num_bits=0,0
        add_register(circuit,*registers)
        return
    quantum,classical=integer_index(quantum),integer_index(classical)
    if quantum<0 or classical<0:raise ValueError("Circuit bit counts must be nonnegative.")
    current_gate_limits().check("max_circuit_bits",quantum+classical)
    circuit._qregs=() if not quantum else (QuantumRegister(quantum,"q"),)
    circuit._cregs=() if not classical else (ClassicalRegister(classical,"c"),)
    circuit._qubits=() if not quantum else tuple(circuit._qregs[0])
    circuit._clbits=() if not classical else tuple(circuit._cregs[0])
    circuit.num_qubits,circuit.num_bits=quantum,classical
    install_bits(circuit,circuit._qubits,circuit._clbits,circuit._qregs,circuit._cregs)


def add_bits(circuit,bits):
    incoming=bounded_items(bits)
    for bit in incoming:validate_bit(bit)
    current_gate_limits().check("max_circuit_bits",circuit.num_qubits+circuit.num_bits+len(incoming))
    existing=set(circuit._qubits+circuit._clbits)
    if any(bit in existing for bit in incoming):raise CircuitError("Bit is already in this Circuit.")
    if len(set(incoming))!=len(incoming):raise ValueError("Duplicate bits in add_bits.")
    quantum=tuple(bit_view(bit) for bit in incoming if isinstance(bit,Qubit))
    classical=tuple(bit_view(bit) for bit in incoming if type(bit) is Clbit)
    install_bits(circuit,circuit._qubits+quantum,circuit._clbits+classical,circuit._qregs,circuit._cregs)


def add_register(circuit,*registers):
    quantum,classical=list(circuit._qubits),list(circuit._clbits)
    qregs,cregs=list(circuit._qregs),list(circuit._cregs)
    names={reg.name for reg in qregs+cregs}
    for register in registers:
        validate_register(register)
        if register.name in names:raise CircuitError("Register name already exists.")
        current_gate_limits().check("max_circuit_bits",len(quantum)+len(classical)+len(register))
        incoming=register._bits
        for bit in incoming:validate_bit(bit,register._bit_type)
        if len(set(incoming))!=len(incoming):raise CircuitError("Register contains duplicate bits.")
        destination,groups=(classical,cregs) if type(register) is ClassicalRegister else (quantum,qregs)
        known=set(destination)
        destination.extend(bit_view(bit) for bit in incoming if bit not in known)
        groups.append(register);names.add(register.name)
    install_bits(circuit,tuple(quantum),tuple(classical),tuple(qregs),tuple(cregs))


def find_bit(circuit,bit):
    validate_bit(bit)
    quantum=isinstance(bit,Qubit)
    positions,registers=(circuit._qubit_indices,circuit._qregs) if quantum else (circuit._clbit_indices,circuit._cregs)
    try:index=positions[bit._key]
    except KeyError:raise CircuitError("Bit has not been added to this Circuit.") from None
    memberships=[(reg,i) for reg in registers for i,member in enumerate(reg) if member==bit]
    return BitLocations(index,memberships)


def operand_index(circuit,bit,quantum):
    from quantumbridge.execution.resource_context import current_resources
    if is_bit(bit):
        validate_bit(bit,Qubit if quantum else Clbit)
        try:return (circuit._qubit_indices if quantum else circuit._clbit_indices)[bit._key]
        except KeyError:raise CircuitError("Bit has not been added to this Circuit.") from None
    if current_resources() is not None and not any(type(bit) is kind for kind in (int,bool)):
        from quantumbridge.execution.array_ownership import _defined_type
        if not _defined_type(type(bit),"numpy"):
            from .classical import UnsupportedFeature
            raise UnsupportedFeature("managed_bit_index")
    try:value=integer_index(bit)
    except TypeError:raise CircuitError("Bit operand must be an integer index or a native bit.") from None
    width=circuit.num_qubits if quantum else circuit.num_bits
    if value<0:value+=width
    if not 0<=value<width:raise CircuitError("Bit index is outside this Circuit.")
    return value


def copy_bits(source,target):
    from quantumbridge.execution.resource_context import current_resources
    if current_resources() is not None:
        from .circuit import is_native_circuit
        from .classical import UnsupportedFeature
        if not is_native_circuit(source):raise UnsupportedFeature("managed_bit_circuit_type")
        from .composition import _admit_graph
        _admit_graph((source._qubits,source._clbits,source._qregs,source._cregs))
    install_bits(target,source._qubits,source._clbits,source._qregs,source._cregs)


def install_bits(circuit,quantum,classical,qregs,cregs):
    if any(type(items) is not tuple for items in (quantum,classical,qregs,cregs)):
        raise CircuitError("Native Circuit bit collections must be admitted tuples.")
    for bit in quantum:validate_bit(bit,Qubit)
    for bit in classical:validate_bit(bit,Clbit)
    for reg in qregs+cregs:validate_register(reg)
    qindices={bit._key:i for i,bit in enumerate(quantum)}
    cindices={bit._key:i for i,bit in enumerate(classical)}
    circuit._qubits,circuit._clbits=quantum,classical
    circuit._qregs,circuit._cregs=qregs,cregs
    circuit._qubit_indices,circuit._clbit_indices=qindices,cindices
    circuit.num_qubits,circuit.num_bits=len(quantum),len(classical)


def bit_graph(circuit):
    """Native IR structural record; canonical integer circuits need no extension."""
    if (type(circuit.num_qubits) is not int or type(circuit.num_bits) is not int
            or type(circuit._qubits) is not tuple or type(circuit._clbits) is not tuple
            or type(circuit._qregs) is not tuple or type(circuit._cregs) is not tuple
            or circuit.num_qubits!=len(circuit._qubits) or circuit.num_bits!=len(circuit._clbits)):
        raise CircuitError("Native Circuit widths disagree with their bit collections.")
    current_gate_limits().check("max_circuit_bits",circuit.num_qubits+circuit.num_bits)
    for bit in circuit._qubits:validate_bit(bit,Qubit)
    for bit in circuit._clbits:validate_bit(bit,Clbit)
    for reg in circuit._qregs+circuit._cregs:validate_register(reg)
    canonical_q=tuple(("Qubit","QuantumRegister","q",circuit.num_qubits,i) for i in range(circuit.num_qubits))
    canonical_c=tuple(("Clbit","ClassicalRegister","c",circuit.num_bits,i) for i in range(circuit.num_bits))
    if (tuple(bit._key for bit in circuit._qubits)==canonical_q
            and tuple(bit._key for bit in circuit._clbits)==canonical_c
            and [(reg.name,len(reg)) for reg in circuit._qregs]==([] if not canonical_q else [("q",len(canonical_q))])
            and [(reg.name,len(reg)) for reg in circuit._cregs]==([] if not canonical_c else [("c",len(canonical_c))])):
        return None
    groups=[]
    work=circuit.num_qubits+circuit.num_bits
    for reg in circuit._qregs+circuit._cregs:
        work+=len(reg);current_gate_limits().check("max_decomposition_work",work)
        positions=circuit._clbit_indices if type(reg) is ClassicalRegister else circuit._qubit_indices
        groups.append({"kind":type(reg).__name__,"name":reg.name,"bits":[positions[bit._key] for bit in reg]})
    return {"version":1,"quantum":[list(bit._key) for bit in circuit._qubits],
        "classical":[list(bit._key) for bit in circuit._clbits],"groups":groups}


def restore_bit_graph(circuit,data,*,limits=None):
    if data is None:return
    from quantumbridge.execution.resource_context import current_resources
    if current_resources() is not None:
        from .composition import _admit_graph
        _admit_graph(data)
    if (type(data) is not dict or any(type(k) is not str for k in data)
            or set(data)!={"version","quantum","classical","groups"}
            or type(data["version"]) is not int or data["version"]!=1
            or any(type(data[k]) is not list for k in ("quantum","classical","groups"))
            or len(data["quantum"])!=circuit.num_qubits or len(data["classical"])!=circuit.num_bits):
        raise CircuitError("Malformed native IR bit graph.")
    if limits is not None:limits.check("max_registers",len(data["groups"]),format="ir")
    kinds={"Qubit":Qubit,"AncillaQubit":AncillaQubit,"Clbit":Clbit}
    values=[]
    for records,quantum in ((data["quantum"],True),(data["classical"],False)):
        decoded=[]
        for key in records:
            if (type(key) is not list or not key or type(key[0]) is not str or key[0] not in kinds
                    or any(type(item) is not str and type(item) is not int for item in key)):
                raise CircuitError("Malformed native IR bit key.")
            bit=object.__new__(kinds[key[0]])
            object.__setattr__(bit,"_key",tuple(key))
            validate_bit(bit,Qubit if quantum else Clbit)
            decoded.append(bit)
        if len(set(decoded))!=len(decoded):raise CircuitError("Duplicate native IR bit identity.")
        values.append(tuple(decoded))
    groups={"QuantumRegister":QuantumRegister,"ClassicalRegister":ClassicalRegister,"AncillaRegister":AncillaRegister}
    qregs,cregs,names=[],[],set()
    work=circuit.num_qubits+circuit.num_bits
    for record in data["groups"]:
        if (type(record) is not dict or any(type(k) is not str for k in record)
                or set(record)!={"kind","name","bits"} or type(record["kind"]) is not str
                or record["kind"] not in groups or type(record["name"]) is not str
                or record["name"] in names or type(record["bits"]) is not list):
            raise CircuitError("Malformed native IR register group.")
        members=values[int(record["kind"]=="ClassicalRegister")]
        work+=len(record["bits"]);current_gate_limits().check("max_decomposition_work",work)
        if any(type(index) is not int or not 0<=index<len(members) for index in record["bits"]):
            raise CircuitError("Native IR register index is outside its bit collection.")
        reg=groups[record["kind"]](name=record["name"],bits=[members[i] for i in record["bits"]])
        (cregs if record["kind"]=="ClassicalRegister" else qregs).append(reg)
        names.add(reg.name)
    install_bits(circuit,*values,tuple(qregs),tuple(cregs))
