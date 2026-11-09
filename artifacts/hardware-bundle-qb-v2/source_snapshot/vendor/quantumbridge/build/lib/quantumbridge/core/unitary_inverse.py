# Copyright 2026 QuantumBridge Contributors. Apache-2.0.
"""Owned qualification for the existing native primitive matrix inverse path."""

import hashlib

import numpy as np


def _failure():
    from .classical import UnsupportedFeature
    raise UnsupportedFeature("native_inverse_matrix_qualification",
        "A derived matrix needs its validated native inverse owner and unchanged local payload.")


def _digest(matrix):
    from .gate_limits import current_gate_limits
    current_gate_limits().check("max_decomposition_work",matrix.size)
    return hashlib.sha256(memoryview(matrix).cast("B")).digest()


def inverse_matrix_owner(operation):
    from .operations import Operation
    from .standard_gates import UnitaryGate
    from .gate_limits import current_gate_limits
    if type(operation) is not Operation: _failure()
    fields=object.__getattribute__(operation,"__dict__")
    owner=fields.get("_native_inverse_owner")
    qualification=fields.get("_native_inverse_qualification")
    if owner is None and qualification is None: return None
    if type(owner) is not UnitaryGate or type(qualification) is not tuple or len(qualification)!=10: _failure()
    if (type(qualification[5]) is not str or type(qualification[6]) is not str
            or qualification[7] is not None and type(qualification[7]) is not str
            or type(qualification[8]) is not bytes or qualification[9] is not None): _failure()
    metadata=fields.get("metadata")
    if type(metadata) is not dict or any(type(key) is not str for key in metadata): _failure()
    if (any(type(fields.get(key)) is not tuple for key in ("targets","controls","params"))
            or any(type(wire) is not int or wire<0 for wire in fields["targets"])
            or len(set(fields["targets"]))!=len(fields["targets"])
            or "operation_index" in metadata and type(metadata["operation_index"]) is not int): _failure()
    if (type(fields.get("name")) is not str or type(metadata.get("inverse_of")) is not str
            or metadata.get("label") is not None and type(metadata["label"]) is not str
            or fields.get("_public_matrix_basis") is not None): _failure()
    if (qualification[0] is not metadata or qualification[1] is not fields.get("targets")
            or qualification[2] is not fields.get("controls") or qualification[3] is not fields.get("params")
            or qualification[4] is not owner or fields.get("name")!=qualification[5]
            or metadata.get("inverse_of")!=qualification[6] or metadata.get("label")!=qualification[7]
            or fields.get("_public_matrix_basis")!=qualification[9]): _failure()
    if (fields.get("controls") or fields.get("params") or "_public_frame" in fields or "_shared_owner" in fields
            or set(metadata)-{"matrix","inverse_of","label","operation_index"}): _failure()
    source=object.__getattribute__(owner,"__dict__")
    params=source.get("_params")
    if (type(source.get("name")) is not str or type(source.get("_public_matrix_basis")) is not str
            or source.get("label") is not None and type(source["label"]) is not str): _failure()
    if (source.get("name")!="unitary" or type(source.get("metadata")) is not dict or source["metadata"]
            or source.get("_public_matrix_basis")!="native" or source.get("label")!=qualification[7]
            or type(params) is not tuple or len(params)!=1 or type(params[0]) is not np.ndarray): _failure()
    matrix=params[0]
    width=source.get("num_qubits")
    if type(width) is not int or width!=len(fields["targets"]): _failure()
    if width>current_gate_limits().max_matrix_dimension.bit_length()-1:
        raise MemoryError("Inverse matrix exceeds max_matrix_dimension before allocation.")
    if (metadata.get("matrix") is not matrix or matrix.shape!=(1<<width,)*2
            or matrix.dtype!=np.dtype(complex) or not matrix.flags.c_contiguous or matrix.flags.writeable
            or _digest(matrix)!=qualification[8]): _failure()
    if type(metadata.get("inverse_of")) is not str or type(fields.get("name")) is not str: _failure()
    if fields["name"]!=metadata["inverse_of"]+"_dg": _failure()
    return owner


def seal_inverse_copy(operation):
    """Seal an already admitted, memo-reconstructed witness before it escapes."""
    fields=object.__getattribute__(operation,"__dict__")
    owner=fields["_native_inverse_owner"]
    params=object.__getattribute__(owner,"__dict__")["_params"]
    params[0].flags.writeable=False
    inverse_matrix_owner(operation)


def primitive_matrix_inverse(operation):
    """Qualify only a real constructor invocation, never deserialized metadata."""
    from .operations import Operation
    from .standard_gates import UnitaryGate
    from .gate_limits import current_gate_limits
    from quantumbridge.execution.array_reservations import numeric_shape,numeric_array
    from quantumbridge.execution.array_ownership import retained_array
    from quantumbridge.execution.resource_context import current_resources
    if type(operation) is not Operation: return None
    prior=inverse_matrix_owner(operation)
    metadata=operation.metadata
    if prior is None:
        if (operation.name!="unitary" or type(metadata) is not dict
                or any(type(key) is not str for key in metadata)
                or set(metadata)-{"matrix","label","operation_index"} or "matrix" not in metadata): return None
    if operation.controls or operation.params: return None
    if object.__getattribute__(operation,"__dict__").get("_public_matrix_basis") not in (None,"native"): return None
    label=metadata.get("label")
    if label is not None and type(label) is not str: _failure()
    if "operation_index" in metadata and type(metadata["operation_index"]) is not int: _failure()
    width=len(operation.targets)
    if width>current_gate_limits().max_matrix_dimension.bit_length()-1:
        raise MemoryError("Inverse matrix exceeds max_matrix_dimension before allocation.")
    shape=numeric_shape(metadata["matrix"])
    if shape!=(1<<width,)*2: _failure()
    current_gate_limits().check("max_decomposition_work",(1<<width)**2)
    matrix=numeric_array(metadata["matrix"])
    scope=current_resources()
    adjoint=retained_array(None if scope is None else scope.ledger,"native_inverse_matrix_adjoint",shape,complex)
    np.conjugate(matrix.T,out=adjoint)
    owner=UnitaryGate(adjoint,label=label)
    object.__setattr__(owner,"_public_matrix_basis","native")
    snapshot=object.__getattribute__(owner,"_params")[0]
    snapshot.flags.writeable=False
    derived={key:value for key,value in metadata.items() if key!="matrix"}
    derived.update(matrix=snapshot,inverse_of=operation.name)
    result=Operation(operation.name+"_dg",operation.targets,metadata=derived)
    object.__setattr__(result,"_native_inverse_owner",owner)
    qualification=(result.metadata,result.targets,result.controls,result.params,owner,
        result.name,operation.name,label,_digest(snapshot),None)
    object.__setattr__(result,"_native_inverse_qualification",qualification)
    return result


def canonical_inverse_matrix(operation,limits=None):
    from .operations import Operation
    owner=inverse_matrix_owner(operation)
    if owner is None: return None
    if limits is not None:
        size=object.__getattribute__(owner,"_params")[0].size
        limits.check("max_numeric_elements",size,format="qpy",version=13)
        limits.check("max_input_bytes",size*16,format="qpy",version=13)
    # Existing owned-copy admission keeps the private witness off public handles.
    snapshot=owner.copy()
    metadata={key:value for key,value in operation.metadata.items() if key in {"label","operation_index"}}
    result=Operation("unitary",operation.targets,metadata={**metadata,"gate":snapshot})
    object.__setattr__(result,"_public_matrix_basis","native")
    return result
