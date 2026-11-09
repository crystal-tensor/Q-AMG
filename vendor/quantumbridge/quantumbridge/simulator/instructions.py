# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# Executable native events, stored as ordinary Circuit operations and safe IR data.
from copy import deepcopy
import numpy as np
from quantumbridge.core.classical import UnsupportedFeature
from quantumbridge.ir.qb_ir import _matrix_to_ir


def save(circuit,kind,wires=None,*,label=None,pershot=False,conditional=False,unnormalized=False,indices=None,observable=None):
    from quantumbridge.devices.execution import validate_wires
    selected=validate_wires(wires,circuit.num_qubits)
    if not selected: raise ValueError("Save instruction needs nonempty wires.")
    if any(not isinstance(v,bool) for v in (pershot,conditional,unnormalized)): raise TypeError("Save modes must be boolean.")
    label=kind if label is None else label
    if not isinstance(label,str) or not label: raise ValueError("Save label must be nonempty text.")
    data={"kind":"save_"+kind,"label":label,"pershot":pershot,"conditional":conditional,"unnormalized":unnormalized}
    if indices is not None:
        from numbers import Integral
        indices=tuple(indices)
        if not indices or any(isinstance(i,bool) or not isinstance(i,Integral) or not 0<=i<1<<circuit.num_qubits for i in indices): raise ValueError("Invalid saved amplitude indices.")
        data["indices"]=list(map(int,indices))
    if observable is not None:
        from quantumbridge.devices.matrix_product_state_device import MatrixProductStateDevice
        operator=MatrixProductStateDevice._paulis(observable,len(selected))
        if operator is not None: data["paulis"]=[[label,float(coefficient.real)] for label,coefficient in operator.terms]
        else:
            from quantumbridge.utils.math import observable_matrix
            if len(selected)>10: raise MemoryError("Dense saved observable exceeds construction budget.")
            matrix=observable_matrix(observable,len(selected))
            if not np.all(np.isfinite(matrix)) or not np.allclose(matrix,matrix.conj().T,atol=1e-10,rtol=0): raise ValueError("Saved observable must be finite Hermitian.")
            data["observable"]=_matrix_to_ir(matrix)
    return circuit.append("save_"+kind,selected,metadata={"simulator_instruction":data})


def set_state(circuit,kind,value):
    if kind=="matrix_product_state":
        from quantumbridge.information import MatrixProductState
        state=MatrixProductState.from_dict(value) if isinstance(value,dict) else value
        if not isinstance(state,MatrixProductState) or state.num_qubits!=circuit.num_qubits: raise ValueError("MPS replacement width/type mismatch.")
        payload=state.to_dict()
    elif kind=="stabilizer":
        from quantumbridge.information import StabilizerState,Clifford
        state=StabilizerState(Clifford.from_dict(value)) if isinstance(value,dict) else StabilizerState(value)
        if state.num_qubits!=circuit.num_qubits or not state.is_valid(): raise ValueError("Invalid replacement stabilizer.")
        payload=state.clifford.to_dict()
    else:
        array=np.asarray(getattr(value,"data",value),complex)
        dimension=1<<circuit.num_qubits
        if array.nbytes>256*1024*1024: raise MemoryError("Set state exceeds construction memory budget.")
        expected=(dimension,) if kind=="statevector" else (dimension,dimension) if kind in {"density_matrix","unitary"} else (dimension**2,dimension**2)
        if array.shape!=expected or not np.all(np.isfinite(array)): raise ValueError("Set state shape/finite-value mismatch.")
        if kind=="statevector" and not np.isclose(np.vdot(array,array),1,atol=1e-10,rtol=0): raise ValueError("Replacement vector is not normalized.")
        if kind=="density_matrix":
            from quantumbridge.information import DensityMatrix
            if not DensityMatrix(array).is_valid(atol=1e-10,rtol=0): raise ValueError("Replacement density is not positive Hermitian trace-one.")
        if kind=="unitary" and not np.allclose(array.conj().T@array,np.eye(dimension),atol=1e-10,rtol=0): raise ValueError("Replacement unitary is not unitary.")
        if kind=="superop":
            from quantumbridge.information import SuperOp
            if not SuperOp(array).is_cptp(): raise ValueError("Replacement superoperator must be CPTP.")
        payload=_matrix_to_ir(array)
    return circuit.append("set_"+kind,tuple(range(circuit.num_qubits)),metadata={"simulator_instruction":{"kind":"set_"+kind,"value":deepcopy(payload)}})
