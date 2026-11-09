# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# Native CPTP algebra. Error matrices use public little-endian qarg ordering.
from itertools import product
from numbers import Integral, Real
import numpy as np
from quantumbridge.utils.math import validate_kraus_operators, pauli_matrix
from quantumbridge.execution.resource_context import current_resources


class QuantumError:
    def __init__(self, operators, name="kraus", *, max_kraus_ops=256, max_dimension=64):
        scope = current_resources()
        if scope is None:operators = tuple(operators)
        else:
            from quantumbridge.execution.array_reservations import numeric_operator_sequence, numeric_shape
            operators = numeric_operator_sequence(operators, max_count=max_kraus_ops, copy_dtype=complex)
        if not operators or len(operators) > max_kraus_ops:
            raise ValueError("Quantum error exceeds the Kraus construction budget.")
        if scope is None:
            first = np.asarray(operators[0])
            shape = first.shape
        else:shape = numeric_shape(operators[0])
        dimension = shape[0] if len(shape) == 2 else 0
        if dimension < 2 or dimension & (dimension-1) or dimension > max_dimension:
            raise ValueError("Quantum error needs a bounded power-of-two dimension.")
        validated = validate_kraus_operators(operators, dimension)
        self._operators = tuple(op.copy() for op in validated) if scope is None else validated
        self.num_qubits = dimension.bit_length()-1
        self.name = str(name)
        self.provenance = {"implementation":"quantumbridge_native", "mathematics":"CPTP Kraus", "matrix_order":"little-endian qargs"}

    def kraus(self):
        if current_resources() is not None:
            from quantumbridge.execution.array_reservations import numeric_array_copies
            return numeric_array_copies(self._operators, category="quantum_error_export")
        return tuple(op.copy() for op in self._operators)

    def compose(self, other, *, front=False):
        if not isinstance(other, QuantumError) or self.num_qubits != other.num_qubits:
            raise ValueError("QuantumError composition requires equal widths.")
        if len(self._operators)*len(other._operators) > 256:
            raise MemoryError("Composed error exceeds the Kraus construction budget.")
        left, right = (self,other) if front else (other,self)
        if current_resources() is not None:return self._managed_products(left, right, "compose")
        return QuantumError([a@b for a in left._operators for b in right._operators], "compose")

    def tensor(self, other):
        if not isinstance(other, QuantumError): raise TypeError("Expected native QuantumError.")
        if len(self._operators)*len(other._operators)>256 or self.num_qubits+other.num_qubits>6:
            raise MemoryError("Tensor error exceeds the construction budget.")
        if current_resources() is not None:return self._managed_products(self, other, "tensor")
        return QuantumError([np.kron(a,b) for a in self._operators for b in other._operators], "tensor")

    @staticmethod
    def _managed_products(left, right, kind):
        from quantumbridge.execution.array_reservations import owned_arrays, numeric_array, matmul_into, kron_into
        scope = current_resources()
        dimension = 1 << (left.num_qubits if kind == "compose" else left.num_qubits+right.num_qubits)
        count = len(left._operators)*len(right._operators)
        scope.ledger.preflight(memory_bytes=2*count*16*dimension**2+33*dimension)
        outputs = owned_arrays([((dimension, dimension), complex)]*count, category="quantum_error_"+kind)
        for out, (a, b) in zip(outputs, product(left._operators, right._operators)):
            a, b = numeric_array(a), numeric_array(b)
            if kind == "compose":matmul_into(a, b, out, scope=scope)
            else:kron_into(a,b,out,scope=scope)
        return QuantumError(outputs, kind)

    def expand(self, other): return other.tensor(self)

    def to_quantumchannel(self):
        from quantumbridge.information import Kraus
        return Kraus(self.kraus())

    def to_dict(self):
        return {"format":"qb-quantum-error-v1", "name":self.name,
                "kraus":[{"real":op.real.tolist(),"imag":op.imag.tolist()} for op in self._operators]}

    @classmethod
    def from_dict(cls, value):
        from quantumbridge.ir.qb_ir import matrix_from_ir
        if value.get("format")!="qb-quantum-error-v1": raise ValueError("Unknown quantum-error schema.")
        return cls([matrix_from_ir(op) for op in value["kraus"]],value["name"])


def probability(value):
    if isinstance(value,bool) or not isinstance(value,Real) or not np.isfinite(value) or not 0<=value<=1:
        raise ValueError("Expected a finite noise probability in [0,1].")
    return float(value)


def kraus_error(operators, canonical_kraus=False):
    error = QuantumError(operators)
    if not isinstance(canonical_kraus,bool): raise TypeError("canonical_kraus must be boolean.")
    if not canonical_kraus: return error
    if error.num_qubits>4:raise MemoryError("Canonical Kraus Choi diagonalization exceeds dimension budget.")
    if current_resources() is not None:
        from quantumbridge.execution import mps_arrays as arrays
        from quantumbridge.execution.array_ownership import retained_array
        from quantumbridge.execution.array_reservations import owned_arrays
        dimension = 1 << error.num_qubits
        size = dimension**2
        choi = retained_array(current_resources().ledger, "canonical_kraus_choi", (size, size), complex)
        for row in range(size):
            for column in range(size):
                choi[row, column] = sum(complex(op[row % dimension, row // dimension])*complex(op[column % dimension, column // dimension]).conjugate() for op in error._operators)
        values, vectors = arrays.decomposed(choi, "eigh")
        selected = [index for index, value in enumerate(values) if value > 1e-14]
        outputs = owned_arrays([((dimension, dimension), complex)]*len(selected), category="canonical_kraus_operator")
        for output, index in zip(outputs, selected):
            np.multiply(vectors[:, index].reshape(dimension, dimension, order="F"), np.sqrt(values[index]), out=output)
        return QuantumError(outputs)
    choi = sum(np.outer(op.reshape(-1,order="F"),op.reshape(-1,order="F").conj()) for op in error.kraus())
    values,vectors = np.linalg.eigh(choi)
    dimension = 1<<error.num_qubits
    return QuantumError([np.sqrt(value)*vector.reshape(dimension,dimension,order="F") for value,vector in zip(values,vectors.T) if value>1e-14])


def mixed_unitary_error(entries):
    scope = current_resources()
    if scope is not None and type(entries) is not tuple and type(entries) is not list:
        from quantumbridge.core.classical import UnsupportedFeature
        raise UnsupportedFeature("managed_unitary_mixture", "Managed unitary mixtures require a builtin entry sequence.")
    entries = tuple(entries)
    if not entries or len(entries)>256: raise ValueError("Invalid mixed-unitary size.")
    probabilities = [probability(p) for _,p in entries]
    if not np.isclose(sum(probabilities),1,atol=1e-12,rtol=0): raise ValueError("Mixture weights must sum to one.")
    operators=[]
    if scope is not None:
        from quantumbridge.execution.array_reservations import numeric_shape, numeric_array, owned_arrays
        from quantumbridge.utils.math import validate_unitary_matrix
        shapes = [numeric_shape(entry[0]) for entry in entries]
        if any(len(shape)!=2 or shape[0]!=shape[1] or shape[0]>64 for shape in shapes):
            raise ValueError("Mixture entries must be finite unitaries.")
        size = sum(16*shape[0]*shape[1] for shape in shapes)
        scope.ledger.preflight(memory_bytes=2*size+33*max(shape[0] for shape in shapes))
        outputs = owned_arrays([(shape, complex) for shape in shapes], category="mixed_unitary_operator")
        for (raw, _), p, output in zip(entries, probabilities, outputs):
            matrix = numeric_array(raw, category="mixed_unitary_conversion")
            if not validate_unitary_matrix(matrix):raise ValueError("Mixture entries must be finite unitaries.")
            np.multiply(matrix, np.sqrt(p), out=output)
        return QuantumError(outputs,"mixed_unitary")
    for (matrix,_),p in zip(entries,probabilities):
        matrix=np.asarray(matrix,complex)
        if matrix.ndim!=2 or matrix.shape[0]!=matrix.shape[1] or matrix.shape[0]>64 or not np.all(np.isfinite(matrix)) or not np.allclose(matrix.conj().T@matrix,np.eye(len(matrix)),atol=1e-10,rtol=0):
            raise ValueError("Mixture entries must be finite unitaries.")
        operators.append(np.sqrt(p)*matrix)
    return QuantumError(operators,"mixed_unitary")


def coherent_unitary_error(matrix): return mixed_unitary_error([(matrix,1.)])


def pauli_error(entries):
    from quantumbridge.operators import Pauli
    entries=tuple(entries)
    if not entries or len(entries)>256: raise ValueError("Invalid Pauli mixture size.")
    width=len(entries[0][0])
    if not 1<=width<=6 or any(not isinstance(label,str) or len(label)!=width or set(label)-set("IXYZ") for label,_ in entries):
        raise ValueError("Pauli errors require equal bounded nonempty Pauli labels.")
    error=mixed_unitary_error([(Pauli(label).to_matrix(),p) for label,p in entries])
    error.name="pauli"
    return error


def depolarizing_error(param, num_qubits):
    if isinstance(num_qubits,bool) or not isinstance(num_qubits,Integral) or not 1<=num_qubits<=4:
        raise ValueError("Depolarizing width exceeds the explicit construction budget [1,4].")
    count=4**num_qubits
    if isinstance(param,bool) or not isinstance(param,Real) or not np.isfinite(param) or not 0<=param<=count/(count-1):
        raise ValueError("Invalid depolarizing parameter for this width.")
    entries=[("".join(label),float(param)/count) for label in product("IXYZ",repeat=num_qubits)]
    entries[0]=(entries[0][0],1-param+param/count)
    return pauli_error(entries)


def reset_error(prob0, prob1=0):
    p0,p1=probability(prob0),probability(prob1)
    if p0+p1>1: raise ValueError("Reset probabilities exceed one.")
    a,b,c=np.sqrt(1-p0-p1),np.sqrt(p0),np.sqrt(p1)
    return QuantumError([[[a,0],[0,a]],[[b,0],[0,0]],[[0,b],[0,0]],
                         [[0,0],[c,0]],[[0,0],[0,c]]],"reset")


def phase_amplitude_damping_error(param_amp, param_phase, excited_state_population=0, canonical_kraus=True):
    a,b,p=probability(param_amp),probability(param_phase),probability(excited_state_population)
    if a+b>1: raise ValueError("Amplitude and phase damping probabilities exceed one.")
    root=np.sqrt(1-a-b);low,high=np.sqrt(1-p),np.sqrt(p)
    down,phase=np.sqrt(a),np.sqrt(b)
    ops=[[[low,0],[0,low*root]],[[0,low*down],[0,0]],[[0,0],[0,low*phase]],
         [[high*root,0],[0,high]],[[0,0],[high*down,0]],[[high*phase,0],[0,0]]]
    result=kraus_error(ops,canonical_kraus); result.name="phase_amplitude_damping"
    return result


def amplitude_damping_error(param_amp, excited_state_population=0, canonical_kraus=True):
    return phase_amplitude_damping_error(param_amp,0,excited_state_population,canonical_kraus)


def phase_damping_error(param_phase, canonical_kraus=True):
    return phase_amplitude_damping_error(0,param_phase,0,canonical_kraus)


def thermal_relaxation_error(t1,t2,time,excited_state_population=0):
    if any(isinstance(v,bool) or not isinstance(v,Real) or np.isnan(v) or v<=0 for v in (t1,t2)) or t2>2*t1:
        raise ValueError("Thermal times require T1,T2>0 and T2<=2*T1.")
    if isinstance(time,bool) or not isinstance(time,Real) or not np.isfinite(time) or time<0:
        raise ValueError("Thermal duration must be finite and nonnegative.")
    p=probability(excited_state_population)
    gamma=-np.expm1(-time/t1)
    rate=max(0.,1/t2-1/(2*t1))
    dephase=-np.expm1(-2*time*rate)
    error=amplitude_damping_error(gamma,p).compose(phase_damping_error(dephase))
    error.name="thermal_relaxation"
    return error
