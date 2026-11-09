# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is part of QuantumBridge SDK.
# This implementation is developed for the QuantumBridge native architecture.

from __future__ import annotations

from functools import lru_cache
from numbers import Integral

import numpy as np

from quantumbridge.core import Circuit
from quantumbridge.core.operations import Instruction
from quantumbridge.information.operator import Operator
from quantumbridge.operators import Pauli
from quantumbridge.operators.pauli import _PHASE_FACTORS, _validate_qargs
from quantumbridge.utils.math import gate_matrix
from quantumbridge.execution import mps_arrays as arrays
from quantumbridge.execution.array_reservations import numeric_array, numeric_shape, numeric_into, _numeric_leaves
from quantumbridge.execution.array_ownership import copied_array, retain_reservation
from quantumbridge.execution.resource_context import current_resources


def _workspace_notice():
    scope=current_resources()
    if scope is not None:
        scope.ledger.unavailable("tableau_python_numpy_workspace",
            "Known ndarray roots are admitted; Python labels, RNG state, scalar/iterator workspace and bounded immutable gate-cache objects are not measured RSS.")


def _clifford_matrix_origin(owner, *, dense=False):
    from quantumbridge.core.classical import UnsupportedFeature
    from quantumbridge.core.composition import _admit_graph
    from quantumbridge.core.gate_limits import current_gate_limits
    if type(owner) is not Clifford:raise UnsupportedFeature("clifford_matrix_owner")
    fields = object.__getattribute__(owner, "__dict__")
    width, tableau, basis = (fields.get(key) for key in ("num_qubits", "tableau", "_public_matrix_basis"))
    if basis is not None and (type(basis) is not str or basis not in ("native", "unresolved")):
        raise UnsupportedFeature("clifford_matrix_origin")
    if type(width) is not int or width < 0:raise ValueError("Clifford width must be a native nonnegative integer.")
    limits = current_gate_limits()
    limits.check("max_circuit_bits", width)
    if type(tableau) is not np.ndarray or tableau.dtype != np.dtype(bool):
        raise UnsupportedFeature("clifford_matrix_storage")
    if tableau.shape != (2 * width, 2 * width + 1):raise ValueError("Clifford width differs from its tableau shape.")
    limits.check("max_decomposition_work", tableau.size)
    if dense:
        if width > limits.max_matrix_dimension.bit_length() - 1:
            raise MemoryError("Clifford matrix exceeds max_matrix_dimension before synthesis.")
        limits.check("max_decomposition_work", (1 << width) ** 2)
        # Validate the signed symplectic owner without an array workspace.
        for left in range(2 * width):
            for right in range(left + 1, 2 * width):
                if _symplectic_product(tableau[left, :-1], tableau[right, :-1]) != int(right == left + width):
                    raise ValueError("Clifford tableau is not symplectic.")
    _admit_graph(owner)
    return basis


def _with_clifford_origin(result, origin):
    if origin is not None:result._public_matrix_basis = origin
    return result


def _matrix_input(value,category):
    return (np.asarray(value,dtype=complex) if current_resources() is None else
        numeric_array(value,category=category))


def _label_array(labels, width, category):
    dtype=np.dtype(f"U{max(1,width)}")
    amount=len(labels)*dtype.itemsize
    if amount>np.iinfo(np.intp).max:raise MemoryError("Tableau label array exceeds addressable storage.")
    scope=current_resources()
    ticket=None if scope is None else scope.ledger.reserve(category,memory_bytes=amount)
    try:
        output=np.asarray(labels,dtype=dtype)
        if ticket is not None:
            ticket.observe(output.nbytes)
            retain_reservation(ticket,output)
        return output
    except BaseException:
        if ticket is not None:ticket.release()
        raise


def _stack_binary(*matrices):
    columns=matrices[0].shape[1]
    if any(matrix.ndim!=2 or matrix.shape[1]!=columns for matrix in matrices):
        raise ValueError("Binary equation widths must agree.")
    result=arrays.owned("tableau_equation_stack",(sum(len(matrix) for matrix in matrices),columns),bool)
    offset=0
    for matrix in matrices:
        np.copyto(result[offset:offset+len(matrix)],matrix)
        offset+=len(matrix)
    return result


def _swap_halves(matrix):
    width=matrix.shape[1]//2
    result=arrays.owned("tableau_symplectic_constraints",matrix.shape,bool)
    np.copyto(result[:,:width],matrix[:,width:])
    np.copyto(result[:,width:],matrix[:,:width])
    return result


def _symplectic_product(left, right) -> int:
    width = len(left) // 2
    return sum((bool(left[index]) and bool(right[width+index]))^
        (bool(left[width+index]) and bool(right[index])) for index in range(width))%2


def _gf2_basis(rows) -> np.ndarray:
    matrix = numeric_array(rows,dtype=bool,copy=True,category="tableau_gf2_basis")
    if matrix.ndim!=2:raise ValueError("Binary row reduction requires a matrix.")
    rank = 0
    for column in range(matrix.shape[1]):
        pivot=next((row for row in range(rank,len(matrix)) if matrix[row,column]),None)
        if pivot is None:
            continue
        if pivot!=rank:
            for index in range(matrix.shape[1]):matrix[rank,index],matrix[pivot,index]=matrix[pivot,index],matrix[rank,index]
        for row in range(len(matrix)):
            if row!=rank and matrix[row,column]:np.logical_xor(matrix[row],matrix[rank],out=matrix[row])
        rank += 1
        if rank == matrix.shape[0]:
            break
    return matrix[:rank]


def _gf2_solve(matrix, rhs) -> np.ndarray:
    coefficients=numeric_array(matrix,dtype=bool,category="tableau_equation_input")
    target=numeric_array(rhs,dtype=bool,category="tableau_target_input")
    if coefficients.ndim!=2 or target.shape!=(len(coefficients),):raise ValueError("Binary equation target dimensions do not match.")
    augmented=arrays.owned("tableau_gf2_augmented",(len(coefficients),coefficients.shape[1]+1),bool)
    np.copyto(augmented[:,:-1],coefficients)
    np.copyto(augmented[:,-1],target)
    rank = 0
    pivots = []
    for column in range(coefficients.shape[1]):
        pivot=next((row for row in range(rank,len(augmented)) if augmented[row,column]),None)
        if pivot is None:
            continue
        if pivot!=rank:
            for index in range(augmented.shape[1]):augmented[rank,index],augmented[pivot,index]=augmented[pivot,index],augmented[rank,index]
        for row in range(len(augmented)):
            if row!=rank and augmented[row,column]:np.logical_xor(augmented[row],augmented[rank],out=augmented[row])
        pivots.append(column)
        rank += 1
        if rank == len(augmented):
            break
    if any(not any(row[:-1]) and row[-1] for row in augmented):
        raise ValueError("QuantumBridge binary constraints are inconsistent.")
    solution=arrays.filled("tableau_gf2_solution",(coefficients.shape[1],),bool,False)
    for row, column in enumerate(pivots):
        solution[column] = augmented[row, -1]
    return solution


def _matrix_pauli(matrix, atol: float = 1e-8) -> Pauli:
    data=_matrix_input(matrix,"tableau_pauli_matrix_input")
    if data.ndim != 2:
        raise ValueError("QuantumBridge Pauli matrix must be two-dimensional.")
    dimension = data.shape[0]
    if data.shape != (dimension, dimension) or dimension == 0 or dimension & (dimension - 1):
        raise ValueError("QuantumBridge Pauli matrix must have power-of-two square dimensions.")
    width = dimension.bit_length() - 1
    first_column=[index for index in range(dimension) if abs(data[index,0])>atol]
    if len(first_column) != 1:
        raise ValueError("QuantumBridge unitary does not normalize the Pauli group.")
    shift = int(first_column[0])
    amplitude = data[shift, 0]
    symbols=[]
    zmask=0
    for qubit in range(width):
        column = 1 << qubit
        ratio = data[shift ^ column, column] / amplitude
        if abs(ratio+1)<=atol:
            zmask|=1<<qubit
        elif not abs(ratio-1)<=atol:
            raise ValueError("QuantumBridge unitary does not normalize the Pauli group.")
        xb,zb=bool(shift&(1<<qubit)),bool(zmask&(1<<qubit))
        symbols.append("Y" if xb and zb else "X" if xb else "Z" if zb else "I")
    factor=amplitude/(1j**symbols.count("Y"))
    matches=[phase for phase,value in enumerate(_PHASE_FACTORS) if abs(factor-value)<=atol]
    if not matches:
        raise ValueError("QuantumBridge Pauli image has an invalid phase.")
    result=Pauli("".join(reversed(symbols)))
    result.phase=matches[0]
    phase=_PHASE_FACTORS[result.phase]*(1j**symbols.count("Y"))
    for column in range(dimension):
        value=-phase if (column&zmask).bit_count()%2 else phase
        for row in range(dimension):
            if not abs(data[row,column]-(value if row==column^shift else 0))<=atol:
                raise ValueError("QuantumBridge unitary does not normalize the Pauli group.")
    return result


@lru_cache(maxsize=128)
def _small_gate_table(matrix_bytes: bytes, width: int):
    dimension = 1 << width
    matrix = np.frombuffer(matrix_bytes, dtype=np.complex128).reshape(dimension, dimension)
    from quantumbridge.utils.math import validate_unitary_matrix
    if not validate_unitary_matrix(matrix,atol=1e-8,rtol=0):
        raise ValueError("QuantumBridge Clifford gate must be unitary.")
    result=[]
    adjoint=arrays.conjugated(matrix).T
    for code in range(4 ** width):
        digits = [(code >> (2 * qubit)) & 3 for qubit in range(width)]
        symbols="".join(reversed(["Y" if digit==3 else "Z" if digit==2 else "X" if digit==1 else "I" for digit in digits]))
        image=arrays.contracted(arrays.contracted(matrix,Pauli(symbols).to_matrix()),adjoint)
        pauli=_matrix_pauli(image)
        if pauli.phase % 2:
            raise ValueError("QuantumBridge Clifford images must be Hermitian.")
        result.append(tuple(char in "XY" for char in reversed(pauli._body))+
            tuple(char in "ZY" for char in reversed(pauli._body))+(bool(pauli.phase//2),))
    return tuple(result)


class Clifford:
    """Native Clifford group element stored as a signed binary symplectic tableau."""

    CLIFFORD_GATES = {"id", "i", "h", "s", "sdg", "sx", "sxdg", "x", "y", "z", "cx", "cy", "cz", "swap", "dcx", "ecr", "iswap"}

    def __setattr__(self, name, value):
        if name == "_public_matrix_basis" and name in self.__dict__:
            raise AttributeError("Clifford matrix origin cannot be overwritten.")
        object.__setattr__(self, name, value)

    def __init__(self, data, validate: bool = True, copy: bool = True):
        from quantumbridge.core.gates import Gate, ControlledGate
        from quantumbridge.core.gates import is_native_instruction
        from quantumbridge.core.circuit import is_native_circuit
        _workspace_notice()
        origin = None
        if type(data) is Clifford:
            origin = _clifford_matrix_origin(data)
            tableau=copied_array(data.tableau,category="clifford_tableau_copy") if copy else data.tableau
        elif is_native_circuit(data) or is_native_instruction(data):
            converted = self.from_circuit(data)
            origin = _clifford_matrix_origin(converted)
            tableau = converted.tableau
        elif type(data) is Pauli:
            tableau = self.from_label(data._body).tableau
        elif issubclass(type(data), Operator):
            from quantumbridge.information.operator import _admit_public_arithmetic_owner
            origin = _admit_public_arithmetic_owner(data)
            if origin == "hybrid":
                data = data.as_native_operator()
                origin = "native"
            elif origin == "native" and data.num_qubits is None:
                from math import isfinite
                size = data.data.shape[0]
                if (data.data.shape != (size, size) or size & (size - 1)
                        or data._input_dims != data._output_dims):
                    raise ValueError("Mixed-radix Clifford construction needs matching power-of-two square partitions.")
                if any(not isfinite(value.real) or not isfinite(value.imag) for value in data.data.flat):
                    raise ValueError("Mixed-radix Clifford input must be finite.")
                # A qudit partition is not the tableau's binary partition.
                # Materialize its declared public frame before interpreting
                # the numerical map as a projective qubit Clifford.
                data = data.as_qiskit_operator()
                origin = None
            tableau = self.from_matrix(data.data).tableau
        elif type(data) is dict:
            tableau = self.from_dict(data).tableau
        else:
            if current_resources() is not None:
                shape=numeric_shape(data)
                raw=data
                if issubclass(type(data),np.ndarray):
                    raw=np.ndarray.view(data,np.ndarray)
                    complex_input=raw.dtype.kind=="c"
                else:
                    complex_input=any(isinstance(value,(complex,np.complexfloating)) for value in _numeric_leaves(data))
            else:
                raw=np.asarray(data)
                shape=raw.shape
                complex_input=np.iscomplexobj(raw)
            if complex_input:
                tableau = self.from_matrix(raw).tableau
            else:
                if len(shape)!=2 or shape[0]%2 or shape[1] not in (shape[0],shape[0]+1):
                    raise ValueError("QuantumBridge Clifford tableau must have shape (2n, 2n) or (2n, 2n+1).")
                native=issubclass(type(raw),np.ndarray)
                if native:raw=np.ndarray.view(raw,np.ndarray)
                if not (native and raw.dtype==np.dtype(bool)) and any(value not in (0,1) for value in _numeric_leaves(raw)):
                    raise ValueError("QuantumBridge Clifford tableau entries must be binary.")
                if shape[1]==shape[0]:
                    tableau=arrays.filled("clifford_tableau_with_phase",(shape[0],shape[0]+1),bool,False)
                    numeric_into(tableau[:,:-1],raw)
                elif native:
                    tableau=raw if not copy and raw.dtype==np.dtype(bool) else arrays.copied(raw,dtype=bool,category="clifford_tableau_input")
                else:tableau=numeric_array(raw,dtype=bool,copy=True,category="clifford_tableau_input")
        self.tableau = tableau
        self.num_qubits = len(tableau) // 2
        if validate and not self.is_unitary():
            raise ValueError("QuantumBridge Clifford tableau is not symplectic.")
        _with_clifford_origin(self, origin)

    @classmethod
    def identity(cls, num_qubits: int) -> "Clifford":
        if isinstance(num_qubits,bool) or not isinstance(num_qubits, Integral) or num_qubits < 0:
            raise ValueError("QuantumBridge Clifford width must be a non-negative integer.")
        width=int(num_qubits)
        tableau=arrays.filled("clifford_identity",(2*width,2*width+1),bool,False)
        for index in range(2*width):tableau[index,index]=True
        return cls(tableau,validate=False,copy=False)

    @classmethod
    def from_label(cls, label: str) -> "Clifford":
        if type(label) is not str or any(char not in "IXYZHS" for char in label):
            raise ValueError("QuantumBridge Clifford labels support I, X, Y, Z, H and S.")
        result = cls.identity(len(label))
        for wire, char in enumerate(label):
            if char != "I":
                result._append_gate(char.lower(), (len(label) - 1 - wire,))
        return result

    @classmethod
    def from_circuit(cls, circuit) -> "Clifford":
        from quantumbridge.core.circuit import is_native_circuit
        from quantumbridge.core.gates import is_native_instruction
        from quantumbridge.core.composition import _admit_graph
        from quantumbridge.information.operator import _operator_matrix_origin
        if not is_native_circuit(circuit) and not is_native_instruction(circuit):
            raise TypeError("Clifford.from_circuit requires a native circuit or instruction owner.")
        _admit_graph(circuit)
        origin = _operator_matrix_origin(circuit)
        if isinstance(circuit, Instruction):
            if circuit.metadata.get("condition") is not None or circuit.metadata.get("control_flow") is not None:
                raise ValueError("QuantumBridge Clifford cannot represent classical control flow.")
            if circuit.definition is not None:
                circuit = circuit.definition
            else:
                circuit = Circuit(circuit.num_qubits).append(circuit, range(circuit.num_qubits))
        if not isinstance(circuit, Circuit):
            raise TypeError("QuantumBridge Clifford.from_circuit requires a native Circuit or Instruction.")
        circuit = circuit.lowered()
        if circuit.measurements:
            raise ValueError("QuantumBridge Clifford cannot represent measurement or reset.")
        result = cls.identity(circuit.num_qubits)
        for operation in circuit.operations:
            if operation.metadata.get("condition") is not None or operation.metadata.get("control_flow") is not None:
                raise ValueError("QuantumBridge Clifford cannot represent classical control flow.")
            if operation.name in {"barrier", "delay"} and operation.metadata.get("directive"):
                continue
            if operation.name in {"measure", "reset"}:
                raise ValueError("QuantumBridge Clifford cannot represent measurement or reset.")
            try:
                matrix = gate_matrix(operation.name, operation.params, operation.metadata)
                qubits = tuple(circuit.num_qubits - 1 - wire for wire in reversed(operation.wires))
                result._append_matrix(matrix, qubits)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"QuantumBridge operation {operation.name!r} is not a supported Clifford gate.") from exc
        return _with_clifford_origin(result, origin)

    @classmethod
    def from_matrix(cls, matrix) -> "Clifford":
        data=_matrix_input(matrix,"clifford_matrix_input")
        if data.ndim != 2 or data.shape[0] != data.shape[1]:
            raise ValueError("QuantumBridge Clifford matrix must be square.")
        dimension = data.shape[0]
        if dimension == 0 or dimension & (dimension - 1):
            raise ValueError("QuantumBridge Clifford matrix dimensions must be powers of two.")
        from quantumbridge.utils.math import validate_unitary_matrix
        if not validate_unitary_matrix(data,atol=1e-8,rtol=0):
            raise ValueError("QuantumBridge Clifford matrix must be unitary.")
        result = cls.identity(dimension.bit_length() - 1)
        adjoint=arrays.conjugated(data).T
        for row in range(2 * result.num_qubits):
            image=_matrix_pauli(arrays.contracted(arrays.contracted(data,result._row_pauli(row).to_matrix()),adjoint))
            result._set_row(row, image)
        return result

    @classmethod
    def from_operator(cls, operator) -> "Clifford":
        if issubclass(type(operator), (Operator, Clifford)):
            if issubclass(type(operator), Clifford):_clifford_matrix_origin(operator)
            return cls(operator)
        if current_resources() is not None:
            from quantumbridge.information.channels import _operator_data
            return cls.from_matrix(operator.to_matrix() if type(operator) in (Clifford,Pauli) else _operator_data(operator))
        matrix = operator.to_matrix() if hasattr(operator, "to_matrix") else operator
        return cls.from_matrix(matrix)

    @classmethod
    def from_dict(cls, obj) -> "Clifford":
        stabilizers = [Pauli(label) for label in obj["stabilizer"]]
        destabilizers = [Pauli(label) for label in obj["destabilizer"]]
        if len(stabilizers) != len(destabilizers):
            raise ValueError("QuantumBridge Clifford stabilizer and destabilizer counts must match.")
        width = len(stabilizers)
        if any(pauli.num_qubits != width for pauli in destabilizers + stabilizers):
            raise ValueError("QuantumBridge Clifford generator widths must match their count.")
        result = cls.identity(width)
        for row, pauli in enumerate(destabilizers + stabilizers):
            result._set_row(row, pauli)
        if not result.is_unitary():
            raise ValueError("QuantumBridge Clifford generators do not form a symplectic basis.")
        return result

    @property
    def dim(self):
        return 1 << self.num_qubits, 1 << self.num_qubits

    @property
    def symplectic_matrix(self):
        return self.tableau[:, :-1]

    @property
    def x(self):
        return self.tableau[:, :self.num_qubits]

    @property
    def z(self):
        return self.tableau[:, self.num_qubits:2 * self.num_qubits]

    @property
    def phase(self):
        return self.tableau[:, -1]

    @property
    def destab(self):
        return self.tableau[:self.num_qubits]

    @property
    def stab(self):
        return self.tableau[self.num_qubits:]

    @property
    def destab_x(self):
        return self.x[:self.num_qubits]

    @property
    def destab_z(self):
        return self.z[:self.num_qubits]

    @property
    def destab_phase(self):
        return self.phase[:self.num_qubits]

    @property
    def stab_x(self):
        return self.x[self.num_qubits:]

    @property
    def stab_z(self):
        return self.z[self.num_qubits:]

    @property
    def stab_phase(self):
        return self.phase[self.num_qubits:]

    def input_dims(self, qargs=None):
        selected = tuple(range(self.num_qubits) if qargs is None else qargs)
        _validate_qargs(selected, len(selected), self.num_qubits)
        return (2,) * len(selected)

    output_dims = input_dims

    def _row_pauli(self, row: int) -> Pauli:
        body="".join("Y" if x and z else "X" if x else "Z" if z else "I"
            for x,z in zip(reversed(self.x[row]),reversed(self.z[row])))
        result=Pauli(body)
        result.phase=2*int(self.phase[row])
        return result

    def _set_row(self, row: int, pauli: Pauli) -> None:
        if pauli.num_qubits != self.num_qubits or pauli.phase % 2:
            raise ValueError("QuantumBridge Clifford generators must be Hermitian Paulis of matching width.")
        for qubit,symbol in enumerate(reversed(pauli._body)):
            self.tableau[row,qubit]=symbol in "XY"
            self.tableau[row,self.num_qubits+qubit]=symbol in "ZY"
        self.phase[row] = bool(pauli.phase // 2)

    def _append_matrix(self, matrix, qubits) -> None:
        selected = _validate_qargs(qubits, len(qubits), self.num_qubits)
        width = len(selected)
        data=_matrix_input(matrix,"clifford_local_gate_input")
        if data.shape != (1 << width, 1 << width):
            raise ValueError("QuantumBridge Clifford gate matrix does not match its qubits.")
        if width > 2:
            composed = self.compose(Clifford.from_matrix(data), qargs=selected)
            self.tableau[:] = composed.tableau
            return
        actions = _small_gate_table(data.tobytes(), width)
        for row in range(2*self.num_qubits):
            code=sum((int(self.x[row,qubit])+2*int(self.z[row,qubit]))<<(2*offset) for offset,qubit in enumerate(selected))
            image=actions[code]
            for offset,qubit in enumerate(selected):
                self.tableau[row,qubit]=image[offset]
                self.tableau[row,self.num_qubits+qubit]=image[width+offset]
            self.tableau[row,-1]^=image[-1]

    def _append_gate(self, name: str, qubits) -> None:
        # Native two-qubit gate matrices list control first (the high local bit).
        selected = tuple(qubits)
        local = selected[::-1] if name in {"cx", "cy"} else selected
        self._append_matrix(gate_matrix(name), local)

    def is_unitary(self) -> bool:
        width = self.num_qubits
        scratch=arrays.owned("clifford_symplectic_validation",(width,),bool)
        for left in range(2*width):
            for right in range(left+1,2*width):
                np.logical_and(self.x[left],self.z[right],out=scratch)
                first=int(np.count_nonzero(scratch))
                np.logical_and(self.z[left],self.x[right],out=scratch)
                actual=(first+int(np.count_nonzero(scratch)))%2
                if actual!=int(right==left+width):return False
        return True

    def copy(self) -> "Clifford":
        return Clifford(self, validate=False)

    def to_labels(self, array: bool = False, mode: str = "B"):
        if mode not in {"B", "D", "S"}:
            raise ValueError("QuantumBridge Clifford label mode must be B, D or S.")
        rows = range(2 * self.num_qubits)
        if mode == "D":
            rows = range(self.num_qubits)
        elif mode == "S":
            rows = range(self.num_qubits, 2 * self.num_qubits)
        labels = [("-" if self.phase[row] else "+") + self._row_pauli(row)._body for row in rows]
        if not array:return labels
        if not labels:return arrays.owned("clifford_empty_labels",(0,),float)
        return _label_array(labels,self.num_qubits+1,"clifford_label_export")

    def to_dict(self):
        return {"stabilizer": self.to_labels(mode="S"), "destabilizer": self.to_labels(mode="D")}

    def conjugate_pauli(self, pauli) -> Pauli:
        operator = pauli if isinstance(pauli, Pauli) else Pauli(pauli)
        if operator.num_qubits != self.num_qubits:
            raise ValueError("QuantumBridge Pauli and Clifford widths must match.")
        result = Pauli("I" * self.num_qubits)
        result.phase = operator.phase + 3 * operator._body.count("Y")
        for qubit,symbol in enumerate(reversed(operator._body)):
            if symbol in "XY":
                result = result.dot(self._row_pauli(qubit))
            if symbol in "ZY":
                result = result.dot(self._row_pauli(self.num_qubits + qubit))
        return result

    def _embedded(self, qargs, width: int) -> "Clifford":
        origin = _clifford_matrix_origin(self)
        selected = _validate_qargs(qargs, self.num_qubits, width)
        result = Clifford.identity(width)
        for index, qubit in enumerate(selected):
            result._set_row(qubit, self._row_pauli(index)._embedded(selected, width))
            result._set_row(width + qubit, self._row_pauli(self.num_qubits + index)._embedded(selected, width))
        return _with_clifford_origin(result, origin)

    def compose(self, other, qargs=None, front: bool = False) -> "Clifford":
        origin = _clifford_matrix_origin(self)
        operand = other if isinstance(other, Clifford) else Clifford(other)
        other_origin = _clifford_matrix_origin(operand)
        if qargs is not None:
            operand = operand._embedded(qargs, self.num_qubits)
        if self.num_qubits != operand.num_qubits:
            raise ValueError("QuantumBridge Clifford composition requires matching widths.")
        first, second = (operand, self) if front else (self, operand)
        result = Clifford.identity(self.num_qubits)
        for row in range(2 * self.num_qubits):
            result._set_row(row, second.conjugate_pauli(first._row_pauli(row)))
        return _with_clifford_origin(result, origin if origin == other_origin else "unresolved")

    def dot(self, other, qargs=None) -> "Clifford":
        return self.compose(other, qargs=qargs, front=True)

    def adjoint(self) -> "Clifford":
        origin = _clifford_matrix_origin(self)
        width = self.num_qubits
        result=Clifford.identity(width)
        for row in range(2*width):
            for column in range(2*width):
                result.tableau[row,column]=self.tableau[(column+width)%(2*width),(row+width)%(2*width)]
        for row in range(2 * width):
            image = self.conjugate_pauli(result._row_pauli(row))
            result.phase[row] = bool(image.phase // 2)
        return _with_clifford_origin(result, origin)

    def conjugate(self) -> "Clifford":
        result = self.copy()
        for row in range(2*self.num_qubits):
            result.tableau[row,-1]^=sum(bool(x and z) for x,z in zip(result.x[row],result.z[row]))%2==1
        return result

    def transpose(self) -> "Clifford":
        return self.adjoint().conjugate()

    def tensor(self, other) -> "Clifford":
        origin = _clifford_matrix_origin(self)
        operand = other if isinstance(other, Clifford) else Clifford(other)
        other_origin = _clifford_matrix_origin(operand)
        width = self.num_qubits + operand.num_qubits
        result = Clifford.identity(width)
        for source, positions in ((operand, tuple(range(operand.num_qubits))), (self, tuple(range(operand.num_qubits, width)))):
            for offset, qubit in enumerate(positions):
                result._set_row(qubit, source._row_pauli(offset)._embedded(positions, width))
                result._set_row(width + qubit, source._row_pauli(source.num_qubits + offset)._embedded(positions, width))
        return _with_clifford_origin(result, origin if origin == other_origin else "unresolved")

    def expand(self, other) -> "Clifford":
        return Clifford(other).tensor(self)

    def power(self, n: int) -> "Clifford":
        origin = _clifford_matrix_origin(self)
        if not isinstance(n, Integral) or n < 1:
            raise ValueError("QuantumBridge Clifford powers must be positive integers; use adjoint() for the inverse.")
        factor = self.copy()
        remaining = int(n)
        result = Clifford.identity(self.num_qubits)
        _with_clifford_origin(result, origin)
        while remaining:
            if remaining & 1:
                result = result.compose(factor)
            remaining >>= 1
            if remaining:
                factor = factor.compose(factor)
        return result

    def to_circuit(self) -> Circuit:
        reduced = self.copy()
        width = self.num_qubits
        reductions = []

        def apply(name, *qubits):
            reduced._append_gate(name, qubits)
            reductions.append((name, qubits))

        # Reduce each conjugate X/Z generator pair, preserving earlier pairs.
        for index in range(width):
            pivot=next((qubit for qubit in range(index,width) if reduced.x[index,qubit] or reduced.z[index,qubit]),None)
            if pivot is None:
                raise ValueError("QuantumBridge Clifford tableau is not symplectic.")
            if reduced.z[index, pivot]:
                apply("sdg" if reduced.x[index, pivot] else "h", pivot)
            if pivot != index:
                apply("swap", index, pivot)
            for qubit in range(index + 1, width):
                if not (reduced.x[index, qubit] or reduced.z[index, qubit]):
                    continue
                if reduced.z[index, qubit]:
                    apply("sdg" if reduced.x[index, qubit] else "h", qubit)
                apply("cx", index, qubit)

            stabilizer_row = width + index
            if reduced.x[stabilizer_row, index]:
                apply("h", index)
                apply("s", index)
                apply("h", index)
            for qubit in range(index + 1, width):
                if not (reduced.x[stabilizer_row, qubit] or reduced.z[stabilizer_row, qubit]):
                    continue
                if reduced.x[stabilizer_row, qubit]:
                    if reduced.z[stabilizer_row, qubit]:
                        apply("sdg", qubit)
                    apply("h", qubit)
                apply("cx", qubit, index)
            if reduced.phase[index]:
                apply("z", index)
            if reduced.phase[stabilizer_row]:
                apply("x", index)

        if not all(bool(reduced.tableau[row,column])==(row==column)
                for row in range(2*width) for column in range(2*width+1)):
            raise ValueError("QuantumBridge Clifford synthesis did not reduce to identity.")
        circuit = Circuit(width)
        inverse_names = {"s": "sdg", "sdg": "s"}
        for name, qubits in reversed(reductions):
            wires = [width - 1 - qubit for qubit in qubits]
            getattr(circuit, inverse_names.get(name, name))(*wires)
        return circuit

    def to_matrix(self) -> np.ndarray:
        _clifford_matrix_origin(self, dense=True)
        scope=current_resources()
        if scope is not None:
            from quantumbridge.execution.array_reservations import array_layout
            scope.ledger.preflight(memory_bytes=array_layout((1<<self.num_qubits,)*2,complex)[2])
        if self.num_qubits == 0:
            return arrays.filled("clifford_zero_qubit_matrix",(1,1),complex,1)
        return Operator.from_circuit(self.to_circuit()).data

    def to_operator(self) -> Operator:
        origin = _clifford_matrix_origin(self, dense=True)
        result = Operator(self.to_matrix())
        if origin is not None:result._public_matrix_basis = origin
        return result

    def to_instruction(self) -> Instruction:
        return self.to_circuit().to_instruction(name="clifford")

    def equiv(self, other, atol: float = 1e-10) -> bool:
        if isinstance(other, Clifford):
            return self == other
        if isinstance(other, (Circuit, Instruction)):
            return self == Clifford(other)
        return self.to_operator().equiv(other, atol=atol)

    def __array__(self, dtype=None, copy=None):
        return arrays.copied(self.tableau,dtype=bool if dtype is None else dtype,category="clifford_array_export")

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self,memo):
        from quantumbridge.execution.array_ownership import native_object_deepcopy
        _workspace_notice()
        return native_object_deepcopy(self,memo)

    def __matmul__(self, other):
        return self.dot(other)

    def __xor__(self, other):
        return self.tensor(other)

    def __eq__(self, other) -> bool:
        return isinstance(other,Clifford) and self.tableau.shape==other.tableau.shape and all(
            first==second for first,second in zip(self.tableau.flat,other.tableau.flat))

    def __repr__(self):
        return f"Clifford({self.to_dict()!r})"
