# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
"""Native public standard objects sharing the existing Gate execution engine."""

from .gates import Gate, ControlledGate, _copy, _integer
from .operations import Instruction
from .parameters import parameter_set
from .gate_limits import current_gate_limits
import math


def invocation_params(operation):
    """Matrix-valued object parameters are not scalar execution arguments."""
    from .annotated_operation import AnnotatedOperation
    seen=set()
    while type(operation) is AnnotatedOperation or isinstance(operation,ControlledGate) and type(operation) is not CUGate:
        if id(operation) in seen:raise ValueError("Cyclic operation base parameter graph.")
        seen.add(id(operation));current_gate_limits().check("max_definition_depth",len(seen))
        operation=operation.base_op if type(operation) is AnnotatedOperation else operation.base_gate
    return () if type(operation) is UnitaryGate else tuple(operation.params)


class _StandardGate(Gate):
    _operation = None
    _width = 1
    _arity = 0

    def _initialize(self, params, label, *, frozen=False):
        from .parameters import Parameter
        from quantumbridge.circuit_library.standard_gates import standard_gate_definition
        formals = tuple(Parameter("_qb_standard_"+str(i)) for i in range(self._arity))
        body = standard_gate_definition(self._operation, *formals)
        Gate.__init__(self, self._operation, self._width, formals, label, definition=body)
        if params: self.params = params
        object.__setattr__(self, "_mutable", not frozen)

    def __setattr__(self, name, value):
        if not self.mutable: raise TypeError("This standard Gate is immutable; use to_mutable().")
        super().__setattr__(name, value)

    def _validate_parameter_arity(self, values):
        if len(values) != self._arity: raise ValueError("Standard Gate parameter count differs from its public signature.")

    def validate_parameter(self, value):
        from .errors import CircuitError
        try: return Gate.validate_parameter(self, value)
        except (TypeError, ValueError) as exc: raise CircuitError(str(exc)) from exc

    def assign_parameters(self, mapping, *, strict=True):
        from .instruction_parameters import admit_owner
        admit_owner(self)
        bound = Gate.assign_parameters(self, mapping, strict=strict)
        result = self.to_mutable()
        object.__getattribute__(result, "__dict__").update(object.__getattribute__(bound, "__dict__"))
        object.__setattr__(result, "_mutable", self.mutable)
        return result

    def inverse(self, annotated=False):
        from .instruction_parameters import admit_owner
        admit_owner(self)
        if self._arity == 1: return type(self)(-self.params[0])
        if self._arity == 0:
            partner={"s":SdgGate,"sdg":SGate,"t":TdgGate,"tdg":TGate,"sx":SXdgGate,"sxdg":SXGate}.get(self._operation)
            return type(self)() if partner is None else partner()
        theta, phi, lam = self.params
        return UGate(-theta, -lam, -phi)

    def power(self, exponent, annotated=False):
        from .instruction_parameters import admit_owner, admit_payloads
        admit_owner(self)
        admit_payloads((exponent,))
        if self._arity == 1:
            from .parameters import _coerce_expression
            _coerce_expression(exponent)
            if isinstance(exponent, complex): raise ValueError("A rotation power must be real.")
            return type(self)(self.params[0]*exponent)
        return Gate.power(self, exponent, annotated=annotated)

    def control(self, num_ctrl_qubits=1, label=None, ctrl_state=None, annotated=None):
        from .instruction_parameters import admit_owner
        admit_owner(self)
        count = _integer(num_ctrl_qubits, "Control width")
        if count == 1 and self._operation == "u": return CUGate(*self.params, 0, label=label, ctrl_state=ctrl_state)
        if count == 2 and self._operation in {"x", "z"}:
            kind = CCXGate if self._operation == "x" else CCZGate
            return kind(label=label, ctrl_state=ctrl_state, _force_mutable=self._operation == "x")
        if count == 1 and self._operation in _CONTROLLED_CLASSES:
            options = {"_force_mutable": True} if self._operation == "x" else {}
            return _CONTROLLED_CLASSES[self._operation](*self.params, label=label, ctrl_state=ctrl_state, **options)
        return Gate.control(self, count, label, ctrl_state, annotated)


class _AngleGate(_StandardGate):
    _arity = 1

    def __init__(self, theta, label=None):
        from .instruction_parameters import admit_owner
        admit_owner(self)
        _StandardGate._initialize(self, (theta,), label)


class RXGate(_AngleGate): _operation = "rx"
class RYGate(_AngleGate): _operation = "ry"
class RZGate(_AngleGate): _operation = "rz"
class RXXGate(_AngleGate): _operation, _width = "rxx", 2
class RYYGate(_AngleGate): _operation, _width = "ryy", 2
class RZZGate(_AngleGate): _operation, _width = "rzz", 2
class PhaseGate(_AngleGate):
    _operation = "phase"

    def __init__(self, theta, label=None):
        super().__init__(theta, label)
        object.__setattr__(self, "name", "p")


class UGate(_StandardGate):
    _operation, _arity = "u", 3

    def __init__(self, theta, phi, lam, label=None):
        from .instruction_parameters import admit_owner
        admit_owner(self)
        _StandardGate._initialize(self, (theta, phi, lam), label)


class _FixedGate(_StandardGate):
    def __init__(self, label=None, *, _force_mutable=False):
        from .instruction_parameters import admit_owner
        admit_owner(self)
        _StandardGate._initialize(self, (), label, frozen=label is None and not _force_mutable)


class HGate(_FixedGate): _operation = "h"
class IGate(_FixedGate): _operation = "id"
class SGate(_FixedGate): _operation = "s"
class SdgGate(_FixedGate): _operation = "sdg"
class TGate(_FixedGate): _operation = "t"
class TdgGate(_FixedGate): _operation = "tdg"
class SXGate(_FixedGate): _operation = "sx"
class SXdgGate(_FixedGate): _operation = "sxdg"
class XGate(_FixedGate): _operation = "x"
class YGate(_FixedGate): _operation = "y"
class ZGate(_FixedGate):
    _operation = "z"

    def power(self, exponent, annotated=False):
        from .instruction_parameters import admit_owner, admit_payloads
        admit_owner(self)
        admit_payloads((exponent,))
        return PhaseGate(math.pi*exponent)
class SwapGate(_FixedGate): _operation, _width = "swap", 2


class _ControlledStandard(ControlledGate):
    _base_type = None
    _arity = 0
    _num_ctrl = 1

    def _initialize(self, params, label, ctrl_state, frozen=False):
        ControlledGate.__init__(self, self._base_type(*params), self._num_ctrl, ctrl_state, label)
        if self._num_ctrl == 2:
            object.__setattr__(self, "name", "cc"+self.base_gate.name+("_o"+str(self.ctrl_state) if self.ctrl_state != 3 else ""))
        object.__setattr__(self, "_mutable", not frozen)

    def __setattr__(self, name, value):
        if not self.mutable: raise TypeError("This standard Gate is immutable; use to_mutable().")
        super().__setattr__(name, value)

    def assign_parameters(self, mapping, *, strict=True):
        base = self.base_gate.assign_parameters(mapping, strict=strict)
        result = type(self)(*base.params, label=self.label, ctrl_state=self.ctrl_state)
        if self.definition is not None:
            from .instruction_parameters import bind_definition
            object.__setattr__(result, "_definition", bind_definition(self.definition,mapping))
        return result

    def inverse(self, annotated=False):
        return type(self)(*self.base_gate.inverse().params, ctrl_state=self.ctrl_state)

    def power(self, exponent, annotated=False):
        from .instruction_parameters import admit_owner, admit_payloads
        admit_owner(self)
        admit_payloads((exponent, annotated))
        if self._base_type is PhaseGate:
            return type(self)(*self.base_gate.power(exponent).params, ctrl_state=self.ctrl_state)
        if self.is_parameterized() and not annotated:
            raise TypeError("Bind controlled Gate parameters or request an annotated power.")
        return Gate.power(self, exponent, annotated=annotated)

    def control(self, num_ctrl_qubits=1, label=None, ctrl_state=None, annotated=None):
        count = _integer(num_ctrl_qubits, "Control width")
        if count == 1 and self.num_ctrl_qubits == 1 and self._base_type in (XGate, ZGate):
            from .annotated_operation import ControlModifier
            state = ControlModifier(1, ctrl_state).ctrl_state | self.ctrl_state<<1
            kind = CCXGate if self._base_type is XGate else CCZGate
            return kind(label=label, ctrl_state=state, _force_mutable=self._base_type is XGate)
        return Gate.control(self, count, label, ctrl_state, annotated)


class _ControlledAngle(_ControlledStandard):
    _arity = 1

    def __init__(self, theta, label=None, ctrl_state=None):
        from .instruction_parameters import admit_owner
        admit_owner(self)
        _ControlledStandard._initialize(self, (theta,), label, ctrl_state)


class CRXGate(_ControlledAngle): _base_type = RXGate
class CRYGate(_ControlledAngle): _base_type = RYGate
class CRZGate(_ControlledAngle): _base_type = RZGate
class CPhaseGate(_ControlledAngle): _base_type = PhaseGate


class _ControlledFixed(_ControlledStandard):
    def __init__(self, label=None, ctrl_state=None, *, _force_mutable=False):
        from .instruction_parameters import admit_owner
        admit_owner(self)
        from .annotated_operation import ControlModifier
        state = ControlModifier(self._num_ctrl, ctrl_state).ctrl_state
        _ControlledStandard._initialize(self, (), label, state,
            frozen=label is None and state == (1<<self._num_ctrl)-1 and not _force_mutable)


class CXGate(_ControlledFixed): _base_type = XGate
class CYGate(_ControlledFixed): _base_type = YGate
class CZGate(_ControlledFixed): _base_type = ZGate
class CHGate(_ControlledFixed): _base_type = HGate
class CSwapGate(_ControlledFixed): _base_type = SwapGate
class CCXGate(_ControlledFixed): _base_type, _num_ctrl = XGate, 2
class CCZGate(_ControlledFixed): _base_type, _num_ctrl = ZGate, 2


class CUGate(_ControlledStandard):
    _base_type, _arity = UGate, 4

    def __init__(self, theta, phi, lam, gamma, label=None, ctrl_state=None):
        from .instruction_parameters import admit_owner, owned_copy
        from .parameters import Parameter
        from .circuit import Circuit
        admit_owner(self)
        formals = tuple(Parameter("_qb_cu_"+str(i)) for i in range(4))
        ControlledGate.__init__(self, UGate(*formals[:3]), 1, ctrl_state, label)
        body = Circuit(2)
        if self.ctrl_state == 0: body.x(0)
        body.phase(formals[3], 0)
        if self.ctrl_state == 0: body.x(0)
        body = body.compose(self.definition)
        object.__setattr__(self, "name", "cu"+("_o0" if self.ctrl_state == 0 else ""))
        object.__setattr__(self, "_params", formals)
        object.__setattr__(self, "_definition", owned_copy(body))
        object.__setattr__(self, "_parameter_plan", None)
        self.params = (theta, phi, lam, gamma)

    @property
    def params(self): return Instruction.params.fget(self)

    @params.setter
    def params(self, values):
        from .instruction_parameters import validated_values
        values = validated_values(self, values)
        if len(values) != 4: raise ValueError("CUGate requires theta, phi, lambda and controlled phase.")
        base = UGate(*values[:3])
        Gate.params.fset(self, values)
        object.__setattr__(self, "base_gate", base)

    def assign_parameters(self, mapping, *, strict=True):
        bound = Gate.assign_parameters(self, mapping, strict=strict)
        result = self.copy()
        object.__getattribute__(result, "__dict__").update(object.__getattribute__(bound, "__dict__"))
        object.__setattr__(result, "base_gate", self.base_gate.assign_parameters(mapping, strict=False))
        return result

    def inverse(self, annotated=False):
        theta, phi, lam, gamma = self.params
        return CUGate(-theta, -lam, -phi, -gamma, ctrl_state=self.ctrl_state)


class UnitaryGate(Gate):
    """An owned matrix-valued Gate, with no hidden scalar execution parameters."""

    def __init__(self, data, label=None, check_input=True, *, num_qubits=None):
        from .instruction_parameters import admit_owner
        admit_owner(self)
        matrix = UnitaryGate.validate_parameter(self, data)
        width = matrix.shape[0].bit_length()-1
        if num_qubits is not None and num_qubits != width: raise ValueError("Unitary width differs from its matrix.")
        Gate.__init__(self, "unitary", width, (matrix,), label)

    def validate_parameter(self, parameter):
        from quantumbridge.execution.array_reservations import numeric_shape, numeric_array
        from quantumbridge.utils.math import validate_unitary_matrix
        shape = numeric_shape(parameter)
        if len(shape) != 2 or shape[0] != shape[1] or shape[0] < 1 or shape[0] & (shape[0]-1):
            raise ValueError("UnitaryGate needs a nonempty square power-of-two matrix.")
        current_gate_limits().check("max_matrix_dimension", shape[0])
        matrix = numeric_array(parameter, category="native_unitary_parameter", copy=True)
        if not validate_unitary_matrix(matrix): raise ValueError("UnitaryGate input must be finite and unitary.")
        return matrix

    @property
    def definition(self):
        from .circuit import Circuit
        result=Circuit(self.num_qubits).unitary(self._params[0], range(self.num_qubits))
        basis=getattr(self,"_public_matrix_basis",None)
        if basis is not None:object.__setattr__(result.operations[0],"_public_matrix_basis",basis)
        return result

    @definition.setter
    def definition(self, value):
        if value is not None: raise AttributeError("UnitaryGate's executable definition is derived from its owned matrix.")
        object.__setattr__(self, "_definition", None)

    @property
    def params(self): return Instruction.params.fget(self)

    @params.setter
    def params(self, values):
        from .instruction_parameters import admit_owner, parameter_values, admit_payloads
        admit_owner(self)
        values = parameter_values(values)
        if len(values) != 1: raise ValueError("UnitaryGate requires exactly one matrix parameter.")
        admit_payloads(values)
        matrix = self.validate_parameter(values[0])
        if matrix.shape != (1<<self.num_qubits,)*2: raise ValueError("Matrix updates cannot change Gate width.")
        object.__setattr__(self, "_params", (matrix,))

    def assign_parameters(self, mapping, *, strict=True):
        from .circuit import _parameter_mapping_from_values
        _parameter_mapping_from_values((), mapping, strict)
        return self.copy()

    def to_matrix(self):
        from quantumbridge.utils.math import validate_unitary_matrix
        current_gate_limits().check("max_matrix_dimension", self._params[0].shape[0])
        if not validate_unitary_matrix(self._params[0]): raise ValueError("The live matrix parameter is no longer unitary.")
        return _copy(self._params[0])

    def inverse(self, annotated=False):
        from .instruction_parameters import admit_owner
        admit_owner(self)
        result=UnitaryGate(self.to_matrix().conj().T)
        basis=getattr(self,"_public_matrix_basis",None)
        if basis is not None:object.__setattr__(result,"_public_matrix_basis",basis)
        return result

    def power(self, exponent, annotated=False):
        result=super().power(exponent,annotated=annotated)
        if getattr(self,"_public_matrix_basis",None)=="native" and type(result) is UnitaryGate:
            object.__setattr__(result,"_public_matrix_basis","native")
        if getattr(self,"_public_matrix_basis",None)=="little" and type(result) is UnitaryGate:
            # Integer powers lower the definition; fractional powers use the
            # displayed matrix directly. Keep both results in the source basis.
            if float(exponent).is_integer():
                from quantumbridge.information.operator import _reverse_qubit_order
                result=UnitaryGate(_reverse_qubit_order(result.to_matrix(),self.num_qubits),label=result.label)
            object.__setattr__(result,"_public_matrix_basis","little")
        return result


_CONTROLLED_CLASSES = {"rx": CRXGate, "ry": CRYGate, "rz": CRZGate, "phase": CPhaseGate,
    "x": CXGate, "y": CYGate, "z": CZGate, "h": CHGate, "swap": CSwapGate}
STANDARD_GATE_TYPES = (RXGate, RYGate, RZGate, PhaseGate, UGate, HGate, XGate, YGate,
    ZGate, SwapGate, CRXGate, CRYGate, CRZGate, CPhaseGate, CXGate, CYGate, CZGate,
    CHGate, CSwapGate, CCXGate, CCZGate, CUGate, UnitaryGate,
    IGate, SGate, SdgGate, TGate, TdgGate, SXGate, SXdgGate, RXXGate, RYYGate, RZZGate)

STANDARD_GATE_NAMES={
    "id":IGate,"i":IGate,"h":HGate,"x":XGate,"y":YGate,"z":ZGate,
    "s":SGate,"sdg":SdgGate,"t":TGate,"tdg":TdgGate,"sx":SXGate,"sxdg":SXdgGate,
    "rx":RXGate,"ry":RYGate,"rz":RZGate,"phase":PhaseGate,"p":PhaseGate,"u":UGate,"u3":UGate,
    "cx":CXGate,"cy":CYGate,"cz":CZGate,"ch":CHGate,"swap":SwapGate,"cswap":CSwapGate,
    "crx":CRXGate,"cry":CRYGate,"crz":CRZGate,"cp":CPhaseGate,"cphase":CPhaseGate,
    "ccx":CCXGate,"ccz":CCZGate,"cu":CUGate,"rxx":RXXGate,"ryy":RYYGate,"rzz":RZZGate,
}


def standard_type_name(gate):
    return next((kind.__name__ for kind in STANDARD_GATE_TYPES if type(gate) is kind), None)


def standard_from_record(name, params, label=None, ctrl_state=None):
    kind = next((kind for kind in STANDARD_GATE_TYPES if kind.__name__ == name), None)
    if kind is None: raise ValueError("Unknown native standard Gate type.")
    if issubclass(kind, _ControlledStandard): return kind(*params, label=label, ctrl_state=ctrl_state)
    result=kind(*params, label=label)
    # Decoded records and canonical format checks use native matrix coordinates.
    if kind is UnitaryGate:object.__setattr__(result,"_public_matrix_basis","native")
    return result
