# Copyright 2026 QuantumBridge Contributors. Apache-2.0.
"""Branch-consistent fixed-projector powers lowered into existing native gates.

For constant spectral projectors P_j, f(U)=sum_j exp(i*a*arg(lambda_j))*P_j.
This is a verified special case of matrix-function differentiation, not the
incorrect commuting derivative formula for arbitrary moving projectors.
"""

from dataclasses import dataclass
import math
from numbers import Real
from uuid import uuid4

import numpy as np

from quantumbridge.core import Circuit, Parameter
from quantumbridge.core.annotated_operation import (
    AnnotatedOperation,
    ControlModifier,
    InverseModifier,
    PowerModifier,
    resolve_annotated_operation,
    validate_annotated_operation,
)
from quantumbridge.core.classical import UnsupportedFeature
from quantumbridge.core.gate_limits import current_gate_limits
from quantumbridge.core.gates import ControlledGate, Gate, _append_controlled, lower
from quantumbridge.core.operations import execution_operation
from quantumbridge.core.parameters import parameter_set, resolve_parameter


@dataclass
class _Spectrum:
    axes: tuple
    phases: list
    angle_power: bool = False


def _charge(work, amount):
    work[0] += amount
    current_gate_limits().check("max_decomposition_work", work[0])


def _admit(width, work):
    limits = current_gate_limits()
    if width >= limits.max_matrix_dimension.bit_length():
        raise MemoryError("Symbolic power local spectrum exceeds max_matrix_dimension.")
    dimension = 1 << width
    _charge(work, dimension * max(1, width))
    limits.check("max_total_operations", 8 * dimension * max(1, width))
    from quantumbridge.execution.resource_context import current_resources

    scope = current_resources()
    if scope is not None:
        scope.retain("symbolic_power_spectral_graph", 64 * dimension * max(1, width))
    return dimension


def _principal(value):
    module = type(value).__module__
    if module.startswith(("jax", "jaxlib", "tensorflow")):
        from quantumbridge.interfaces.symbolic_math import principal_angle

        return principal_angle(value)
    from .scalars import scalar_primal

    point = scalar_primal(value)
    wrapped = math.atan2(math.sin(point), math.cos(point))
    if abs(abs(wrapped) - math.pi) <= 64 * np.finfo(float).eps:
        raise UnsupportedFeature(
            "symbolic_power_branch",
            "Fractional power is nondifferentiable at this branch switch.",
        )
    from .second_trace import SecondDual
    from .trace import Dual

    if type(value) is SecondDual:
        return value._unary(wrapped, 1.0, 0.0)
    if type(value) is Dual:
        return value._new(wrapped, lambda: value.derivative.copy())
    if module.startswith("torch"):
        return value - value.detach() + wrapped
    if module.startswith("autograd"):
        from autograd.tracer import getval

        return value - getval(value) + wrapped
    return wrapped


def _same_definition(source, expected):
    if source is None or expected is None:
        return source is expected
    # Existing bounded native IR preserves actual UUIDs and detects edited
    # definitions; never trust a standard class name over its executable body.
    return source.to_ir().to_dict() == expected.to_ir().to_dict()


def _control(spectrum, count, state, work):
    width = len(spectrum.axes)
    _admit(width + count, work)
    phases = []
    for prefix in range(1 << count):
        public_state = sum(
            ((prefix >> (count - 1 - wire)) & 1) << wire for wire in range(count)
        )
        phases.extend(
            spectrum.phases if public_state == state else [0.0] * (1 << width)
        )
    return _Spectrum(("Z",) * count + spectrum.axes, phases)


def _primitive(name, parameters, width, evaluate):
    if (
        name
        not in {
            "rx",
            "ry",
            "rz",
            "rxx",
            "ryy",
            "rzz",
            "phase",
            "p",
            "u1",
            "global_phase",
        }
        or len(parameters) != 1
    ):
        raise UnsupportedFeature(
            "symbolic_power_generator",
            "No fixed-projector symbolic power rule for this generator.",
        )
    angle = evaluate(parameters[0])
    if name in {"phase", "p", "u1"}:
        if width != 1:
            raise ValueError("Phase primitive has invalid width.")
        return _Spectrum(("Z",), [0.0, angle])
    if name == "global_phase":
        return _Spectrum((None,) * width, [angle] * (1 << width))
    expected = 2 if name in {"rxx", "ryy", "rzz"} else 1
    if width != expected:
        raise ValueError("Rotation primitive has invalid width.")
    return _Spectrum(
        (name[-1].upper(),) * width,
        [
            -angle / 2 if index.bit_count() % 2 == 0 else angle / 2
            for index in range(1 << width)
        ],
    )


def _gate_spectrum(operation, evaluate, work, depth=0):
    current_gate_limits().check("max_definition_depth", depth)
    _admit(operation.num_qubits, work)
    if type(operation) is AnnotatedOperation:
        result = _gate_spectrum(operation.base_op, evaluate, work, depth + 1)
        for modifier in operation.modifiers:
            if type(modifier) is ControlModifier:
                result = _control(
                    result, modifier.num_ctrl_qubits, modifier.ctrl_state, work
                )
            elif type(modifier) is InverseModifier:
                result.phases = [-value for value in result.phases]
            else:
                exponent = evaluate(modifier.power)
                integral = (
                    not parameter_set((modifier.power,))
                    and isinstance(exponent, Real)
                    and float(exponent).is_integer()
                )
                if integral or result.angle_power:
                    result.phases = [exponent * value for value in result.phases]
                else:
                    result.phases = [
                        exponent * _principal(value) for value in result.phases
                    ]
                if integral:
                    result.angle_power = False
        return result
    if isinstance(operation, ControlledGate):
        expected = ControlledGate(
            operation.base_gate, operation.num_ctrl_qubits, operation.ctrl_state
        )
        if not _same_definition(operation.definition, expected.definition):
            raise UnsupportedFeature(
                "symbolic_power_generator",
                "An edited controlled definition requires its own projector rule.",
            )
        base = _gate_spectrum(operation.base_gate, evaluate, work, depth + 1)
        result = _control(base, operation.num_ctrl_qubits, operation.ctrl_state, work)
        from quantumbridge.core.standard_gates import CPhaseGate

        result.angle_power = type(operation) is CPhaseGate
        return result
    from quantumbridge.core.standard_gates import (
        RXGate,
        RYGate,
        RZGate,
        RXXGate,
        RYYGate,
        RZZGate,
        PhaseGate,
    )

    standard = {
        RXGate: "rx",
        RYGate: "ry",
        RZGate: "rz",
        RXXGate: "rxx",
        RYYGate: "ryy",
        RZZGate: "rzz",
        PhaseGate: "phase",
    }
    if type(operation) in standard:
        expected = type(operation)(*operation.params)
        if not _same_definition(operation.definition, expected.definition):
            raise UnsupportedFeature(
                "symbolic_power_generator",
                "An edited standard definition requires its own projector rule.",
            )
        result = _primitive(
            standard[type(operation)], operation.params, operation.num_qubits, evaluate
        )
        result.angle_power = True
        return result
    body = operation.definition
    if body is None:
        return _primitive(
            operation.name, operation.params, operation.num_qubits, evaluate
        )
    _charge(
        work,
        max(1, len(body.operations)) * (1 << body.num_qubits) * max(1, body.num_qubits),
    )
    phases = [evaluate(body.global_phase)] * (1 << body.num_qubits)
    axes = [None] * body.num_qubits
    for raw in body.operations:
        item = execution_operation(raw)
        child = item.metadata.get(
            "annotated_operation",
            item.metadata.get("gate", item.metadata.get("instruction_object")),
        )
        if child is None:
            if set(item.metadata) - {"controlled_base"}:
                raise UnsupportedFeature(
                    "symbolic_power_generator",
                    "Custom primitive metadata has no projector rule.",
                )
            from .adjoint import gate_derivative_spec

            name, controls, inverse = gate_derivative_spec(item)
            form = _primitive(name, item.params, len(item.wires) - controls, evaluate)
            if controls:
                form = _control(form, controls, (1 << controls) - 1, work)
            if inverse:
                form.phases = [-value for value in form.phases]
        else:
            form = _gate_spectrum(child, evaluate, work, depth + 1)
        if len(form.axes) != len(item.wires):
            raise ValueError("Projector frame and invocation width disagree.")
        for wire, axis in zip(item.wires, form.axes):
            if axis is None:
                continue
            if axes[wire] is not None and axes[wire] != axis:
                raise UnsupportedFeature(
                    "symbolic_power_generator",
                    "These generators need a more general projector/Frechet rule.",
                )
            axes[wire] = axis
        for index in range(len(phases)):
            local = 0
            for wire in item.wires:
                local = 2 * local + ((index >> (body.num_qubits - 1 - wire)) & 1)
            phases[index] = phases[index] + form.phases[local]
    return _Spectrum(tuple(axes), phases)


def lower_recorded_powers(circuit, evaluate, values):
    """Use the existing structural lowerer with a trace-preserving annotation rule."""

    work = [0]

    def scalar(expression):
        value = evaluate(expression)
        kind = type(value)
        from quantumbridge.core.parameters import ParameterExpression
        if isinstance(value, (Parameter, ParameterExpression)):
            # A frozen symbolic input has no numeric principal phase yet.
            # Preserve the native power occurrence for runtime tensor math.
            raise UnsupportedFeature("symbolic_power_generator")
        from .trace import Dual
        from .second_trace import SecondDual

        if kind in (int, float, Dual, SecondDual):
            return value
        if kind is np.ndarray:
            if value.shape or value.dtype.kind not in "fiu":
                raise ValueError("Power angles require real scalars.")
            return value.item()
        if isinstance(value, np.generic) and kind.__module__.startswith("numpy"):
            if value.dtype.kind not in "fiu":
                raise ValueError("Power angles require real scalars.")
            return value.item()
        if kind.__module__.startswith(
            ("torch", "autograd", "jax", "jaxlib", "tensorflow")
        ):
            return value
        raise UnsupportedFeature(
            "symbolic_power_scalar_type",
            "Power angles require native real scalars or a supported trace interface.",
        )

    def resolve(annotation):
        validate_annotated_operation(annotation)
        current, fractional = annotation, False
        while type(current) is AnnotatedOperation:
            for modifier in current.modifiers:
                if type(modifier) is PowerModifier:
                    if parameter_set((modifier.power,)):
                        fractional = True
                    else:
                        value = resolve_parameter(modifier.power, {})
                        fractional |= not float(value).is_integer()
            current = current.base_op
        if not annotation.parameters or not fractional:
            return resolve_annotated_operation(annotation)
        try:
            result = _gate_spectrum(annotation, scalar, work)
        except UnsupportedFeature as error:
            if error.feature != "symbolic_power_generator":
                raise
            base = annotation
            while type(base) is AnnotatedOperation:
                base = base.base_op
            if type(base) not in (Gate, ControlledGate):
                raise
            from .matrix_power_jet import recorded_matrix_power
            return recorded_matrix_power(annotation)
        width = len(result.axes)
        _admit(width, work)
        out = Circuit(width)

        def slot(value):
            if isinstance(value, Real):
                return float(value)
            parameter = Parameter("__qb_power_" + uuid4().hex)
            values[parameter] = value
            return parameter

        for wire, axis in enumerate(result.axes):
            if axis == "X":
                out.h(wire)
            elif axis == "Y":
                out.sdg(wire).h(wire)
        if not width:
            out.global_phase = slot(result.phases[0])
        # Keep each local projector phase scoped to its invocation. Extracting
        # a global phase would incorrectly escape a classically conditioned gate.
        for index, value in enumerate(result.phases if width else ()):
            if isinstance(value, Real) and value == 0:
                continue
            opened = [
                wire for wire in range(width) if not index & (1 << (width - 1 - wire))
            ]
            for wire in opened:
                out.x(wire)
            _append_controlled(
                out, "phase", (width - 1,), slot(value), tuple(range(width - 1)), {}
            )
            for wire in reversed(opened):
                out.x(wire)
        for wire, axis in enumerate(result.axes):
            if axis == "X":
                out.h(wire)
            elif axis == "Y":
                out.h(wire).s(wire)
        return Gate(
            "native_symbolic_power", width, tuple(out.parameters), definition=out
        )

    return lower(circuit, _annotation_resolver=resolve)
