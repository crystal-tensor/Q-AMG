# Copyright 2026 QuantumBridge Contributors. Apache-2.0.
"""Local TF operators from existing native plans.

This kernel performs no Device execution. The whole-plan consumer separately
admits contractions and derivative rules before selecting the local kernel.
"""

import math
from numbers import Real

from quantumbridge.core import Parameter
from quantumbridge.core.annotated_operation import AnnotatedOperation, ControlModifier, InverseModifier, PowerModifier
from quantumbridge.core.classical import UnsupportedFeature
from quantumbridge.core.gate_limits import current_gate_limits
from quantumbridge.core.operations import execution_operation
from quantumbridge.core.parameters import resolve_parameter
from quantumbridge.diff.compiled_parameters import evaluate_compiled_parameter, record_compiled_parameters
from quantumbridge.diff.qnode_engine import AngleRecorder, ExecutionPlan
from quantumbridge.diff.matrix_power_jet import PRIMITIVE, _admit, _decode, _source_body


def _integer_power(tf, matrix, exponent):
    result, count = tf.eye(2, dtype=tf.complex128), abs(exponent)
    if exponent < 0:
        matrix = tf.linalg.adjoint(matrix)
    while count:
        if count & 1:
            result = matrix @ result
        count >>= 1
        if count:
            matrix = matrix @ matrix
    return result


class TensorFlowLocalOperator:
    """A compiled local-kernel view of one native_matrix_power occurrence.

    Inputs follow the existing plan recorder, including distinct occurrences
    of the same source expression. No new parameter namespace or circuit IR
    is created. TensorFlow is not imported by structural admission.
    """

    def __init__(self, plan, operation_index, *, readonly=False):
        if type(plan) is not ExecutionPlan:
            raise TypeError("A local TF kernel requires an existing native ExecutionPlan.")
        compiled = plan.circuit.lowered()
        if type(operation_index) is not int or not 0 <= operation_index < len(compiled.operations):
            raise ValueError("Local TF operation index is outside the compiled plan.")
        operation = execution_operation(compiled.operations[operation_index])
        if operation.name != PRIMITIVE:
            raise UnsupportedFeature("tensorflow_local_operation", "The local power kernel consumes native matrix-power plan occurrences only.")
        annotation, symbols, _ = _decode((1.,)*len(operation.params), operation.metadata)
        if len(operation.wires) != annotation.num_qubits or len(set(operation.wires)) != len(operation.wires):
            raise ValueError("Local TF operand frame disagrees with its native source width.")
        body = _source_body(annotation)
        if body.num_qubits != 1:
            raise UnsupportedFeature("tensorflow_local_power_width", "General multi-qubit powered bodies require a validated tensor rule.")
        supported = {"rx", "ry", "rz", "id", "i", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "sx", "sxdg", "p", "phase", "u1", "u", "u2", "u3", "global_phase"}
        primitives = tuple(execution_operation(raw) for raw in body.operations)
        for primitive in primitives:
            if primitive.metadata or primitive.name not in supported or primitive.wires not in {(), (0,)}:
                raise UnsupportedFeature("tensorflow_local_power_primitive", "The source contains an unvalidated local primitive or metadata.")
        chain, cursor = [], annotation
        while type(cursor) is AnnotatedOperation:
            chain.append(cursor)
            cursor = cursor.base_op
        modifiers, controls, fractional = [], [], False
        for node in reversed(chain):
            for modifier in node.modifiers:
                if type(modifier) is ControlModifier:
                    controls.append((modifier.num_ctrl_qubits, modifier.ctrl_state))
                elif type(modifier) is InverseModifier:
                    modifiers.append(("inverse", None))
                else:
                    if type(modifier) is not PowerModifier:
                        raise UnsupportedFeature("tensorflow_local_modifier")
                    exponent = resolve_parameter(modifier.power, {})
                    if isinstance(exponent, bool) or not isinstance(exponent, Real) or not math.isfinite(exponent):
                        raise ValueError("Local TF powers need finite fixed real exponents.")
                    exponent = float(exponent)
                    if not exponent.is_integer():
                        if fractional:
                            raise UnsupportedFeature("tensorflow_local_repeated_fractional_power", "Sequential principal fractional powers require a separately validated phase rule.")
                        fractional = True
                    else:
                        current_gate_limits().check("max_repeat_count", abs(int(exponent)))
                    modifiers.append(("power", exponent))
        indices = {parameter: index for index, parameter in enumerate(plan.recorder.parameters)}
        if len(indices) != len(plan.recorder.parameters):
            raise ValueError("Local TF plan recorder contains duplicate parameter identities.")
        bindings = []
        for value in operation.params:
            if isinstance(value, Parameter):
                if value not in indices:
                    raise ValueError("Local occurrence identity is absent from its plan recorder.")
                bindings.append((indices[value], None))
            elif isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value):
                bindings.append((None, float(value)))
            else:
                raise UnsupportedFeature("tensorflow_local_unrecorded_parameter")
        dimension = _admit(annotation.num_qubits)
        work = 16*dimension**3*(1+len(primitives)+sum(1+int(abs(value)).bit_length() for kind, value in modifiers if kind == "power"))
        current_gate_limits().check("max_decomposition_work", work)
        self.workspace_bytes = 16*128*(4*dimension)**2
        self._work = work
        self._readonly = readonly
        self._body, self._symbols = body, symbols
        self._bindings, self._modifiers, self._controls = tuple(bindings), tuple(modifiers), tuple(controls)
        self.input_size = len(plan.recorder.parameters)
        self.wires = operation.wires
        self.source_parameter_uuids = tuple(str(parameter.uuid) for parameter in symbols)
        self.occurrence_parameter_uuids = tuple(str(value.uuid) if isinstance(value, Parameter) else None for value in operation.params)
        self.compiled_operation_names = tuple(execution_operation(raw).name for raw in compiled.operations)

    def __call__(self, point):
        import tensorflow as tf
        from . import tensorflow_native_math as numeric
        if not tf.is_tensor(point) or point.dtype != tf.float64 or point.shape != (self.input_size,):
            raise ValueError("A local TF kernel requires the declared static float64 plan vector.")
        from quantumbridge.execution.resource_context import current_resources
        scope = current_resources()
        _admit(len(self.wires), reserved_workspace_bytes=getattr(scope, "tensorflow_tensor_workspace_admitted", 0))
        current_gate_limits().check("max_decomposition_work", self._work)
        for kind, value in self._modifiers:
            if kind == "power" and value.is_integer():
                current_gate_limits().check("max_repeat_count", abs(int(value)))
        if scope is not None:
            if self.workspace_bytes > getattr(scope, "tensorflow_tensor_workspace_admitted", 0):
                scope.ledger.preflight(memory_bytes=self.workspace_bytes)
            scope.ledger.unavailable("tensorflow_allocator_memory", "Local tensor workspace is a conservative admission bound, not a measured framework allocation or graph-wide derivative account.")
        valid = _finite_parameters()(point)
        point = tf.where(valid, point, tf.zeros_like(point))
        if self._readonly:
            body = self._body
            mapping = dict(zip(self._symbols, (point[index] if index is not None else value for index, value in self._bindings)))
        else:
            recorder = AngleRecorder()
            recorder.parameters = list(self._symbols)
            recorder.angles = [point[index] if index is not None else value for index, value in self._bindings]
            body, recorded = record_compiled_parameters(self._body, recorder)
            mapping = dict(zip(recorded.parameters, recorded.angles))
        def angle(value):
            result = evaluate_compiled_parameter(value, mapping) if self._readonly else resolve_parameter(value, mapping)
            return tf.convert_to_tensor(result, tf.float64)
        matrix = tf.eye(2, dtype=tf.complex128)
        global_phase = angle(body.global_phase)
        for raw in body.operations:
            operation = execution_operation(raw)
            local, local_phase = _primitive(numeric, operation.name, [angle(value) for value in operation.params])
            matrix = local @ matrix
            global_phase += local_phase
        for kind, value in self._modifiers:
            if kind == "inverse":
                matrix, global_phase = tf.linalg.adjoint(matrix), -global_phase
            elif value.is_integer():
                matrix, global_phase = _integer_power(tf, matrix, int(value)), value*global_phase
            else:
                matrix = numeric.fractional_su2(matrix, value, global_phase)
                global_phase = tf.constant(0., tf.float64)
        # Inverse/integer operations commute with an already incorporated
        # scalar phase. Controls can be deferred since f(diag(I,U))=diag(I,f(U)).
        matrix = numeric.phase(global_phase)*matrix
        valid = _finite_parameters()(tf.concat([tf.reshape(tf.math.real(matrix), (-1,)), tf.reshape(tf.math.imag(matrix), (-1,))], 0))
        matrix = tf.where(valid, matrix, tf.zeros_like(matrix))
        for count, state in self._controls:
            prefix = sum(((state >> bit) & 1) << (count-1-bit) for bit in range(count))
            projector = tf.linalg.diag(tf.constant([int(index == prefix) for index in range(1 << count)], tf.complex128))
            old = matrix.shape[0]
            matrix = tf.eye(old*(1 << count), dtype=tf.complex128)+numeric.kron(projector, matrix-tf.eye(old, dtype=tf.complex128))
        return matrix


def _primitive(numeric, name, values):
    import tensorflow as tf
    zero, pi = tf.constant(0., tf.float64), tf.constant(math.pi, tf.float64)
    if name in {"rx", "ry", "rz"}:
        return numeric.rotation(name[-1], values[0]), zero
    if name in {"phase", "p", "u1"}:
        return numeric.rotation("z", values[0]), values[0]/2
    if name == "global_phase":
        return tf.eye(2, dtype=tf.complex128), values[0]
    if name in {"id", "i"}:
        return tf.eye(2, dtype=tf.complex128), zero
    if name in {"s", "sdg", "t", "tdg", "sx", "sxdg"}:
        theta = pi/(4 if name in {"t", "tdg"} else 2)
        if name in {"sdg", "tdg", "sxdg"}:
            theta = -theta
        return numeric.rotation("x" if name in {"sx", "sxdg"} else "z", theta), theta/2
    if name in {"x", "y", "z", "h"}:
        from quantumbridge.utils.math import gate_matrix
        return -1j*tf.constant(gate_matrix(name), tf.complex128), pi/2
    theta, phi, lam = (pi/2, *values) if name == "u2" else values
    return numeric.rotation("z", phi) @ numeric.rotation("y", theta) @ numeric.rotation("z", lam), (phi+lam)/2


_checked = None


def _finite_parameters():
    global _checked
    if _checked is None:
        import tensorflow as tf
        @tf.function(autograph=False, jit_compile=False)
        def check(point):
            with tf.control_dependencies([tf.debugging.assert_all_finite(point, "Nonfinite local TF tensor")]):
                return tf.constant(True)
        _checked = check
    return _checked
