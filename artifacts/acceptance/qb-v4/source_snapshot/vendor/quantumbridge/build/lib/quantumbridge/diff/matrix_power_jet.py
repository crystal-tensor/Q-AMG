# Copyright 2026 QuantumBridge Contributors. Apache-2.0.
"""Local directional native matrix-function jets for moving eigenprojectors.

The transient primitive carries a bounded native IR source, not an upstream
object or a callable. The user's symbolic annotation remains unchanged.
"""

from contextvars import ContextVar
import math

import numpy as np

from quantumbridge.core import Circuit
from quantumbridge.core.annotated_operation import (
    AnnotatedOperation,
    ControlModifier,
    InverseModifier,
    PowerModifier,
    _base_gate,
    resolve_annotated_operation,
)
from quantumbridge.core.classical import UnsupportedFeature
from quantumbridge.core.gate_limits import current_gate_limits
from quantumbridge.core.gates import Gate, ControlledGate, lower
from quantumbridge.core.operations import execution_operation
from quantumbridge.core.parameters import parameter_set, resolve_parameter, _coerce_expression
from quantumbridge.execution import mps_arrays as arrays
from quantumbridge.execution.resource_context import current_resources
from quantumbridge.information.matrix_functions import (
    _Work,
    _finite,
    _product,
    _real,
    matrix_power,
    unitary_power_jet,
    unitary_power_mixed_jet,
)

PRIMITIVE = "native_matrix_power"
_DECODING = ContextVar("native_matrix_power_decoding", default=False)


def _admit(width, *, reserved_workspace_bytes=0):
    limits = current_gate_limits()
    if type(width) is not int or width < 0:
        raise ValueError("Matrix-power source width must be a nonnegative integer.")
    if width >= limits.max_matrix_dimension.bit_length():
        raise MemoryError("Matrix-power local source exceeds max_matrix_dimension.")
    dimension = 1 << width
    limits.check("max_matrix_dimension", 4 * dimension)
    if type(reserved_workspace_bytes) is not int or reserved_workspace_bytes < 0:
        raise ValueError("Reserved matrix-power workspace requires nonnegative integer bytes.")
    scope = current_resources()
    if scope is not None and reserved_workspace_bytes < 16 * 128 * (4 * dimension) ** 2:
        scope.ledger.preflight(memory_bytes=16 * 128 * (4 * dimension) ** 2)
    return dimension


def recorded_matrix_power(annotation):
    """Keep every source parameter as its own scalar execution occurrence."""
    base = annotation
    while type(base) is AnnotatedOperation:
        if any(
            type(m) is PowerModifier and parameter_set((m.power,))
            for m in base.modifiers
        ):
            raise UnsupportedFeature(
                "matrix_power_variable_exponent",
                "Moving-projector power jets currently require a fixed real exponent.",
            )
        base = base.base_op
    if type(base) not in (Gate, ControlledGate) or not annotation.parameters:
        raise UnsupportedFeature(
            "symbolic_power_generator",
            "General power jets require a parameterized native generic Gate.",
        )
    _admit(annotation.num_qubits)
    parameters = tuple(annotation.parameters)
    current_gate_limits().check("max_gate_parameters", len(parameters))
    source = Circuit(annotation.num_qubits).append(
        annotation, range(annotation.num_qubits)
    )
    program = source.to_ir().to_json()
    from quantumbridge.serialization.limits import SerializationLimits

    SerializationLimits().check("max_metadata_bytes", len(program.encode("utf-8")))
    description = {"program": program, "num_qubits": annotation.num_qubits}
    if len(parameters) == 1:
        description["parameter_uuid"] = str(parameters[0].uuid)
    else:
        description["parameter_uuids"] = [
            str(parameter.uuid) for parameter in parameters
        ]
    metadata = {"matrix_power_source": description}
    body = Circuit(annotation.num_qubits).append(
        PRIMITIVE, range(annotation.num_qubits), params=parameters, metadata=metadata
    )
    return Gate(PRIMITIVE, annotation.num_qubits, parameters, definition=body)


def _description(metadata):
    if (
        type(metadata) is not dict
        or any(type(key) is not str for key in metadata)
        or set(metadata) != {"matrix_power_source"}
    ):
        raise ValueError("Malformed matrix-power primitive metadata.")
    description = metadata["matrix_power_source"]
    if (
        type(description) is not dict
        or any(type(key) is not str for key in description)
        or set(description)
        not in (
            {"program", "parameter_uuid", "num_qubits"},
            {"program", "parameter_uuids", "num_qubits"},
        )
        or type(description["program"]) is not str
    ):
        raise ValueError("Malformed matrix-power source description.")
    if "parameter_uuid" in description:
        identities = [description["parameter_uuid"]]
    else:
        identities = description["parameter_uuids"]
    if (
        type(identities) is not list
        or not identities
        or any(type(value) is not str for value in identities)
    ):
        raise ValueError("Malformed matrix-power parameter identities.")
    current_gate_limits().check("max_gate_parameters", len(identities))
    if any(
        len(value) != 36 or set(value) - set("0123456789abcdef-")
        for value in identities
    ):
        raise ValueError("Malformed matrix-power parameter UUID.")
    from quantumbridge.serialization.limits import SerializationLimits

    SerializationLimits().check(
        "max_metadata_bytes", len(description["program"]) + 40 * len(identities)
    )
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate matrix-power parameter identities.")
    return description, identities


def matrix_power_parameter_count(metadata):
    return len(_description(metadata)[1])


def _decode(parameters, metadata):
    from quantumbridge.serialization.limits import SerializationLimits
    from quantumbridge.ir.qb_ir import IRProgram
    from quantumbridge.compat.qiskit_adapter import circuit_from_ir
    from quantumbridge.execution.array_reservations import numeric_shape

    if _DECODING.get():
        raise UnsupportedFeature(
            "nested_matrix_power_source",
            "Transient matrix-power sources cannot embed another transient source.",
        )
    description, identities = _description(metadata)
    _admit(description["num_qubits"])
    limits = SerializationLimits()
    limits.check("max_metadata_bytes", len(description["program"]))
    limits.check("max_metadata_bytes", len(description["program"].encode("utf-8")))
    limits.check(
        "max_metadata_bytes",
        len(description["program"].encode("utf-8")) + 40 * len(identities),
    )
    if numeric_shape(parameters, real_only=True) != (len(identities),):
        raise ValueError(
            "Matrix-power parameters must match the source identity count."
        )
    if issubclass(type(parameters), np.ndarray):
        parameters = np.ndarray.view(parameters, np.ndarray)
    point = tuple(_real(value, "Matrix-power parameter") for value in parameters)
    if any(not math.isfinite(value) for value in point):
        raise ValueError("Matrix-power parameters must be finite.")
    token = _DECODING.set(True)
    try:
        source = circuit_from_ir(
            IRProgram.from_json(description["program"], serialization_limits=limits)
        )
        if (
            source.num_qubits != description["num_qubits"]
            or source.num_bits
            or source.measurements
            or source.has_dynamic_operations
            or len(source.operations) != 1
            or source.global_phase != 0
        ):
            raise ValueError("Invalid local matrix-power source circuit.")
        parameters = tuple(source.parameters)
        if [str(parameter.uuid) for parameter in parameters] != identities:
            raise ValueError(
                "Matrix-power source parameter identity disagrees with its descriptor."
            )
        operation = execution_operation(source.operations[0])
        annotation = operation.metadata.get("annotated_operation")
        if (
            type(annotation) is not AnnotatedOperation
            or operation.wires != tuple(range(source.num_qubits))
            or set(operation.metadata) != {"annotated_operation"}
        ):
            raise ValueError("Matrix-power source must be one local native annotation.")
        # Validate the source domain before numeric execution, including nested
        # encoded sources that could otherwise reset graph-depth accounting.
        _source_body(annotation)
        return annotation, parameters, point
    finally:
        _DECODING.reset(token)


def _source_body(annotation):
    base = annotation
    while type(base) is AnnotatedOperation:
        if any(
            type(m) is PowerModifier and parameter_set((m.power,))
            for m in base.modifiers
        ):
            raise UnsupportedFeature("matrix_power_variable_exponent")
        base = base.base_op
    if type(base) not in (Gate, ControlledGate):
        raise UnsupportedFeature("matrix_power_base_type")
    body = lower(_base_gate(base).definition, _annotation_resolver=_inner_annotation)
    for raw in body.operations:
        operation = execution_operation(raw)
        if operation.name == PRIMITIVE or "matrix_power_source" in operation.metadata:
            raise UnsupportedFeature("nested_matrix_power_source")
        if set(operation.metadata) - {"controlled_base"}:
            raise UnsupportedFeature("matrix_power_primitive_metadata")
        cursor = operation.metadata
        while "controlled_base" in cursor:
            description = cursor["controlled_base"]
            if (
                description["name"] == PRIMITIVE
                or "matrix_power_source" in description["metadata"]
            ):
                raise UnsupportedFeature("nested_matrix_power_source")
            cursor = description["metadata"]
    return body


def _inner_annotation(annotation):
    if any(type(m) is PowerModifier for m in annotation.modifiers):
        raise UnsupportedFeature(
            "nested_matrix_power_annotation",
            "A fractional annotation inside a generic Gate body needs a nested jet rule.",
        )
    return resolve_annotated_operation(annotation)


def matrix_power_value(parameters, metadata):
    annotation, symbols, point = _decode(parameters, metadata)
    body = _source_body(annotation)
    work = _Work(1 << annotation.num_qubits)
    mapping = dict(zip(symbols, point))
    value = arrays.filled("matrix_power_local_identity", (1 << body.num_qubits,)*2, complex, 0)
    value.flat[::len(value)+1] = 1
    from quantumbridge.utils.math import gate_matrix
    for raw in body.operations:
        operation = execution_operation(raw)
        local = gate_matrix(operation.name,
            tuple(resolve_parameter(p, mapping) for p in operation.params), operation.metadata)
        value = _product(_embed(local, operation.wires, body.num_qubits, work), value, work)
    value *= np.exp(1j*float(resolve_parameter(body.global_phase, mapping)))
    chain, current = [], annotation
    while type(current) is AnnotatedOperation:
        chain.append(current)
        current = current.base_op
    for node in reversed(chain):
        for modifier in node.modifiers:
            if type(modifier) is InverseModifier:
                value = arrays.conjugated(value).T
            elif type(modifier) is ControlModifier:
                old = len(value)
                dimension = old*(1 << modifier.num_ctrl_qubits)
                _admit(dimension.bit_length()-1)
                prefix = sum(((modifier.ctrl_state >> bit) & 1)
                    << (modifier.num_ctrl_qubits-1-bit) for bit in range(modifier.num_ctrl_qubits))
                controlled = arrays.filled("matrix_power_value_control", (dimension, dimension), complex, 0)
                controlled.flat[::dimension+1] = 1
                controlled[prefix*old:(prefix+1)*old, prefix*old:(prefix+1)*old] = value
                value = controlled
            else:
                value = matrix_power(value, _real(resolve_parameter(modifier.power, {}),
                    "Matrix-power exponent"), 0, assume_unitary=True, _work=work)
    return value


def _linear(*terms):
    result = arrays.filled("matrix_jet_sum", terms[0][1].shape, complex, 0)
    scratch = arrays.filled("matrix_jet_sum_scratch", result.shape, complex, 0)
    for coefficient, matrix in terms:
        np.multiply(matrix, coefficient, out=scratch)
        np.add(result, scratch, out=result)
    _finite(result)
    return result


def _scalar_directions(expression, mapping, left, right, work):
    value = _real(resolve_parameter(expression, mapping), "Gate angle")
    dependencies = parameter_set((expression,))
    first = other = mixed = 0.0
    for parameter in dependencies:
        a, b = left.get(parameter, 0.0), right.get(parameter, 0.0)
        if not a and not b:
            continue
        work.charge(1)
        derivative = _coerce_expression(expression)._gradient_expression(parameter)
        numeric = _real(resolve_parameter(derivative, mapping), "Gate-angle derivative")
        first += a * numeric
        other += b * numeric
        if a:
            for second in dependencies:
                weight = right.get(second, 0.0)
                if weight:
                    work.charge(1)
                    mixed += (
                        a
                        * weight
                        * _real(
                            resolve_parameter(derivative._gradient_expression(second), mapping),
                            "Gate-angle mixed derivative",
                        )
                    )
    return value, first, other, mixed


def _embed(matrix, wires, width, work):
    dimension, local = 1 << width, len(matrix)
    work.charge(dimension * local)
    result = arrays.filled(
        "matrix_jet_local_embedding", (dimension, dimension), complex, 0
    )
    for column in range(dimension):
        local_column = 0
        for wire in wires:
            local_column = 2 * local_column + ((column >> (width - 1 - wire)) & 1)
        for local_row in range(local):
            row = column
            for offset, wire in enumerate(wires):
                mask = 1 << (width - 1 - wire)
                row = (row & ~mask) | (
                    ((local_row >> (len(wires) - 1 - offset)) & 1) << (width - 1 - wire)
                )
            result[row, column] = matrix[local_row, local_column]
    return result


def _body_directions(body, mapping, left, right, work):
    from quantumbridge.utils.math import gate_matrix
    from .adjoint import gate_derivative, gate_second_derivative

    dimension = 1 << body.num_qubits
    value = arrays.filled("matrix_jet_identity", (dimension, dimension), complex, 0)
    value.flat[:: dimension + 1] = 1
    first = arrays.filled("matrix_jet_first", value.shape, complex, 0)
    other = arrays.filled("matrix_jet_other", value.shape, complex, 0)
    second = arrays.filled("matrix_jet_second", value.shape, complex, 0)
    for raw in body.operations:
        operation = execution_operation(raw)
        jets = [
            _scalar_directions(angle, mapping, left, right, work)
            for angle in operation.params
        ]
        parameters = tuple(jet[0] for jet in jets)
        matrix = gate_matrix(operation.name, parameters, operation.metadata)
        zero = arrays.filled("matrix_jet_zero", matrix.shape, complex, 0)
        local_first = local_other = local_second = zero
        if any(any(value != 0 for value in jet[1:]) for jet in jets):
            if len(jets) != 1:
                raise UnsupportedFeature(
                    "matrix_power_multi_angle_primitive",
                    "Decompose multi-angle primitives before differentiating a general matrix power.",
                )
            derivative = gate_derivative(operation, parameters)
            local_first = _linear((jets[0][1], derivative))
            local_other = _linear((jets[0][2], derivative))
            local_second = _linear(
                (jets[0][3], derivative),
                (
                    jets[0][1] * jets[0][2],
                    gate_second_derivative(operation, parameters),
                ),
            )
        matrix, local_first, local_other, local_second = [
            _embed(item, operation.wires, body.num_qubits, work)
            for item in (matrix, local_first, local_other, local_second)
        ]
        second_next = _linear(
            (1, _product(local_second, value, work)),
            (1, _product(local_first, other, work)),
            (1, _product(local_other, first, work)),
            (1, _product(matrix, second, work)),
        )
        first_next = _linear(
            (1, _product(local_first, value, work)), (1, _product(matrix, first, work))
        )
        other_next = _linear(
            (1, _product(local_other, value, work)), (1, _product(matrix, other, work))
        )
        value, first, other, second = (
            _product(matrix, value, work),
            first_next,
            other_next,
            second_next,
        )
    phase, dphase, ophase, ddphase = _scalar_directions(
        body.global_phase, mapping, left, right, work
    )
    factor = np.exp(1j * phase)
    return (
        _linear((factor, value)),
        _linear((factor, first), (factor * 1j * dphase, value)),
        _linear((factor, other), (factor * 1j * ophase, value)),
        _linear(
            (factor, second),
            (1j * factor * dphase, other),
            (1j * factor * ophase, first),
            (factor * (1j * ddphase - dphase * ophase), value),
        ),
    )


def _directions(values, count):
    from quantumbridge.execution.array_reservations import numeric_shape

    if numeric_shape(values, real_only=True) != (count,):
        raise ValueError("Matrix-power direction must match its parameter count.")
    if issubclass(type(values), np.ndarray):
        values = np.ndarray.view(values, np.ndarray)
    return tuple(_real(value, "Matrix-power direction") for value in values)


def matrix_power_directions(parameters, metadata, left, right):
    annotation, symbols, point = _decode(parameters, metadata)
    left, right = _directions(left, len(symbols)), _directions(right, len(symbols))
    work = _Work(4 * (1 << annotation.num_qubits))
    body = _source_body(annotation)
    result = _body_directions(
        body,
        dict(zip(symbols, point)),
        dict(zip(symbols, left)),
        dict(zip(symbols, right)),
        work,
    )
    chain, current = [], annotation
    while type(current) is AnnotatedOperation:
        chain.append(current)
        current = current.base_op
    for node in reversed(chain):
        for modifier in node.modifiers:
            if type(modifier) is InverseModifier:
                result = tuple(arrays.conjugated(item).T for item in result)
            elif type(modifier) is ControlModifier:
                old = len(result[0])
                dimension = old * (1 << modifier.num_ctrl_qubits)
                _admit(dimension.bit_length() - 1)
                prefix = sum(
                    ((modifier.ctrl_state >> bit) & 1)
                    << (modifier.num_ctrl_qubits - 1 - bit)
                    for bit in range(modifier.num_ctrl_qubits)
                )
                start = prefix * old
                controlled = []
                for index, item in enumerate(result):
                    output = arrays.filled(
                        "matrix_jet_control", (dimension, dimension), complex, 0
                    )
                    if index == 0:
                        output.flat[:: dimension + 1] = 1
                    output[start : start + old, start : start + old] = item
                    controlled.append(output)
                result = tuple(controlled)
            else:
                exponent = _real(
                    resolve_parameter(modifier.power, {}), "Matrix-power exponent"
                )
                if left == right:
                    value, first, second = unitary_power_jet(
                        result[0], result[1], result[3], exponent, _work=work
                    )
                    result = value, first, first, second
                else:
                    result = unitary_power_mixed_jet(*result, exponent, _work=work)
    return result


def matrix_power_derivative(
    parameters, metadata, order, *, parameter_index=0, direction=None
):
    if type(order) is not int or order not in (1, 2):
        raise ValueError("Matrix-power jets support derivative orders one and two.")
    count = matrix_power_parameter_count(metadata)
    if type(parameter_index) is not int or not 0 <= parameter_index < count:
        raise ValueError("Matrix-power derivative parameter index is out of range.")
    left = tuple(float(index == parameter_index) for index in range(count))
    right = left if direction is None else _directions(direction, count)
    result = matrix_power_directions(parameters, metadata, left, right)
    return result[1 if order == 1 else 3]


def matrix_power_directional(parameters, metadata, direction):
    count = matrix_power_parameter_count(metadata)
    return matrix_power_directions(parameters, metadata, direction, (0.0,) * count)[1]
