# Copyright 2026 QuantumBridge Contributors. Apache-2.0.
"""Construction-only scalar GraphDef data import into the existing native AST.

No GraphDef is executed or retained. A shape/work pass bounds tree expansion
before a second pass creates native expressions. Integer data is layout only.
"""

from collections import deque
from dataclasses import dataclass
from types import MappingProxyType
import hashlib
import math
import struct
from numbers import Real

import numpy as np

from quantumbridge.core import Parameter, ParameterExpression
from quantumbridge.core.classical import UnsupportedFeature
from quantumbridge.core.parameters import validate_parameter_names
from quantumbridge.core.symbolic_limits import current_symbolic_limits
from quantumbridge.diff.compiled_parameters import FrozenParameterBinding
from quantumbridge.diff.higher_order_limits import resolve_higher_order_limits


_ATTRS = {
    "Placeholder": {"dtype": "type", "shape": "shape", "_user_specified_name": "s"},
    "Const": {"dtype": "type", "value": "tensor"},
    "Identity": {"T": "type"},
    "Pack": {"T": "type", "N": "i", "axis": "i"},
    "Unpack": {"T": "type", "num": "i", "axis": "i"},
    "Reshape": {"T": "type", "Tshape": "type"},
    "ReverseV2": {"T": "type", "Tidx": "type"},
    "StridedSlice": {"T": "type", "Index": "type", **{key: "i" for key in
        ("begin_mask", "end_mask", "ellipsis_mask", "new_axis_mask", "shrink_axis_mask")}},
    **{op: {"T": "type"} for op in ("Add", "AddV2", "Sub", "Mul", "Pow", "Sin", "Cos", "Exp")},
}

_PROOF_OPS = {"Neg":"neg", "RealDiv":"/", "Tan":"tan", "Log":"log",
    "Atan":"arctan", "Conj":"conjugate"}
_SCALAR_OPS = {"Add":"+", "AddV2":"+", "Sub":"-", "Mul":"*", "Pow":"**",
    "Sin":"sin", "Cos":"cos", "Exp":"exp"}


def _occurrence_signature(op, *operands):
    return hashlib.sha256(op.encode("ascii")+b"\0"+b"".join(operands)).digest()


def _literal_signature(value):
    return _occurrence_signature("float64",struct.pack("!d",float(value)))


class ScalarGraphCapabilityMiss(UnsupportedFeature):
    """No native lowering; only an independently admitted old query may run."""


def _capability(reason):
    raise ScalarGraphCapabilityMiss("tensorflow_scalar_capability", reason)


def _refuse(reason):
    raise UnsupportedFeature("tensorflow_scalar_graph", reason)


def _ref(value):
    if not value or value.startswith("^"):
        _refuse("Control dependencies and empty tensor references are not admitted.")
    name, separator, port = value.rpartition(":")
    if not separator:
        return value, 0
    if not name or not port.isdecimal():
        _refuse("Tensor ports require nonnegative static indices.")
    return name, int(port)


@dataclass
class _Tensor:
    dtype: int
    data: np.ndarray
    weight: np.ndarray
    depth: np.ndarray


def lower_scalar_graph(graph, *, input_name, output_name, parameters,
                       limits=None, bridge_limits=None, ledger=None, allow_square=False,
                       verify_occurrence_only=None):
    """Return a native frozen binding and immutable construction-work evidence.

    Unsupported sources raise UnsupportedFeature; resource failures propagate.
    The first domain is one static float64 vector input and vector output.
    """
    import tensorflow as tf
    from .tensorflow_autodiff import TensorFlowBridgeLimits
    from quantumbridge.execution.resource_context import current_resources
    from quantumbridge.execution.resource_ledger import UnifiedResourceLedger

    if type(graph) is not tf.compat.v1.GraphDef:
        _refuse("Expected public TensorFlow GraphDef data.")
    higher = resolve_higher_order_limits(limits)
    bridge = TensorFlowBridgeLimits() if bridge_limits is None else bridge_limits
    if not isinstance(bridge, TensorFlowBridgeLimits):
        raise TypeError("bridge_limits requires TensorFlowBridgeLimits.")
    byte_count, node_count = graph.ByteSize(), len(graph.node)
    bridge.check("max_scalar_graph_bytes", byte_count)
    higher.check("max_parameter_graph_nodes", node_count)
    snapshot_memory = 2*byte_count+1024*node_count
    higher.check("max_memory_bytes", snapshot_memory)
    active = current_resources()
    if ledger is None:
        ledger = UnifiedResourceLedger(parent=None if active is None else active.ledger)
    elif active is not None:
        account = ledger
        while account is not None and account is not active.ledger:
            account = account.parent
        if account is None:
            _refuse("Scalar construction cannot bypass an unrelated resource scope.")
    with ledger.reserve("tensorflow_scalar_source_snapshot", memory_bytes=snapshot_memory):
        frozen_graph = tf.compat.v1.GraphDef()
        frozen_graph.CopyFrom(graph)
        binding, work = _lower_snapshot(frozen_graph, input_name=input_name, output_name=output_name,
            parameters=parameters, higher=higher, bridge=bridge, ledger=ledger, allow_square=allow_square,
            snapshot_memory=snapshot_memory, verify=verify_occurrence_only)
    return binding, work


def _lower_snapshot(graph, *, input_name, output_name, parameters, higher, bridge, ledger,
                    allow_square, snapshot_memory, verify=None):
    import tensorflow as tf
    symbolic = current_symbolic_limits()
    if type(parameters) not in (tuple, list) or not all(isinstance(p, Parameter) for p in parameters):
        raise TypeError("Use a bounded sequence of existing native Parameter identities.")
    parameters = tuple(parameters)
    if verify is not None and (type(verify) is not FrozenParameterBinding or verify.parameters != parameters):
        _refuse("Source proof requires the readmitted binding's exact root UUID order.")
    if len(set(parameters)) != len(parameters):
        _refuse("Registered input identities must be unique.")
    validate_parameter_names(parameters)
    higher.check("max_input_parameters", len(parameters))
    bridge.check("max_input_elements", len(parameters))
    byte_count, node_count = graph.ByteSize(), len(graph.node)
    bridge.check("max_scalar_graph_bytes", byte_count)
    higher.check("max_parameter_graph_nodes", node_count)
    if graph.library.ByteSize():
        _refuse("Function libraries are outside the scalar source domain.")
    root, result = _ref(input_name), _ref(output_name)
    if root[1] != 0:
        _refuse("The registered input must be placeholder port zero.")

    def shape(proto):
        if proto.unknown_rank or any(d.size < 0 or d.name for d in proto.dim):
            _refuse("Scalar source layouts require fully static unnamed dimensions.")
        dims = tuple(int(d.size) for d in proto.dim)
        bridge.check("max_scalar_rank", len(dims))
        for dim in dims:
            bridge.check("max_scalar_elements", dim)
        bridge.check("max_scalar_elements", math.prod(dims))
        return dims

    nodes, shapes, refs, total_elements = {}, {}, {}, 0
    for node in graph.node:
        if not node.name or ":" in node.name or node.name.startswith("^") or node.name in nodes:
            _refuse("Scalar source nodes require unique tensor-addressable names.")
        schema = _ATTRS.get(node.op)
        if verify is not None and node.op in _PROOF_OPS:
            schema = {"T":"type"}
        if schema is None or node.op == "Pow" and allow_square is not True and verify is None:
            _capability("Unapproved scalar operation: " + node.op)
        required = set(schema)-{"_user_specified_name"}
        if not required <= set(node.attr) or set(node.attr)-set(schema)-{"_output_shapes"}:
            _refuse("Unapproved or incomplete scalar operation attributes.")
        if any(node.attr[k].WhichOneof("value") != schema[k] for k in schema if k in node.attr):
            _refuse("Malformed scalar operation attribute type.")
        if node.experimental_debug_info.ByteSize() or node.experimental_type.ByteSize():
            _refuse("Opaque node provenance is not admitted.")
        try:
            device = tf.DeviceSpec.from_string(node.device)
        except ValueError:
            _refuse("Malformed scalar node placement.")
        if (device.job not in (None, "localhost") or device.replica not in (None, 0)
                or device.task not in (None, 0) or device.device_type not in (None, "CPU")
                or device.device_index not in (None, 0)):
            _refuse("Scalar source placement must be local CPU.")
        if "_output_shapes" not in node.attr or node.attr["_output_shapes"].WhichOneof("value") != "list":
            _refuse("Each scalar output must carry static shape evidence.")
        output_shapes = node.attr["_output_shapes"].list
        if any(field.name != "shape" for field, _ in output_shapes.ListFields()):
            _refuse("Malformed output-shape metadata.")
        expected_ports = node.attr["num"].i if node.op == "Unpack" else 1
        if expected_ports < 1 or len(output_shapes.shape) != expected_ports:
            _refuse("Output arity differs from its static metadata.")
        shapes[node.name] = tuple(shape(s) for s in output_shapes.shape)
        total_elements += sum(math.prod(s) for s in shapes[node.name])
        higher.check("max_memory_bytes", byte_count+256*total_elements)
        nodes[node.name], refs[node.name] = node, tuple(_ref(r) for r in node.input)
        if node.op == "Placeholder" and (node.name != root[0] or node.attr["dtype"].type != 2
                or shape(node.attr["shape"].shape) != (len(parameters),)):
            _refuse("An input is not the registered float64 vector placeholder.")
        if node.op == "Const":
            tensor, dtype = node.attr["value"].tensor, node.attr["dtype"].type
            if dtype not in (2, 3, 9) or tensor.dtype != dtype or shape(tensor.tensor_shape) != shapes[node.name][0]:
                _refuse("Constant dtype/shape is not admitted.")
            payload = {2: "double_val", 3: "int_val", 9: "int64_val"}[dtype]
            if any(f.name not in {"dtype", "tensor_shape", "tensor_content", payload} for f, _ in tensor.ListFields()):
                _refuse("Unsupported constant storage.")
            size, width = math.prod(shapes[node.name][0]), 8 if dtype in (2, 9) else 4
            repeated = getattr(tensor, payload)
            if (tensor.tensor_content and (repeated or len(tensor.tensor_content) != size*width)
                    or len(repeated) > size):
                _refuse("Malformed constant payload length.")
    if root[0] not in nodes or nodes[root[0]].op != "Placeholder" or result[0] not in nodes:
        _refuse("Registered input/output does not exist.")
    dependents, indegree = {name: [] for name in nodes}, {}
    for name, inputs in refs.items():
        indegree[name] = len(inputs)
        for source, port in inputs:
            if source not in nodes or port >= len(shapes[source]):
                _refuse("A scalar tensor reference is missing or has an invalid port.")
            dependents[source].append(name)
    if result[1] >= len(shapes[result[0]]) or len(shapes[result[0]][result[1]]) != 1:
        _refuse("The scalar result must be a static vector.")
    queue, order = deque(name for name in nodes if not indegree[name]), []
    while queue:
        name = queue.popleft()
        order.append(name)
        for target in dependents[name]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    if len(order) != len(nodes):
        _refuse("Cyclic scalar dataflow is not admitted.")
    workspace, scan_work = byte_count+256*total_elements, node_count+total_elements
    higher.check("max_memory_bytes", snapshot_memory+workspace)
    higher.check("max_adjoint_operations", scan_work)

    def run(emit):
        values, tree_work = {}, 0
        for name in order:
            node = nodes[name]
            inputs = [values[s][p] for s, p in refs[name]]
            op, attr = node.op, node.attr
            arity = {"Placeholder": 0, "Const": 0, "Identity": 1, "Unpack": 1,
                "Pack": attr["N"].i if op == "Pack" else 0, "Reshape": 2, "ReverseV2": 2,
                "StridedSlice": 4, "Sin": 1, "Cos": 1, "Exp": 1,
                "Neg":1,"Tan":1,"Log":1,"Atan":1,"Conj":1}.get(op, 2)
            if len(inputs) != arity or op == "Pack" and arity < 1:
                _refuse("Scalar operation input arity mismatch.")
            dtype = attr["dtype"].type if op in ("Placeholder", "Const") else attr["T"].type
            if dtype not in (2, 3, 9) or any(x.dtype != dtype for x in
                    (inputs[:1] if op in ("Reshape", "ReverseV2", "StridedSlice") else inputs)):
                _refuse("Scalar dtype mismatch or implicit conversion.")
            def wrap(data, weight=None, depth=None, out_dtype=dtype):
                data = np.asarray(data, dtype=object if out_dtype == 2 else np.int64)
                return _Tensor(out_dtype, data,
                    np.full(data.shape, 2, dtype=object) if weight is None else np.asarray(weight, dtype=object),
                    np.full(data.shape, 2, dtype=object) if depth is None else np.asarray(depth, dtype=object))
            def layout(fn):
                source = inputs[0]
                return wrap(fn(source.data), fn(source.weight), fn(source.depth))
            def integers(index, type_key):
                source = inputs[index]
                if source.dtype not in (3, 9) or attr[type_key].type != source.dtype:
                    _refuse("Layout metadata must be static int32/int64 data.")
                return source.data
            if op == "Placeholder":
                data = ([ParameterExpression("parameter", (p,)) for p in parameters] if emit else
                    [_occurrence_signature("input",i.to_bytes(8,"big")) for i in range(len(parameters))]
                    if verify is not None else np.zeros(len(parameters)))
                out = [wrap(data)]
            elif op == "Const":
                data = tf.make_ndarray(attr["value"].tensor)
                if dtype == 2 and not np.isfinite(data).all():
                    _refuse("Nonfinite scalar constants are not admitted.")
                if dtype == 2:
                    for x in data.flat:
                        symbolic.check("max_numeric_magnitude", abs(float(x)))
                    if emit:
                        data = np.array([ParameterExpression("const", (float(x),)) for x in data.flat], dtype=object).reshape(data.shape)
                    elif verify is not None:
                        data = np.array([_literal_signature(x) for x in data.flat],dtype=object).reshape(data.shape)
                out = [wrap(data)]
            elif op == "Identity":
                out = [inputs[0]]
            elif op == "Pack":
                if any(x.data.shape != inputs[0].data.shape for x in inputs):
                    _refuse("Pack inputs require identical shapes.")
                axis, rank = attr["axis"].i, inputs[0].data.ndim+1
                if not -rank <= axis < rank:
                    _refuse("Invalid Pack axis.")
                derived = list(inputs[0].data.shape)
                derived.insert(axis % rank, len(inputs))
                if tuple(derived) != shapes[name][0]:
                    _refuse("Pack output metadata disagrees with its operands.")
                out = [wrap(*(np.stack([getattr(x, key) for x in inputs], axis=axis) for key in ("data", "weight", "depth")))]
            elif op == "Unpack":
                rank, axis = inputs[0].data.ndim, attr["axis"].i
                if not -rank <= axis < rank or inputs[0].data.shape[axis] != attr["num"].i:
                    _refuse("Invalid Unpack axis or count.")
                out = [layout(lambda a, i=i: np.take(a, i, axis=axis)) for i in range(attr["num"].i)]
            elif op == "Reshape":
                target = integers(1, "Tshape")
                if target.ndim != 1:
                    _refuse("Reshape metadata must be a vector.")
                target = tuple(int(x) for x in target)
                if sum(x == -1 for x in target) > 1 or any(x < -1 for x in target):
                    _refuse("Invalid reshape dimensions.")
                declared = shapes[name][0]
                if len(target) != len(declared) or any(x != -1 and x != y for x, y in zip(target, declared)):
                    _refuse("Reshape target and output metadata disagree.")
                if math.prod(declared) != inputs[0].data.size or -1 in target and 0 in target:
                    _refuse("Reshape changes element count or has ambiguous zero inference.")
                out = [layout(lambda a: a.reshape(declared))]
            elif op == "ReverseV2":
                axes = integers(1, "Tidx")
                rank = inputs[0].data.ndim
                if axes.ndim != 1 or any(not -rank <= int(a) < rank for a in axes):
                    _refuse("Invalid reverse axes.")
                axes = tuple(int(a) % rank for a in axes)
                if len(set(axes)) != len(axes):
                    _refuse("Repeated reverse axes are not admitted.")
                out = [layout(lambda a: np.flip(a, axis=axes))]
            elif op == "StridedSlice":
                begin, end, strides = [integers(i, "Index") for i in (1, 2, 3)]
                rank = inputs[0].data.ndim
                if any(a.ndim != 1 for a in (begin, end, strides)) or not len(begin) == len(end) == len(strides):
                    _refuse("Slice metadata requires equally sized static vectors.")
                if len(begin) != rank:
                    _capability("First-domain slicing requires one static entry per dimension.")
                masks = {k: attr[k].i for k in ("begin_mask", "end_mask", "ellipsis_mask", "new_axis_mask", "shrink_axis_mask")}
                if any(v < 0 or v >= 1 << rank for v in masks.values()):
                    _refuse("Unvalidated slicing masks.")
                if masks["ellipsis_mask"] or masks["new_axis_mask"]:
                    _capability("Ellipsis and new-axis slicing are outside the native lowering subset.")
                selection = []
                for i, (start, stop, stride) in enumerate(zip(begin, end, strides)):
                    start, stop, stride = int(start), int(stop), int(stride)
                    if stride == 0:
                        _refuse("A slice stride cannot be zero.")
                    if masks["shrink_axis_mask"] & 1 << i:
                        if stride != 1 or not -inputs[0].data.shape[i] <= start < inputs[0].data.shape[i]:
                            _refuse("Invalid shrinking slice index or stride.")
                        selection.append(start)
                    else:
                        selection.append(slice(None if masks["begin_mask"] & 1 << i else start,
                            None if masks["end_mask"] & 1 << i else stop, stride))
                out = [layout(lambda a: a[tuple(selection)])]
            else:
                if dtype != 2:
                    _refuse("Integer metadata cannot be used as numeric arithmetic.")
                if op == "Pow" and verify is None:
                    exponent_node = nodes[refs[name][1][0]]
                    if exponent_node.op != "Const" or inputs[1].data.shape != ():
                        _capability("Only a literal scalar square exponent is admitted.")
                    if float(tf.make_ndarray(exponent_node.attr["value"].tensor)) != 2.:
                        _capability("Only the fixed real square has been validated.")
                try:
                    arrays = np.broadcast_arrays(*(x.data for x in inputs))
                    weights = np.broadcast_arrays(*(x.weight for x in inputs))
                    depths = np.broadcast_arrays(*(x.depth for x in inputs))
                except ValueError:
                    _refuse("Incompatible arithmetic broadcast shapes.")
                if arrays[0].shape != shapes[name][0]:
                    _refuse("Arithmetic output shape disagrees with its operands.")
                weight, depth = np.asarray(1+sum(weights), dtype=object), np.asarray(1+np.maximum.reduce(depths), dtype=object)
                higher.check("max_parameter_graph_nodes", int(sum(weight.flat)))
                symbolic.check("max_nodes", max(weight.flat, default=0))
                symbolic.check("max_depth", max(depth.flat, default=0))
                native_op = (_SCALAR_OPS|_PROOF_OPS)[op]
                data = np.empty(arrays[0].shape, dtype=object)
                if emit:
                    for i in np.ndindex(data.shape):
                        data[i] = ParameterExpression(native_op, tuple(a[i] for a in arrays))
                elif verify is not None:
                    for i in np.ndindex(data.shape):
                        data[i] = _occurrence_signature(native_op,*(a[i] for a in arrays))
                out = [wrap(data, weight, depth)]
            if tuple(x.data.shape for x in out) != shapes[name]:
                _refuse("Derived tensor layouts disagree with captured shape metadata.")
            values[name] = out
            tree_work += sum(int(sum(x.weight.flat)) for x in out)
            higher.check("max_adjoint_operations", scan_work+tree_work)
        output = values[result[0]][result[1]]
        if output.dtype != 2:
            _refuse("A scalar binding result must be float64.")
        higher.check("max_parameter_graph_nodes", int(sum(output.weight.flat)))
        return output, tree_work

    with ledger.reserve("tensorflow_scalar_source_scan", memory_bytes=workspace, derivative_work=scan_work) as scan:
        scan.consume(derivative_work=scan_work)
        output, tree_work = run(False)
        ast_memory = 128*tree_work
        higher.check("max_memory_bytes", snapshot_memory+workspace+ast_memory)
        higher.check("max_adjoint_operations", scan_work+2*tree_work)
        if verify is not None:
            positions = {p:i for i,p in enumerate(parameters)}
            visited = 0
            def signature(value,depth=0):
                nonlocal visited
                visited += 1
                higher.check("max_parameter_graph_nodes",visited)
                symbolic.check("max_depth",depth+1)
                if isinstance(value,Parameter):
                    if value not in positions:
                        _refuse("Source proof found an unknown native input UUID.")
                    return _occurrence_signature("input",positions[value].to_bytes(8,"big"))
                if isinstance(value,Real) and not isinstance(value,bool):
                    return _literal_signature(value)
                if type(value) is not ParameterExpression:
                    _refuse("Source proof requires admitted native scalar syntax.")
                if value.op == "const":
                    return _literal_signature(value.args[0])
                if value.op == "parameter":
                    return signature(value.args[0],depth+1)
                return _occurrence_signature(value.op,*(signature(arg,depth+1) for arg in value.args))
            proof_memory = ast_memory+128*verify.node_count
            higher.check("max_memory_bytes",snapshot_memory+workspace+proof_memory)
            proof_work = tree_work+verify.node_count
            higher.check("max_adjoint_operations",scan_work+proof_work)
            with ledger.reserve("tensorflow_scalar_occurrence_proof",memory_bytes=proof_memory,
                    derivative_work=proof_work) as proof:
                proof.consume(derivative_work=proof_work)
                expected = tuple(signature(value) for value in verify.expressions)
                if expected != tuple(output.data.flat):
                    _refuse("Frozen binding differs from the actual compiled scalar source occurrences.")
            digest = hashlib.sha256(graph.SerializeToString(deterministic=True)).hexdigest()
            return verify, MappingProxyType(dict(graph_bytes=byte_count,graph_nodes=node_count,
                workspace_bytes=snapshot_memory+workspace+proof_memory,construction_work=scan_work+proof_work,
                parameter_nodes=verify.node_count,source_sha256=digest,
                input_uuids=verify.parameter_uuids,output_occurrences=len(expected),
                execution="read-only ordered occurrence proof; no AST emission or graph execution",
                allocator_memory="ESTIMATED_ONLY"))
        with ledger.reserve("tensorflow_scalar_native_binding", memory_bytes=ast_memory, derivative_work=2*tree_work) as build:
            build.consume(derivative_work=2*tree_work)
            output, _ = run(True)
            used = set().union(*(value.parameters for value in output.data.flat))
            if used < set(parameters):
                _capability("Incomplete registered input provenance: unused input positions require the old query path.")
            try:
                binding = FrozenParameterBinding.create(parameters, tuple(output.data.flat), limits=higher)
            except ValueError as error:
                from quantumbridge.core.symbolic_limits import SymbolicResourceLimitError
                if isinstance(error, SymbolicResourceLimitError):
                    raise
                _refuse("Incomplete registered input provenance: " + str(error))
            digest = hashlib.sha256(graph.SerializeToString(deterministic=True)).hexdigest()
    return binding, MappingProxyType(dict(graph_bytes=byte_count, graph_nodes=node_count,
        workspace_bytes=snapshot_memory+workspace+ast_memory, construction_work=scan_work+2*tree_work,
        parameter_nodes=binding.node_count, source_sha256=digest,
        execution="existing native ParameterExpression only", allocator_memory="ESTIMATED_ONLY"))
