# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
"""Bounded native Tape batching; execution reuses the existing QNode plan."""

from quantumbridge.execution.resource_context import copy_execution_payload as deepcopy
from dataclasses import dataclass
from itertools import islice
from numbers import Integral
from uuid import uuid4

import numpy as np

from quantumbridge.core.classical import UnsupportedFeature
from quantumbridge.diff.qnode_engine import AngleRecorder, ExecutionPlan, map_measurements
from quantumbridge.qml.measurements import MeasurementRequest
from quantumbridge.qml.qnode import QNode
from quantumbridge.qml.shot_vector import ShotSpec
from quantumbridge.qml.tape import Tape
from quantumbridge.utils.math import Hamiltonian


@dataclass(frozen=True)
class TransformLimits:
    max_transforms: int = 32
    max_tapes: int = 128
    max_operations_per_tape: int = 100000
    max_total_operations: int = 500000
    max_expansion_steps: int = 200000
    max_structure_depth: int = 64
    max_postprocessing_nodes: int = 100000
    max_total_shots: int = 1000000
    max_sample_values: int = 2000000

    def __post_init__(self):
        if any(isinstance(value, bool) or not isinstance(value, Integral) or value < 1
               for value in self.__dict__.values()):
            raise ValueError("Transform limits must be positive integers.")

    def check(self, name, value):
        if value > getattr(self, name):
            raise MemoryError("Native transform exceeded " + name)


def resolve_limits(limits):
    if limits is None:
        return TransformLimits()
    if not isinstance(limits, TransformLimits):
        raise TypeError("Native transforms require TransformLimits.")
    return limits


def _identity_angle(value):
    return value


def copy_for_transform(tape):
    result = tape.copy()
    # Transforms edit parameter ASTs, not the source builder's mutable recorder.
    result.parameter_transform = _identity_angle
    return result


def check_structure(tree, limits):
    pending, count = [(tree, 0)], 0
    while pending:
        item, depth = pending.pop()
        count += 1
        limits.check("max_structure_depth", depth)
        if isinstance(item, (tuple, list, dict)):
            values = item.values() if isinstance(item, dict) else item
            limits.check("max_postprocessing_nodes", count + len(item) + len(pending))
            pending.extend((value, depth + 1) for value in values)
        elif isinstance(item, np.ndarray):
            count += item.size
        limits.check("max_postprocessing_nodes", count)


def output_tree(tape):
    if tape.output_tree is not None:
        return tape.output_tree
    if not tape.measurements:
        return ()
    return tape.measurements[0] if len(tape.measurements) == 1 else tuple(tape.measurements)


def operation_count(circuit, limits):
    pending, count = [(circuit, 0, frozenset())], 0
    while pending:
        body, depth, ancestors = pending.pop()
        limits.check("max_structure_depth", depth)
        if id(body) in ancestors:
            raise ValueError("Cyclic native control bodies are unsupported.")
        count += len(body.operations) + len(body.measurements)
        limits.check("max_operations_per_tape", count)
        ancestors = ancestors | {id(body)}
        for operation in body.operations:
            node = operation.metadata.get("control_node")
            if node is not None:
                pending.extend((child, depth + 1, ancestors) for child in node.bodies)
    return count


def check_batch(tapes, limits, *, shot_budget=True):
    limits.check("max_tapes", len(tapes))
    operations, shots, sample_values = 0, 0, 0
    for tape in tapes:
        if not isinstance(tape, Tape):
            raise TypeError("Transform children must be native Tape objects.")
        operations += operation_count(tape.circuit, limits)
        limits.check("max_total_operations", operations)
        tree = output_tree(tape)
        check_structure(tree, limits)
        requests = []
        map_measurements(tree, lambda request: requests.append(request))
        if any(not isinstance(request, MeasurementRequest) for request in requests):
            raise TypeError("Transform outputs must contain measurement requests.")
        legacy_analytic = any(request.kind in {"state", "density_matrix"} for request in requests)
        for partition in ShotSpec.parse(tape.shots).partitions:
            for request in requests:
                count = request.shots if request.shots is not None else partition
                if legacy_analytic and request.kind in {"state", "density_matrix", "expval", "var", "probs"}:
                    count = None
                if count is not None and (isinstance(count, bool) or not isinstance(count, Integral) or count < 1):
                    raise ValueError("Transform measurement shots must be positive integers.")
                if request.kind == "pauli_group":
                    from quantumbridge.qml.pauli_group import PauliGroup
                    group = request.observable
                    if not isinstance(group, PauliGroup):
                        raise TypeError("Joint measurement requires a native PauliGroup.")
                    group.validate(tape.circuit.num_qubits)
                    sample_values += (count or 0) * (len(group.basis) + len(group.words)) + len(group.words) ** 2
                    limits.check("max_sample_values", sample_values)
                if count is not None:
                    terms = len(request.observable.terms) if request.kind == "expval" and isinstance(request.observable, Hamiltonian) else 1
                    shots += count * terms
                    if request.kind in {"sample", "counts", "probs"}:
                        width = tape.circuit.num_qubits if request.wires is None else len(request.wires)
                        width += len((request.metadata or {}).get("measurement_records", ()))
                        sample_values += count * max(1, width)
                        limits.check("max_sample_values", sample_values)
        if shot_budget:
            limits.check("max_total_shots", shots)
    return operations, shots


class TransformError(RuntimeError):
    """A user transform, child execution or reconstruction failed at this stage."""


@dataclass(frozen=True)
class PreparedTransform:
    tapes: tuple
    processing_fn: object
    metadata: dict
    processing_levels: tuple = ()


class TransformProgram:
    """Compose Tape -> (Tapes, processor) functions with reverse reconstruction.

    Processors receive one result per child Tape, including that Tape's shot
    partitions. They must be pure result functions, not execute more circuits.
    Changed shots require a validated complete native group-allocation receipt.
    """

    def __init__(self, transforms=(), *, limits=None):
        self.limits = resolve_limits(limits)
        self.transforms = tuple(islice(transforms, self.limits.max_transforms + 1))
        self.limits.check("max_transforms", len(self.transforms))
        if not all(callable(function) for function in self.transforms):
            raise TypeError("TransformProgram stages must be callable.")
        self.last_metadata = {}

    def prepare(self, tape):
        limits = self.limits
        check_batch((tape,), limits, shot_budget=False)
        batch, levels, trace, expansion = (copy_for_transform(tape),), [], [], 0
        for stage, function in enumerate(self.transforms):
            children, processors = [], []
            name = getattr(function, "__name__", type(function).__name__)
            for index, parent in enumerate(batch):
                context = f"transform stage {stage} ({name}), parent {index}"
                try:
                    returned = function(copy_for_transform(parent))
                    if not isinstance(returned, (tuple, list)) or len(returned) != 2:
                        raise TypeError("A transform must return (tapes, processing_fn).")
                    emitted, processor = returned
                    if not callable(processor):
                        raise TypeError("A transform result processor must be callable.")
                    remaining = limits.max_tapes - len(children)
                    emitted = tuple(islice(emitted, remaining + 1))
                    limits.check("max_tapes", len(children) + len(emitted))
                    added_operations, _ = check_batch(emitted, limits)
                    expansion += added_operations + len(emitted)
                    limits.check("max_expansion_steps", expansion)
                    parent_shots = ShotSpec.parse(parent.shots)
                    if any(ShotSpec.parse(child.shots) != parent_shots for child in emitted):
                        from .qwc import validate_shot_allocation
                        if not validate_shot_allocation(parent, emitted):
                            raise UnsupportedFeature("transform_shot_allocation", "Changed shot partitions require a validated native joint-group allocation receipt.")
                    start = len(children)
                    children.extend(copy_for_transform(child) for child in emitted)
                    processors.append((start, len(children), processor))
                except MemoryError as exc:
                    raise MemoryError(context + ": " + str(exc)) from exc
                except Exception as exc:
                    raise TransformError(context + ": " + str(exc)) from exc
            operation_total, shot_total = check_batch(children, limits)
            trace.append({"stage": stage, "name": name, "input_tape_count": len(batch),
                          "output_tape_count": len(children), "operation_count": operation_total,
                          "sampled_shots": shot_total,
                          "parent_child_slices": [(start, stop) for start, stop, _ in processors]})
            levels.append(tuple(processors))
            batch = tuple(children)

        def process(results, *, parameter_values=None, differentiating=False):
            if len(results) != len(batch):
                raise ValueError("Postprocessing result count must match executed Tapes.")
            current = tuple(results)
            check_structure(current, limits)
            for stage in reversed(range(len(levels))):
                reconstructed = []
                runtime_metadata = []
                for parent, (start, stop, processor) in enumerate(levels[stage]):
                    try:
                        if hasattr(processor, "native_evaluate"):
                            value = processor.native_evaluate(current[start:stop], parameters=(parameter_values or {}).get((stage,parent), ()), differentiating=differentiating)
                        else:
                            value = processor(current[start:stop])
                        if not differentiating:
                            runtime_metadata.append(deepcopy(getattr(processor, "last_metadata", {})))
                        check_structure(value, limits)
                        reconstructed.append(value)
                    except MemoryError:
                        raise
                    except Exception as exc:
                        raise TransformError(f"postprocessing stage {stage}, parent {parent}: {exc}") from exc
                current = tuple(reconstructed)
                if not differentiating:
                    trace[stage]["postprocessing_metadata"] = runtime_metadata
                check_structure(current, limits)
            if len(current) != 1:
                raise ValueError("TransformProgram must reconstruct one original Tape result.")
            return current[0]

        return PreparedTransform(batch, process, {"transform_count": len(self.transforms), "stages": trace,
                                                  "postprocessing_order": list(reversed(range(len(levels))))}, tuple(levels))

    def __call__(self, target, *, qnode_options=None):
        if not isinstance(target, Tape):
            from .dispatch import bind_qnode
            return bind_qnode(target, self, qnode_options)
        if qnode_options is not None:
            raise TypeError("Tape dispatch does not accept QNode options.")
        prepared = self.prepare(target)
        return prepared.tapes, prepared.processing_fn

    def execute_qnode(self, node, *args, **kwargs):
        transformed = self(node)
        value = transformed(*args, **kwargs)
        self.last_metadata = deepcopy(transformed.last_metadata)
        return value


_UNSET = object()


def _is_live(value):
    return type(value).__module__.startswith(("torch","autograd","jax","jaxlib","tensorflow")) or hasattr(value,"derivative")


def _reject_live(values):
    if any(_is_live(value) for value in values):
        raise UnsupportedFeature("raw_transform_autodiff", "Raw live transform gate values require a recorded QNode for native VJP execution.")


def _reject_unsupported_interface(values):
    if any(type(value).__module__.startswith(("jax", "jaxlib", "tensorflow")) for value in values):
        raise UnsupportedFeature("transform_interface", "Native transformed JAX/TF VJP execution is not implemented.")


class TransformedExecutionPlan:
    """Own child native plans and reconstruction dependencies for this call only."""

    def __init__(self, node, tape, original_recorder, program, bindings=None):
        from quantumbridge.diff.compiled_parameters import record_compiled_parameters
        self.source_tree = output_tree(tape)
        self.node, self.prepared, self.limits = node, program.prepare(tape), program.limits
        self.original_recorder, self.recorder = original_recorder, AngleRecorder()
        self.child_plans, self.parameter_slices = [], []
        self.last_spec, self.metadata = None, {}
        self.forward_physical_shots, self.forward_evaluation_count, self.last_forward_executions = 0,0,[]
        for level in self.prepared.processing_levels:
            for _, _, processor in level:
                validate = getattr(processor, "native_validate_device", None)
                if validate is not None:
                    validate(node.device)
        mapping = dict(bindings or {})
        if any(_is_live(value) for value in mapping.values()):
            raise UnsupportedFeature("raw_transform_autodiff", "Raw Tape supports numeric bindings; use a QNode for framework input VJPs.")
        seed_sequence = np.random.SeedSequence(node.seed)
        children = self.prepared.tapes
        seeds = seed_sequence.spawn(len(children)) if node.seed is not None else [None] * len(children)
        for index, (child, child_seed) in enumerate(zip(children, seeds)):
            pending = [child.circuit]
            while pending:
                body = pending.pop()
                _reject_live((body.global_phase,) + tuple(value for operation in body.operations for value in operation.params))
                for operation in body.operations:
                    control = operation.metadata.get("control_node")
                    if control is not None:
                        pending.extend(control.bodies)
            if original_recorder is None:
                child.circuit = child.circuit.assign_parameters(mapping, strict=False)
                if child.circuit.parameters:
                    raise ValueError(f"Transform child {index} has unbound parameters.")
                recorder = AngleRecorder()
            else:
                child.circuit, recorder = record_compiled_parameters(child.circuit, original_recorder)
            start = len(self.recorder.parameters)
            for angle in recorder.angles:
                self.recorder(angle)
            self.parameter_slices.append((start, len(self.recorder.parameters)))
            seed_value = child.seed if child.seed != node.seed else None if child_seed is None else int(child_seed.generate_state(1)[0])
            from copy import copy
            child_node = copy(node)
            child_node.num_qubits, child_node.shots, child_node.seed = child.circuit.num_qubits, child.shots, seed_value
            self.child_plans.append(ExecutionPlan(child_node, child, output_tree(child), recorder))
        check_batch(children, self.limits)
        self.processing_parameter_slices = {}
        for stage, level in enumerate(self.prepared.processing_levels):
            for parent, (_, _, processor) in enumerate(level):
                start = len(self.recorder.parameters)
                for value in getattr(processor, "native_parameters", ()):
                    if original_recorder is None:
                        raise UnsupportedFeature("raw_coefficient_graph", "Raw Tape execution needs numeric coefficients; use a QNode for a native coefficient VJP.")
                    self.recorder(value)
                self.processing_parameter_slices[(stage,parent)] = (start,len(self.recorder.parameters))

    def _processing_values(self, values):
        from quantumbridge.diff.scalars import scalar_primal
        result = {}
        for key,(start,stop) in self.processing_parameter_slices.items():
            original = [scalar_primal(value) for value in self.recorder.angles[start:stop]]
            if not np.array_equal(values[start:stop],original):
                raise UnsupportedFeature("coefficient_rebinding", "Coefficient primals changed after grouping/allocation; construct a fresh QNode plan.")
            result[key] = values[start:stop]
        return result

    def _reconstruct(self, values, results, executions):
        value = self.prepared.processing_fn(tuple(results), parameter_values=self._processing_values(values))
        self.metadata = {**deepcopy(self.prepared.metadata), "native_execution": True,
                         "executed_tape_count": len(executions), "executions": executions,
                         "gate_parameter_count": sum(stop-start for start,stop in self.parameter_slices),
                         "processing_parameter_count": sum(stop-start for start,stop in self.processing_parameter_slices.values()),
                         "source_gate_parameter_count": 0 if self.original_recorder is None else len(self.original_recorder.parameters),
                         "execution_semantics": "one independent native ExecutionPlan evaluation per child Tape",
                         "gradient_support": "native child quantum Jacobians and bounded native postprocessing chain rule"}
        self.metadata["deferred_measurements"] = [
            {"tape_index": index, **deepcopy(plan.circuit.metadata["deferred_measurements"])}
            for index, plan in enumerate(self.child_plans) if "deferred_measurements" in plan.circuit.metadata]
        from quantumbridge.diff.trace import flatten_tree
        try:
            _, self.last_spec = flatten_tree(value)
        except UnsupportedFeature:
            self.last_spec = None
        return value

    def evaluate(self, values):
        if len(values) != len(self.recorder.parameters):
            raise ValueError("Transformed QNode angle shape mismatch.")
        self._processing_values(values)
        self.limits.check("max_total_shots",self.forward_physical_shots+check_batch(self.prepared.tapes,self.limits)[1])
        results, executions = [], []
        for index, (plan, (start, stop)) in enumerate(zip(self.child_plans, self.parameter_slices)):
            execution_id = str(uuid4())
            try:
                value = plan.evaluate(values[start:stop])
            except Exception as exc:
                raise TransformError(f"execution child {index} ({execution_id}): {exc}") from exc
            check_structure(value, self.limits)
            results.append(value)
            executions.append({"tape_index": index, "execution_id": execution_id,
                               "gate_angle_slice": (start, stop),
                               "operation_count": operation_count(plan.circuit, self.limits), **deepcopy(plan.metadata)})
        self.forward_physical_shots += sum(part.get("physical_shots",0) for execution in executions for part in execution.get("partitions",()))
        self.forward_evaluation_count += 1
        self.last_forward_executions = deepcopy(executions)
        value = self._reconstruct(values, results, executions)
        self.metadata.update(forward_physical_shots=self.forward_physical_shots,forward_evaluation_count=self.forward_evaluation_count)
        return value

    def jacobian(self, values, method=None, step=1e-6):
        from quantumbridge.diff.processing import processing_jacobian, result_payload
        from quantumbridge.diff.trace import flatten_tree
        if self.node.max_diff not in {1, 2}:
            raise UnsupportedFeature("higher_order_derivatives", "Transformed native derivatives implement orders at most two.")
        if len(values) != len(self.recorder.parameters):
            raise ValueError("Transformed QNode angle shape mismatch.")
        self._processing_values(values)
        results, matrices, executions, derivative_receipts = [], [], [], []
        selected = [plan.gradient_capability(method or self.node.diff_method) for plan in self.child_plans]
        repetitions = [1+len(plan.recorder.parameters) if chosen == "mps-tangent" else 2+sum(4 if operation.name in {"crx","cry","crz"} and chosen == "parameter-shift" else 2
                               for operation in plan._occurrences().values()) if chosen != "adjoint" else 1
                       for plan,chosen in zip(self.child_plans,selected)]
        total_shots = sum(check_batch((child,),self.limits)[1]*count for child,count in zip(self.prepared.tapes,repetitions))
        self.limits.check("max_total_shots",total_shots+self.forward_physical_shots)
        storage = 0
        for child,count in zip(self.prepared.tapes,repetitions):
            for request in child.measurements:
                if request.kind == "pauli_group":
                    group = request.observable
                    storage += sum((shots or 0)*(len(group.words)+len(group.basis))+len(group.words)**2 for shots in ShotSpec.parse(child.shots).partitions)*count
        self.limits.check("max_sample_values",storage)
        if any(request.seed is not None and (request.shots is not None or any(shots is not None for shots in plan.shots.partitions)) for plan in self.child_plans for request in plan.requests):
            raise UnsupportedFeature("local_seed_gradient", "Fixed local measurement seeds cannot provide independent shifted sampling streams.")
        if any(request.kind == "var" and (request.shots is not None or any(shots is not None for shots in plan.shots.partitions)) for plan in self.child_plans for request in plan.requests):
            raise UnsupportedFeature("sampled_variance_gradient", "Finite-shot transformed variance derivatives have no verified estimator contract yet.")
        for level in self.prepared.processing_levels:
            for _,_,processor in level:
                if getattr(processor,"native_variance",False) and any(shots is not None for shots in ShotSpec.parse(self.node.shots).partitions):
                    raise UnsupportedFeature("sampled_variance_gradient", "Grouped finite-shot variance derivatives have no verified estimator contract yet.")
        for index,(plan,(start,stop),chosen) in enumerate(zip(self.child_plans,self.parameter_slices,selected)):
            base,matrix,_ = plan.jacobian(values[start:stop],chosen,step)
            results.append(base)
            matrices.append(matrix)
            executions.append({"tape_index":index,"execution_id":str(uuid4()),"gate_angle_slice":(start,stop),
                               "operation_count":operation_count(plan.circuit,self.limits),**deepcopy(plan.metadata)})
            derivative_receipts.append({"tape_index":index,**deepcopy(plan.metadata)})
        value = self._reconstruct(values,results,executions)
        _, processing, spec, quantum_width = processing_jacobian(self.prepared,tuple(results),self._processing_values(values))
        flat,_ = flatten_tree(value)
        if len(flat)*max(1,len(values)) > 2000000:
            raise MemoryError("Transformed QNode Jacobian exceeds its output budget.")
        matrix,offset = np.zeros((len(flat),len(values))),0
        for child_matrix,(start,stop),result in zip(matrices,self.parameter_slices,results):
            width = len(flatten_tree(result_payload(result))[0])
            matrix[:,start:stop] = processing[:,offset:offset+width] @ child_matrix
            offset += width
        offset = quantum_width
        for start,stop in self.processing_parameter_slices.values():
            matrix[:,start:stop] = processing[:,offset:offset+stop-start]
            offset += stop-start
        actual_shots = sum(plan.metadata.get("gradient_physical_shots",0) for plan in self.child_plans)
        self.metadata.update(postprocessing_chain_rule=True,native_postprocessing_derivative=True,
                             gradient_method=selected[0] if len(set(selected)) == 1 else "mixed" if selected else "classical-only",
                             shifted_evaluation_count=sum(plan.metadata.get("shifted_evaluation_count",0) for plan in self.child_plans),
                             derivative_evaluation_count=sum(plan.metadata.get("derivative_evaluation_count",0) for plan in self.child_plans),
                             derivative_order=1,gradient_executions=derivative_receipts,
                             gradient_semantics="finite-shot estimate" if total_shots else "analytic",
                             gradient_physical_shots=actual_shots,preceding_forward_physical_shots=self.forward_physical_shots,
                             preceding_forward_evaluation_count=self.forward_evaluation_count,
                             preceding_forward_executions=deepcopy(self.last_forward_executions),
                             total_physical_shots=actual_shots+self.forward_physical_shots,
                             differentiation_path="native child quantum means -> native postprocessing Jacobian -> source classical graph")
        return value,matrix,spec

    compute_jacobian = jacobian

    def second_derivatives(self, values, method=None, *, direction=None, classical=None, classical_second=None, limits=None):
        from quantumbridge.diff.second_order import transformed_second_derivatives
        return transformed_second_derivatives(self, values, method, direction=direction, classical=classical,
                                              classical_second=classical_second, limits=limits)

    def execute_and_compute_derivatives(self, values, method=None, step=1e-6):
        return self.jacobian(values,method,step)

    def execute_and_compute_vjp(self, cotangent, values, method=None, step=1e-6):
        from quantumbridge.diff.trace import flatten_tree
        value,matrix,spec = self.jacobian(values,method,step)
        flat,actual = flatten_tree(cotangent)
        if actual != spec:
            raise ValueError("Cotangent tree must match the transformed output.")
        return value,flat @ matrix

    def compute_vjp(self, cotangent, values, method=None, step=1e-6):
        return self.execute_and_compute_vjp(cotangent,values,method,step)[1]


def execute_transform(tape, transform_fn, device=None, *, bindings=None, shots=_UNSET,
                      seed=_UNSET, limits=None, return_metadata=False):
    """Execute every transformed child with the existing native QNode plan.

    Returns its reconstructed QNode-style value. With return_metadata=True,
    returns (value, metadata). Execution IDs denote real child plan evaluations,
    not claims that a multi-measurement device uses only one simulator run.
    Raw parameter bindings are numeric; native/framework derivatives use the
    recorded QNode entry points, not raw live gate values or Tape bindings.
    """
    if not isinstance(tape, Tape):
        raise TypeError("execute_transform requires a native Tape.")
    if tape.output_tree is None and not tape.measurements:
        raise ValueError("Raw transform execution needs terminal measurements or an explicit output tree.")
    limits = resolve_limits(limits if limits is not None else getattr(transform_fn, "limits", None))
    check_batch((tape,), limits, shot_budget=False)
    source = tape.copy()
    if shots is not _UNSET:
        source.shots = shots
    if seed is not _UNSET:
        source.seed = seed
    if isinstance(transform_fn, TransformProgram):
        program = TransformProgram(transform_fn.transforms, limits=limits)
    else:
        program = TransformProgram((transform_fn,), limits=limits)
    node = QNode(lambda _: None, source.circuit.num_qubits, device=device, shots=source.shots, seed=source.seed)
    plan = TransformedExecutionPlan(node, source, None, program, bindings)
    value = plan.evaluate(())
    return (value, plan.metadata) if return_metadata else value
