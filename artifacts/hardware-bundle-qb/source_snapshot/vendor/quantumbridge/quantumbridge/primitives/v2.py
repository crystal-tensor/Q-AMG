# Copyright 2026 QuantumBridge Contributors. Apache-2.0.
"""Public primitive execution containers, isolated from the Legacy result ABI."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import prod
from numbers import Integral
from threading import RLock
from uuid import uuid4
from weakref import WeakValueDictionary

import numpy as np

from quantumbridge.core import Circuit
from quantumbridge.core.circuit_metadata import execution_metadata
from quantumbridge.devices import StatevectorDevice
from quantumbridge.execution.array_reservations import array_layout, owned_arrays
from quantumbridge.execution.resource_context import (
    ResourceScope,
    current_resources,
    measurement_shots,
)
from quantumbridge.execution.resource_ledger import UnifiedResourceLedger
from quantumbridge.providers.job import Job
from quantumbridge.utils.math import gate_matrix

from .bindings_array import _shape_tuple
from .pubs import EstimatorPub, SamplerPub, _precision
from .packed_bits import BitArray, allocate


class QiskitError(Exception):
    """Native counterpart of the public primitive error; no upstream dependency."""

    def __init__(self, *message):
        self.message = " ".join(map(str, message))
        super().__init__(self.message)

    def __str__(self):
        return repr(self.message)


class DataBin(Mapping):
    def __init__(self, *, shape=(), **fields):
        self._shape = _shape_tuple(shape)
        for name, value in fields.items():
            if hasattr(type(self), name) or name.startswith("_"):
                raise ValueError(
                    "Primitive data field conflicts with a container attribute: " + name
                )
            if tuple(getattr(value, "shape", ()))[: len(self._shape)] != self._shape:
                raise ValueError(
                    "Primitive data field does not have the declared batch shape."
                )
        self._fields = dict(fields)

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return prod(self.shape)

    def __getattr__(self, name):
        try:
            return object.__getattribute__(self, "_fields")[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __getitem__(self, name):
        return self._fields[name]

    def __iter__(self):
        return iter(self._fields)

    def __len__(self):
        return len(self._fields)


@dataclass(frozen=True)
class PubResult:
    data: DataBin
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "metadata", dict(self.metadata or {}))


class SamplerPubResult(PubResult):
    def join_data(self, names=None):
        names = tuple(self.data) if names is None else tuple(names)
        if not names:
            raise ValueError("At least one register name must be selected.")
        values = [self.data[name] for name in names]
        if all(isinstance(value, BitArray) for value in values):
            return BitArray.concatenate_bits(values)
        if all(isinstance(value, np.ndarray) and value.ndim >= 1 for value in values):
            first = values[0]
            if any(value.shape[:-1] != first.shape[:-1] for value in values):
                raise ValueError("Joined arrays require matching leading dimensions.")
            output = allocate(
                first.shape[:-1] + (sum(value.shape[-1] for value in values),),
                np.result_type(*values),
                first,
            )
            np.concatenate(values, axis=-1, out=output)
            return output
        raise TypeError(
            "Joined primitive fields must all be BitArrays or all numeric ndarrays."
        )


class PrimitiveResult(Sequence):
    def __init__(self, pub_results, metadata=None):
        self._pub_results = tuple(pub_results)
        self.metadata = dict(metadata or {})

    def __getitem__(self, index):
        return self._pub_results[index]

    def __len__(self):
        return len(self._pub_results)


class PrimitiveJob(Job):
    """A local synchronous Job with cached results and original error types."""

    def __init__(self, job_id, executor):
        super().__init__(job_id, executor=executor)
        self._executor = None
        if self._error is not None:
            # A failed job must not retain the executor's numerical workspaces.
            self._error.__traceback__ = None
            self._error.__context__ = None
            self._error.__cause__ = None

    def result(self):
        if self._error is not None:
            raise self._error.with_traceback(None)
        return self._result


@dataclass
class _SamplerPlan:
    circuit: Circuit
    values: np.ndarray
    parameters: tuple
    shape: tuple
    shots: int
    registers: tuple
    selected: tuple
    work_per_point: int
    has_reset: bool = False
    error: Exception | None = None


def _sampler_plans(pubs, shots, ledger):
    plans = []
    admitted_points, admitted_shots = 0, 0
    for index, item in enumerate(pubs):
        if index >= ledger.limits.max_total_tapes:
            raise ValueError("Primitive PUB count exceeds the execution limit.")
        pub = SamplerPub.coerce(item, shots=shots)
        admitted_points += pub.size
        admitted_shots += pub.size * int(pub.shots)
        ledger.preflight(tapes=admitted_points, shots=admitted_shots)
        pub.parameter_values._validate_circuit(pub.circuit)
        circuit = pub.circuit.copy().lowered()
        mid_measurement = any(
            item.metadata.get("operation_index", len(circuit.operations))
            < len(circuit.operations)
            for item in circuit.measurements
        )
        error = (
            QiskitError("StatevectorSampler cannot handle mid-circuit measurements")
            if mid_measurement
            else None
        )
        has_reset = any(op.name == "reset" for op in circuit.operations)
        if error is None and (
            execution_metadata(circuit).get("classical_scope")
            or any(
                any(
                    key in op.metadata
                    for key in (
                        "condition",
                        "control_flow",
                        "control_node",
                        "control_flow_stack",
                        "simulator_instruction",
                    )
                )
                for op in circuit.operations
            )
        ):
            raise ValueError(
                "Native public StatevectorSampler does not yet support classical control or simulator directives."
            )
        if circuit.has_dynamic_operations and not has_reset and error is None:
            raise ValueError(
                "Unsupported nonunitary operation in native public StatevectorSampler."
            )
        registers = tuple(
            (register.name, tuple(circuit.find_bit(bit).index for bit in register))
            for register in circuit.cregs
        )
        DataBin(**{name: None for name, _ in registers})
        for measurement in circuit.measurements:
            if not (
                0 <= measurement.wire < circuit.num_qubits
                and 0 <= measurement.bit < circuit.num_bits
            ):
                raise ValueError(
                    "Primitive measurement is outside its circuit bit graph."
                )
        parameters = tuple(circuit.parameters)
        values = pub.parameter_values.as_array(parameters)
        if any(not np.isfinite(value) for value in values.flat):
            raise ValueError("Primitive parameter snapshot contains a nonfinite value.")
        selected = tuple(
            sorted(
                {measurement.wire for measurement in circuit.measurements}, reverse=True
            )
        )
        reset_work = 3 * sum(op.name == "reset" for op in circuit.operations)
        work = (
            (
                max(1, len(circuit.operations) + reset_work)
                * (array_layout((1 << circuit.num_qubits,), complex)[2] // 16)
            )
            if pub.size
            else 0
        )
        plans.append(
            _SamplerPlan(
                circuit,
                values,
                parameters,
                pub.shape,
                int(pub.shots),
                registers,
                selected,
                work,
                has_reset,
                error,
            )
        )
    layouts, workspace, points, total_shots, total_work = [], 0, 0, 0, 0
    for plan in plans:
        size = prod(plan.shape)
        points += size
        total_shots += size * plan.shots
        total_work += size * plan.work_per_point
        layouts.extend(
            (plan.shape + (plan.shots, (len(bits) + 7) // 8), np.uint8)
            for _, bits in plan.registers
        )
        if not size:
            continue
        state_bytes = array_layout((1 << plan.circuit.num_qubits,), complex)[2]
        largest_gate = max(
            (len(op.controls) + len(op.targets) for op in plan.circuit.operations),
            default=0,
        )
        gate_bytes = array_layout((1 << largest_gate, 1 << largest_gate), complex)[2]
        outcomes = 1 << len(plan.selected)
        # Conservative payload headroom for existing state/matmul/sampling,
        # probability labels and shot strings. This is not a Python RSS bound.
        current = (
            8 * state_bytes + 4 * gate_bytes + array_layout((outcomes,), float)[2] * 4
        )
        if plan.has_reset:
            current += 4 * state_bytes
        current += plan.shots * (32 + 4 * (len(plan.selected) + plan.circuit.num_bits))
        current += outcomes * max(8, 4 * len(plan.selected))
        workspace = max(workspace, current)
    packed_bytes = sum(array_layout(shape, dtype)[2] for shape, dtype in layouts)
    ledger.preflight(
        memory_bytes=packed_bytes + workspace,
        tapes=points,
        shots=total_shots,
        derivative_work=total_work,
    )
    # Validate all parameter-specific lowering/gate legality before the first
    # quantum execution, without storing an object array of every bound circuit.
    for plan in plans:
        if plan.error is not None:
            continue
        for loc in np.ndindex(plan.shape):
            bound = plan.circuit.bind(
                dict(zip(plan.parameters, plan.values[loc]))
            ).lowered()
            for operation in bound.operations:
                if operation.name == "reset":
                    if (
                        len(operation.targets) != 1
                        or operation.controls
                        or operation.params
                    ):
                        raise ValueError(
                            "Reset must target exactly one qubit without controls or parameters."
                        )
                    continue
                if operation.metadata.get("directive") and operation.name in {
                    "barrier",
                    "delay",
                }:
                    continue
                gate_matrix(operation.name, operation.params, operation.metadata)
            if plan.has_reset and plan.error is None:
                from quantumbridge.execution.control_flow_engine import (
                    DynamicExecutionEngine,
                )

                DynamicExecutionEngine(
                    StatevectorDevice(), bound.remove_final_measurements(inplace=False)
                )
    return plans, layouts, points, total_shots, total_work


def _execute_sampler(plans, layouts, points, shots, work, seed, ledger):
    # Public errors are observed through Job.result, but no earlier PUB may
    # execute when a later PUB is statically known to be unsupported.
    for plan in plans:
        if plan.error is not None:
            raise plan.error
    outputs = iter(owned_arrays(layouts, category="primitive_packed_memory"))
    buffers = []
    for plan in plans:
        fields = {name: next(outputs) for name, _ in plan.registers}
        for buffer in fields.values():
            buffer.fill(0)
        buffers.append(fields)
    device, results = StatevectorDevice(), []
    ledger.unavailable(
        "primitive_sampler_python_workspace",
        "State/sampling/packed ndarray leases and conservative payload headroom are admitted; Python headers, RNG internals and process RSS are not measured.",
    )
    with ledger.reserve(
        "primitive_sampler_work", tapes=points, derivative_work=work
    ) as ticket:
        with measurement_shots(ledger, shots):
            for plan, fields in zip(plans, buffers):
                positions = {wire: index for index, wire in enumerate(plan.selected)}
                classical = {
                    measurement.bit: measurement.wire
                    for measurement in plan.circuit.measurements
                }
                for loc in np.ndindex(plan.shape):
                    bound = plan.circuit.bind(
                        dict(zip(plan.parameters, plan.values[loc]))
                    ).lowered()
                    try:
                        # Actual native shot memory with existing marginal
                        # sampling and completion metering, never counts expansion.
                        if plan.has_reset:
                            from quantumbridge.execution.control_flow_engine import (
                                DynamicExecutionEngine,
                            )
                            from quantumbridge.information import Statevector

                            trajectories = DynamicExecutionEngine(
                                device, bound.remove_final_measurements(inplace=False)
                            ).trajectories(1)
                            try:
                                state, _, _ = next(trajectories)
                                # Public sampler seed controls terminal samples,
                                # not reset preparation (verified in both SDKs).
                                memory = Statevector(state).sample_memory(
                                    plan.shots, qargs=plan.selected, seed=seed
                                )
                            finally:
                                trajectories.close()
                        else:
                            memory = device.sample(
                                bound, wires=plan.selected, shots=plan.shots, seed=seed
                            )
                    except BaseException:
                        ledger.unavailable(
                            "primitive_failed_quantum_work",
                            "Known completed samples remain charged. Additional work in a failed native call is unavailable, not zero or physically rolled back.",
                        )
                        raise
                    ticket.consume(tapes=1, derivative_work=plan.work_per_point)
                    for name, bits in plan.registers:
                        packed = fields[name][loc]
                        for shot, sample in enumerate(memory):
                            for local, global_bit in enumerate(bits):
                                wire = classical.get(global_bit)
                                if wire is not None and sample[positions[wire]] == "1":
                                    packed[shot, packed.shape[-1] - 1 - local // 8] |= (
                                        np.uint8(1 << (local % 8))
                                    )
                data = DataBin(
                    shape=plan.shape,
                    **{
                        name: BitArray(fields[name], len(bits))
                        for name, bits in plan.registers
                    },
                )
                results.append(
                    SamplerPubResult(
                        data,
                        {
                            "shots": plan.shots,
                            "circuit_metadata": dict(plan.circuit.metadata or {}),
                        },
                    )
                )
    return PrimitiveResult(results, {"version": 2})


class _LocalPrimitive:
    # Admit synchronous local execution before any instance or ledger lock.
    # Nested A->B/B->A calls otherwise form a cross-thread lock cycle. A shared
    # RLock permits same-thread reentry while serializing local admissions.
    _execution_lock = RLock()

    def __init__(self, seed):
        np.random.default_rng(seed)
        self._seed = seed
        self._jobs, self._pending = WeakValueDictionary(), set()
        self._job_lock, self._rng_lock = RLock(), RLock()

    @property
    def seed(self):
        return self._seed

    def _new_id(self):
        with self._job_lock:
            for _ in range(4):
                identifier = str(uuid4())
                if identifier not in self._pending and identifier not in self._jobs:
                    self._pending.add(identifier)
                    return identifier
        raise RuntimeError("Repeated primitive job UUID collision.")

    def _submit(self, executor):
        identifier = self._new_id()
        try:
            with self._rng_lock:
                job = PrimitiveJob(identifier, executor)
            with self._job_lock:
                self._jobs[identifier] = job
            return job
        finally:
            with self._job_lock:
                self._pending.discard(identifier)


class StatevectorSampler(_LocalPrimitive):
    """Native public V2 sampling; Legacy Sampler remains a separate entrance."""

    def __init__(self, *, default_shots=1024, seed=None):
        if isinstance(default_shots, bool) or not isinstance(default_shots, Integral):
            raise TypeError("default_shots must be an integer.")
        if default_shots <= 0:
            raise ValueError("default_shots must be positive.")
        super().__init__(seed)
        self._default_shots = int(default_shots)

    @property
    def default_shots(self):
        return self._default_shots

    def run(self, pubs, *, shots=None):
        shots = self.default_shots if shots is None else shots
        if isinstance(shots, bool) or not isinstance(shots, Integral):
            raise TypeError("Sampler shots must be an integer.")
        if shots <= 0:
            raise ValueError("Sampler shots must be positive.")
        resources = current_resources()
        ledger = UnifiedResourceLedger() if resources is None else resources.ledger
        # The existing account lock preserves whole-job workspace headroom
        # while synchronous native allocation uses this same account.
        # Waiting calls must not hold a ledger lock while another call's
        # allocation/GC can release storage belonging to that waiting ledger.
        with self._execution_lock, self._rng_lock, ledger._lock, ResourceScope(ledger):
            plans, layouts, points, count, work = _sampler_plans(
                pubs, int(shots), ledger
            )
            return self._submit(lambda: _execute_sampler(
                plans, layouts, points, count, work, self.seed, ledger
            ))


@dataclass
class _EstimatorPlan:
    circuit: Circuit
    observables: object
    values: np.ndarray
    parameters: tuple
    parameter_shape: tuple
    shape: tuple
    precision: float
    has_reset: bool
    work_per_cell: int
    error: Exception | None = None


def _broadcast_coordinate(loc, shape):
    return tuple(0 if width == 1 else index for index, width in zip(loc[len(loc)-len(shape):], shape)) if shape else ()


def _estimator_plans(pubs, precision, ledger):
    plans, layouts, points, total_work, workspace = [], [], 0, 0, 0
    for index, item in enumerate(pubs):
        if index >= ledger.limits.max_total_tapes:
            raise MemoryError("Estimator PUB count exceeds the execution limit.")
        pub = EstimatorPub.coerce(item, precision=precision)
        points += pub.size
        ledger.preflight(tapes=points)
        pub.parameter_values._validate_circuit(pub.circuit)
        pub.observables.validate_for_execution()
        circuit = pub.circuit.copy().lowered()
        observables = pub.observables.copy()
        parameters = tuple(circuit.parameters)
        values = pub.parameter_values.as_array(parameters)
        error = None
        if pub.size and circuit.num_qubits >= 63:
            raise MemoryError("Estimator statevector exceeds checked array dimensions.")
        non_pauli = next((label for loc in np.ndindex(observables.shape) for label in observables._at(loc) if set(label) - set("IXYZ")), None) if pub.size else None
        if non_pauli is not None:
            error = QiskitError(f'Pauli string label "{non_pauli}" is not valid.')
        elif circuit.measurements:
            error = QiskitError("Cannot apply instruction with classical bits: measure")
        elif execution_metadata(circuit).get("classical_scope") or any(
            any(key in op.metadata for key in ("condition", "control_flow", "control_node", "control_flow_stack", "simulator_instruction"))
            for op in circuit.operations
        ):
            error = QiskitError("Native public Estimator does not yet support classical control or simulator directives")
        has_reset = any(op.name == "reset" for op in circuit.operations)
        work = 0
        if pub.size:
            width = circuit.num_qubits
            if width >= 63:
                raise MemoryError("Estimator statevector exceeds checked array dimensions.")
            state_bytes = array_layout((1 << width,), complex)[2]
            largest_gate = max(1, max((len(op.controls) + len(op.targets) for op in circuit.operations), default=0))
            gate_bytes = array_layout((1 << largest_gate,) * 2, complex)[2]
            term_count = max((len(observables._at(loc)) for loc in np.ndindex(observables.shape)), default=0)
            work = (max(1, len(circuit.operations)) + 3*sum(op.name == "reset" for op in circuit.operations) + 2*width*term_count + 1) * (state_bytes // 16)
            workspace = max(workspace, (12 if has_reset else 8)*state_bytes + 4*gate_bytes + pub.size*8)
        total_work += pub.size*work
        layouts.extend(((pub.shape, float), (pub.shape, float)))
        plans.append(_EstimatorPlan(circuit, observables, values, parameters,
                                    pub.parameter_values.shape, pub.shape, pub.precision,
                                    has_reset, work, error))
    ledger.preflight(memory_bytes=sum(array_layout(shape, dtype)[2] for shape, dtype in layouts) + workspace,
                     tapes=points, derivative_work=total_work)
    # Lower and validate every parameter point before any PUB can evolve a state.
    for plan in plans:
        if plan.error is not None or not prod(plan.shape):
            continue
        for loc in np.ndindex(plan.parameter_shape):
            bound = plan.circuit.bind(dict(zip(plan.parameters, plan.values[loc]))).lowered()
            for operation in bound.operations:
                if operation.name == "reset":
                    if len(operation.targets) != 1 or operation.controls or operation.params:
                        raise ValueError("Reset requires one target and no parameters or controls.")
                elif operation.metadata.get("directive") and operation.name in {"barrier", "delay"}:
                    continue
                else:
                    gate_matrix(operation.name, operation.params, operation.metadata)
            if plan.has_reset:
                from quantumbridge.execution.control_flow_engine import DynamicExecutionEngine
                DynamicExecutionEngine(StatevectorDevice(), bound)
    return plans, layouts, points, total_work


def _observable_expectation(state, width, terms, ledger):
    from quantumbridge.utils.math import Hamiltonian, PauliString, expectation_observable

    if all(not set(label) - set("IXYZ") for label in terms):
        observable = Hamiltonian([
            (coefficient, PauliString([(width-1-index, char) for index, char in enumerate(label) if char != "I"]))
            for label, coefficient in terms.items()
        ])
        return expectation_observable(state, width, observable).real
    from quantumbridge.execution.array_ownership import retained_array
    from quantumbridge.utils.math import apply_unitary, pauli_matrix

    total = 0j
    projectors = {"0": ("Z", 1), "1": ("Z", -1), "+": ("X", 1), "-": ("X", -1), "r": ("Y", 1), "l": ("Y", -1)}
    for label, coefficient in terms.items():
        evolved = state
        for index, char in enumerate(label):
            if char == "I":
                continue
            if char in projectors:
                axis, sign = projectors[char]
                matrix = retained_array(ledger, "primitive_observable_projector", (2, 2), complex)
                np.multiply(pauli_matrix(axis), sign*0.5, out=matrix)
                matrix[0, 0] += 0.5
                matrix[1, 1] += 0.5
            else:
                matrix = pauli_matrix(char)
            evolved = apply_unitary(evolved, matrix, (width-1-index,), width)
        total += coefficient*np.vdot(state, evolved)
    return total.real


def _execute_estimator(plans, layouts, points, work, seed, ledger):

    for plan in plans:
        if plan.error is not None:
            raise plan.error
    buffers = owned_arrays(layouts, category="primitive_estimator_output")
    device, results = StatevectorDevice(), []
    ledger.unavailable("primitive_estimator_workspace", "Owned numerical arrays and conservative state/gate headroom are admitted; Python objects, RNG internal allocation and process RSS are not measured.")
    with ledger.reserve("primitive_estimator_work", tapes=points, derivative_work=work) as ticket:
        for pub_index, plan in enumerate(plans):
            evs, stds = buffers[2*pub_index:2*pub_index+2]
            stds.fill(0)
            rng = np.random.default_rng(seed)
            for loc in np.ndindex(plan.shape):
                parameter_loc = _broadcast_coordinate(loc, plan.parameter_shape)
                observable_loc = _broadcast_coordinate(loc, plan.observables.shape)
                bound = plan.circuit.bind(dict(zip(plan.parameters, plan.values[parameter_loc]))).lowered()
                terms = plan.observables._at(observable_loc)
                trajectories = None
                try:
                    if plan.has_reset:
                        from quantumbridge.execution.control_flow_engine import DynamicExecutionEngine
                        trajectories = DynamicExecutionEngine(device, bound).trajectories(1, seed=rng)
                        state, _, _ = next(trajectories)
                    else:
                        state = device.statevector(bound)
                    evs[loc] = _observable_expectation(state, bound.num_qubits, terms, ledger)
                    ticket.consume(tapes=1, derivative_work=plan.work_per_cell)
                    if plan.precision:
                        evs[loc] = rng.normal(evs[loc], plan.precision)
                except BaseException:
                    ledger.unavailable("primitive_failed_estimator_work", "Completed cells remain charged; partial work in the failed native call is unknown, not physically rolled back.")
                    raise
                finally:
                    if trajectories is not None:
                        trajectories.close()
                    state = None
            results.append(PubResult(DataBin(shape=plan.shape, evs=evs, stds=stds),
                                     {"target_precision": plan.precision, "circuit_metadata": dict(plan.circuit.metadata or {})}))
    return PrimitiveResult(results, {"version": 2})


class StatevectorEstimator(_LocalPrimitive):
    """Native public V2 expectation execution, independent of Legacy Estimator."""

    def __init__(self, *, default_precision=0.0, seed=None):
        precision = _precision(default_precision)
        if precision is None:
            raise ValueError("default_precision must be a nonnegative number.")
        super().__init__(seed)
        self._default_precision = precision

    @property
    def default_precision(self):
        return self._default_precision

    def run(self, pubs, *, precision=None):
        precision = self.default_precision if precision is None else _precision(precision)
        resources = current_resources()
        ledger = UnifiedResourceLedger() if resources is None else resources.ledger
        with self._execution_lock, self._rng_lock, ledger._lock, ResourceScope(ledger):
            plans, layouts, points, work = _estimator_plans(pubs, precision, ledger)
            return self._submit(lambda: _execute_estimator(plans, layouts, points, work, self.seed, ledger))
