# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# Independent structured interpreter over native quantum-state representations.

from copy import deepcopy
from dataclasses import dataclass, field
import numpy as np

from quantumbridge.core import Circuit, Measurement
from quantumbridge.core.classical import ClassicalStore, classical_eval, classical_assign, ClassicalExecutionError, UnsupportedFeature
from quantumbridge.core.control_flow import ExecutionLimits, validate_control_flow
from quantumbridge.core.parameters import resolve_parameter
from quantumbridge.devices.execution import prepare_program, condition_enabled, readout_probability
from quantumbridge.results import Result
from quantumbridge.execution.resource_context import meter_native_samples, record_completed_native_shot, result_shots
from quantumbridge.utils.math import apply_unitary, apply_density_channel, gate_matrix, validate_kraus_operators, validate_unitary_matrix


@dataclass
class Branch:
    state: object
    store: ClassicalStore
    weight: float = 1.0
    decisions: dict = field(default_factory=dict)
    signal: str = ""
    steps: int = 0
    iterations: int = 0

    def child(self, state, probability=1.0):
        return Branch(state, self.store.copy(), self.weight * probability, self.decisions.copy(), self.signal, self.steps, self.iterations)


class ExecutionRecords(list):
    pass


def _state_copy(state):
    from quantumbridge.execution.resource_context import current_resources
    if current_resources() is not None and issubclass(type(state),np.ndarray):
        from quantumbridge.execution import mps_arrays as arrays
        return arrays.copied(state,category="dynamic_state_copy")
    return state.copy()


class QuantumStateAdapter:
    def __init__(self, device, width):
        self.device, self.width = device, width
        self.kind = getattr(device, "simulation", "statevector")

    def initial(self):
        if hasattr(self.device, "_initial"): return self.device._initial(self.width)
        raise UnsupportedFeature("dynamic_device", "QuantumBridge device lacks a native dynamic state protocol.")

    def gate(self, state, matrix, wires):
        if self.kind in {"unitary","superop"}:return self.device.map_gate(state,matrix,wires,self.width)
        if self.kind == "matrix_product_state": return state.apply(matrix, wires)
        if self.kind == "exact_density_matrix": return self.device._apply(state, "gate", matrix, wires, self.width)
        if self.kind == "exact_stabilizer":
            from quantumbridge.information import Clifford
            try: Clifford.from_matrix(matrix)
            except ValueError as exc: raise UnsupportedFeature("non_clifford_tableau", "QuantumBridge tableau execution requires Clifford gates.") from exc
            state.clifford._append_matrix(matrix, tuple(self.width - 1 - wire for wire in reversed(wires)))
            return state
        return apply_unitary(state, matrix, tuple(wires), self.width)

    def project(self, state, wire):
        results = []
        for outcome in (0, 1):
            if self.kind == "exact_density_matrix":
                child, probability = self.device._project(state, wire, outcome, self.width)
            elif self.kind == "matrix_product_state":
                child = state.copy()
                probability = child.project(wire, outcome)
            elif self.kind == "exact_stabilizer":
                child = state.copy()
                _, probability = child._measure_z(self.width - 1 - wire, forced=outcome)
            else:
                from quantumbridge.execution.resource_context import current_resources
                if current_resources() is None:
                    mask = ((np.arange(1 << self.width) >> (self.width - 1 - wire)) & 1) == outcome
                    child = state * mask
                else:
                    from quantumbridge.execution import mps_arrays as arrays
                    child=arrays.filled("dynamic_vector_projection",state.shape,complex,0)
                    for index in range(1<<self.width):
                        if (index>>(self.width-1-wire))&1==outcome:child[index]=state[index]
                probability = float(np.vdot(child, child).real)
                if probability:
                    if current_resources() is None:child = child / np.sqrt(probability)
                    else:np.divide(child,np.sqrt(probability),out=child)
            if probability > 0: results.append((probability, child, outcome))
        return results

    def channel(self, state, operators, wires):
        if self.kind in {"unitary","superop"}:return [(1.,self.device.map_channel(state,operators,wires,self.width))]
        if self.kind == "exact_density_matrix":
            return [(1.0, apply_density_channel(state, operators, wires, self.width))]
        if self.kind == "matrix_product_state":
            return [(probability, state.copy().apply(operator, wires))
                    for probability, operator in self.device._kraus_choices(state, operators, wires)]
        results = []
        for operator in operators:
            if self.kind == "exact_stabilizer":
                dimension = operator.shape[0]
                from quantumbridge.execution.resource_context import current_resources
                if current_resources() is None:probability=float(np.trace(operator.conj().T@operator).real/dimension)
                else:
                    from quantumbridge.execution import mps_arrays as arrays
                    probability=arrays.norm_squared(operator)/dimension
                if probability <= 0: continue
                unitary = operator / np.sqrt(probability) if current_resources() is None else arrays.divided(operator,np.sqrt(probability))
                if not validate_unitary_matrix(unitary, dimension):
                    raise UnsupportedFeature("tableau_kraus", "QuantumBridge tableau supports unitary Clifford mixtures, not arbitrary selective Kraus maps.")
                results.append((probability, self.gate(state.copy(), unitary, wires)))
            else:
                child = apply_unitary(state, operator, tuple(wires), self.width)
                probability = float(np.vdot(child, child).real)
                if probability > 0:
                    from quantumbridge.execution.resource_context import current_resources
                    if current_resources() is None:child=child/np.sqrt(probability)
                    else:np.divide(child,np.sqrt(probability),out=child)
                    results.append((probability,child))
        return results

    def sample(self, state, rng):
        label = []
        for wire in range(self.width):
            choices = self.project(state, wire)
            from quantumbridge.execution.resource_context import current_resources
            if current_resources() is None:
                weights = np.asarray([p for p, _, _ in choices])
                _, state, outcome = choices[int(rng.choice(len(choices), p=weights / weights.sum()))]
            else:_,state,outcome=DynamicExecutionEngine._choose(choices,rng)[0]
            p = readout_probability(getattr(self.device, "noise_model", None), wire, outcome)
            label.append(str(outcome ^ int(p > 0 and rng.random() < p)))
        return "".join(label)


class DynamicExecutionEngine:
    def __init__(self, device, circuit, parameters=None, limits=None, snapshot_context=None):
        if not isinstance(circuit, Circuit): raise TypeError("QuantumBridge execution requires a native Circuit.")
        self.device = device
        self.circuit = (circuit.bind(parameters) if parameters else circuit).lowered()
        configured = limits or getattr(device, "execution_limits", None)
        self.limits = configured or ExecutionLimits(max_branches=getattr(device, "max_branches", 4096))
        if not isinstance(self.limits, ExecutionLimits): raise TypeError("QuantumBridge execution_limits requires ExecutionLimits.")
        if getattr(device, "max_qubits", None) is not None and circuit.num_qubits > device.max_qubits: raise ValueError("Circuit exceeds device qubit limit.")
        validate_control_flow(self.circuit, limits=self.limits)
        from quantumbridge.execution.resource_context import current_resources
        resources = current_resources()
        if resources is not None:
            resources.ledger.unavailable("dynamic_classical_python_storage",
                "Control dispatch and loop tests consume shared work; Python classical stores, branch headers and allocator RSS are not measured numeric buffers.")
        self.adapter = QuantumStateAdapter(device, circuit.num_qubits)
        self.snapshots=snapshot_context
        if self.snapshots is None and circuit.has_simulator_instructions:
            from quantumbridge.simulator.snapshots import SnapshotContext
            method={"exact_density_matrix":"density_matrix","exact_stabilizer":"stabilizer"}.get(self.adapter.kind,self.adapter.kind)
            self.snapshots=SnapshotContext(circuit,method)
        self.noise_model = getattr(device, "noise_model", None)
        if self.noise_model is not None and hasattr(self.noise_model,"validate"): self.noise_model.validate(circuit.num_qubits)
        self.noise = []
        if self.noise_model is not None:
            if self.adapter.kind == "exact_stabilizer":
                for _, choices, wires, _ in device._noise_instructions(circuit.num_qubits):
                    from quantumbridge.execution.resource_context import current_resources
                    if current_resources() is None:operators=tuple(np.sqrt(p)*matrix for p,matrix in choices)
                    else:
                        from quantumbridge.execution.array_reservations import numeric_shape, numeric_array, owned_arrays
                        shapes=[numeric_shape(matrix) for _,matrix in choices]
                        operators=owned_arrays([(shape,complex) for shape in shapes],category="dynamic_noise_weighted_operator")
                        for (p,matrix),output in zip(choices,operators):
                            np.multiply(numeric_array(matrix),np.sqrt(p),out=output)
                    self.noise.append(("channel", operators, tuple(circuit.num_qubits - 1 - wire for wire in reversed(wires)), {}))
            else:
                _, self.noise = prepare_program(Circuit(circuit.num_qubits, circuit.num_bits), noise_model=self.noise_model,_leaf_only=True)

    def _gate_error(self,branches,name,wires,rng):
        model=self.noise_model
        error=model.error_for(name,wires) if model is not None and hasattr(model,"error_for") else None
        if error is None: return branches
        # Error matrices follow little-endian public qargs; native tensors are q0-first.
        return [child for branch in branches for child in self._quantum(branch,"channel",error.kraus(),tuple(reversed(wires)),rng)]

    def _program(self, circuit, top=False):
        result, inserted = [], False
        from quantumbridge.core import Operation
        from quantumbridge.core.classical_scope import entries
        from quantumbridge.core.control_flow import ControlBlock
        for variable,scope,ordered in entries(circuit):
            if not ordered and scope != "capture" and variable.op == "var":
                result.append(Operation("control_flow",(),metadata={"control_node":ControlBlock("declare",{"target":variable,"initial":None})}))
        from numbers import Real
        phase = circuit.global_phase
        if not isinstance(phase, Real) or phase != 0:
            result.append(Operation("global_phase", (0,), params=(phase,)))
        for item in circuit.iter_instructions():
            if top and isinstance(item, Measurement) and item.metadata.get("operation_index", len(circuit.operations)) == len(circuit.operations) and not inserted:
                result.extend(self.noise)
                inserted = True
            result.append(item)
        if top and not inserted: result.extend(self.noise)
        return self._group_readout(result)

    def _group_readout(self,items):
        correlated=getattr(self.noise_model,"correlated_readout_errors",{})
        if not correlated:return items
        result=[];index=0
        while index<len(items):
            if not isinstance(items[index],Measurement):result.append(items[index]);index+=1;continue
            end=index
            while end<len(items) and isinstance(items[end],Measurement):end+=1
            pending=items[index:end];by_wire={item.wire:item for item in pending};used=set()
            for wires,error in correlated.items():
                if not set(wires)&set(by_wire):continue
                if not set(wires)<=set(by_wire) or set(wires)&used or len(by_wire)!=len(pending):
                    raise UnsupportedFeature("correlated_readout_group","Correlated readout requires one consecutive complete nonoverlapping measurement group.")
                if len({m.bit for m in pending})!=len(pending):raise UnsupportedFeature("correlated_readout_mapping","Correlated blocks require distinct classical destinations.")
                group=tuple(by_wire[q] for q in wires)
                if len({m.bit for m in group})!=len(group) or any(set(m.metadata)-{"operation_index"} for m in group):
                    raise UnsupportedFeature("correlated_readout_mapping")
                result.append(("measure_group",group,error,{}));used.update(wires)
            result.extend(item for item in pending if item.wire not in used)
            index=end
        return result

    def _measure_group(self,branch,measurements,error,rng):
        active=[(branch,0)]
        for index,item in enumerate(measurements):
            updated=[]
            for current,truth in active:
                for noisy in self._gate_error([current],"measure",(item.wire,),rng):
                    for p,state,outcome in self._choose(self.adapter.project(noisy.state,item.wire),rng):
                        updated.append((noisy.child(state,p),truth|(outcome<<index)))
            active=updated;self._cap([current for current,_ in active])
        result=[]
        for current,truth in active:
            choices=[(p,index) for index,p in enumerate(error.probabilities[truth]) if p>0]
            for p,recorded in self._choose(choices,rng):
                child=current.child(_state_copy(current.state),p)
                for index,item in enumerate(measurements):child.store.bits[item.bit]=(recorded>>index)&1
                result.append(child)
        return result

    def _tick(self, branch, *, control=False):
        branch.steps += 1
        if branch.steps > self.limits.max_instruction_executions:
            raise ClassicalExecutionError("max_instruction_executions", "QuantumBridge dynamic instruction budget exhausted.")
        if control:
            from quantumbridge.execution.resource_context import current_resources
            resources = current_resources()
            if resources is not None:
                # Classical dispatch/loop tests are actual runtime work, not
                # another quantum tape or an extra measurement shot.
                resources.ledger.consume(derivative_work=1, category="dynamic_control_step")

    def _cap(self, branches):
        if len(branches) > self.limits.max_branches:
            raise ClassicalExecutionError("max_branches", "QuantumBridge analytic branches exceed max_branches; use finite-shot execution.")
        if branches and self.adapter.kind == "matrix_product_state":
            branches[0].state._guard(sum(tensor.size for branch in branches for tensor in branch.state._tensors) * 3)
        if branches and self.adapter.kind == "exact_density_matrix": self.device._check_memory(self.circuit.num_qubits, len(branches))
        if branches and self.adapter.kind == "statevector":
            limit = getattr(self.device, "max_memory_mb", None)
            if limit is not None and sum(branch.state.nbytes for branch in branches) * 8 > limit * 1024 ** 2:
                raise MemoryError("QuantumBridge dynamic statevector branches exceed max_memory_mb.")
        return branches

    @staticmethod
    def _choose(choices, rng):
        if rng is None: return choices
        from quantumbridge.execution.resource_context import current_resources
        from quantumbridge.execution.array_ownership import retained_array
        resources = current_resources()
        weights = retained_array(None if resources is None else resources.ledger,
            "dynamic_choice_probabilities", (len(choices),), float,
            lambda: np.fromiter((choice[0] for choice in choices), dtype=float, count=len(choices)))
        np.divide(weights, weights.sum(), out=weights)
        selected = choices[int(rng.choice(len(choices), p=weights))]
        return [(1.0, *selected[1:])]

    def _quantum(self, branch, kind, value, wires, rng):
        if kind == "gate":
            branch.state = self.adapter.gate(branch.state, value, wires)
            return [branch]
        if kind == "channel":
            return [branch.child(state, p) for p, state in self._choose(self.adapter.channel(branch.state, value, wires), rng)]
        children = []
        for probability, state, outcome in self._choose(self.adapter.project(branch.state, value), rng):
            if kind == "reset":
                if outcome: state = self.adapter.gate(state, gate_matrix("x"), (value,))
                children.append(branch.child(state, probability))
                continue
            readout = readout_probability(self.noise_model, value, outcome)
            flips = self._choose([(1 - readout, 0), (readout, 1)], rng)
            for factor, flip in flips:
                if factor:
                    child = branch.child(_state_copy(state), probability * factor)
                    child.store.bits[wires] = outcome ^ flip
                    children.append(child)
        return children

    def _leaf(self, item, branch, rng, binding):
        if isinstance(item, tuple):
            if item[0]=="measure_group":return self._measure_group(branch,item[1],item[2],rng)
            return self._quantum(branch, *item[:3], rng)
        if not condition_enabled(item.metadata, branch.store.bits, branch.decisions): return [branch]
        if not isinstance(item,Measurement) and "simulator_instruction" in item.metadata:
            if item.name.startswith("save_"): return self.snapshots.observe(item,branch,rng)
            if item.name.startswith("set_"): return self.snapshots.replace(item,branch)
            raise UnsupportedFeature("simulator_instruction")
        if isinstance(item, Measurement):
            return [child for current in self._gate_error([branch],"measure",(item.wire,),rng) for child in self._quantum(current,"measure",item.wire,item.bit,rng)]
        if item.metadata.get("directive") and item.name in {"barrier", "delay"}: return [branch]
        if item.name == "reset":
            if self.adapter.kind=="superop":
                reset=validate_kraus_operators(([[1,0],[0,0]],[[0,1],[0,0]]),2)
                return self._gate_error(self._quantum(branch,"channel",reset,item.wires,rng),"reset",item.wires,rng)
            return self._gate_error(self._quantum(branch,"reset",item.targets[0],None,rng),"reset",item.wires,rng)
        if item.name == "kraus":
            operators = validate_kraus_operators(item.metadata.get("kraus", ()), 1 << len(item.wires))
            return self._quantum(branch, "channel", operators, item.wires, rng)
        parameters = tuple(resolve_parameter(p, binding) for p in item.params)
        if item.metadata.get("requires_zero_input"):
            from quantumbridge.devices.execution import validate_preparation_input
            validate_preparation_input(branch.state,item.wires,self.circuit.num_qubits,self.adapter.kind)
        try: matrix = gate_matrix(item.name, parameters, item.metadata)
        except ValueError as exc: raise UnsupportedFeature("quantum_operation", str(exc)) from exc
        dimension = 1 << len(item.wires)
        if not validate_unitary_matrix(matrix, dimension):
            raise ValueError("QuantumBridge dynamic gate must be a finite unitary matching its wires.")
        return self._gate_error(self._quantum(branch,"gate",matrix,item.wires,rng),item.name,item.wires,rng)

    def _body(self, body, branches, rng, depth, binding):
        if body is None: return branches
        existing = [set(branch.store.types) for branch in branches]
        if not branches: return []
        # Every branch has the same lexical declarations on block entry.
        keep = existing[0]
        result = self._walk(self._program(body), branches, rng, depth + 1, binding)
        for branch in result:
            for key in set(branch.store.types) - keep:
                branch.store.types.pop(key)
                branch.store.values.pop(key, None)
                branch.store.names.pop(key, None)
        return result

    def _node(self, node, branch, rng, depth, binding):
        data, kind = node.data, node.kind
        if kind == "declare": branch.store.declare(data["target"], data["initial"])
        elif kind == "store": classical_assign(branch.store, data["target"], data["value"])
        elif kind in {"break", "continue"}: branch.signal = kind
        elif kind == "if":
            selected = data["body"] if classical_eval(data["condition"], branch.store) else data["alternate"]
            return self._body(selected, [branch], rng, depth, binding)
        elif kind == "switch":
            value = classical_eval(data["selector"], branch.store)
            selected = next((body for values, body in data["cases"] if value in values), data["default"])
            return self._body(selected, [branch], rng, depth, binding)
        elif kind in {"for", "while"}:
            active, finished, iteration = [branch], [], 0
            variable = data.get("variable")
            saved = None if variable is None else (variable.value[0], branch.store.types.get(variable.value[0]), branch.store.values.get(variable.value[0]), branch.store.names.get(variable.value[0]))
            if variable is not None and variable.value[0] not in branch.store.types: branch.store.declare(variable)
            while active:
                entering = []
                for current in active:
                    self._tick(current, control=True)
                    done = iteration >= len(data["values"]) if kind == "for" else not classical_eval(data["condition"], current.store)
                    if done: finished.append(current)
                    else: entering.append(current)
                if not entering: break
                if iteration >= self.limits.max_loop_iterations:
                    raise ClassicalExecutionError("max_loop_iterations", "QuantumBridge loop exceeds max_loop_iterations.")
                local = dict(binding)
                if kind == "for":
                    value = data["values"][iteration]
                    if data["parameter"] is not None: local[data["parameter"]] = value
                    if variable is not None:
                        for current in entering: classical_assign(current.store, variable, value)
                for current in entering: current.iterations += 1
                after = self._body(data["body"], entering, rng, depth, local)
                active = []
                for current in after:
                    if current.signal == "break":
                        current.signal = ""
                        finished.append(current)
                    else:
                        current.signal = ""
                        active.append(current)
                iteration += 1
                self._cap(active + finished)
            if saved is not None:
                key, dtype, value, name = saved
                for current in finished:
                    if dtype is None:
                        current.store.types.pop(key, None); current.store.values.pop(key, None); current.store.names.pop(key, None)
                    else:
                        current.store.types[key], current.store.names[key] = dtype, name
                        if value is None: current.store.values.pop(key, None)
                        else: current.store.values[key] = value
            return finished
        return [branch]

    def _walk(self, items, branches, rng, depth=0, binding=None):
        if depth > self.limits.max_control_depth: raise ClassicalExecutionError("max_control_depth", "QuantumBridge control depth exhausted.")
        for item in items:
            updated = []
            for branch in branches:
                if branch.signal: updated.append(branch); continue
                node = None if isinstance(item, (tuple, Measurement)) else item.metadata.get("control_node")
                conditional = not isinstance(item, tuple) and any(key in item.metadata for key in
                    ("condition", "gate_condition", "control_flow", "control_flow_stack"))
                self._tick(branch, control=node is not None or conditional)
                if node is not None:
                    if condition_enabled(item.metadata, branch.store.bits, branch.decisions): updated.extend(self._node(node, branch, rng, depth, binding or {}))
                    else: updated.append(branch)
                else: updated.extend(self._leaf(item, branch, rng, binding or {}))
                self._cap(updated)
            branches = updated
        return branches

    def branches(self):
        result = self._walk(self._program(self.circuit, top=True), [Branch(self.adapter.initial(), ClassicalStore(self.circuit.num_bits))], None)
        records = ExecutionRecords((branch.state, branch.store.bits, branch.decisions, branch.weight) for branch in result)
        records.metadata = {"dynamic_control_flow": True, "max_instruction_executions_observed": max(branch.steps for branch in result),
            "execution_limits": dict(self.limits.__dict__),
            "classical_ensemble": [{"weight": branch.weight, **branch.store.to_dict()} for branch in result]}
        if self.adapter.kind=="matrix_product_state":
            records.metadata.update(max_bond_observed=max(b.state.peak_bond_dimension for b in result),
                max_discarded_weight=max(b.state.discarded_weight for b in result),
                tensor_storage_bytes=sum(t.nbytes for b in result for t in b.state._tensors),**self.device.state_options)
        if self.snapshots is not None:
            records.metadata.update(self.snapshots.metadata())
            records.mps_snapshots=self.snapshots.native_mps_snapshots()
        return records

    def trajectories(self, shots, seed=None):
        if isinstance(shots, bool) or not isinstance(shots, (int, np.integer)) or not 0 <= shots <= self.limits.max_shots:
            raise ValueError("QuantumBridge dynamic shots exceed max_shots or are invalid.")
        rng = np.random.default_rng(seed)
        if self.snapshots is not None:self.snapshots.finite_shots=True
        for _ in range(shots):
            result = self._walk(self._program(self.circuit, top=True), [Branch(self.adapter.initial(), ClassicalStore(self.circuit.num_bits))], rng)
            yield result[0].state, result[0].store.bits, rng

    def quantum_samples(self, wires, shots, seed=None):
        from quantumbridge.devices.execution import validate_wires
        if getattr(self.noise_model,"correlated_readout_errors",{}):
            raise UnsupportedFeature("correlated_quantum_sample","Correlated readout requires explicit complete measurement groups and shots().")
        selected = validate_wires(wires, self.circuit.num_qubits)
        samples = []
        for state, _, rng in self.trajectories(shots, seed):
            label = []
            for wire in selected:
                choices = self._choose(self.adapter.project(state, wire), rng)
                _, state, outcome = choices[0]
                p = readout_probability(self.noise_model, wire, outcome)
                label.append(str(outcome ^ int(p > 0 and rng.random() < p)))
            samples.append("".join(label))
        return samples

    @meter_native_samples(result_count=result_shots)
    def shots(self, shots, seed=None, memory=False):
        if isinstance(shots, bool) or not isinstance(shots, (int, np.integer)) or not 0 < shots <= self.limits.max_shots:
            raise ValueError("QuantumBridge dynamic shots must be positive and within max_shots.")
        if not self.circuit.has_measurements and getattr(self.noise_model,"correlated_readout_errors",{}):
            raise UnsupportedFeature("correlated_implicit_measurement","Correlated readout requires explicit complete measurement groups.")
        rng, samples, stores, peak, discarded, steps, loops = np.random.default_rng(seed), [], [], 1, 0.0, 0, 0
        tensor_bytes=0
        if self.snapshots is not None:self.snapshots.finite_shots=True
        bond_profile = [1] * max(0, self.circuit.num_qubits - 1)
        for _ in range(shots):
            branches = self._walk(self._program(self.circuit, top=True), [Branch(self.adapter.initial(), ClassicalStore(self.circuit.num_bits))], rng)
            branch = branches[0]
            if self.adapter.kind == "matrix_product_state":
                bond_profile = [max(old, current) for old, current in zip(bond_profile, branch.state.bond_dimensions)]
            label = "".join(map(str, branch.store.bits)) if self.circuit.has_measurements else self.adapter.sample(branch.state, rng)
            samples.append(label)
            record_completed_native_shot()
            if memory: stores.append(branch.store.to_dict())
            steps, loops = max(steps, branch.steps), max(loops, branch.iterations)
            if self.adapter.kind == "matrix_product_state":
                peak, discarded = max(peak, branch.state.peak_bond_dimension), max(discarded, branch.state.discarded_weight)
                tensor_bytes=max(tensor_bytes,sum(t.nbytes for t in branch.state._tensors))
        counts = {}
        for label in samples: counts[label] = counts.get(label, 0) + 1
        metadata = {"device": type(self.device).__name__, "simulation": self.adapter.kind, "dynamic_control_flow": True,
            "execution_limits": dict(self.limits.__dict__),
            "shots": int(shots), "seed": seed, "num_bits": self.circuit.num_bits if self.circuit.has_measurements else self.circuit.num_qubits,
            "max_instruction_executions_observed": steps, "max_loop_iterations_observed": loops,
            "qubit_order": "q0-first bitstrings", "classical_order": "c0-first bitstrings"}
        if self.adapter.kind == "matrix_product_state": metadata.update(max_bond_observed=peak, max_discarded_weight=discarded,
            max_bond_dimensions_observed=bond_profile,tensor_storage_bytes=tensor_bytes, **self.device.state_options)
        if memory: metadata.update(memory=samples, classical_memory=stores)
        if self.snapshots is not None: metadata.update(self.snapshots.metadata())
        return Result(counts_data=counts, probabilities_data={key: count / shots for key, count in counts.items()}, metadata_data=metadata,
                      mps_snapshots={} if self.snapshots is None else self.snapshots.native_mps_snapshots())


def dynamic_branches(device, circuit, parameters=None):
    return DynamicExecutionEngine(device, circuit, parameters).branches()


def dynamic_shots(device, circuit, shots, seed=None, parameters=None, memory=False):
    return DynamicExecutionEngine(device, circuit, parameters).shots(shots, seed, memory)
