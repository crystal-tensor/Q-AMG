# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# Independently implemented for the QuantumBridge native architecture.

from __future__ import annotations

from numbers import Integral

import numpy as np

from quantumbridge.information.matrix_product_state import MatrixProductState
from quantumbridge.results import Result
from quantumbridge.execution.device_derivatives import NativeDeviceDerivatives
from quantumbridge.execution.resource_context import meter_native_samples, record_completed_native_shot, result_shots
from quantumbridge.execution import mps_arrays as arrays
from quantumbridge.utils.math import Hamiltonian, PauliString, observable_matrix
from .execution import (
    condition_enabled, decision_liveness, prepare_program, readout_probability, needs_dynamic_engine,
)


class MatrixProductStateDevice(NativeDeviceDerivatives):
    """Native SVD MPS execution, exact ensembles or seeded quantum trajectories."""

    simulation = "matrix_product_state"
    supports_classical_control = True

    def __init__(self, max_qubits=None, max_bond_dim=None, cutoff=1e-16, max_memory_mb=1024,
                 max_branches=4096, noise_model=None, initial_state=None, execution_limits=None):
        if max_qubits is not None and (isinstance(max_qubits, bool) or not isinstance(max_qubits, Integral) or max_qubits < 1):
            raise ValueError("QuantumBridge MPS max_qubits must be a positive integer or None.")
        if isinstance(max_branches, bool) or not isinstance(max_branches, Integral) or max_branches < 1:
            raise ValueError("QuantumBridge max_branches must be a positive integer.")
        validated = MatrixProductState.zero(1, max_bond_dim=max_bond_dim, cutoff=cutoff, max_memory_mb=max_memory_mb)
        self.max_qubits = None if max_qubits is None else int(max_qubits)
        self.max_branches = int(max_branches)
        self.state_options = dict(max_bond_dim=validated.max_bond_dim, cutoff=validated.cutoff, max_memory_mb=validated.max_memory_mb)
        self.noise_model, self.initial_state = noise_model, initial_state
        self.execution_limits = execution_limits
        self.last_execution_metadata = {}

    def _configured(self, **changes):
        options = dict(self.state_options, max_qubits=self.max_qubits, max_branches=self.max_branches,
                       noise_model=self.noise_model, initial_state=self.initial_state, execution_limits=self.execution_limits)
        options.update(changes)
        return MatrixProductStateDevice(**options)

    def with_noise_model(self, noise_model):
        return self._configured(noise_model=noise_model)

    def with_initial_state(self, initial_state):
        return self._configured(initial_state=initial_state)

    def _prepare(self, circuit, parameters=None):
        bound, program = prepare_program(circuit, parameters, self.max_qubits, self.noise_model)
        if not needs_dynamic_engine(bound,self.noise_model) and bound.global_phase != 0:
            matrix = arrays.filled("mps_global_phase_matrix", (2, 2), complex, 0)
            matrix[0, 0] = matrix[1, 1] = np.exp(1j * float(bound.global_phase))
            program.insert(0, ("gate", matrix, (0,), {}))
        return bound, program

    def _initial(self, width):
        if self.initial_state is None:
            return MatrixProductState.zero(width, **self.state_options)
        if isinstance(self.initial_state, MatrixProductState):
            state = MatrixProductState(self.initial_state.tensors, **self.state_options)
            state.discarded_weight = self.initial_state.discarded_weight
            state.truncations = self.initial_state.truncations
            state.peak_bond_dimension = max(state.peak_bond_dimension, self.initial_state.peak_bond_dimension)
        else:
            state = MatrixProductState.from_statevector(self.initial_state, **self.state_options)
        if state.num_qubits != width:
            raise ValueError("QuantumBridge MPS initial state width must match the circuit.")
        return state

    @staticmethod
    def _gate(state, matrix, wires, metadata):
        if metadata.get("requires_zero_input"):
            from .execution import validate_preparation_input
            validate_preparation_input(state,wires,state.num_qubits,"matrix_product_state")
        return state.apply(matrix,wires)

    @staticmethod
    def _kraus_choices(state, operators, wires):
        density = state.reduced_density_matrix(wires)
        probabilities = []
        for operator in operators:
            first = arrays.contracted(operator, density)
            product = arrays.contracted(first, arrays.conjugated(operator).T)
            probabilities.append(max(0.0, float(np.trace(product).real)))
        return [(p, operator) for p, operator in zip(probabilities, operators) if p > 0]

    def _branches(self, circuit, program):
        if needs_dynamic_engine(circuit,self.noise_model):
            from quantumbridge.execution.control_flow_engine import dynamic_branches
            return dynamic_branches(self, circuit)
        branches = [(self._initial(circuit.num_qubits), [0] * circuit.num_bits, {}, 1.0)]
        live = decision_liveness(program)
        for index, (kind, value, wires, metadata) in enumerate(program):
            updated = []
            for state, bits, decisions, weight in branches:
                if not condition_enabled(metadata, bits, decisions):
                    updated.append((state, bits, decisions, weight))
                elif kind == "gate":
                    updated.append((self._gate(state,value,wires,metadata), bits, decisions, weight))
                elif kind == "channel":
                    for probability, operator in self._kraus_choices(state, value, wires):
                        self._guard_branches(state, len(branches) + len(updated) + 1)
                        child = state.copy().apply(operator, wires)
                        updated.append((child, bits.copy(), decisions.copy(), weight * probability))
                else:
                    for outcome in (0, 1):
                        self._guard_branches(state, len(branches) + len(updated) + 1)
                        child = state.copy()
                        probability = child.project(value, outcome)
                        if not probability:
                            continue
                        readout = readout_probability(self.noise_model, value, outcome)
                        for flip, factor in ((0, 1 - readout), (1, readout)):
                            if factor:
                                recorded = bits.copy()
                                recorded[wires] = outcome ^ flip
                                updated.append((child.copy(), recorded, decisions.copy(), weight * probability * factor))
                if len(updated) > self.max_branches:
                    raise RuntimeError("QuantumBridge MPS analytic ensemble exceeded max_branches; use shot execution.")
            branches = [(state, bits, {key: value for key, value in decisions.items() if key in live[index]}, weight)
                        for state, bits, decisions, weight in updated]
        return branches

    @staticmethod
    def _guard_branches(state, count):
        state._guard(sum(tensor.size for tensor in state._tensors) * (count + 2))

    def _trajectories(self, circuit, program, shots, seed):
        if needs_dynamic_engine(circuit,self.noise_model):
            from quantumbridge.execution.control_flow_engine import DynamicExecutionEngine
            yield from DynamicExecutionEngine(self, circuit).trajectories(shots, seed)
            return
        rng, prefix, offset = np.random.default_rng(seed), self._initial(circuit.num_qubits), 0
        for kind, value, wires, metadata in program:
            if kind != "gate" or metadata.get("condition") or metadata.get("control_flow_stack"):
                break
            self._gate(prefix,value,wires,metadata)
            offset += 1
        for _ in range(shots):
            state, bits, decisions = prefix.copy(), [0] * circuit.num_bits, {}
            for kind, value, wires, metadata in program[offset:]:
                if not condition_enabled(metadata, bits, decisions):
                    continue
                if kind == "gate":
                    self._gate(state,value,wires,metadata)
                elif kind == "channel":
                    choices = self._kraus_choices(state, value, wires)
                    weights = arrays.owned("mps_trajectory_probabilities", (len(choices),), float,
                        lambda: np.fromiter((p for p, _ in choices), dtype=float, count=len(choices)))
                    np.divide(weights, weights.sum(), out=weights)
                    selected = int(rng.choice(len(choices), p=weights))
                    state.apply(choices[selected][1], wires)
                else:
                    outcome = state.measure(value, rng)
                    p = readout_probability(self.noise_model, value, outcome)
                    bits[wires] = outcome ^ int(p > 0 and rng.random() < p)
            yield state, bits, rng

    def matrix_product_state(self, circuit, parameters=None):
        bound, program = self._prepare(circuit, parameters)
        if any(kind != "gate" or metadata.get("condition") or metadata.get("control_flow_stack") for kind, _, _, metadata in program):
            raise ValueError("QuantumBridge single MPS state requires unitary execution without measurements/channels; use ensemble or shots.")
        state = self._initial(bound.num_qubits)
        for _, matrix, wires, metadata in program:
            self._gate(state,matrix,wires,metadata)
        self.last_execution_metadata = self._metadata(bound,[state])
        return state

    def statevector(self, circuit, parameters=None):
        return self.matrix_product_state(circuit, parameters).statevector()

    state = statevector

    def reduced_density_matrix(self, circuit, wires=None, parameters=None):
        bound, program = self._prepare(circuit, parameters)
        branches = self._branches(bound, program)
        result = None
        for state, _, _, weight in branches:
            density = state.reduced_density_matrix(wires)
            weighted = arrays.owned("mps_weighted_density", density.shape, density.dtype)
            np.multiply(density, weight, out=weighted)
            if result is None: result = weighted
            else: np.add(result, weighted, out=result)
        self.last_execution_metadata = self._metadata(bound,[state for state,_,_,_ in branches])
        return 0 if result is None else result

    def density_matrix(self, circuit, parameters=None):
        return self.reduced_density_matrix(circuit, parameters=parameters)

    def probabilities(self, circuit, wires=None, parameters=None):
        bound, program = self._prepare(circuit, parameters)
        result = {}
        states = []
        for state, _, _, weight in self._branches(bound, program):
            states.append(state)
            for label, probability in state.probabilities(wires).items():
                result[label] = result.get(label, 0.0) + weight * probability
        self.last_execution_metadata = self._metadata(bound,states)
        return result

    @staticmethod
    def _paulis(observable, width):
        from quantumbridge.operators import Pauli, SparsePauliOperator
        from quantumbridge.operators.pauli import _PHASE_FACTORS

        if isinstance(observable, str):
            observable = Pauli(observable)
        if isinstance(observable, Pauli):
            observable = SparsePauliOperator([(observable._body, _PHASE_FACTORS[observable.phase])])
        if isinstance(observable, (PauliString, Hamiltonian)):
            terms = [(1, observable)] if isinstance(observable, PauliString) else observable.terms
            converted = []
            for coefficient, pauli in terms:
                label = ["I"] * width
                for wire, char in pauli.terms:
                    if not 0 <= wire < width:
                        raise ValueError("QuantumBridge observable wires are outside the MPS.")
                    label[wire] = char
                converted.append(("".join(label), coefficient))
            observable = SparsePauliOperator(converted)
        if not isinstance(observable, SparsePauliOperator):
            return None
        if observable.terms and observable.num_qubits != width:
            raise ValueError("QuantumBridge MPS observable width must match its state.")
        result = observable.simplify()
        if any(abs(coefficient.imag) > 1e-12 for _, coefficient in result.terms):
            raise ValueError("QuantumBridge MPS observables must be Hermitian.")
        return result

    def _moments(self, circuit, observable, parameters=None, variance=False):
        bound, program = self._prepare(circuit, parameters)
        operator = self._paulis(observable, bound.num_qubits)
        squared = operator.dot(operator).simplify() if operator is not None and variance else None
        first = second = 0.0
        states = []
        for state, _, _, weight in self._branches(bound, program):
            states.append(state)
            if operator is not None:
                first += weight * sum(coefficient * state.expectation_pauli(label) for label, coefficient in operator.terms).real
                if variance:
                    second += weight * sum(coefficient * state.expectation_pauli(label) for label, coefficient in squared.terms).real
            else:
                state._guard((1 << (2 * bound.num_qubits)) * 8)
                matrix = observable_matrix(observable, bound.num_qubits)
                from quantumbridge.utils.math import validate_observable_matrix
                if not validate_observable_matrix(matrix, bound.num_qubits):
                    raise ValueError("QuantumBridge MPS observables must be finite Hermitian matrices matching the state.")
                density = state.reduced_density_matrix()
                product = arrays.contracted(density, matrix)
                first += weight * np.trace(product).real
                if variance:
                    second += weight * np.trace(arrays.contracted(product, matrix)).real
        self.last_execution_metadata = self._metadata(bound,states)
        return max(0.0, float(second - first ** 2)) if variance else float(first)

    def expectation(self, circuit, observable, parameters=None):
        return self._moments(circuit, observable, parameters)

    def variance(self, circuit, observable, parameters=None):
        return self._moments(circuit, observable, parameters, variance=True)

    @staticmethod
    def _shots(shots, allow_zero=False):
        if isinstance(shots, bool) or not isinstance(shots, Integral) or shots < int(not allow_zero):
            raise ValueError("QuantumBridge MPS shots must be a positive integer (or zero for sample/counts).")

    @meter_native_samples
    def sample(self, circuit, wires=None, shots=1, seed=None, parameters=None):
        self._shots(shots, allow_zero=True)
        bound, program = self._prepare(circuit, parameters)
        selected = self._initial(bound.num_qubits)._wires(wires)
        memory = []
        resource = self._metadata(bound,[])
        for state, _, rng in self._trajectories(bound, program, shots, seed):
            current = self._metadata(bound,[state])
            for key in ("max_bond_observed","max_discarded_weight","max_tensor_bytes","max_truncations"):
                resource[key] = max(resource[key],current[key])
            resource["max_bond_dimensions_observed"] = [max(a,b) for a,b in zip(resource["max_bond_dimensions_observed"],current["max_bond_dimensions_observed"])]
            outcomes = []
            for wire in selected:
                outcome = state.measure(wire, rng)
                p = readout_probability(self.noise_model, wire, outcome)
                outcomes.append(str(outcome ^ int(p > 0 and rng.random() < p)))
            memory.append("".join(outcomes))
            record_completed_native_shot()
        self.last_execution_metadata = {**resource,"shots":shots,"seed":seed}
        return memory

    def counts(self, circuit, wires=None, shots=1024, seed=None, parameters=None):
        return self._counts(self.sample(circuit, wires, shots, seed, parameters))

    @staticmethod
    def _counts(memory):
        result = {}
        for label in memory:
            result[label] = result.get(label, 0) + 1
        return result

    def _metadata(self, bound, states):
        return {"device": "MatrixProductStateDevice", "simulation": self.simulation,
                "num_qubits": bound.num_qubits, "qubit_order": "q0-first bitstrings", "classical_order": "c0-first bitstrings",
                **self.state_options, "max_branches": self.max_branches,
                "max_bond_observed": max((state.peak_bond_dimension for state in states), default=1),
                "max_discarded_weight": max((state.discarded_weight for state in states), default=0),
                "max_tensor_bytes":max((state.tensor_bytes for state in states),default=0),
                "tensor_bytes_semantics":"maximum observed final MPS storage, not peak SVD workspace",
                "max_truncations":max((state.truncations for state in states),default=0),
                "max_bond_dimensions_observed":[max((state.bond_dimensions[index] for state in states),default=1) for index in range(bound.num_qubits-1)],
                "noise_model_metadata": {} if self.noise_model is None else self.noise_model.summary()}

    @meter_native_samples(result_count=result_shots)
    def run_shots(self, circuit, shots=1024, seed=None, parameters=None, memory=False):
        if needs_dynamic_engine(circuit,self.noise_model):
            from quantumbridge.execution.control_flow_engine import dynamic_shots
            return dynamic_shots(self, circuit, shots, seed, parameters, memory)
        self._shots(shots)
        bound, program = self._prepare(circuit, parameters)
        samples, max_bond, discarded = [], 1, 0.0
        bond_profile = [1] * max(0, bound.num_qubits - 1)
        for state, bits, rng in self._trajectories(bound, program, shots, seed):
            bond_profile = [max(old, current) for old, current in zip(bond_profile, state.bond_dimensions)]
            if not bound.measurements:
                bits = []
                for wire in range(bound.num_qubits):
                    outcome = state.measure(wire, rng)
                    p = readout_probability(self.noise_model, wire, outcome)
                    bits.append(outcome ^ int(p > 0 and rng.random() < p))
            samples.append("".join(map(str, bits)))
            record_completed_native_shot()
            max_bond = max(max_bond, state.peak_bond_dimension)
            discarded = max(discarded, state.discarded_weight)
        counts = self._counts(samples)
        metadata = {**self._metadata(bound, []), "shots": int(shots), "seed": seed,
                    "max_bond_observed": max_bond, "max_discarded_weight": discarded,
                    "max_bond_dimensions_observed": bond_profile}
        if memory:
            metadata["memory"] = samples
        return Result(counts_data=counts, probabilities_data={key: value / shots for key, value in counts.items()}, metadata_data=metadata)

    def run(self, circuit, parameters=None, observable=None):
        bound, program = self._prepare(circuit, parameters)
        branches = self._branches(bound, program)
        probabilities = None
        if bound.has_measurements:
            probabilities = {}
            for _, bits, _, weight in branches:
                label = "".join(map(str, bits))
                probabilities[label] = probabilities.get(label, 0.0) + weight
        metadata = {**self._metadata(bound, [state for state, _, _, _ in branches]), **getattr(branches, "metadata", {})}
        pure = branches[0][0] if len(branches) == 1 else None
        if pure is None:
            metadata["mps_ensemble"] = [{"weight": weight, "classical_bits": bits, "state": state.to_dict()} for state, bits, _, weight in branches]
        return Result(mps=pure, probabilities_data=probabilities, metadata_data=metadata,
                      mps_snapshots=getattr(branches, "mps_snapshots", {}),
                      expectation_data=None if observable is None else self.expectation(bound, observable))
