# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This implementation is developed for the QuantumBridge native architecture.

from __future__ import annotations

from numbers import Integral

import numpy as np

from quantumbridge.core import Circuit, Measurement
from quantumbridge.results import Result
from quantumbridge.execution.device_derivatives import NativeDeviceDerivatives
from quantumbridge.execution.resource_context import meter_native_samples
from quantumbridge.utils.math import gate_matrix
from .execution import apply_readout_distribution, condition_enabled, validate_conditions, validate_wires, needs_dynamic_engine


class StabilizerDevice(NativeDeviceDerivatives):
    """Tableau execution, including ordered measurement, reset and classical branches."""

    simulation = "exact_stabilizer"
    supports_classical_control = True

    def __init__(self, max_qubits=None, noise_model=None, max_branches=100000, execution_limits=None):
        if max_qubits is not None and (not isinstance(max_qubits, Integral) or max_qubits < 1):
            raise ValueError("QuantumBridge stabilizer max_qubits must be a positive integer.")
        if not isinstance(max_branches, Integral) or max_branches < 1:
            raise ValueError("QuantumBridge max_branches must be a positive integer.")
        self.max_qubits = max_qubits
        self.noise_model = noise_model
        self.max_branches = int(max_branches)
        self.execution_limits = execution_limits

    def _prepare(self, circuit, parameters=None):
        from quantumbridge.information.clifford import Clifford

        if not isinstance(circuit, Circuit):
            raise TypeError("QuantumBridge StabilizerDevice requires a native Circuit.")
        bound = (circuit.bind(parameters) if parameters else circuit).lowered()
        if self.max_qubits is not None and bound.num_qubits > self.max_qubits:
            raise ValueError("QuantumBridge circuit exceeds this device qubit limit.")
        if needs_dynamic_engine(bound,self.noise_model):
            return bound, [("structured", bound, None, {})]
        program = []
        noise = self._noise_instructions(bound.num_qubits)
        inserted_noise = False
        for item in bound.iter_instructions():
            metadata = item.metadata
            self._validate_conditions(metadata, bound.num_bits)
            if isinstance(item, Measurement):
                bound._check_wire(item.wire)
                bound._check_bit(item.bit)
                if not inserted_noise and metadata.get("operation_index", len(bound.operations)) == len(bound.operations):
                    program.extend(noise)
                    inserted_noise = True
                program.append(("measure", item.wire, item.bit, metadata))
            elif item.metadata.get("directive") and item.name in {"barrier", "delay"}:
                continue
            elif item.name == "reset":
                if item.controls or len(item.targets) != 1:
                    raise ValueError("QuantumBridge reset requires one target and no controls.")
                for wire in item.wires:
                    program.append(("reset", wire, None, metadata))
            else:
                try:
                    matrix = gate_matrix(item.name, item.params, metadata)
                    Clifford.from_matrix(matrix)
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"QuantumBridge stabilizer operation {item.name!r} is not a bound Clifford gate.") from exc
                selected = tuple(bound.num_qubits - 1 - wire for wire in reversed(item.wires))
                program.append(("gate", matrix, selected, metadata))
        if not inserted_noise:
            program.extend(noise)
        return bound, program

    _validate_conditions = staticmethod(validate_conditions)
    _enabled = staticmethod(condition_enabled)

    def with_noise_model(self, noise_model):
        return StabilizerDevice(self.max_qubits, noise_model, self.max_branches, self.execution_limits)

    def _noise_instructions(self, num_qubits):
        from quantumbridge.information.clifford import Clifford
        from quantumbridge.noise.channels import PhaseDampingChannel

        if self.noise_model is None:
            return []
        if not hasattr(self.noise_model, "channels") or not hasattr(self.noise_model, "readout_errors"):
            raise TypeError("QuantumBridge stabilizer noise requires a native NoiseModel.")
        result = []
        for entry in self.noise_model.channels:
            channel = entry["channel"]
            if isinstance(channel, PhaseDampingChannel):
                p = (1 - np.sqrt(1 - channel.probability)) / 2
                kraus = [np.sqrt(1 - p) * np.eye(2), np.sqrt(p) * np.diag([1, -1])]
            else:
                kraus = channel.kraus()
            choices = []
            for operator in kraus:
                operator = np.asarray(operator, dtype=complex)
                if operator.shape != (2, 2):
                    raise ValueError("QuantumBridge native NoiseModel channels act on single wires.")
                weight = float(np.trace(operator.conj().T @ operator).real / 2)
                if weight <= 1e-15:
                    continue
                unitary = operator / np.sqrt(weight)
                try:
                    Clifford.from_matrix(unitary)
                except ValueError as exc:
                    raise ValueError("QuantumBridge stabilizer noise must be a mixture of Clifford unitaries.") from exc
                choices.append((weight, unitary))
            if not choices or not np.isclose(sum(weight for weight, _ in choices), 1, atol=1e-10, rtol=0):
                raise ValueError("QuantumBridge stabilizer noise must be trace preserving.")
            wires = range(num_qubits) if entry["wires"] is None else entry["wires"]
            for wire in wires:
                self._wires([wire], num_qubits)
                result.append(("noise", choices, (num_qubits - 1 - wire,), {}))
        for wire, error in self.noise_model.readout_errors.items():
            self._wires([wire], num_qubits)
            if any(not np.isfinite(value) or not 0 <= value <= 1 for value in (error.p0_to_1, error.p1_to_0)):
                raise ValueError("QuantumBridge readout error probabilities must be between zero and one.")
        return result

    def _readout_probability(self, wire, outcome):
        if self.noise_model is None:
            return 0.0
        error = self.noise_model.readout_errors.get(wire)
        return 0.0 if error is None else error.p1_to_0 if outcome else error.p0_to_1

    @staticmethod
    def _initial(width):
        from quantumbridge.information import Clifford, StabilizerState

        return StabilizerState(Clifford.identity(width))

    _wires = staticmethod(validate_wires)

    def _ensemble(self, circuit, program):
        if needs_dynamic_engine(circuit,self.noise_model):
            from quantumbridge.execution.control_flow_engine import dynamic_branches
            return dynamic_branches(self, circuit)
        branches = [(self._initial(circuit.num_qubits), [0] * circuit.num_bits, {}, 1.0)]
        for kind, value, args, metadata in program:
            updated = []
            for state, bits, decisions, weight in branches:
                if not self._enabled(metadata, bits, decisions):
                    updated.append((state, bits, decisions, weight))
                elif kind == "gate":
                    state.clifford._append_matrix(value, args)
                    updated.append((state, bits, decisions, weight))
                elif kind == "noise":
                    for probability, matrix in value:
                        child = state.copy()
                        child.clifford._append_matrix(matrix, args)
                        updated.append((child, bits.copy(), decisions.copy(), weight * probability))
                else:
                    qubit = circuit.num_qubits - 1 - value
                    for outcome in (0, 1):
                        child = state.copy()
                        _, probability = child._measure_z(qubit, forced=outcome)
                        if not probability:
                            continue
                        if kind == "reset":
                            if outcome:
                                child.clifford._append_gate("x", (qubit,))
                            updated.append((child, bits.copy(), decisions.copy(), weight * probability))
                        else:
                            readout = self._readout_probability(value, outcome)
                            for flip, readout_weight in ((0, 1 - readout), (1, readout)):
                                if readout_weight:
                                    child_bits = bits.copy()
                                    child_bits[args] = outcome ^ flip
                                    updated.append((child.copy(), child_bits, decisions.copy(), weight * probability * readout_weight))
                if len(updated) > self.max_branches:
                    raise RuntimeError("QuantumBridge exact stabilizer ensemble exceeded max_branches; use shot sampling instead.")
            branches = updated
        return branches

    def _trajectories(self, circuit, program, shots, seed):
        if needs_dynamic_engine(circuit,self.noise_model):
            from quantumbridge.execution.control_flow_engine import DynamicExecutionEngine
            yield from DynamicExecutionEngine(self, circuit).trajectories(shots, seed)
            return
        rng = np.random.default_rng(seed)
        prefix = self._initial(circuit.num_qubits)
        offset = 0
        for kind, value, args, metadata in program:
            if kind != "gate" or metadata.get("condition") or metadata.get("control_flow_stack"):
                break
            prefix.clifford._append_matrix(value, args)
            offset += 1
        for _ in range(shots):
            state, bits, decisions = prefix.copy(), [0] * circuit.num_bits, {}
            for kind, value, args, metadata in program[offset:]:
                if not self._enabled(metadata, bits, decisions):
                    continue
                if kind == "gate":
                    state.clifford._append_matrix(value, args)
                elif kind == "noise":
                    weights = np.array([weight for weight, _ in value])
                    index = int(rng.choice(len(value), p=weights / weights.sum()))
                    state.clifford._append_matrix(value[index][1], args)
                else:
                    qubit = circuit.num_qubits - 1 - value
                    outcome, _ = state._measure_z(qubit, rng=rng)
                    if kind == "reset":
                        if outcome:
                            state.clifford._append_gate("x", (qubit,))
                    else:
                        probability = self._readout_probability(value, outcome)
                        bits[args] = outcome ^ int(probability > 0 and rng.random() < probability)
            yield state, bits, rng

    def stabilizer_state(self, circuit, parameters=None):
        from quantumbridge.information import StabilizerState

        bound, _ = self._prepare(circuit, parameters)
        if self.noise_model is not None:
            raise ValueError("QuantumBridge noisy circuits require ensemble or shot execution, not a single stabilizer state.")
        return StabilizerState(bound.remove_final_measurements())

    def probabilities(self, circuit, wires=None, parameters=None):
        bound, program = self._prepare(circuit, parameters)
        selected = self._wires(wires, bound.num_qubits)
        qargs = [bound.num_qubits - 1 - wire for wire in reversed(selected)]
        probabilities = {}
        for state, _, _, weight in self._ensemble(bound, program):
            for label, probability in state.probabilities_dict(qargs).items():
                probabilities[label] = probabilities.get(label, 0.0) + weight * probability
        return dict(sorted(probabilities.items()))

    def _observable(self, observable, width):
        from quantumbridge.operators import Pauli, SparsePauliOperator
        from quantumbridge.utils.math import Hamiltonian, PauliString

        if isinstance(observable, str):
            observable = Pauli(observable)
        if isinstance(observable, Pauli):
            from quantumbridge.operators.pauli import _PHASE_FACTORS

            result = SparsePauliOperator([(observable._body, _PHASE_FACTORS[observable.phase])])
        elif isinstance(observable, SparsePauliOperator):
            result = observable
        elif isinstance(observable, (PauliString, Hamiltonian)):
            terms = [(1, observable)] if isinstance(observable, PauliString) else observable.terms
            converted = []
            for coefficient, pauli in terms:
                label = ["I"] * width
                for wire, char in pauli.terms:
                    self._wires([wire], width)
                    label[wire] = char
                converted.append(("".join(label), coefficient))
            result = SparsePauliOperator(converted)
        else:
            raise TypeError("QuantumBridge stabilizer observables must use Pauli, SparsePauliOperator, PauliString or Hamiltonian.")
        if result.terms and result.num_qubits != width:
            raise ValueError("QuantumBridge stabilizer observable width must match the circuit.")
        result = result.simplify()
        if any(abs(coefficient.imag) > 1e-12 for _, coefficient in result.terms):
            raise ValueError("QuantumBridge device observables must be Hermitian.")
        return result

    def expectation(self, circuit, observable, parameters=None):
        bound, program = self._prepare(circuit, parameters)
        operator = self._observable(observable, bound.num_qubits)
        return float(sum(weight * state.expectation_value(operator).real for state, _, _, weight in self._ensemble(bound, program)))

    def variance(self, circuit, observable, parameters=None):
        bound, program = self._prepare(circuit, parameters)
        operator = self._observable(observable, bound.num_qubits)
        squared = operator.dot(operator).simplify()
        first = second = 0.0
        for state, _, _, weight in self._ensemble(bound, program):
            first += weight * state.expectation_value(operator).real
            second += weight * state.expectation_value(squared).real
        return max(0.0, float(second - first ** 2))

    @meter_native_samples
    def sample(self, circuit, wires=None, shots=1, seed=None, parameters=None):
        if not isinstance(shots, Integral) or shots < 0:
            raise ValueError("QuantumBridge stabilizer shots must be a non-negative integer.")
        bound, program = self._prepare(circuit, parameters)
        selected = self._wires(wires, bound.num_qubits)
        memory = []
        for state, _, rng in self._trajectories(bound, program, shots, seed):
            outcomes = []
            for wire in selected:
                outcome, _ = state._measure_z(bound.num_qubits - 1 - wire, rng=rng)
                probability = self._readout_probability(wire, outcome)
                outcomes.append(str(outcome ^ int(probability > 0 and rng.random() < probability)))
            memory.append("".join(outcomes))
        return memory

    def counts(self, circuit, wires=None, shots=1024, seed=None, parameters=None):
        return self._counts(self.sample(circuit, wires=wires, shots=shots, seed=seed, parameters=parameters))

    @staticmethod
    def _counts(memory):
        counts = {}
        for label in memory:
            counts[label] = counts.get(label, 0) + 1
        return counts

    def _metadata(self, circuit):
        return {"device": "StabilizerDevice", "simulation": self.simulation, "num_qubits": circuit.num_qubits,
                "qubit_order": "q0-first bitstrings", "classical_order": "c0-first bitstrings",
                "noise_model_metadata": {} if self.noise_model is None else self.noise_model.summary()}

    def run_shots(self, circuit, shots=1024, seed=None, parameters=None, memory=False):
        if needs_dynamic_engine(circuit,self.noise_model):
            from quantumbridge.execution.control_flow_engine import dynamic_shots
            return dynamic_shots(self, circuit, shots, seed, parameters, memory)
        if not isinstance(shots, Integral) or shots < 1:
            raise ValueError("QuantumBridge stabilizer shots must be a positive integer.")
        bound, program = self._prepare(circuit, parameters)
        if not bound.measurements:
            samples = self.sample(bound, shots=shots, seed=seed)
        else:
            samples = ["".join(map(str, bits)) for _, bits, _ in self._trajectories(bound, program, shots, seed)]
        counts = self._counts(samples)
        metadata = {**self._metadata(bound), "shots": int(shots), "seed": seed,
                    "num_bits": bound.num_bits if bound.measurements else bound.num_qubits}
        if memory:
            metadata["memory"] = samples
        return Result(counts_data=counts, probabilities_data={key: value / shots for key, value in counts.items()}, metadata_data=metadata)

    def run(self, circuit, parameters=None, observable=None):
        bound, program = self._prepare(circuit, parameters)
        metadata = self._metadata(bound)
        records = self._ensemble(bound, program) if needs_dynamic_engine(bound,self.noise_model) else None
        if records is not None: metadata.update(records.metadata)
        if observable is not None:
            value = self.expectation(bound, observable)
            return Result(expectation_data=value, metadata_data=metadata)
        if bound.has_measurements:
            probabilities = {}
            for _, bits, _, weight in records if records is not None else self._ensemble(bound, program):
                label = "".join(map(str, bits))
                probabilities[label] = probabilities.get(label, 0.0) + weight
        else:
            probabilities = apply_readout_distribution(self.probabilities(bound), self.noise_model, range(bound.num_qubits))
        return Result(probabilities_data=probabilities, metadata_data=metadata)

    def statevector(self, circuit, parameters=None):
        return self.stabilizer_state(circuit, parameters).to_statevector().data

    def density_matrix(self, circuit, parameters=None):
        bound, program = self._prepare(circuit, parameters)
        return sum(weight * state.to_density_matrix().data for state, _, _, weight in self._ensemble(bound, program))
