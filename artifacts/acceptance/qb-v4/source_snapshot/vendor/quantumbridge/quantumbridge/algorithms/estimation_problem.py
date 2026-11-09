# Copyright 2026 QuantumBridge Contributors. Apache-2.0.
"""Native amplitude-estimation inputs and coherent Grover reflections."""
from math import asin, isfinite, pi
from numbers import Integral, Real
import warnings

from quantumbridge.core import Circuit
from quantumbridge.core.gate_limits import current_gate_limits
from quantumbridge.core.operations import instruction_definition_work


def _unitary_circuit(value, name):
    if not isinstance(value, Circuit):
        raise TypeError(name+" must be a Native Circuit.")
    if value.num_bits or value.measurements:
        raise ValueError(name+" must have no classical bits or measurements.")
    value.to_gate()
    return value


def _phase_flip(circuit, wires):
    gate = Circuit(1).z(0).to_gate()
    if len(wires) > 1:
        gate = gate.control(len(wires)-1)
    circuit.append(gate, wires)


class EstimationProblem:
    """State preparation, objective wires, coherent oracle and result mapping."""

    def __init__(self, state_preparation, objective_qubits, grover_operator=None,
                 post_processing=None, is_good_state=None):
        self._objective_qubits, self._grover_operator = None, None
        self.state_preparation = state_preparation
        self.objective_qubits = objective_qubits
        self.grover_operator = grover_operator
        self.post_processing = post_processing
        self.is_good_state = is_good_state

    @property
    def state_preparation(self):
        return self._state_preparation

    @state_preparation.setter
    def state_preparation(self, value):
        _unitary_circuit(value, "state_preparation")
        objectives = getattr(self, "_objective_qubits", None)
        if objectives is not None and any(wire >= value.num_qubits for wire in objectives):
            raise ValueError("Objective qubits exceed the new preparation width.")
        grover = getattr(self, "_grover_operator", None)
        if grover is not None and grover.num_qubits != value.num_qubits:
            raise ValueError("Grover operator and preparation must have the same width.")
        self._state_preparation = value

    @property
    def objective_qubits(self):
        return list(self._objective_qubits)

    @objective_qubits.setter
    def objective_qubits(self, value):
        if isinstance(value, Integral) and not isinstance(value, bool):
            value = [value]
        if type(value) not in (tuple, list) or not value:
            raise ValueError("objective_qubits must be a nonempty list of wire indices.")
        if any(isinstance(wire, bool) or not isinstance(wire, Integral) for wire in value):
            raise TypeError("Objective qubits must be integers.")
        wires = tuple(map(int, value))
        if len(set(wires)) != len(wires) or any(wire < 0 or wire >= self.state_preparation.num_qubits for wire in wires):
            raise ValueError("Objective qubits must be distinct and in range.")
        self._objective_qubits = wires

    @property
    def grover_operator(self):
        if self._grover_operator is not None:
            return self._grover_operator
        preparation, limits = self.state_preparation, current_gate_limits()
        _unitary_circuit(preparation, "state_preparation")
        width = preparation.num_qubits
        ancillary = set(preparation.ancillas)
        reflection_wires = tuple(index for index, bit in enumerate(preparation.qubits) if bit not in ancillary)
        limits.check("max_control_qubits", max(len(reflection_wires)-1, len(self._objective_qubits)-1))
        work = 2*instruction_definition_work(preparation)+2*len(reflection_wires)+2
        limits.check("max_total_operations", work)
        limits.check("max_decomposition_work", work)
        circuit = Circuit(width, name="Grover")
        _phase_flip(circuit, self._objective_qubits)
        circuit.compose(preparation.inverse(), inplace=True)
        for wire in reflection_wires:
            circuit.x(wire)
        if reflection_wires:
            _phase_flip(circuit, reflection_wires)
        else:
            circuit.global_phase = circuit.global_phase+pi
        for wire in reflection_wires:
            circuit.x(wire)
        circuit.compose(preparation, inplace=True)
        # The zero reflection above is I-2|0><0|; Q needs its negative.
        circuit.global_phase = circuit.global_phase+pi
        return circuit

    @grover_operator.setter
    def grover_operator(self, value):
        if value is not None:
            _unitary_circuit(value, "grover_operator")
            if value.num_qubits != self.state_preparation.num_qubits:
                raise ValueError("Grover operator and preparation must have the same width.")
        self._grover_operator = value

    @property
    def post_processing(self):
        return (lambda value: value) if self._post_processing is None else self._post_processing

    @post_processing.setter
    def post_processing(self, value):
        if value is not None and not callable(value):
            raise TypeError("post_processing must be callable or None.")
        self._post_processing = value

    @property
    def has_good_state(self):
        return self._is_good_state is not None

    @property
    def is_good_state(self):
        return (lambda label: all(bit == "1" for bit in label)) if self._is_good_state is None else self._is_good_state

    @is_good_state.setter
    def is_good_state(self, value):
        if value is not None and not callable(value):
            raise TypeError("is_good_state must be callable or None.")
        self._is_good_state = value

    def rescale(self, scaling_factor):
        if (isinstance(scaling_factor, bool) or not isinstance(scaling_factor, Real)
                or not isfinite(scaling_factor) or not 0 <= scaling_factor <= 1):
            raise ValueError("scaling_factor must be a finite real number in [0, 1].")
        width = self.state_preparation.num_qubits
        current_gate_limits().check("max_circuit_bits", width+1)
        if self._grover_operator is not None:
            warnings.warn("Rescaling discards the custom Grover operator and rebuilds its reflections.", UserWarning, stacklevel=2)
        preparation = Circuit(width+1).compose(self.state_preparation, qubits=range(width))
        preparation.ry(2*asin(scaling_factor), width)
        predicate = self.is_good_state
        return EstimationProblem(preparation, [*self.objective_qubits, width], post_processing=self.post_processing,
            is_good_state=lambda label: label[0] == "1" and predicate(label[1:]))
