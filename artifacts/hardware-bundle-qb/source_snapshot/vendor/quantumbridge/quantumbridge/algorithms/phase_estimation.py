# Copyright 2026 QuantumBridge Contributors. Apache-2.0.
"""Circuit-based phase estimation through Native controlled gates and Sampler V2."""
from collections.abc import Mapping
from math import isfinite
from numbers import Integral, Real

import numpy as np

from quantumbridge.core import Circuit, ClassicalRegister
from quantumbridge.core.gate_limits import current_gate_limits
from quantumbridge.core.gates import validate_definition
from quantumbridge.core.operations import instruction_definition_work
from quantumbridge.circuit_library.templates import qft


class AlgorithmError(RuntimeError):
    """A quantum algorithm's primitive execution or result protocol failed."""


def _evaluation_bits(value):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("num_evaluation_qubits must be an integer.")
    if value < 1:
        raise ValueError("num_evaluation_qubits must be positive.")
    current_gate_limits().check("max_circuit_bits", value)
    return int(value)


def phase_estimation_circuit(num_evaluation_qubits, unitary, state_preparation=None):
    """Build coherent QPE; evaluation wire zero is the phase's most significant bit."""
    bits = _evaluation_bits(num_evaluation_qubits)
    if not isinstance(unitary, Circuit):
        raise TypeError("Phase estimation requires a Native unitary Circuit.")
    if state_preparation is not None and not isinstance(state_preparation, Circuit):
        raise TypeError("state_preparation must be a Native Circuit or None.")
    if state_preparation is not None and (state_preparation.num_qubits != unitary.num_qubits
            or state_preparation.num_bits or state_preparation.measurements):
        raise ValueError("State preparation must match the unitary width and have no classical bits.")
    limits = current_gate_limits()
    limits.check("max_circuit_bits", bits+unitary.num_qubits)
    if bits > limits.max_repeat_count.bit_length():
        raise MemoryError("Phase-estimation queries exceed max_repeat_count before expansion.")
    queries = (1 << bits)-1
    limits.check("max_repeat_count", queries)
    controlled = unitary.to_gate().control()
    body_work = validate_definition(controlled.definition, owner=controlled)
    preparation_work = 0 if state_preparation is None else instruction_definition_work(state_preparation)
    work = queries*max(1, body_work)+preparation_work+2*bits+bits*(bits-1)//2+bits//2
    limits.check("max_total_operations", work)
    limits.check("max_decomposition_work", work)
    circuit = Circuit(bits+unitary.num_qubits, name="QPE")
    targets = tuple(range(bits, circuit.num_qubits))
    if state_preparation is not None:
        circuit.compose(state_preparation, qubits=targets, inplace=True)
    for wire in range(bits):
        circuit.h(wire)
    # Repeated coherent queries avoid eigenvalue or dense matrix-power shortcuts.
    for wire in range(bits):
        for _ in range(1 << (bits-1-wire)):
            circuit.append(controlled, (wire, *targets))
    circuit.compose(qft(bits, inverse=True), qubits=range(bits), inplace=True)
    return circuit


class PhaseEstimationResult:
    """Measured QPE distribution, distinct from the historical exact helper result."""

    def __init__(self, num_evaluation_qubits, circuit_result, phases):
        self._num_evaluation_qubits = _evaluation_bits(num_evaluation_qubits)
        if isinstance(phases, Mapping):
            current_gate_limits().check("max_decomposition_work", len(phases))
            for key in phases:
                if type(key) is not str or len(key) != self._num_evaluation_qubits or set(key)-{"0", "1"}:
                    raise ValueError("Phase keys must be evaluation-width binary strings.")
            values = phases.values()
        elif type(phases) is np.ndarray:
            if phases.shape != (1 << self._num_evaluation_qubits,):
                raise ValueError("Phase array must have one weight per evaluation outcome.")
            current_gate_limits().check("max_decomposition_work", phases.size)
            values = phases.flat
        else:
            raise TypeError("phases must be a mapping or a one-dimensional ndarray.")
        for value in values:
            if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
                raise ValueError("Phase weights must be finite real numbers.")
        self._phases, self._circuit_result = phases, circuit_result

    @property
    def num_evaluation_qubits(self):
        return self._num_evaluation_qubits

    @property
    def circuit_result(self):
        return self._circuit_result

    @property
    def phases(self):
        return self._phases

    @property
    def phase(self):
        if isinstance(self.phases, Mapping):
            return int(max(self.phases, key=self.phases.get), 2)/(1 << self.num_evaluation_qubits)
        return int(np.argmax(self.phases))/(1 << self.num_evaluation_qubits)

    def filter_phases(self, cutoff=0., as_float=True):
        if isinstance(cutoff, bool) or not isinstance(cutoff, Real) or not isfinite(cutoff):
            raise ValueError("Phase cutoff must be a finite real number.")
        if type(as_float) is not bool:
            raise TypeError("as_float must be boolean.")
        entries = self.phases.items() if isinstance(self.phases, Mapping) else (
            (format(index, "0"+str(self.num_evaluation_qubits)+"b"), value) for index, value in enumerate(self.phases))
        return {(int(label, 2)/(1 << self.num_evaluation_qubits) if as_float else label): value
            for label, value in entries if value > cutoff}

    def combine(self, result):
        if result is None:
            raise TypeError("Cannot combine a phase-estimation result with None.")
        if not isinstance(result, PhaseEstimationResult):
            raise TypeError("Expected a circuit phase-estimation result.")
        self._num_evaluation_qubits = result.num_evaluation_qubits
        self._circuit_result, self._phases = result.circuit_result, result.phases


class PhaseEstimation:
    """Run controlled-query QPE using an explicitly supplied Sampler V2."""

    def __init__(self, num_evaluation_qubits, sampler=None, *, transpiler=None, transpiler_options=None):
        self._num_evaluation_qubits = _evaluation_bits(num_evaluation_qubits)
        from quantumbridge.primitives.sampler import Sampler as LegacySampler
        if sampler is None or not callable(getattr(sampler, "run", None)):
            raise AlgorithmError("A Sampler V2 with run() must be provided.")
        if isinstance(sampler, LegacySampler):
            raise AlgorithmError("Phase estimation requires Sampler V2, not the Legacy sampler.")
        if transpiler is not None and not callable(getattr(transpiler, "run", None)):
            raise TypeError("transpiler must expose run().")
        if transpiler_options is not None and not isinstance(transpiler_options, Mapping):
            raise TypeError("transpiler_options must be a mapping or None.")
        self._sampler, self._transpiler = sampler, transpiler
        self._transpiler_options = dict(transpiler_options or {})

    def construct_circuit(self, unitary, state_preparation=None):
        return phase_estimation_circuit(self._num_evaluation_qubits, unitary, state_preparation)

    def estimate(self, unitary, state_preparation=None, *, parameter_values=None, shots=None):
        return self.estimate_from_pe_circuit(self.construct_circuit(unitary, state_preparation),
            parameter_values=parameter_values, shots=shots)

    def estimate_from_pe_circuit(self, pe_circuit, *, parameter_values=None, shots=None):
        if not isinstance(pe_circuit, Circuit):
            raise TypeError("pe_circuit must be a Native Circuit.")
        bits = self._num_evaluation_qubits
        if pe_circuit.num_qubits < bits or pe_circuit.measurements:
            raise ValueError("QPE input must have the evaluation register and no measurements.")
        if shots is not None and (isinstance(shots, bool) or not isinstance(shots, Integral) or shots < 1):
            raise ValueError("shots must be a positive integer or None.")
        circuit = pe_circuit.copy()
        names, name, suffix = {register.name for register in circuit.cregs}, "meas", 0
        while name in names:
            suffix += 1
            name = "meas_"+str(suffix)
        register = ClassicalRegister(bits, name)
        circuit.add_register(register)
        for wire, bit in enumerate(register):
            circuit.measure(wire, circuit.find_bit(bit).index)
        if self._transpiler is not None:
            circuit = self._transpiler.run(circuit, **self._transpiler_options)
            if not isinstance(circuit, Circuit) or not any(reg.name == name and len(reg) == bits for reg in circuit.cregs):
                raise ValueError("Transpiler must preserve the QPE classical output register.")
        pub = (circuit,) if parameter_values is None else (circuit, parameter_values)
        try:
            job = self._sampler.run([pub], **({} if shots is None else {"shots": int(shots)}))
            raw = job.result()
            if len(raw) != 1:
                raise ValueError("QPE requires one primitive PUB result.")
            samples = raw[0].data[name]
            sample_shots = samples.num_shots
            if (samples.shape != () or samples.num_bits != bits or isinstance(sample_shots, bool)
                    or not isinstance(sample_shots, Integral) or sample_shots < 1):
                raise ValueError("QPE requires one scalar bit-array with the evaluation width.")
            counts = samples.get_counts()
            if not isinstance(counts, Mapping) or not counts:
                raise ValueError("QPE sampler returned no measured counts.")
            current_gate_limits().check("max_decomposition_work", len(counts))
            phases, total = {}, 0
            for label, count in counts.items():
                if type(label) is not str or len(label) != bits or set(label)-{"0", "1"}:
                    raise ValueError("QPE sampler returned malformed evaluation bits.")
                if isinstance(count, bool) or not isinstance(count, Integral) or count < 0:
                    raise ValueError("QPE sampler counts must be nonnegative integers.")
                total += int(count)
                # BitArray displays the high classical bit first; phase wire 0 is MSB.
                phases[label[::-1]] = int(count)/sample_shots
            if total != sample_shots:
                raise ValueError("QPE sampler counts disagree with its number of shots.")
        except MemoryError:
            raise
        except Exception as error:
            raise AlgorithmError("Phase-estimation sampler job or result protocol failed.") from error
        return PhaseEstimationResult(bits, raw, phases)
