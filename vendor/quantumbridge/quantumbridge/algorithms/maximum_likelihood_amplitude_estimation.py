# Copyright 2026 QuantumBridge Contributors. Apache-2.0.
"""Independent multi-circuit maximum-likelihood amplitude estimation."""
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite, pi, sin
from numbers import Integral, Real
import warnings

import numpy as np

from quantumbridge.core import Circuit, ClassicalRegister
from quantumbridge.core.gate_limits import current_gate_limits
from quantumbridge.core.operations import instruction_definition_work
from .binomial_amplitude_statistics import (BinomialLikelihood, confidence_interval,
    evaluation_schedule as _evaluation_schedule, fisher_information, search_boundaries)
from .estimation_problem import EstimationProblem, _unitary_circuit
from .phase_estimation import AlgorithmError


@dataclass
class MaximumLikelihoodAmplitudeEstimationResult:
    circuit_results: object = None
    primitive_result: object = None
    estimation: float | None = None
    estimation_processed: object = None
    theta: float | None = None
    evaluation_schedule: list | None = None
    good_counts: list | None = None
    total_counts: list | None = None
    fisher_information: float | None = None
    num_oracle_queries: int | float | None = None
    shots: int | float | None = None
    confidence_interval: tuple | None = None
    confidence_interval_processed: tuple | None = None
    confidence_interval_method: str | None = None
    post_processing: object = None
    minimizer: object = None

    def combine(self, result):
        if result is None:
            raise TypeError("Cannot combine a maximum-likelihood result with None.")
        for name in vars(self):
            if hasattr(result, name):
                setattr(self, name, getattr(result, name))


def _success_counts(results, task, num_state_qubits=None):
    if not isinstance(task, EstimationProblem):
        raise TypeError("estimation_problem must be a Native EstimationProblem.")
    if type(results) not in (tuple, list) or not results:
        raise ValueError("Circuit results must be a nonempty list.")
    width, objectives = task.state_preparation.num_qubits, task.objective_qubits
    if num_state_qubits is not None:
        warnings.warn("num_state_qubits is deprecated; width is taken from the Native problem.", DeprecationWarning, stacklevel=2)
        if isinstance(num_state_qubits, bool) or not isinstance(num_state_qubits, Integral) or num_state_qubits != width:
            raise ValueError("num_state_qubits must match the problem's full width.")
    good, totals, predicate = [], [], task.is_good_state
    for row in results:
        successes, total = 0, 0
        if isinstance(row, Mapping):
            current_gate_limits().check("max_decomposition_work", len(row))
            if not row:
                raise ValueError("Count results must not be empty.")
            for label, count in row.items():
                if isinstance(count, bool) or not isinstance(count, Real) or not isfinite(count) or count < 0:
                    raise ValueError("Counts must be finite nonnegative real numbers.")
                if type(label) is not str or set(label.replace(" ", "")) - {"0", "1"}:
                    raise ValueError("Count labels must be binary strings.")
                label = label.replace(" ", "")
                if len(label) != len(objectives):
                    raise ValueError("Count width must match the objective measurement register.")
                total += count
                if predicate(label):
                    successes += count
        elif isinstance(row, np.ndarray):
            current_gate_limits().check("max_decomposition_work", row.size)
            if row.ndim != 1 or row.size != 1 << width or not np.all(np.isfinite(row)):
                raise ValueError("Native statevector must match the problem's full width.")
            probabilities = np.abs(row)**2
            if not np.isclose(np.sum(probabilities), 1., atol=1e-12, rtol=1e-12):
                raise ValueError("Statevector must be normalized.")
            for index, probability in enumerate(probabilities):
                label = "".join(str((index >> (width-1-wire)) & 1) for wire in reversed(objectives))
                if predicate(label):
                    successes += float(probability)
            total = 1
        else:
            raise TypeError("Circuit results must contain counts or Native ndarray statevectors.")
        if total <= 0:
            raise ValueError("Circuit results require positive exposure.")
        good.append(successes)
        totals.append(total)
    return good, totals


class MaximumLikelihoodAmplitudeEstimation:
    """Sample A, QA, Q**2 A, ... and fit their joint binomial likelihood."""

    def __init__(self, evaluation_schedule, minimizer=None, sampler=None, *, transpiler=None, transpiler_options=None):
        self._schedule = _evaluation_schedule(evaluation_schedule)
        if minimizer is not None and not callable(minimizer):
            raise TypeError("minimizer must be callable or None.")
        if transpiler is not None and not callable(getattr(transpiler, "run", None)):
            raise TypeError("transpiler must expose run().")
        if transpiler_options is not None and not isinstance(transpiler_options, Mapping):
            raise TypeError("transpiler_options must be a mapping or None.")
        self._minimizer, self._sampler = minimizer, sampler
        self._transpiler = transpiler
        self._transpiler_options = dict(transpiler_options or {})

    @property
    def evaluation_schedule(self):
        return list(self._schedule)

    @property
    def sampler(self):
        return self._sampler

    @sampler.setter
    def sampler(self, value):
        self._sampler = value

    def construct_circuits(self, estimation_problem, measurement=False):
        if not isinstance(estimation_problem, EstimationProblem):
            raise TypeError("estimation_problem must be a Native EstimationProblem.")
        if type(measurement) is not bool:
            raise TypeError("measurement must be boolean.")
        _evaluation_schedule(self._schedule)
        preparation = _unitary_circuit(estimation_problem.state_preparation, "state_preparation")
        grover = estimation_problem.grover_operator if any(self._schedule) else None
        if grover is not None:
            _unitary_circuit(grover, "grover_operator")
        work = len(self._schedule)*(instruction_definition_work(preparation)+len(estimation_problem.objective_qubits))
        if grover is not None:
            work += sum(self._schedule)*instruction_definition_work(grover)
        current_gate_limits().check("max_total_operations", work)
        current_gate_limits().check("max_decomposition_work", work)
        circuits = []
        for power in self._schedule:
            circuit = Circuit(preparation.num_qubits).compose(preparation)
            for _ in range(power):
                circuit.compose(grover, inplace=True)
            if measurement:
                circuit.add_register(ClassicalRegister(len(estimation_problem.objective_qubits), "meas"))
                for bit, wire in enumerate(estimation_problem.objective_qubits):
                    circuit.measure(wire, bit)
            circuits.append(circuit)
        return circuits

    def _fit(self, likelihood):
        minimize = self._minimizer or self._default_minimizer
        bounds = [[0., pi/2]] if self._minimizer is None else [[1e-15, pi/2-1e-15]]
        theta = minimize(likelihood.negative, bounds)
        if isinstance(theta, bool) or not isinstance(theta, Real) or not isfinite(theta) or not 0 <= theta <= pi/2:
            raise ValueError("minimizer must return one finite angle in [0, pi/2].")
        if not isfinite(likelihood(theta)):
            raise ValueError("minimizer returned a point with zero likelihood.")
        return float(theta)

    def compute_mle(self, circuit_results, estimation_problem, num_state_qubits=None, return_counts=False):
        if type(return_counts) is not bool:
            raise TypeError("return_counts must be boolean.")
        good, totals = _success_counts(circuit_results, estimation_problem, num_state_qubits)
        likelihood = BinomialLikelihood(self._schedule, good, totals)
        theta = self._fit(likelihood)
        return (theta, good) if return_counts else theta

    def estimate(self, estimation_problem, *, parameter_values=None, shots=None):
        from quantumbridge.primitives.sampler import Sampler as LegacySampler
        if self.sampler is None or isinstance(self.sampler, LegacySampler) or not callable(getattr(self.sampler, "run", None)):
            raise AlgorithmError("An explicit Sampler V2 is required for maximum-likelihood amplitude estimation.")
        if shots is not None and (isinstance(shots, bool) or not isinstance(shots, Integral) or shots < 1):
            raise ValueError("shots must be a positive integer or None.")
        _evaluation_schedule(self._schedule)
        if self._minimizer is None:
            search_boundaries(self._schedule)
        circuits = self.construct_circuits(estimation_problem, measurement=True)
        prepared = []
        for circuit in circuits:
            if parameter_values is not None:
                circuit = circuit.assign_parameters(parameter_values)
            if circuit.parameters:
                raise ValueError("One fully bound parameter point is required.")
            if self._transpiler is not None:
                circuit = self._transpiler.run(circuit, **self._transpiler_options)
                if not isinstance(circuit, Circuit) or not any(reg.name == "meas" and len(reg) == len(estimation_problem.objective_qubits) for reg in circuit.cregs):
                    raise ValueError("Transpiler must preserve the objective readout register.")
            prepared.append(circuit)
        try:
            raw = self.sampler.run(prepared, **({} if shots is None else {"shots": int(shots)})).result()
            if len(raw) != len(self._schedule):
                raise ValueError("Sampler must return one PUB result per Grover power.")
            counts, exposures = [], []
            for pub in raw:
                samples = pub.data.meas
                if (samples.shape != () or samples.num_bits != len(estimation_problem.objective_qubits)
                        or isinstance(samples.num_shots, bool) or not isinstance(samples.num_shots, Integral) or samples.num_shots < 1):
                    raise ValueError("Scalar objective BitArray output is required for each PUB.")
                row = samples.get_counts()
                if not isinstance(row, Mapping) or not row or any(isinstance(value, bool) or not isinstance(value, Integral) or value < 0 for value in row.values()):
                    raise ValueError("Sampler counts must be nonnegative integers.")
                if any(type(label) is not str or len(label) != len(estimation_problem.objective_qubits)
                       or set(label)-{"0", "1"} for label in row):
                    raise ValueError("Sampler must return objective-width binary labels.")
                if sum(row.values()) != samples.num_shots:
                    raise ValueError("Sampler counts disagree with the shot total.")
                counts.append(row)
                exposures.append(int(samples.num_shots))
        except MemoryError:
            raise
        except Exception as error:
            raise AlgorithmError("Maximum-likelihood sampler job or result protocol failed.") from error
        good, totals = _success_counts(counts, estimation_problem)
        likelihood = BinomialLikelihood(self._schedule, good, totals)
        theta = self._fit(likelihood)
        amplitude, process = sin(theta)**2, estimation_problem.post_processing
        result = MaximumLikelihoodAmplitudeEstimationResult(circuit_results=counts, primitive_result=raw,
            estimation=amplitude, estimation_processed=process(amplitude), theta=theta,
            evaluation_schedule=self.evaluation_schedule, good_counts=good, total_counts=totals,
            shots=exposures[0] if len(set(exposures)) == 1 else None,
            num_oracle_queries=sum(power*total for power, total in zip(self._schedule, totals)),
            fisher_information=fisher_information(likelihood, theta), post_processing=process,
            minimizer=self._minimizer or self._default_minimizer)
        result.confidence_interval_method = "fisher" if 0 in self._schedule and 0 < amplitude < 1 else "likelihood_ratio"
        result.confidence_interval = confidence_interval(result, .05, result.confidence_interval_method)
        result.confidence_interval_processed = tuple(process(value) for value in result.confidence_interval)
        return result

    def _default_minimizer(self, function, bounds):
        from .amplitude_statistics import _maximize
        if len(bounds) != 1 or len(bounds[0]) != 2:
            raise ValueError("Default likelihood minimizer needs one scalar bound.")
        likelihood = getattr(function, "__self__", None)
        if isinstance(likelihood, BinomialLikelihood) and bounds == [[0., pi/2]]:
            return likelihood.maximum()
        lower, upper = bounds[0]
        boundaries = sorted({lower, upper, *(point for point in search_boundaries(self._schedule) if lower < point < upper)})
        maxima = [_maximize(lambda theta: -function(theta), start, end) for start, end in zip(boundaries, boundaries[1:])]
        return max(maxima, key=lambda item: item[1])[0]

    @staticmethod
    def compute_confidence_interval(result, alpha, kind="fisher", apply_post_processing=False):
        if type(apply_post_processing) is not bool:
            raise TypeError("apply_post_processing must be boolean.")
        raw = confidence_interval(result, alpha, kind)
        return tuple(result.post_processing(value) for value in raw) if apply_post_processing else raw
