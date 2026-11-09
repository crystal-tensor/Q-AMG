# Copyright 2026 QuantumBridge Contributors. Apache-2.0.
"""Shared, offline application-study evidence with explicit measured estimates."""

from dataclasses import dataclass, field

import numpy as np

from quantumbridge.core import Circuit
from quantumbridge.devices import StatevectorDevice
from quantumbridge.finance.kernels import labels
from quantumbridge.qml.subroutines import trotter_product
from quantumbridge.serialization import serialize_circuit
from quantumbridge.utils.math import Hamiltonian, PauliString


@dataclass
class ApplicationCase:
    direction: str
    domain: str
    title: str
    inputs: dict
    intermediate: dict
    reference: dict
    experiments: dict[str, Circuit]
    estimands: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)
    optimization: bool = False


def diagonal_cost_hamiltonian(costs):
    values = np.asarray(costs, float)
    if values.shape != (16,) or not np.all(np.isfinite(values)):
        raise ValueError(
            "The bounded application cost table needs sixteen finite values."
        )
    terms = []
    for mask in range(16):
        coefficient = (
            sum(
                value * (-1 if (index & mask).bit_count() % 2 else 1)
                for index, value in enumerate(values)
            )
            / 16
        )
        if abs(coefficient) > 1e-14:
            terms.append(
                (
                    float(coefficient),
                    PauliString(
                        [(wire, "Z") for wire in range(4) if mask & (1 << (3 - wire))]
                    ),
                )
            )
    return Hamiltonian(terms or [(0.0, PauliString([]))])


def diagonal_qaoa_circuit(costs, gamma, beta):
    """One layer using the existing exact commuting-Pauli evolution template."""
    if not np.all(np.isfinite([gamma, beta])):
        raise ValueError("QAOA angles must be finite.")
    circuit = Circuit(4, name="application-diagonal-qaoa-p1")
    for wire in range(4):
        circuit.h(wire)
    trotter_product(circuit, diagonal_cost_hamiltonian(costs), -float(gamma), n=1)
    for wire in range(4):
        circuit.rx(2 * float(beta), wire)
    return circuit


def train_diagonal_qaoa(costs):
    values = np.asarray(costs, float)
    hamiltonian = diagonal_cost_hamiltonian(values)
    normalized = (values - values.min()) / max(float(np.ptp(values)), 1e-12)
    device, trace = StatevectorDevice(), []
    best = (float("inf"), 0.0, 0.0)

    def evaluate(gamma, beta):
        probabilities = device.probabilities(
            diagonal_qaoa_circuit(normalized, gamma, beta)
        )
        return sum(
            values[int(label, 2)] * probability
            for label, probability in probabilities.items()
        )

    for gamma in np.linspace(0, 4 * np.pi, 17):
        for beta in np.linspace(0, np.pi, 13, endpoint=False):
            value = evaluate(gamma, beta)
            if value < best[0]:
                best = float(value), float(gamma), float(beta)
                trace.append(
                    {
                        "improvement": len(trace),
                        "expected_penalized_cost": best[0],
                        "gamma": best[1],
                        "beta": best[2],
                    }
                )
    for step in (0.2, 0.1, 0.05, 0.025):
        for dg, db in ((step, 0), (-step, 0), (0, step), (0, -step)):
            gamma, beta = best[1] + dg, best[2] + db
            value = evaluate(gamma, beta)
            if value < best[0]:
                best = float(value), gamma, beta
                trace.append(
                    {
                        "improvement": len(trace),
                        "expected_penalized_cost": best[0],
                        "gamma": best[1],
                        "beta": best[2],
                    }
                )
    return (
        diagonal_qaoa_circuit(normalized, best[1], best[2]),
        trace,
        max(len(pauli.terms) for _, pauli in hamiltonian.terms),
    )


def optimization_case(
    direction, domain, title, inputs, candidates, *, intermediate=None
):
    if len(candidates) != 16 or [row["bitstring"] for row in candidates] != list(
        labels(4)
    ):
        raise ValueError(
            "Application optimization examples require the ordered four-bit truth table."
        )
    feasible = [row for row in candidates if row["feasible"]]
    if not feasible:
        raise ValueError("The model has no feasible candidate.")
    costs = np.array([row["penalized_cost"] for row in candidates], float)
    if not np.all(np.isfinite(costs)):
        raise ValueError("Candidate costs must be finite.")
    if not candidates[int(np.argmin(costs))]["feasible"]:
        raise ValueError("The chosen penalty does not separate infeasible minima.")
    best = min(row["objective"] for row in feasible)
    circuit, trace, degree = train_diagonal_qaoa(costs)
    return ApplicationCase(
        direction,
        domain,
        title,
        inputs,
        {
            "candidates": candidates,
            "optimizer_trace": trace,
            "cost_pauli_degree": degree,
            "encoding": "bounded exact diagonal truth-table; degree>2 is not a QUBO",
            **(intermediate or {}),
        },
        {
            "best_objective": best,
            "best_bitstrings": [
                row["bitstring"]
                for row in feasible
                if abs(row["objective"] - best) < 1e-12
            ],
            "feasible_count": len(feasible),
            "baseline": "complete classical enumeration of sixteen states",
        },
        {"qaoa": circuit},
        {"qaoa": costs.tolist()},
        optimization=True,
    )


def run_local_case(case, *, shots=1024, seed=20261023):
    """Sample native circuits, with no credential lookup or cloud client path."""
    if type(shots) is not int or not 2 <= shots <= 100000:
        raise ValueError("shots must be an integer from 2 to 100000.")
    if type(seed) is not int:
        raise ValueError("A reproducible integer seed is required.")
    device, observations, estimates = StatevectorDevice(), {}, {}
    for index, (name, source) in enumerate(case.experiments.items()):
        if not isinstance(source, Circuit) or not 1 <= source.num_qubits <= 6:
            raise ValueError(
                "Application experiments require native circuits with one to six wires."
            )
        counts = device.counts(source, shots=shots, seed=seed + index)
        if sum(counts.values()) != shots:
            raise ValueError("Native count total differs from the requested shots.")
        states = labels(source.num_qubits)
        probabilities = np.array([counts.get(label, 0) / shots for label in states])
        exact = device.probabilities(source)
        measured = source.copy()
        if not measured.measurements:
            measured.measure_all()
        qasm = serialize_circuit(measured, "qasm2", metadata_policy="external").data
        observations[name] = {
            "counts": counts,
            "probabilities": probabilities.tolist(),
            "ideal_probabilities": [exact.get(label, 0) for label in states],
            "shots": shots,
            "seed": seed + index,
            "qubits": source.num_qubits,
            "gate_count": len(source.operations),
            "openqasm2": qasm,
            "bit_order": "q0-first",
        }
        if name in case.estimands:
            values = np.asarray(case.estimands[name], float)
            if values.shape != probabilities.shape or not np.all(np.isfinite(values)):
                raise ValueError(
                    "Finite diagonal estimand must match its experiment width."
                )
            mean = float(values @ probabilities)
            variance = float((values - mean) ** 2 @ probabilities)
            estimates[name] = {
                "estimate": mean,
                "standard_error": float(np.sqrt(max(0, variance) / (shots - 1))),
                "ideal_reference": float(
                    values @ np.array(observations[name]["ideal_probabilities"])
                ),
            }
    summary = dict(case.outputs)
    if case.optimization:
        p = observations["qaoa"]["probabilities"]
        rows = case.intermediate["candidates"]
        observed = [row for i, row in enumerate(rows) if row["feasible"] and p[i] > 0]
        best = min(observed, key=lambda row: row["objective"]) if observed else None
        summary.update(
            best_observed_feasible=best,
            observed_objective_gap=None
            if best is None
            else best["objective"] - case.reference["best_objective"],
            feasible_probability=sum(
                p[i] for i, row in enumerate(rows) if row["feasible"]
            ),
            optimal_probability=sum(
                p[i]
                for i, row in enumerate(rows)
                if row["bitstring"] in case.reference["best_bitstrings"]
            ),
            decision_rule="best feasible actually sampled candidate; not the most frequent string",
        )
    if "pauli_energy_constant" in case.outputs:
        terms = case.outputs["pauli_energy_terms"]
        summary["sampled_energy_hartree"] = case.outputs["pauli_energy_constant"] + sum(
            coefficient * estimates[name]["estimate"]
            for name, coefficient in terms.items()
        )
        summary["energy_standard_error_hartree"] = float(
            np.sqrt(
                sum(
                    coefficient**2 * estimates[name]["standard_error"] ** 2
                    for name, coefficient in terms.items()
                )
            )
        )
    return {
        "execution_mode": "local_simulator" if observations else "local_classical",
        "hardware_access": False,
        "new_cloud_tasks": 0,
        "experiments": observations,
        "estimates": estimates,
        "summary": summary,
    }
