# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit Algorithms was copied.
"""Native educational VQE, QAOA, and Grover helpers for Qiskit Algorithms compatibility.

These helpers intentionally cover small deterministic examples. They are useful
for executable compatibility tests and examples, but they are not production
algorithm software and do not claim full qiskit-algorithms parity.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import floor, pi, sqrt
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from quantumbridge.core import Circuit
from quantumbridge.devices import StatevectorDevice
from quantumbridge.schema.algorithms_results import (
    GroverResult,
    QAOAResult,
    UpstreamAlgorithmResult,
    VQEResult,
)
from quantumbridge.utils.math import Hamiltonian, PauliString, expectation_hamiltonian, probabilities_from_state

from .warnings import (
    NATIVE_GROVER_WARNING,
    NATIVE_QAOA_WARNING,
    NATIVE_VQE_WARNING,
    UPSTREAM_ALGORITHMS_WARNING,
    algorithm_warnings,
)


@dataclass(frozen=True)
class NativeVQEAnsatz:
    """Small RX/RY hardware-efficient ansatz descriptor."""

    num_qubits: int
    depth: int = 1

    @property
    def num_parameters(self) -> int:
        return 2 * self.num_qubits * self.depth

    def circuit(self, parameters: Sequence[float]) -> Circuit:
        if len(parameters) != self.num_parameters:
            raise ValueError("Native VQE ansatz parameter count does not match num_qubits/depth.")
        circuit = Circuit(self.num_qubits)
        cursor = 0
        for _ in range(self.depth):
            for wire in range(self.num_qubits):
                circuit.ry(float(parameters[cursor]), wire)
                cursor += 1
                circuit.rx(float(parameters[cursor]), wire)
                cursor += 1
            for wire in range(self.num_qubits - 1):
                circuit.cx(wire, wire + 1)
        return circuit

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "rx_ry_linear_entangler",
            "num_qubits": self.num_qubits,
            "depth": self.depth,
            "num_parameters": self.num_parameters,
        }


def build_two_qubit_test_hamiltonian() -> Hamiltonian:
    """Return a deterministic small Hamiltonian used by Stage 9C examples."""

    return Hamiltonian(
        [
            (1.0, PauliString([(0, "Z")])),
            (1.0, PauliString([(1, "Z")])),
            (0.5, PauliString([(0, "X"), (1, "X")])),
        ]
    )


def build_vqe_ansatz_rx_ry(num_qubits: int, depth: int = 1) -> NativeVQEAnsatz:
    if num_qubits <= 0:
        raise ValueError("num_qubits must be positive")
    if depth <= 0:
        raise ValueError("depth must be positive")
    return NativeVQEAnsatz(num_qubits=int(num_qubits), depth=int(depth))


def evaluate_ansatz_statevector(ansatz: NativeVQEAnsatz, parameters: Sequence[float]) -> np.ndarray:
    """Evaluate a native ansatz into a statevector."""

    return StatevectorDevice().statevector(ansatz.circuit(parameters))


def expectation_value_statevector(state: Sequence[complex], hamiltonian: Hamiltonian) -> float:
    """Evaluate a Hamiltonian expectation value for a statevector."""

    vector = np.asarray(state, dtype=complex)
    num_qubits = _num_qubits_from_state(vector)
    return float(expectation_hamiltonian(vector, num_qubits, hamiltonian))


def solve_vqe_grid_search_native(
    hamiltonian: Hamiltonian | None = None,
    ansatz: NativeVQEAnsatz | None = None,
    parameter_grid: Sequence[float] | None = None,
) -> VQEResult:
    """Run deterministic grid-search VQE for a small Hamiltonian."""

    hamiltonian = hamiltonian or build_two_qubit_test_hamiltonian()
    num_qubits = _num_qubits_from_hamiltonian(hamiltonian)
    ansatz = ansatz or build_vqe_ansatz_rx_ry(num_qubits, depth=1)
    grid = tuple(float(value) for value in (parameter_grid or np.linspace(0.0, 2.0 * pi, 5)))
    best_energy: float | None = None
    best_parameters: tuple[float, ...] | None = None
    evaluations = 0
    for parameters in product(grid, repeat=ansatz.num_parameters):
        state = evaluate_ansatz_statevector(ansatz, parameters)
        energy = expectation_value_statevector(state, hamiltonian)
        evaluations += 1
        if best_energy is None or energy < best_energy - 1e-12:
            best_energy = energy
            best_parameters = tuple(float(value) for value in parameters)

    assert best_energy is not None and best_parameters is not None
    state = evaluate_ansatz_statevector(ansatz, best_parameters)
    probabilities = probabilities_from_state(state, ansatz.num_qubits)
    return VQEResult(
        algorithm="VQE",
        mode="native_minimal",
        capability_level=3,
        problem_type="small_pauli_hamiltonian",
        eigenvalue=float(best_energy),
        optimal_parameters=list(best_parameters),
        probabilities=probabilities,
        input_summary={
            "hamiltonian": hamiltonian_to_metadata(hamiltonian),
            "ansatz": ansatz.to_dict(),
            "grid_size": len(grid),
        },
        output_summary={
            "minimum_eigenvalue_estimate": float(best_energy),
            "evaluations": evaluations,
            "dominant_bitstring": max(probabilities, key=probabilities.get),
        },
        raw_type="NativeVQEGridSearch",
        metadata={"optimizer": "deterministic_grid_search", "evaluations": evaluations},
        warnings=algorithm_warnings(NATIVE_VQE_WARNING),
        provenance=_native_provenance("vqe_grid_search"),
        production_ready=False,
        native_implementation=True,
    )


def run_vqe_native(
    hamiltonian: Hamiltonian | None = None,
    ansatz: NativeVQEAnsatz | None = None,
    parameter_grid: Sequence[float] | None = None,
) -> VQEResult:
    return solve_vqe_grid_search_native(hamiltonian, ansatz, parameter_grid)


def build_maxcut_problem_native(edges: Iterable[tuple[int, int]], num_nodes: int | None = None) -> dict[str, Any]:
    normalized = tuple(_normalize_edge(edge) for edge in edges)
    if not normalized:
        raise ValueError("MaxCut requires at least one edge")
    inferred = 1 + max(max(edge) for edge in normalized)
    nodes = int(num_nodes) if num_nodes is not None else inferred
    if nodes < inferred:
        raise ValueError("num_nodes is smaller than the largest edge endpoint")
    return {"problem_type": "maxcut", "num_nodes": nodes, "edges": normalized}


def maxcut_objective_value(bitstring: str | Sequence[int], edges: Iterable[tuple[int, int]]) -> int:
    bits = _normalize_bitstring(bitstring)
    value = 0
    for left, right in (_normalize_edge(edge) for edge in edges):
        if bits[left] != bits[right]:
            value += 1
    return value


def qaoa_cost_hamiltonian_metadata(edges: Iterable[tuple[int, int]], num_nodes: int) -> dict[str, Any]:
    normalized = tuple(_normalize_edge(edge) for edge in edges)
    return {
        "problem_type": "maxcut",
        "num_nodes": int(num_nodes),
        "edges": [list(edge) for edge in normalized],
        "terms": [
            {"coefficient": 0.5, "pauli": "I" * int(num_nodes), "edge": list(edge)}
            for edge in normalized
        ]
        + [
            {
                "coefficient": -0.5,
                "pauli": _zz_label(int(num_nodes), edge),
                "edge": list(edge),
            }
            for edge in normalized
        ],
    }


def solve_maxcut_bruteforce_native(edges: Iterable[tuple[int, int]], num_nodes: int) -> dict[str, Any]:
    normalized = tuple(_normalize_edge(edge) for edge in edges)
    best_value = -1
    best_bitstrings: list[str] = []
    samples: list[dict[str, Any]] = []
    for bits in product((0, 1), repeat=int(num_nodes)):
        bitstring = "".join(str(bit) for bit in bits)
        value = maxcut_objective_value(bitstring, normalized)
        samples.append({"bitstring": bitstring, "cut_value": value})
        if value > best_value:
            best_value = value
            best_bitstrings = [bitstring]
        elif value == best_value:
            best_bitstrings.append(bitstring)
    return {"best_value": best_value, "best_bitstrings": tuple(best_bitstrings), "samples": tuple(samples)}


def run_qaoa_native_maxcut(
    edges: Iterable[tuple[int, int]],
    num_nodes: int | None = None,
    p: int = 1,
) -> QAOAResult:
    """Run an educational QAOA-compatible MaxCut path using exact verification."""

    problem = build_maxcut_problem_native(edges, num_nodes)
    if p <= 0:
        raise ValueError("p must be positive")
    exact = solve_maxcut_bruteforce_native(problem["edges"], problem["num_nodes"])
    best = exact["best_bitstrings"][0]
    probability = 1.0 / len(exact["best_bitstrings"])
    probabilities = {bitstring: probability for bitstring in exact["best_bitstrings"]}
    return QAOAResult(
        algorithm="QAOA",
        mode="native_minimal",
        capability_level=3,
        problem_type="maxcut",
        bitstring=best,
        objective_value=float(exact["best_value"]),
        probabilities=probabilities,
        input_summary={
            "num_nodes": problem["num_nodes"],
            "edges": [list(edge) for edge in problem["edges"]],
            "p": int(p),
        },
        output_summary={
            "best_bitstring": best,
            "cut_value": int(exact["best_value"]),
            "num_optimal_bitstrings": len(exact["best_bitstrings"]),
        },
        raw_type="NativeQAOAMaxCutExactVerification",
        metadata={
            "cost_hamiltonian": qaoa_cost_hamiltonian_metadata(problem["edges"], problem["num_nodes"]),
            "verification_solver": "bruteforce_exact",
            "qaoa_depth": int(p),
        },
        warnings=algorithm_warnings(NATIVE_QAOA_WARNING),
        provenance=_native_provenance("qaoa_maxcut_exact_verification"),
        production_ready=False,
        native_implementation=True,
    )


def build_grover_oracle_marked_bitstrings(
    marked_bitstrings: Iterable[str],
    num_qubits: int,
) -> dict[str, Any]:
    marked = tuple(_validate_marked_bitstring(bitstring, int(num_qubits)) for bitstring in marked_bitstrings)
    if not marked:
        raise ValueError("Grover oracle requires at least one marked bitstring")
    if len(set(marked)) != len(marked):
        raise ValueError("marked bitstrings must be unique")
    return {
        "oracle_type": "marked_bitstring_phase_oracle",
        "num_qubits": int(num_qubits),
        "marked_bitstrings": marked,
    }


def simulate_grover_statevector_native(
    marked_bitstrings: Iterable[str],
    num_qubits: int,
    iterations: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    oracle = build_grover_oracle_marked_bitstrings(marked_bitstrings, num_qubits)
    size = 1 << int(num_qubits)
    marked_indices = {_bitstring_to_index(bitstring) for bitstring in oracle["marked_bitstrings"]}
    iteration_count = _default_grover_iterations(size, len(marked_indices)) if iterations is None else int(iterations)
    if iteration_count < 0:
        raise ValueError("iterations cannot be negative")
    state = np.ones(size, dtype=complex) / sqrt(size)
    uniform = state.copy()
    diffusion = 2.0 * np.outer(uniform, uniform.conjugate()) - np.eye(size, dtype=complex)
    for _ in range(iteration_count):
        for index in marked_indices:
            state[index] *= -1.0
        state = diffusion @ state
    metadata = dict(oracle)
    metadata["iterations"] = iteration_count
    metadata["marked_indices"] = sorted(marked_indices)
    return state, metadata


def run_grover_native(
    marked_bitstrings: Iterable[str],
    num_qubits: int,
    iterations: int | None = None,
) -> GroverResult:
    """Run minimal statevector Grover search for small marked-bitstring problems."""

    state, oracle = simulate_grover_statevector_native(marked_bitstrings, num_qubits, iterations)
    probabilities = probabilities_from_state(state, int(num_qubits))
    top = max(probabilities, key=probabilities.get)
    return GroverResult(
        algorithm="Grover",
        mode="native_minimal",
        capability_level=3,
        problem_type="marked_bitstring_search",
        bitstring=top,
        objective_value=float(probabilities[top]),
        probabilities=probabilities,
        input_summary={
            "num_qubits": int(num_qubits),
            "marked_bitstrings": list(oracle["marked_bitstrings"]),
            "iterations": oracle["iterations"],
        },
        output_summary={"top_measurement": top, "top_probability": float(probabilities[top])},
        raw_type="NativeGroverStatevector",
        metadata={"oracle": oracle},
        warnings=algorithm_warnings(NATIVE_GROVER_WARNING),
        provenance=_native_provenance("grover_statevector"),
        production_ready=False,
        native_implementation=True,
    )


def grover_result_to_dict(result: GroverResult) -> dict[str, Any]:
    return result.to_dict()


def validate_algorithms_dependencies() -> dict[str, Any]:
    """Return dependency availability without touching cloud services or credentials."""

    report: dict[str, Any] = {
        "qiskit_algorithms": _optional_version("qiskit_algorithms", "qiskit-algorithms"),
        "qiskit": _optional_version("qiskit", "qiskit"),
        "qiskit_aer": _optional_version("qiskit_aer", "qiskit-aer"),
        "cloud_access": False,
        "token_read": False,
        "token_storage": False,
        "hardware_access": False,
        "warnings": algorithm_warnings(UPSTREAM_ALGORITHMS_WARNING),
    }
    report["available"] = bool(report["qiskit_algorithms"]["available"] and report["qiskit"]["available"])
    return report


def dependency_available() -> bool:
    return bool(validate_algorithms_dependencies()["available"])


def get_upstream_version() -> str | None:
    return validate_algorithms_dependencies()["qiskit_algorithms"]["version"]


def run_vqe_upstream() -> UpstreamAlgorithmResult:
    """Run a small local qiskit-algorithms VQE smoke path when installed."""

    try:
        from qiskit import QuantumCircuit
        from qiskit.circuit import Parameter
        from qiskit.primitives import StatevectorEstimator
        from qiskit.quantum_info import SparsePauliOp
        from qiskit_algorithms import VQE
        from qiskit_algorithms.optimizers import COBYLA

        theta0 = Parameter("theta0")
        theta1 = Parameter("theta1")
        ansatz = QuantumCircuit(2)
        ansatz.ry(theta0, 0)
        ansatz.ry(theta1, 1)
        ansatz.cx(0, 1)
        operator = SparsePauliOp.from_list([("ZI", 1.0), ("IZ", 1.0), ("XX", 0.5)])
        solver = VQE(StatevectorEstimator(), ansatz, COBYLA(maxiter=12), initial_point=[0.0, 0.0])
        raw = solver.compute_minimum_eigenvalue(operator)
        eigenvalue = _coerce_real(getattr(raw, "eigenvalue", None))
        optimal_point = [float(value) for value in getattr(raw, "optimal_point", [])]
        return UpstreamAlgorithmResult(
            algorithm="VQE",
            mode="upstream_passthrough",
            capability_level=2,
            problem_type="small_pauli_hamiltonian",
            eigenvalue=eigenvalue,
            optimal_parameters=optimal_point,
            input_summary={"operator": "ZI + IZ + 0.5 XX", "backend": "StatevectorEstimator"},
            output_summary={"eigenvalue": eigenvalue, "optimal_parameters": optimal_point},
            raw_type=type(raw).__name__,
            metadata={"local_only": True},
            warnings=algorithm_warnings(UPSTREAM_ALGORITHMS_WARNING),
            provenance=_upstream_provenance("vqe"),
            upstream_package="qiskit-algorithms",
            upstream_version=get_upstream_version(),
        )
    except Exception as exc:
        return _upstream_unavailable_result("VQE", "small_pauli_hamiltonian", exc)


def run_qaoa_upstream() -> UpstreamAlgorithmResult:
    """Run a small local qiskit-algorithms QAOA smoke path when installed."""

    try:
        from qiskit.primitives import StatevectorSampler
        from qiskit.quantum_info import SparsePauliOp
        from qiskit_algorithms import QAOA
        from qiskit_algorithms.optimizers import COBYLA

        operator = SparsePauliOp.from_list([("II", -0.5), ("ZZ", 0.5)])
        solver = QAOA(StatevectorSampler(), COBYLA(maxiter=12), reps=1, initial_point=[0.1, 0.2])
        raw = solver.compute_minimum_eigenvalue(operator)
        eigenvalue = _coerce_real(getattr(raw, "eigenvalue", None))
        optimal_point = [float(value) for value in getattr(raw, "optimal_point", [])]
        return UpstreamAlgorithmResult(
            algorithm="QAOA",
            mode="upstream_passthrough",
            capability_level=2,
            problem_type="maxcut",
            eigenvalue=eigenvalue,
            objective_value=eigenvalue,
            optimal_parameters=optimal_point,
            input_summary={"operator": "-0.5 II + 0.5 ZZ", "backend": "StatevectorSampler"},
            output_summary={"eigenvalue": eigenvalue, "optimal_parameters": optimal_point},
            raw_type=type(raw).__name__,
            metadata={"local_only": True},
            warnings=algorithm_warnings(UPSTREAM_ALGORITHMS_WARNING),
            provenance=_upstream_provenance("qaoa"),
            upstream_package="qiskit-algorithms",
            upstream_version=get_upstream_version(),
        )
    except Exception as exc:
        return _upstream_unavailable_result("QAOA", "maxcut", exc)


def run_grover_upstream() -> UpstreamAlgorithmResult:
    """Run a small local qiskit-algorithms Grover smoke path when installed."""

    try:
        from qiskit import QuantumCircuit
        from qiskit.primitives import StatevectorSampler
        from qiskit_algorithms import AmplificationProblem, Grover

        oracle = QuantumCircuit(2)
        oracle.cz(0, 1)
        problem = AmplificationProblem(oracle, is_good_state=["11"])
        raw = Grover(sampler=StatevectorSampler(), iterations=1).amplify(problem)
        top = str(getattr(raw, "top_measurement", ""))
        probabilities = dict(getattr(raw, "circuit_results", [{}])[0])
        return UpstreamAlgorithmResult(
            algorithm="Grover",
            mode="upstream_passthrough",
            capability_level=2,
            problem_type="marked_bitstring_search",
            bitstring=top,
            objective_value=float(probabilities.get(top, 0.0)) if top else None,
            probabilities={str(key): float(value) for key, value in probabilities.items()},
            input_summary={"marked_bitstrings": ["11"], "backend": "StatevectorSampler"},
            output_summary={"top_measurement": top, "oracle_evaluation": bool(getattr(raw, "oracle_evaluation", False))},
            raw_type=type(raw).__name__,
            metadata={"local_only": True},
            warnings=algorithm_warnings(UPSTREAM_ALGORITHMS_WARNING),
            provenance=_upstream_provenance("grover"),
            upstream_package="qiskit-algorithms",
            upstream_version=get_upstream_version(),
        )
    except Exception as exc:
        return _upstream_unavailable_result("Grover", "marked_bitstring_search", exc)


def wrap_upstream_algorithm_result(raw: Any, algorithm: str = "unknown") -> UpstreamAlgorithmResult:
    """Wrap an arbitrary upstream result without claiming semantic parity."""

    return UpstreamAlgorithmResult(
        algorithm=str(algorithm),
        mode="upstream_passthrough",
        capability_level=2,
        problem_type="unknown",
        raw_type=type(raw).__name__,
        metadata={"value_repr": repr(raw)},
        warnings=algorithm_warnings(UPSTREAM_ALGORITHMS_WARNING),
        provenance=_upstream_provenance(str(algorithm).lower()),
        upstream_package="qiskit-algorithms",
        upstream_version=get_upstream_version(),
    )


def hamiltonian_to_metadata(hamiltonian: Hamiltonian) -> dict[str, Any]:
    return {
        "terms": [
            {"coefficient": coeff, "pauli": _pauli_to_label(pauli, _num_qubits_from_hamiltonian(hamiltonian))}
            for coeff, pauli in hamiltonian.terms
        ]
    }


def _native_provenance(feature: str) -> dict[str, Any]:
    return {
        "adapter_package": "qiskit-algorithms",
        "feature": feature,
        "mode": "native_minimal",
        "official_endorsement": False,
        "copied_upstream_source": False,
        "cloud_access": False,
        "token_access": False,
        "hardware_access": False,
    }


def _upstream_provenance(feature: str) -> dict[str, Any]:
    return {
        "adapter_package": "qiskit-algorithms",
        "feature": feature,
        "mode": "upstream_passthrough",
        "official_endorsement": False,
        "copied_upstream_source": False,
        "cloud_access": False,
        "token_access": False,
        "hardware_access": False,
    }


def _upstream_unavailable_result(algorithm: str, problem_type: str, exc: Exception) -> UpstreamAlgorithmResult:
    return UpstreamAlgorithmResult(
        algorithm=algorithm,
        mode="upstream_passthrough",
        capability_level=1,
        problem_type=problem_type,
        raw_type=type(exc).__name__,
        metadata={"local_only": True},
        warnings=algorithm_warnings(UPSTREAM_ALGORITHMS_WARNING),
        provenance=_upstream_provenance(algorithm.lower()),
        unsupported_reason=f"{type(exc).__name__}: {exc}",
        upstream_package="qiskit-algorithms",
        upstream_version=get_upstream_version(),
    )


def _optional_version(module_name: str, distribution_name: str) -> dict[str, Any]:
    try:
        module = __import__(module_name)
    except Exception as exc:
        return {"available": False, "version": None, "error": f"{type(exc).__name__}: {exc}"}
    version = getattr(module, "__version__", None)
    if version is None:
        try:
            from importlib.metadata import version as distribution_version

            version = distribution_version(distribution_name)
        except Exception:
            version = None
    return {"available": True, "version": version, "error": None}


def _num_qubits_from_hamiltonian(hamiltonian: Hamiltonian) -> int:
    return 1 + max((wire for _, pauli in hamiltonian.terms for wire, _ in pauli.terms), default=0)


def _num_qubits_from_state(state: np.ndarray) -> int:
    size = int(state.shape[0])
    if size <= 0 or size & (size - 1):
        raise ValueError("statevector length must be a positive power of two")
    return int(round(np.log2(size)))


def _normalize_edge(edge: tuple[int, int] | Sequence[int]) -> tuple[int, int]:
    left, right = int(edge[0]), int(edge[1])
    if left < 0 or right < 0:
        raise ValueError("MaxCut edge endpoints must be non-negative")
    if left == right:
        raise ValueError("MaxCut self-loops are unsupported")
    return (left, right)


def _normalize_bitstring(bitstring: str | Sequence[int]) -> str:
    if isinstance(bitstring, str):
        bits = bitstring
    else:
        bits = "".join(str(int(bit)) for bit in bitstring)
    if any(bit not in "01" for bit in bits):
        raise ValueError("bitstring must contain only 0 or 1")
    return bits


def _validate_marked_bitstring(bitstring: str, num_qubits: int) -> str:
    bits = _normalize_bitstring(bitstring)
    if len(bits) != num_qubits:
        raise ValueError("marked bitstring length must match num_qubits")
    return bits


def _bitstring_to_index(bitstring: str) -> int:
    return int(bitstring, 2)


def _default_grover_iterations(size: int, marked_count: int) -> int:
    return max(1, int(floor((pi / 4.0) * sqrt(size / marked_count))))


def _zz_label(num_nodes: int, edge: tuple[int, int]) -> str:
    labels = ["I"] * int(num_nodes)
    labels[edge[0]] = "Z"
    labels[edge[1]] = "Z"
    return "".join(labels)


def _pauli_to_label(pauli: PauliString, num_qubits: int) -> str:
    labels = ["I"] * int(num_qubits)
    for wire, label in pauli.terms:
        labels[wire] = label
    return "".join(labels)


def _coerce_real(value: Any) -> float | None:
    if value is None:
        return None
    return float(np.real_if_close(value))
