from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import numpy as np
from quantumbridge import Circuit, ParameterVector
from quantumbridge.serialization import serialize_circuit
from scipy.optimize import minimize

from .backends import CircuitBackend, capabilities_dict
from .problem import Qubo, StructureProblem
from .schema import PlannerSettings


@dataclass
class Candidate:
    bits: str
    energy: float
    feasible: bool
    count: int
    first_job_id: str
    first_seen_s: float


@dataclass
class PlanResult:
    method: str
    execution_source: str
    candidates: list[Candidate]
    timings: dict[str, float | None]
    trace: list[dict] = field(default_factory=list)
    resources: dict = field(default_factory=dict)
    parameters: list[float] = field(default_factory=list)
    circuit_qasm3: str | None = None
    target_hit_s: float | None = None
    status: str = "completed"

    @property
    def best(self) -> Candidate | None:
        return next((c for c in self.candidates if c.feasible), None)

    def as_dict(self) -> dict:
        return asdict(self)


class _TargetFound(Exception):
    pass


def qaoa_circuit(qubo: Qubo, depth: int, mixer: str = "xy"):
    gamma, beta = ParameterVector("gamma", depth), ParameterVector("beta", depth)
    circuit = Circuit(qubo.n)
    if mixer == "xy":
        if any(len(g) != 2 for g in qubo.groups):
            raise ValueError("The current feasible-subspace initializer supports binary groups")
        for a, b in qubo.groups:
            # |01>+|10>: one excitation per catalogue choice, not a precomputed solution.
            circuit.x(a)
            circuit.h(b)
            circuit.cx(b, a)
    elif mixer == "x":
        for i in range(qubo.n):
            circuit.h(i)
    else:
        raise ValueError("Unsupported mixer")
    _, h, jz = qubo.ising()
    scale = max(1.0, float(np.max(np.abs(h))), max(map(abs, jz.values()), default=0))
    for layer in range(depth):
        for i, value in enumerate(h):
            if abs(value) > 1e-12:
                circuit.rz(2 * float(value) / scale * gamma[layer], i)
        for (i, j), value in jz.items():
            if abs(value) > 1e-12:
                circuit.rzz(2 * float(value) / scale * gamma[layer], i, j)
        if mixer == "xy":
            for a, b in qubo.groups:
                circuit.rxx(beta[layer], a, b)
                circuit.ryy(beta[layer], a, b)
        else:
            for i in range(qubo.n):
                circuit.rx(2 * beta[layer], i)
    circuit.measure_all()
    return circuit, gamma, beta, scale


class QuantumStructurePrior:
    """A measured, context-conditioned discrete structure prior (not a trained 3D model)."""

    def __init__(self, backend: CircuitBackend):
        self.backend = backend

    def plan(self, problem: StructureProblem, settings: PlannerSettings, seed: int,
             target: Callable[[str], bool] | None = None,
             initial_parameters: list[float] | None = None) -> PlanResult:
        start = time.perf_counter()
        caps = self.backend.capabilities()
        original_qubo = problem.encode()
        compact = settings.encoding == "compact_binary"
        qubo = original_qubo.compact_binary() if compact else original_qubo
        encoded = time.perf_counter()
        if qubo.n > min(settings.max_qubits, caps.max_qubits):
            from .decomposition import plan_decomposed
            return plan_decomposed(self.backend, problem, settings, seed, target)
        mixer = "x" if compact else settings.mixer
        circuit, gamma, beta, scale = qaoa_circuit(qubo, settings.depth, mixer)
        built = time.perf_counter()
        compiled = self.backend.compile(circuit, seed)
        compilation_s = time.perf_counter() - built
        rng = np.random.default_rng(seed)
        initial = np.array(initial_parameters if initial_parameters is not None else
                           list(rng.uniform(.2, 1.0, settings.depth)) +
                           list(rng.uniform(.1, .7, settings.depth)), dtype=float)
        if len(initial) != 2 * settings.depth or not np.isfinite(initial).all():
            raise ValueError("Wrong warm-start parameter shape")
        candidates: dict[str, Candidate] = {}
        trace: list[dict] = []
        backend_s = 0.0
        classical_scoring_s = 0.0
        total_shots = 0
        target_hit = None
        device_fields = ["upload_s", "queue_s", "quantum_execution_s", "readout_s",
                         "simulator_execution_s"]
        device_times: dict[str, float | None] = {key: 0.0 for key in device_fields}
        last_params, last_bound = initial.copy(), None
        best_mean, best_params = float("inf"), initial.copy()

        def evaluate(parameters, role="training"):
            nonlocal backend_s, total_shots, best_mean, best_params, last_bound, last_params
            nonlocal classical_scoring_s, target_hit
            if time.perf_counter() - start > settings.timeout_s:
                raise TimeoutError("Quantum planner exhausted its wall-clock budget")
            last_params = np.asarray(parameters).copy()
            binding = {**dict(zip(gamma, parameters[:settings.depth], strict=True)),
                       **dict(zip(beta, parameters[settings.depth:], strict=True))}
            bound = compiled.assign_parameters(binding)
            last_bound = bound
            shots = settings.training_shots if role == "training" else settings.shots
            before = time.perf_counter()
            batch = self.backend.sample(bound, shots, int(rng.integers(0, 2**31 - 1)))
            backend_s += time.perf_counter() - before
            observed_at = time.perf_counter() - start
            for key in device_fields:
                value = batch.timings.get(key)
                device_times[key] = (None if value is None or device_times[key] is None
                                     else device_times[key] + value)
            total_shots += shots
            scoring_start = time.perf_counter()
            mean, valid_shots, target_shots = 0.0, 0, 0
            decoded_counts = ({problem.bits(tuple(map(int, raw))): count
                               for raw, count in batch.counts.items()} if compact else batch.counts)
            for bits, count in decoded_counts.items():
                energy = original_qubo.energy(bits)
                feasible = problem.feasible(bits)
                valid_shots += count * feasible
                mean += count * energy / shots
                if bits not in candidates:
                    candidates[bits] = Candidate(bits, energy, feasible, count, batch.job_id,
                                                 observed_at)
                else:
                    candidates[bits].count += count
                if target and target(bits):
                    target_shots += count
            classical_scoring_s += time.perf_counter() - scoring_start
            trace.append({"role": role, "job_id": batch.job_id, "shots": shots,
                          "counts": decoded_counts, "raw_measurement_counts": batch.counts,
                          "structure_encoding": "onehot", "physical_encoding": settings.encoding,
                          "observed_at_s": observed_at,
                          "mean_energy": mean, "raw_feasible_rate": valid_shots / shots,
                          "target_shots": target_shots,
                          "timings": batch.timings, "metadata": batch.metadata,
                          "parameters": list(map(float, parameters))})
            if mean < best_mean:
                best_mean, best_params = mean, np.asarray(parameters).copy()
            if target_shots and target_hit is None:
                # Full batch latency + decoding is charged; no invented per-shot timestamps.
                target_hit = time.perf_counter() - start
                raise _TargetFound
            return mean / scale

        status = "completed"
        try:
            if settings.iterations:
                minimize(evaluate, initial, method="COBYLA",
                         options={"maxiter": max(settings.iterations, len(initial) + 2),
                                  "rhobeg": .5, "tol": 1e-3})
            evaluate(best_params, "inference")
        except _TargetFound:
            status = "target_reached"
        wall = time.perf_counter() - start
        artifact = serialize_circuit(last_bound, "qasm3", metadata_policy="external") if last_bound is not None else None
        resources = {"logical_qubits": qubo.n, "ancillas": 0, "groups": len(problem.groups),
                     "compiled_physical_qubits": compiled.num_qubits,
                     "depth": compiled.depth(), "gate_counts": dict(compiled.count_ops()),
                     "two_qubit_gates": sum(len(op.controls + op.targets) == 2 for op in compiled.operations),
                     "shots": total_shots, "backend_calls": len(trace), "mixer": mixer,
                     "encoding": settings.encoding, "original_qubo_variables": original_qubo.n,
                     "capabilities": capabilities_dict(self.backend),
                     "parameter_origin": "provided_warm_start" if initial_parameters else "seeded_initialization",
                     "quantum_sdk": "QuantumBridge",
                     "circuit_serialization_sidecar": artifact.sidecar if artifact else None,
                     "statevector_bytes_lower_bound": caps.complex_bytes_per_amplitude * 2**qubo.n}
        return PlanResult(
            method="qaoa_binary" if compact else f"qaoa_{settings.mixer}", execution_source=caps.execution_source,
            candidates=sorted(candidates.values(), key=lambda c: (not c.feasible, c.energy, -c.count)),
            timings={"encoding_s": encoded-start, "circuit_build_s": built-encoded,
                     "compilation_s": compilation_s, "backend_roundtrip_s": backend_s,
                     "classical_scoring_s": classical_scoring_s,
                     "optimizer_controller_s": max(0, wall-(built-start)-compilation_s-backend_s-classical_scoring_s),
                     "wall_s": wall, **device_times},
            trace=trace, resources=resources, parameters=list(map(float, last_params)),
            circuit_qasm3=artifact.data if artifact else None,
            target_hit_s=target_hit, status=status)


QuantumPlanner = QuantumStructurePrior
