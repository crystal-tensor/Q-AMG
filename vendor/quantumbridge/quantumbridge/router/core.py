# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit, PennyLane, QOS-UQCI, Quafu, or vendor SDKs was copied.
"""Core fidelity-aware routing primitives for QuantumBridge Router."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from quantumbridge.compat.qos_uqci.job_spec import build_qos_uqci_job_spec
from quantumbridge.compat.qiskit_aer.simulator_native import normalize_circuit_to_quantumbridge_ir
from quantumbridge.qasm.importer import loads as circuit_from_qasm


@dataclass
class RouterSLA:
    min_fidelity: float = 0.75
    max_wait: float = 600.0
    max_cost: float = 25.0
    shots: int = 1024
    priority: int = 1
    preferred_backend: str | None = None
    allow_mock: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BackendState:
    backend_id: str
    provider: str
    backend_type: str = "superconducting"
    num_qubits: int = 5
    basis_gates: list[str] = field(default_factory=lambda: ["h", "rx", "ry", "rz", "cx", "cz", "measure"])
    coupling_map: list[list[int]] = field(default_factory=list)
    queue_depth: int = 0
    avg_job_seconds: float = 30.0
    cost_per_shot: float = 0.0005
    cost_floor: float = 0.10
    utilization: float = 0.25
    calibration_age_seconds: float = 900.0
    single_qubit_error: float = 0.001
    two_qubit_error: float = 0.015
    readout_error: float = 0.02
    multiprogramming_slots: int = 1
    access_mode: str = "local_mock"
    real_adapter_ready: bool = False
    dry_run_only: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WorkloadFeatures:
    workload_id: str
    source_type: str
    num_qubits: int
    operation_count: int
    one_qubit_gates: int
    two_qubit_gates: int
    measurement_count: int
    estimated_depth: int
    qasm_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BackendEstimate:
    backend_id: str
    predicted_fidelity: float
    estimated_wait: float
    estimated_cost: float
    queue_score: float
    calibration_freshness: float
    utilization_score: float
    feasible: bool
    reason_codes: list[str] = field(default_factory=list)
    device_spec: dict[str, Any] = field(default_factory=dict)
    calset: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RouterDecision:
    selected_backend: str
    fallback_backend: str | None
    pred_fidelity: float
    estimated_wait: float
    estimated_cost: float
    queue_score: float
    calibration_freshness: float
    reason_codes: list[str]
    decision_trace: list[dict[str, Any]]
    workload_features: dict[str, Any]
    uqci_bundle: dict[str, Any]
    qos_metrics: dict[str, Any]
    backend_snapshot: list[dict[str, Any]]
    warnings: list[str]
    provenance: dict[str, Any]
    schema_version: str = "quantumbridge-router-v0.1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_router_backends() -> list[BackendState]:
    """Return deterministic local backends for the demo benchmark."""

    return [
        BackendState(
            backend_id="quafu_coherence_alpha",
            provider="quafu_dry_run",
            num_qubits=18,
            coupling_map=_line_coupling(18),
            queue_depth=18,
            avg_job_seconds=35,
            cost_per_shot=0.00042,
            utilization=0.72,
            calibration_age_seconds=420,
            single_qubit_error=0.0008,
            two_qubit_error=0.010,
            readout_error=0.018,
            multiprogramming_slots=3,
            access_mode="real_adapter_dry_run",
            real_adapter_ready=True,
        ),
        BackendState(
            backend_id="xianghan_superconducting_beta",
            provider="xianghan_dry_run",
            num_qubits=10,
            coupling_map=_grid_coupling(10),
            queue_depth=6,
            avg_job_seconds=26,
            cost_per_shot=0.00055,
            utilization=0.38,
            calibration_age_seconds=780,
            single_qubit_error=0.0010,
            two_qubit_error=0.014,
            readout_error=0.021,
            multiprogramming_slots=2,
            access_mode="real_adapter_dry_run",
            real_adapter_ready=True,
        ),
        BackendState(
            backend_id="open_mock_gamma",
            provider="qos_uqci_mock",
            num_qubits=5,
            coupling_map=_line_coupling(5),
            queue_depth=1,
            avg_job_seconds=18,
            cost_per_shot=0.00018,
            utilization=0.18,
            calibration_age_seconds=1500,
            single_qubit_error=0.0018,
            two_qubit_error=0.029,
            readout_error=0.035,
            multiprogramming_slots=1,
            access_mode="local_mock",
            real_adapter_ready=False,
        ),
        BackendState(
            backend_id="research_high_fidelity_delta",
            provider="lab_dry_run",
            num_qubits=6,
            coupling_map=_grid_coupling(6),
            queue_depth=31,
            avg_job_seconds=45,
            cost_per_shot=0.0009,
            utilization=0.86,
            calibration_age_seconds=240,
            single_qubit_error=0.0005,
            two_qubit_error=0.007,
            readout_error=0.012,
            multiprogramming_slots=1,
            access_mode="real_adapter_dry_run",
            real_adapter_ready=True,
        ),
    ]


def route_workload(
    qasm_or_circuit: Any,
    sla: RouterSLA | Mapping[str, Any] | None = None,
    backends: list[BackendState] | None = None,
    workload_id: str = "router-demo-job",
) -> RouterDecision:
    """Route a QASM3-like string or QuantumBridge circuit to the best backend."""

    normalized_sla = _normalize_sla(sla)
    circuit, source_type, qasm_sha = _normalize_workload(qasm_or_circuit)
    features = extract_workload_features(circuit, workload_id=workload_id, source_type=source_type, qasm_sha256=qasm_sha)
    candidates = [backend for backend in (backends or default_router_backends()) if normalized_sla.allow_mock or backend.access_mode != "local_mock"]
    estimates = [estimate_backend(features, backend, normalized_sla) for backend in candidates]
    if not estimates:
        raise ValueError("no backend candidates available for RouterSLA")
    selected = _select_estimate(estimates, normalized_sla)
    fallback = _fallback_estimate(estimates, selected.backend_id)
    uqci_bundle = build_qos_uqci_job_spec(circuit, shots=normalized_sla.shots, backend_name=selected.backend_id)
    trace = _decision_trace(estimates, selected.backend_id, normalized_sla)
    return RouterDecision(
        selected_backend=selected.backend_id,
        fallback_backend=None if fallback is None else fallback.backend_id,
        pred_fidelity=selected.predicted_fidelity,
        estimated_wait=selected.estimated_wait,
        estimated_cost=selected.estimated_cost,
        queue_score=selected.queue_score,
        calibration_freshness=selected.calibration_freshness,
        reason_codes=list(selected.reason_codes),
        decision_trace=trace,
        workload_features=features.to_dict(),
        uqci_bundle=uqci_bundle,
        qos_metrics={
            "estimator": {
                "scatter": [
                    {
                        "backend_id": item.backend_id,
                        "x_queue_wait": item.estimated_wait,
                        "y_pred_fidelity": item.predicted_fidelity,
                        "calibration_freshness": item.calibration_freshness,
                    }
                    for item in estimates
                ]
            },
            "scheduler": {
                "pareto": _pareto_points(estimates),
                "objective": "maximize fidelity while respecting wait/cost SLA",
            },
            "multi_programmer": {
                "heatmap": _allocation_heatmap(features, selected.backend_id),
                "effective_slots": max(1, _backend_by_id(candidates, selected.backend_id).multiprogramming_slots),
            },
            "error_mitigator": {
                "mitigation_budget": "placeholder",
                "policy": "reserved for QOS error mitigation hook",
            },
        },
        backend_snapshot=[backend.to_dict() for backend in candidates],
        warnings=[
            "local Router MVP: no cloud access, token reads, or hardware submission",
            "real backend adapters are dry-run interfaces until explicit credentials are supplied outside the repo",
        ],
        provenance={
            "source": "quantumbridge.router",
            "native_implementation": True,
            "copied_upstream_source": False,
            "cloud_access": False,
            "token_access": False,
            "hardware_access": False,
            "official_endorsement": False,
        },
    )


def extract_workload_features(circuit: Any, workload_id: str, source_type: str = "circuit", qasm_sha256: str | None = None) -> WorkloadFeatures:
    normalized = normalize_circuit_to_quantumbridge_ir(circuit)
    one_qubit = 0
    two_qubit = 0
    for op in normalized.operations:
        width = len(op.controls) + len(op.targets)
        if width <= 1:
            one_qubit += 1
        else:
            two_qubit += 1
    depth = max(1, one_qubit + 2 * two_qubit)
    return WorkloadFeatures(
        workload_id=workload_id,
        source_type=source_type,
        num_qubits=normalized.num_qubits,
        operation_count=len(normalized.operations),
        one_qubit_gates=one_qubit,
        two_qubit_gates=two_qubit,
        measurement_count=len(normalized.measurements),
        estimated_depth=depth,
        qasm_sha256=qasm_sha256,
    )


def estimate_backend(features: WorkloadFeatures, backend: BackendState, sla: RouterSLA) -> BackendEstimate:
    reason_codes: list[str] = []
    if features.num_qubits > backend.num_qubits:
        reason_codes.append("insufficient_qubits")
    if backend.access_mode == "local_mock":
        reason_codes.append("mock_backend_allowed")
    if backend.real_adapter_ready and backend.dry_run_only:
        reason_codes.append("real_adapter_dry_run")
    freshness = max(0.35, math.exp(-backend.calibration_age_seconds / 7200.0))
    topology_penalty = 1.0 if _has_enough_connectivity(backend, features) else 0.93
    raw_error = (
        features.one_qubit_gates * backend.single_qubit_error
        + features.two_qubit_gates * backend.two_qubit_error
        + max(1, features.measurement_count) * backend.readout_error
    )
    load_penalty = 1.0 - min(0.08, backend.utilization * 0.05)
    fidelity = math.exp(-raw_error) * freshness * topology_penalty * load_penalty
    if features.num_qubits > backend.num_qubits:
        fidelity = 0.0
    runtime = backend.avg_job_seconds * (0.30 + features.estimated_depth / 40.0)
    effective_slots = max(1, backend.multiprogramming_slots)
    wait = backend.queue_depth * backend.avg_job_seconds / effective_slots + runtime
    cost = max(backend.cost_floor, backend.cost_per_shot * sla.shots * (1.0 + 0.04 * features.two_qubit_gates))
    feasible = fidelity >= sla.min_fidelity and wait <= sla.max_wait and cost <= sla.max_cost and features.num_qubits <= backend.num_qubits
    if fidelity < sla.min_fidelity:
        reason_codes.append("below_min_fidelity")
    if wait > sla.max_wait:
        reason_codes.append("above_max_wait")
    if cost > sla.max_cost:
        reason_codes.append("above_max_cost")
    if feasible:
        reason_codes.append("sla_feasible")
    return BackendEstimate(
        backend_id=backend.backend_id,
        predicted_fidelity=round(float(fidelity), 6),
        estimated_wait=round(float(wait), 3),
        estimated_cost=round(float(cost), 6),
        queue_score=round(1.0 / (1.0 + wait), 6),
        calibration_freshness=round(float(freshness), 6),
        utilization_score=round(1.0 - backend.utilization, 6),
        feasible=feasible,
        reason_codes=reason_codes,
        device_spec=_device_spec_from_backend(backend),
        calset=_calset_from_backend(backend),
    )


def _normalize_sla(sla: RouterSLA | Mapping[str, Any] | None) -> RouterSLA:
    if sla is None:
        return RouterSLA()
    if isinstance(sla, RouterSLA):
        return sla
    data = dict(sla)
    return RouterSLA(
        min_fidelity=float(data.get("min_fidelity", RouterSLA.min_fidelity)),
        max_wait=float(data.get("max_wait", RouterSLA.max_wait)),
        max_cost=float(data.get("max_cost", RouterSLA.max_cost)),
        shots=int(data.get("shots", RouterSLA.shots)),
        priority=int(data.get("priority", RouterSLA.priority)),
        preferred_backend=data.get("preferred_backend"),
        allow_mock=bool(data.get("allow_mock", RouterSLA.allow_mock)),
    )


def _normalize_workload(qasm_or_circuit: Any) -> tuple[Any, str, str | None]:
    if isinstance(qasm_or_circuit, str):
        text = qasm_or_circuit.strip()
        return circuit_from_qasm(text), "qasm", hashlib.sha256(text.encode("utf-8")).hexdigest()
    return normalize_circuit_to_quantumbridge_ir(qasm_or_circuit), "circuit", None


def _select_estimate(estimates: list[BackendEstimate], sla: RouterSLA) -> BackendEstimate:
    feasible = [item for item in estimates if item.feasible]
    pool = feasible or estimates
    if sla.preferred_backend:
        preferred = [item for item in pool if item.backend_id == sla.preferred_backend]
        if preferred:
            return preferred[0]

    def score(item: BackendEstimate) -> float:
        fidelity_term = item.predicted_fidelity * 0.58
        wait_term = min(1.0, sla.max_wait / max(item.estimated_wait, 1.0)) * 0.30
        cost_term = min(1.0, sla.max_cost / max(item.estimated_cost, 0.001)) * 0.08
        freshness_term = item.calibration_freshness * 0.04
        infeasible_penalty = 0.25 if not item.feasible else 0.0
        return fidelity_term + wait_term + cost_term + freshness_term - infeasible_penalty

    return max(pool, key=score)


def _fallback_estimate(estimates: list[BackendEstimate], selected_backend: str) -> BackendEstimate | None:
    remaining = [item for item in estimates if item.backend_id != selected_backend]
    if not remaining:
        return None
    return max(remaining, key=lambda item: (item.feasible, item.predicted_fidelity, -item.estimated_wait))


def _decision_trace(estimates: list[BackendEstimate], selected_backend: str, sla: RouterSLA) -> list[dict[str, Any]]:
    rows = []
    for item in sorted(estimates, key=lambda row: row.backend_id):
        rows.append(
            {
                "backend_id": item.backend_id,
                "selected": item.backend_id == selected_backend,
                "pred_fidelity": item.predicted_fidelity,
                "estimated_wait": item.estimated_wait,
                "estimated_cost": item.estimated_cost,
                "feasible": item.feasible,
                "sla": sla.to_dict(),
                "reason_codes": list(item.reason_codes),
            }
        )
    return rows


def _pareto_points(estimates: list[BackendEstimate]) -> list[dict[str, Any]]:
    ordered = sorted(estimates, key=lambda item: (item.estimated_wait, -item.predicted_fidelity))
    points = []
    best_fidelity = -1.0
    for item in ordered:
        if item.predicted_fidelity > best_fidelity:
            best_fidelity = item.predicted_fidelity
            points.append(
                {
                    "backend_id": item.backend_id,
                    "wait": item.estimated_wait,
                    "fidelity": item.predicted_fidelity,
                    "cost": item.estimated_cost,
                }
            )
    return points


def _allocation_heatmap(features: WorkloadFeatures, backend_id: str) -> list[dict[str, Any]]:
    cells = []
    for index in range(max(features.num_qubits, 1)):
        cells.append(
            {
                "backend_id": backend_id,
                "qubit": index,
                "owner": features.workload_id,
                "role": "active" if index < features.num_qubits else "idle",
                "compatibility_score": round(max(0.1, 1.0 - 0.04 * features.two_qubit_gates - 0.01 * index), 3),
            }
        )
    return cells


def _device_spec_from_backend(backend: BackendState) -> dict[str, Any]:
    return {
        "schema_version": "quantumbridge-router-devicespec-v0.1",
        "name": backend.backend_id,
        "provider": backend.provider,
        "backend_type": backend.backend_type,
        "num_qubits": backend.num_qubits,
        "basis_gates": list(backend.basis_gates),
        "coupling_map": [list(edge) for edge in backend.coupling_map],
        "access_mode": backend.access_mode,
        "cloud_access": False,
        "token_read": False,
        "hardware_access": False,
    }


def _calset_from_backend(backend: BackendState) -> dict[str, Any]:
    return {
        "schema_version": "quantumbridge-router-calset-v0.1",
        "backend_id": backend.backend_id,
        "generated_from_hardware": False,
        "dry_run_snapshot": True,
        "calibration_age_seconds": backend.calibration_age_seconds,
        "mean_single_qubit_error": backend.single_qubit_error,
        "mean_two_qubit_error": backend.two_qubit_error,
        "mean_readout_error": backend.readout_error,
        "cloud_access": False,
        "token_read": False,
        "hardware_access": False,
    }


def _has_enough_connectivity(backend: BackendState, features: WorkloadFeatures) -> bool:
    if features.two_qubit_gates == 0:
        return True
    return len(backend.coupling_map) >= max(1, features.num_qubits - 1)


def _backend_by_id(backends: list[BackendState], backend_id: str) -> BackendState:
    for backend in backends:
        if backend.backend_id == backend_id:
            return backend
    raise KeyError(backend_id)


def _line_coupling(num_qubits: int) -> list[list[int]]:
    return [[index, index + 1] for index in range(max(0, int(num_qubits) - 1))]


def _grid_coupling(num_qubits: int) -> list[list[int]]:
    edges = _line_coupling(num_qubits)
    for index in range(max(0, int(num_qubits) - 2)):
        if index % 2 == 0:
            edges.append([index, index + 2])
    return edges
