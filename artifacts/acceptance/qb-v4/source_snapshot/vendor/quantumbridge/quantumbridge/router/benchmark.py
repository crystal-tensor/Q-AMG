# This file is independently implemented for QuantumBridge SDK.
"""Benchmark engine for QuantumBridge Router."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from typing import Any

from .core import BackendState, RouterSLA, default_router_backends, estimate_backend, extract_workload_features, route_workload
from .workloads import demo_workloads


@dataclass
class RouterBenchmarkReport:
    suite_id: str
    job_count: int
    router_policy: str
    baseline_summaries: dict[str, dict[str, Any]]
    router_summary: dict[str, Any]
    improvement: dict[str, Any]
    jobs: list[dict[str, Any]]
    dashboard: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "quantumbridge-router-benchmark-v0.1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_router_benchmark(job_count: int = 100, seed: int = 7, sla: RouterSLA | None = None) -> RouterBenchmarkReport:
    """Run a deterministic Router-vs-baselines workload simulation."""

    if int(job_count) <= 0:
        raise ValueError("job_count must be positive")
    rng = random.Random(seed)
    normalized_sla = sla or RouterSLA(min_fidelity=0.82, max_wait=900, max_cost=5.0, shots=1024)
    workloads = _expanded_workloads(int(job_count), rng)
    base_backends = default_router_backends()
    router_jobs, router_summary = _simulate_policy("quantumbridge_router", workloads, base_backends, normalized_sla, rng)
    baselines: dict[str, dict[str, Any]] = {}
    for policy in ("round_robin", "least_queue", "best_fidelity_only"):
        _, summary = _simulate_policy(policy, workloads, default_router_backends(), normalized_sla, random.Random(seed))
        baselines[policy] = summary
    improvement = _improvement(router_summary, baselines)
    return RouterBenchmarkReport(
        suite_id="router_100_job_demo",
        job_count=int(job_count),
        router_policy="fidelity-aware + wait-aware + cost-aware",
        baseline_summaries=baselines,
        router_summary=router_summary,
        improvement=improvement,
        jobs=router_jobs[: min(20, len(router_jobs))],
        dashboard=_dashboard_payload(router_jobs, router_summary, baselines),
        warnings=[
            "benchmark uses deterministic local metadata and dry-run backend states",
            "reported wait/fidelity/cost values are estimates for PoC demonstration, not production hardware claims",
            "no cloud access, token reads, or hardware submissions are performed",
        ],
        provenance={
            "source": "quantumbridge.router.benchmark",
            "native_implementation": True,
            "copied_upstream_source": False,
            "cloud_access": False,
            "token_access": False,
            "hardware_access": False,
        },
    )


def _simulate_policy(
    policy: str,
    workloads: list[Any],
    backends: list[BackendState],
    sla: RouterSLA,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    mutable = [BackendState(**backend.to_dict()) for backend in backends]
    for index, workload in enumerate(workloads):
        workload_id = f"job-{index:03d}-{workload.name or 'workload'}"
        if policy == "quantumbridge_router":
            decision = route_workload(workload, sla=sla, backends=mutable, workload_id=workload_id)
            selected_id = decision.selected_backend
            selected_estimate = next(item for item in decision.decision_trace if item["backend_id"] == selected_id)
            fidelity = float(selected_estimate["pred_fidelity"])
            wait = float(selected_estimate["estimated_wait"])
            cost = float(selected_estimate["estimated_cost"])
            reason_codes = list(selected_estimate["reason_codes"])
        else:
            features = extract_workload_features(workload, workload_id=workload_id)
            estimates = [estimate_backend(features, backend, sla) for backend in mutable]
            selected = _select_baseline(policy, estimates, index)
            selected_id = selected.backend_id
            fidelity = selected.predicted_fidelity
            wait = selected.estimated_wait
            cost = selected.estimated_cost
            reason_codes = list(selected.reason_codes) + [f"baseline_{policy}"]
        backend = _backend_by_id(mutable, selected_id)
        backend.queue_depth = max(0, backend.queue_depth + 1 - rng.choice([0, 1, 1, 2]))
        backend.utilization = min(0.98, max(0.05, backend.utilization + rng.uniform(-0.015, 0.025)))
        rows.append(
            {
                "job_id": workload_id,
                "workload": workload.name,
                "selected_backend": selected_id,
                "estimated_wait": round(wait, 3),
                "pred_fidelity": round(fidelity, 6),
                "estimated_cost": round(cost, 6),
                "reason_codes": reason_codes,
            }
        )
    return rows, _summary(policy, rows)


def _select_baseline(policy: str, estimates: list[Any], index: int):
    if policy == "round_robin":
        return sorted(estimates, key=lambda item: item.backend_id)[index % len(estimates)]
    if policy == "least_queue":
        return min(estimates, key=lambda item: item.estimated_wait)
    if policy == "best_fidelity_only":
        return max(estimates, key=lambda item: item.predicted_fidelity)
    raise ValueError(f"unknown benchmark policy: {policy}")


def _summary(policy: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = max(1, len(rows))
    waits = [float(row["estimated_wait"]) for row in rows]
    fidelities = [float(row["pred_fidelity"]) for row in rows]
    costs = [float(row["estimated_cost"]) for row in rows]
    distribution: dict[str, int] = {}
    for row in rows:
        backend_id = str(row["selected_backend"])
        distribution[backend_id] = distribution.get(backend_id, 0) + 1
    return {
        "policy": policy,
        "job_count": len(rows),
        "avg_wait": round(sum(waits) / total, 3),
        "p95_wait": round(sorted(waits)[int(0.95 * (total - 1))], 3),
        "avg_pred_fidelity": round(sum(fidelities) / total, 6),
        "min_pred_fidelity": round(min(fidelities), 6),
        "total_cost": round(sum(costs), 6),
        "backend_distribution": distribution,
    }


def _improvement(router_summary: dict[str, Any], baselines: dict[str, dict[str, Any]]) -> dict[str, Any]:
    round_robin_wait = float(baselines["round_robin"]["avg_wait"])
    router_wait = float(router_summary["avg_wait"])
    wait_reduction = (round_robin_wait - router_wait) / round_robin_wait if round_robin_wait else 0.0
    best_fidelity = float(baselines["best_fidelity_only"]["avg_pred_fidelity"])
    router_fidelity = float(router_summary["avg_pred_fidelity"])
    fidelity_delta = router_fidelity - best_fidelity
    return {
        "wait_reduction_vs_round_robin": round(wait_reduction, 4),
        "wait_reduction_percent": round(wait_reduction * 100.0, 2),
        "router_avg_wait": router_wait,
        "round_robin_avg_wait": round_robin_wait,
        "fidelity_delta_vs_best_fidelity_only": round(fidelity_delta, 6),
        "success_target_met": wait_reduction >= 0.20 and fidelity_delta >= -0.08,
    }


def _dashboard_payload(router_jobs: list[dict[str, Any]], router_summary: dict[str, Any], baselines: dict[str, Any]) -> dict[str, Any]:
    return {
        "summary_cards": [
            {"label": "Router avg wait", "value": router_summary["avg_wait"]},
            {"label": "Router avg fidelity", "value": router_summary["avg_pred_fidelity"]},
            {"label": "Jobs routed", "value": router_summary["job_count"]},
        ],
        "benchmark_comparison": [
            {"policy": "quantumbridge_router", **router_summary},
            *[{"policy": key, **value} for key, value in baselines.items()],
        ],
        "decision_table": router_jobs[:12],
    }


def _expanded_workloads(job_count: int, rng: random.Random) -> list[Any]:
    base = demo_workloads()
    workloads = []
    for _ in range(job_count):
        workloads.append(rng.choice(base).copy())
    return workloads


def _backend_by_id(backends: list[BackendState], backend_id: str) -> BackendState:
    for backend in backends:
        if backend.backend_id == backend_id:
            return backend
    raise KeyError(backend_id)
