# This file is independently implemented for QuantumBridge SDK.
"""Demo orchestration for QuantumBridge Router."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .benchmark import run_router_benchmark
from .core import RouterSLA, route_workload
from .quafu_tasks import quafu_tasks_to_backend_states, summarize_quafu_tasks_response
from .reports import write_router_demo_artifacts
from .workloads import bell_circuit, qaoa_small_circuit


def run_router_demo(output_dir: str | Path | None = None, job_count: int = 100, seed: int = 7) -> dict[str, Any]:
    sla = RouterSLA(min_fidelity=0.75, max_wait=900, max_cost=5.0, shots=1024, allow_mock=True)
    decision = route_workload(qaoa_small_circuit(), sla=sla, workload_id="demo-qaoa-small").to_dict()
    benchmark = run_router_benchmark(job_count=job_count, seed=seed, sla=sla).to_dict()
    quafu_tasks = _quafu_demo_tasks()
    quafu_evidence = summarize_quafu_tasks_response(quafu_tasks).to_dict()
    quafu_sla = RouterSLA(min_fidelity=0.50, max_wait=1200, max_cost=5.0, shots=1024, allow_mock=False)
    quafu_decision = route_workload(
        bell_circuit(),
        sla=quafu_sla,
        backends=quafu_tasks_to_backend_states(quafu_tasks),
        workload_id="quafu-task-list-demo",
    ).to_dict()
    payload = {
        "decision": decision,
        "benchmark": benchmark,
        "quafu_evidence": quafu_evidence,
        "quafu_router_decision": quafu_decision,
    }
    if output_dir is not None:
        payload["written_files"] = [str(path) for path in write_router_demo_artifacts(payload, output_dir)]
    return payload


def _quafu_demo_tasks() -> dict[str, Any]:
    return {
        "total": 8,
        "items": [
            {"taskId": "sample-a", "taskName": "bell", "chipName": "Baihua", "status": "Pending", "submitTime": "2026-06-13-10-00-00"},
            {"taskId": "sample-b", "taskName": "qaoa", "chipName": "Baihua", "status": "Submitted", "submitTime": "2026-06-13-10-02-00"},
            {"taskId": "sample-c", "taskName": "ghz", "chipName": "Baihua", "status": "Transpiled", "submitTime": "2026-06-13-10-04-00"},
            {"taskId": "sample-d", "taskName": "vqe", "chipName": "Dongling", "status": "Finished", "submitTime": "2026-06-13-09-40-00"},
            {"taskId": "sample-e", "taskName": "bell", "chipName": "Yunmeng", "status": "Pending", "submitTime": "2026-06-13-10-05-00"},
            {"taskId": "sample-f", "taskName": "noise-check", "chipName": "Yunmeng", "status": "Failed", "submitTime": "2026-06-13-09-20-00"},
            {"taskId": "sample-g", "taskName": "bell", "chipName": "Miaofeng", "status": "Canceled", "submitTime": "2026-06-13-08-50-00"},
            {"taskId": "sample-h", "taskName": "ghz", "chipName": "Hongluo", "status": "Finished", "submitTime": "2026-06-13-08-40-00"},
        ],
    }
