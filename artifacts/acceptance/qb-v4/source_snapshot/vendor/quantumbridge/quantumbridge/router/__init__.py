# This file is independently implemented for QuantumBridge SDK.
"""QuantumBridge Router-first MVP.

The router package is a local, evidence-oriented layer for QOS-UQCI demos. It
does not access cloud services, read tokens, or submit hardware jobs.
"""

from .benchmark import run_router_benchmark
from .core import BackendState, RouterSLA, route_workload
from .demo import run_router_demo
from .quafu_tasks import quafu_tasks_to_backend_states, summarize_quafu_tasks_response
from .reports import write_router_demo_artifacts

__all__ = [
    "BackendState",
    "RouterSLA",
    "quafu_tasks_to_backend_states",
    "route_workload",
    "run_router_benchmark",
    "run_router_demo",
    "summarize_quafu_tasks_response",
    "write_router_demo_artifacts",
]
