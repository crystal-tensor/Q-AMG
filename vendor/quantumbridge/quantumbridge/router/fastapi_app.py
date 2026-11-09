# This file is independently implemented for QuantumBridge SDK.
"""Optional FastAPI app for QuantumBridge Router."""

from __future__ import annotations

from importlib import util
from typing import Any

from .benchmark import run_router_benchmark
from .core import route_workload


def fastapi_available() -> bool:
    return util.find_spec("fastapi") is not None


def create_fastapi_app_if_available() -> Any:
    if not fastapi_available():
        return {
            "available": False,
            "unsupported_reason": "fastapi is not installed",
            "server_started": False,
            "cloud_access": False,
            "token_access": False,
            "hardware_access": False,
        }
    from fastapi import FastAPI

    app = FastAPI(title="QuantumBridge Router Local API", version="0.1")

    @app.post("/route")
    def route(payload: dict[str, Any]) -> dict[str, Any]:
        qasm = payload.get("qasm3") or payload.get("qasm")
        if not qasm:
            raise ValueError("POST /route requires qasm3 or qasm")
        return route_workload(qasm, sla=payload.get("sla")).to_dict()

    @app.post("/benchmarks/router")
    def benchmark(payload: dict[str, Any] | None = None) -> dict[str, Any]:
        body = payload or {}
        return run_router_benchmark(job_count=int(body.get("job_count", 100)), seed=int(body.get("seed", 7))).to_dict()

    return app
