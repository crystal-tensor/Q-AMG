# This file is independently implemented for QuantumBridge SDK.
"""Optional FastAPI adapter boundary for Studio local services."""

from __future__ import annotations

from importlib import util
from typing import Any


def fastapi_available() -> bool:
    return util.find_spec("fastapi") is not None


def create_fastapi_app_if_available() -> dict[str, Any] | Any:
    if not fastapi_available():
        return {
            "available": False,
            "unsupported_reason": "fastapi is not installed; Studio Stage 10C does not require it",
            "server_started": False,
            "cloud_access": False,
            "token_access": False,
            "hardware_access": False,
        }
    from fastapi import FastAPI

    app = FastAPI(title="QuantumBridge Studio Local API", version="0.1")
    app.state.quantumbridge_stage = "10C"
    app.state.server_started = False
    return app
