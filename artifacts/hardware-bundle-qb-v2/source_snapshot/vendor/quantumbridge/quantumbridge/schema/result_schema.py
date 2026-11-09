# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
# Design source: docs/roadmap/p2_execution_plan_v0.1.md.

from __future__ import annotations

from typing import Any


RESULT_SCHEMA_VERSION = "0.2"


def make_result_payload(**kwargs: Any) -> dict[str, Any]:
    payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "job_id": None,
        "backend_name": None,
        "backend_version": None,
        "experiment_id": None,
        "shots": None,
        "counts": None,
        "probabilities": None,
        "statevector": None,
        "density_matrix": None,
        "expectation_values": None,
        "observable_metadata": {},
        "circuit_metadata": {},
        "noise_model_metadata": {},
        "seed": None,
        "timing": {},
        "errors": [],
        "warnings": [],
        "provenance": {},
        "metadata": {},
    }
    payload.update(kwargs)
    if payload.get("schema_version") is None:
        payload["schema_version"] = RESULT_SCHEMA_VERSION
    return payload
