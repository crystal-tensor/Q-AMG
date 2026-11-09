# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
"""Qiskit Aer result schema adapter."""

from __future__ import annotations

from typing import Any

from quantumbridge.compat.qiskit_aer.aer_adapter import ADAPTER
from quantumbridge.schema.aer_results import AerResult

from .warnings import QISKIT_AER_WARNING, aer_warnings, native_provenance


def wrap_aer_result(obj: Any, metadata: dict[str, Any] | None = None) -> AerResult:
    """Wrap a raw Aer-like object or mapping in the Stage 9F schema."""

    metadata = dict(metadata or {})
    if isinstance(obj, AerResult):
        return obj
    if isinstance(obj, dict):
        counts = {str(key): int(value) for key, value in obj.get("counts", {}).items()}
        shots = sum(counts.values()) or metadata.get("shots")
        probabilities = (
            {key: value / sum(counts.values()) for key, value in counts.items()}
            if counts and sum(counts.values())
            else {str(key): float(value) for key, value in obj.get("probabilities", {}).items()}
        )
        return AerResult(
            workflow=str(metadata.get("workflow", "aer_result_wrapper")),
            backend=str(metadata.get("backend", obj.get("backend", "qiskit_aer"))),
            mode=str(metadata.get("mode", "upstream_passthrough")),
            capability_level=int(metadata.get("capability_level", 2)),
            upstream_package="qiskit-aer",
            upstream_version=ADAPTER.get_upstream_version(),
            production_ready=False,
            native_implementation=bool(metadata.get("native_implementation", False)),
            num_qubits=metadata.get("num_qubits", obj.get("num_qubits")),
            shots=None if shots is None else int(shots),
            final_statevector=[complex(value) for value in obj.get("final_statevector", ())],
            counts=counts,
            probabilities=probabilities,
            noise_model=obj.get("noise_model"),
            raw_type=str(metadata.get("raw_type", type(obj).__name__)),
            metadata={
                **metadata,
                "cloud_access": False,
                "token_read": False,
                "hardware_access": False,
                "production_simulator": False,
            },
            warnings=aer_warnings(QISKIT_AER_WARNING),
            provenance=native_provenance("aer_result_wrapper"),
        )
    return AerResult(
        workflow=str(metadata.get("workflow", "aer_result_wrapper")),
        backend=str(metadata.get("backend", "qiskit_aer")),
        mode=str(metadata.get("mode", "upstream_passthrough")),
        capability_level=int(metadata.get("capability_level", 2)),
        upstream_package="qiskit-aer",
        upstream_version=ADAPTER.get_upstream_version(),
        production_ready=False,
        native_implementation=False,
        raw_type=type(obj).__name__,
        metadata={
            **metadata,
            "object_repr": repr(obj)[:200],
            "cloud_access": False,
            "token_read": False,
            "hardware_access": False,
            "production_simulator": False,
        },
        warnings=aer_warnings(QISKIT_AER_WARNING),
        provenance=native_provenance("aer_result_wrapper"),
    )
