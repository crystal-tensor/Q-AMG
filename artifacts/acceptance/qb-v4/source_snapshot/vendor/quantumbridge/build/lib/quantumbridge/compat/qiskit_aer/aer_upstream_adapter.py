# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit Aer was copied.
"""Optional upstream qiskit-aer passthrough helpers."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from quantumbridge.schema.aer_results import UpstreamAerResult

from .warnings import UPSTREAM_AER_WARNING, aer_warnings, upstream_provenance


def dependency_available() -> bool:
    try:
        import_module("qiskit_aer")
    except Exception:
        return False
    return True


def get_upstream_version() -> str | None:
    try:
        module = import_module("qiskit_aer")
    except Exception:
        return None
    return getattr(module, "__version__", None)


def validate_aer_dependencies() -> dict[str, Any]:
    return {
        "upstream_package": "qiskit-aer",
        "available": dependency_available(),
        "upstream_version": get_upstream_version(),
        "cloud_access": False,
        "token_read": False,
        "hardware_access": False,
        "warnings": aer_warnings(UPSTREAM_AER_WARNING),
    }


def run_statevector_upstream_aer(circuit: Any, **run_options: Any) -> UpstreamAerResult:
    if not dependency_available():
        return _unsupported(
            "upstream_aer_statevector",
            "optional upstream dependency unavailable: qiskit-aer",
        )
    try:
        AerSimulator = getattr(import_module("qiskit_aer"), "AerSimulator")
        working = _copy_qiskit_circuit(circuit)
        if hasattr(working, "save_statevector"):
            working.save_statevector()
        raw = AerSimulator(method="statevector").run(working, **run_options).result()
        state = raw.get_statevector() if hasattr(raw, "get_statevector") else None
        payload = {
            "statevector_length": None if state is None else len(state),
            "result_success": getattr(raw, "success", None),
        }
        return wrap_upstream_aer_result(raw, workflow="upstream_aer_statevector", data=payload)
    except Exception as exc:  # pragma: no cover - optional dependency shape
        return _unsupported("upstream_aer_statevector", f"{type(exc).__name__}: {exc}")


def run_qasm_upstream_aer(circuit: Any, shots: int = 1024, **run_options: Any) -> UpstreamAerResult:
    if not dependency_available():
        return _unsupported("upstream_aer_qasm", "optional upstream dependency unavailable: qiskit-aer")
    try:
        AerSimulator = getattr(import_module("qiskit_aer"), "AerSimulator")
        raw = AerSimulator().run(circuit, shots=int(shots), **run_options).result()
        counts = raw.get_counts() if hasattr(raw, "get_counts") else {}
        return wrap_upstream_aer_result(
            raw,
            workflow="upstream_aer_qasm",
            data={"counts": {str(key): int(value) for key, value in dict(counts).items()}},
            shots=int(shots),
        )
    except Exception as exc:  # pragma: no cover - optional dependency shape
        return _unsupported("upstream_aer_qasm", f"{type(exc).__name__}: {exc}")


def run_noisy_upstream_aer(
    circuit: Any,
    noise_model: Any = None,
    shots: int = 1024,
    **run_options: Any,
) -> UpstreamAerResult:
    if noise_model is None:
        return run_qasm_upstream_aer(circuit, shots=shots, **run_options)
    if not dependency_available():
        return _unsupported(
            "upstream_aer_noisy_qasm",
            "optional upstream dependency unavailable: qiskit-aer",
        )
    try:
        AerSimulator = getattr(import_module("qiskit_aer"), "AerSimulator")
        raw = AerSimulator(noise_model=noise_model).run(circuit, shots=int(shots), **run_options).result()
        counts = raw.get_counts() if hasattr(raw, "get_counts") else {}
        return wrap_upstream_aer_result(
            raw,
            workflow="upstream_aer_noisy_qasm",
            data={
                "counts": {str(key): int(value) for key, value in dict(counts).items()},
                "noise_model_type": type(noise_model).__name__,
            },
            shots=int(shots),
        )
    except Exception as exc:  # pragma: no cover - optional dependency shape
        return _unsupported("upstream_aer_noisy_qasm", f"{type(exc).__name__}: {exc}")


def wrap_upstream_aer_result(
    raw: Any,
    workflow: str = "upstream_aer_passthrough",
    data: dict[str, Any] | None = None,
    shots: int | None = None,
) -> UpstreamAerResult:
    data = dict(data or {})
    counts = dict(data.pop("counts", {}))
    total = sum(int(value) for value in counts.values())
    probabilities = {key: value / total for key, value in counts.items()} if total else {}
    return UpstreamAerResult(
        workflow=workflow,
        backend=str(data.get("backend", "qiskit_aer")),
        mode="upstream_passthrough",
        capability_level=1,
        upstream_package="qiskit-aer",
        upstream_version=get_upstream_version(),
        production_ready=False,
        native_implementation=False,
        shots=shots,
        counts={str(key): int(value) for key, value in counts.items()},
        probabilities=probabilities,
        raw_type=type(raw).__name__,
        metadata={
            **data,
            "cloud_access": False,
            "token_read": False,
            "hardware_access": False,
            "production_simulator": False,
        },
        warnings=aer_warnings(UPSTREAM_AER_WARNING),
        provenance=upstream_provenance(workflow),
    )


def _unsupported(workflow: str, reason: str) -> UpstreamAerResult:
    return UpstreamAerResult(
        workflow=workflow,
        backend="qiskit_aer",
        mode="upstream_passthrough",
        capability_level=1,
        upstream_package="qiskit-aer",
        upstream_version=get_upstream_version(),
        production_ready=False,
        native_implementation=False,
        raw_type="UnsupportedUpstreamAer",
        metadata={
            "cloud_access": False,
            "token_read": False,
            "hardware_access": False,
            "production_simulator": False,
        },
        warnings=aer_warnings(UPSTREAM_AER_WARNING),
        provenance=upstream_provenance(workflow),
        unsupported_reason=reason,
    )


def _copy_qiskit_circuit(circuit: Any) -> Any:
    if hasattr(circuit, "copy"):
        return circuit.copy()
    return circuit
