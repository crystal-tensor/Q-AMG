# This file is independently implemented for QuantumBridge SDK.
# No source code from Quafu, QuarkStudio, or pyquafu was copied.
"""Explicit, credentialed Quafu cloud execution through QuarkStudio.

Importing this module never reads credentials or touches the network. A token is
resolved only when ``QuarkStudioTransport`` is explicitly constructed.
"""

from __future__ import annotations

import hashlib
import os
import time
from numbers import Integral
from collections.abc import Mapping
from typing import Any, Callable, Protocol

from quantumbridge.core import Circuit
from quantumbridge.providers import JobError, JobStatus, JobTimeoutError
from quantumbridge.schema.backend_results import QuafuBackendResult

from .warnings import cloud_provenance, cloud_warnings


QUARK_SHOT_UNIT = 1024
TOKEN_ENVIRONMENT_VARIABLES = ("QUAFU_TOKEN", "QUAFU_API_TOKEN")


class QuafuCloudTransport(Protocol):
    """Minimal transport contract used by the cloud backend and test doubles."""

    transport_name: str

    def submit(
        self,
        openqasm: str,
        *,
        backend_name: str,
        shots: int,
        name: str,
        compile: bool,
    ) -> str: ...

    def retrieve(self, task_id: str) -> Mapping[str, Any]: ...


class QuarkStudioTransport:
    """Adapter for the optional ``quark.Task`` public API."""

    transport_name = "quarkstudio"

    def __init__(
        self,
        token: str | None = None,
        *,
        task_factory: Callable[[str], Any] | None = None,
    ) -> None:
        resolved_token = _resolve_token(token)
        if task_factory is None:
            try:
                from quark import Task as task_factory  # type: ignore[assignment]
            except ImportError as exc:
                raise ImportError(
                    "Quafu cloud execution requires the optional 'quarkcircuit' package; "
                    "install QuantumBridge with the 'quafu-cloud' extra"
                ) from exc
        self._task_manager = task_factory(resolved_token)
        resolved_token = ""  # Do not retain the credential in this adapter.

    def submit(
        self,
        openqasm: str,
        *,
        backend_name: str,
        shots: int,
        name: str,
        compile: bool,
    ) -> str:
        repeat = _shots_to_repeat(shots)
        task = {
            "chip": str(backend_name),
            "name": str(name),
            "circuit": str(openqasm),
            "shots": int(shots),
            "compile": bool(compile),
        }
        task_id = self._task_manager.run(task, repeat=repeat)
        if not task_id:
            raise JobError("QuarkStudio did not return a task identifier")
        return str(task_id)

    def retrieve(self, task_id: str) -> Mapping[str, Any]:
        payload = self._task_manager.result(str(task_id))
        if not isinstance(payload, Mapping):
            raise JobError("QuarkStudio returned a non-mapping task result")
        return payload

    def status(self, task_id: str | None = None) -> Any:
        return self._task_manager.status(0 if task_id is None else str(task_id))

    def cancel(self, task_id: str) -> Any:
        return self._task_manager.cancel(str(task_id))


class QuafuCloudJob:
    """Pollable QuantumBridge job backed by a remote Quafu task."""

    def __init__(
        self,
        *,
        task_id: str,
        transport: QuafuCloudTransport,
        backend_name: str,
        shots: int,
        name: str,
        compile: bool,
        openqasm_sha256: str,
    ) -> None:
        self._task_id = str(task_id)
        self._transport = transport
        self._backend_name = str(backend_name)
        self._shots = int(shots)
        self._name = str(name)
        self._compile = bool(compile)
        self._openqasm_sha256 = str(openqasm_sha256)
        self._status = JobStatus.QUEUED
        self._last_payload: Mapping[str, Any] | None = None
        self._error: BaseException | None = None

    def job_id(self) -> str:
        return self._task_id

    def status(self) -> JobStatus:
        return self._status

    def refresh(self) -> JobStatus:
        if self._status in {JobStatus.DONE, JobStatus.CANCELLED, JobStatus.ERROR}:
            return self._status
        try:
            payload = self._transport.retrieve(self._task_id)
            self._last_payload = payload
            self._status = _status_from_payload(payload)
        except Exception as exc:
            self._error = exc
            self._status = JobStatus.ERROR
        return self._status

    def result(self, timeout: float | None = 300.0, poll_interval: float = 2.0) -> QuafuBackendResult:
        if timeout is not None and timeout < 0:
            raise JobTimeoutError("Quafu cloud job timeout must be non-negative")
        if poll_interval < 0:
            raise ValueError("Quafu cloud poll_interval must be non-negative")
        deadline = None if timeout is None else time.monotonic() + timeout
        while self._status not in {JobStatus.DONE, JobStatus.CANCELLED, JobStatus.ERROR}:
            self.refresh()
            if self._status == JobStatus.DONE:
                break
            if self._status in {JobStatus.CANCELLED, JobStatus.ERROR}:
                break
            if deadline is not None and time.monotonic() >= deadline:
                raise JobTimeoutError(f"Quafu cloud task {self._task_id} did not finish before timeout")
            if poll_interval:
                time.sleep(poll_interval)
        if self._status == JobStatus.CANCELLED:
            raise JobError(f"Quafu cloud task {self._task_id} was cancelled")
        if self._status == JobStatus.ERROR:
            detail = self._error or _error_detail(self._last_payload)
            raise JobError(f"Quafu cloud task {self._task_id} failed: {detail}")
        return self._build_result(self._last_payload or {})

    def done(self) -> bool:
        return self._status == JobStatus.DONE

    def running(self) -> bool:
        return self._status == JobStatus.RUNNING

    def cancel(self) -> bool:
        if self._status in {JobStatus.DONE, JobStatus.CANCELLED, JobStatus.ERROR}:
            return False
        cancel = getattr(self._transport, "cancel", None)
        if not callable(cancel):
            return False
        response = cancel(self._task_id)
        if isinstance(response, Mapping) and response.get("error"):
            raise JobError(f"Quafu cloud task {self._task_id} could not be cancelled: {response['error']}")
        self._status = JobStatus.CANCELLED
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self._task_id,
            "backend_name": self._backend_name,
            "status": self._status.value,
            "shots": self._shots,
            "name": self._name,
            "compile": self._compile,
            "openqasm_sha256": self._openqasm_sha256,
            "transport": self._transport.transport_name,
        }

    def _build_result(self, payload: Mapping[str, Any]) -> QuafuBackendResult:
        counts = _extract_counts(payload)
        if not counts:
            raise JobError(f"Quafu cloud task {self._task_id} completed without counts")
        actual_shots = sum(counts.values())
        probabilities = {key: value / actual_shots for key, value in counts.items()}
        metadata = {
            "backend_name": self._backend_name,
            "task_id": self._task_id,
            "status": self._status.value,
            "requested_shots": self._shots,
            "actual_shots": actual_shots,
            "transport": self._transport.transport_name,
            "provider_fields": sorted(str(key) for key in payload),
            "cloud_access": True,
            "token_read": True,
            "hardware_access": True,
            "production_backend": False,
            "official_endorsement_claim": False,
        }
        provenance = cloud_provenance("quafu_cloud_backend", self._transport.transport_name)
        return QuafuBackendResult(
            workflow="quafu_cloud_backend",
            project="quafu",
            mode="cloud_hardware",
            capability_level=3,
            production_ready=False,
            native_implementation=False,
            job_spec={
                "schema_version": "0.2",
                "job_id": self._task_id,
                "target": "quafu",
                "backend_name": self._backend_name,
                "execution_mode": "cloud_hardware",
                "shots": self._shots,
                "compile": self._compile,
                "openqasm_sha256": self._openqasm_sha256,
            },
            openqasm_artifact={"format": "OPENQASM 2.0", "sha256": self._openqasm_sha256},
            counts=counts,
            probabilities=probabilities,
            shots=actual_shots,
            raw_type=type(payload).__name__,
            metadata=metadata,
            warnings=cloud_warnings(),
            provenance=provenance,
        )


class QuafuCloudBackend:
    """Submit QuantumBridge circuits to a named Quafu cloud backend."""

    def __init__(
        self,
        backend_name: str = "Baihua",
        *,
        token: str | None = None,
        transport: QuafuCloudTransport | None = None,
        read_only: bool = False,
    ) -> None:
        if type(read_only) is not bool:
            raise TypeError("read_only must be a bool")
        if read_only and transport is None:
            raise ValueError("read-only mode requires an explicit transport; it never constructs a credentialed client")
        if not str(backend_name).strip():
            raise ValueError("Quafu cloud backend_name must be non-empty")
        if token is not None and transport is not None:
            raise ValueError("Pass either token or transport, not both")
        self.name = str(backend_name)
        self._read_only = read_only
        self._transport = transport if transport is not None else QuarkStudioTransport(token=token)

    @property
    def read_only(self) -> bool:
        """Submission policy, not a guarantee about an injected transport's internals."""
        return self._read_only

    def status(self) -> Any:
        status = getattr(self._transport, "status", None)
        if not callable(status):
            raise NotImplementedError("The configured Quafu cloud transport does not expose backend status")
        return status()

    def submit(
        self,
        circuit: Circuit,
        *,
        shots: int = QUARK_SHOT_UNIT,
        name: str | None = None,
        compile: bool = True,
        auto_measure: bool = True,
    ) -> QuafuCloudJob:
        if self.read_only:
            raise JobError("read-only cloud backend refuses submission")
        shots = _validate_shots(shots)
        prepared = _prepare_circuit(circuit, auto_measure=auto_measure)
        from quantumbridge.serialization import serialize_circuit
        # The remote protocol carries source only; job name remains a separate field.
        artifact = serialize_circuit(prepared,"qasm2",metadata_policy="external")
        openqasm = artifact.data
        job_name = name or prepared.name or "quantumbridge-job"
        task_id = self._transport.submit(
            openqasm,
            backend_name=self.name,
            shots=shots,
            name=job_name,
            compile=compile,
        )
        return QuafuCloudJob(
            task_id=task_id,
            transport=self._transport,
            backend_name=self.name,
            shots=shots,
            name=job_name,
            compile=compile,
            openqasm_sha256=hashlib.sha256(openqasm.encode("utf-8")).hexdigest(),
        )

    def run(
        self,
        circuit: Circuit,
        *,
        shots: int = QUARK_SHOT_UNIT,
        name: str | None = None,
        compile: bool = True,
        auto_measure: bool = True,
        timeout: float | None = 300.0,
        poll_interval: float = 2.0,
    ) -> QuafuBackendResult:
        job = self.submit(circuit, shots=shots, name=name, compile=compile, auto_measure=auto_measure)
        return job.result(timeout=timeout, poll_interval=poll_interval)


def _resolve_token(token: str | None) -> str:
    if token is not None and str(token).strip():
        return str(token).strip()
    for variable in TOKEN_ENVIRONMENT_VARIABLES:
        value = os.environ.get(variable)
        if value and value.strip():
            return value.strip()
    names = " or ".join(TOKEN_ENVIRONMENT_VARIABLES)
    raise ValueError(f"Quafu cloud execution requires an explicit token or {names}")


def _validate_shots(shots: int) -> int:
    if isinstance(shots, bool) or not isinstance(shots, int) or shots <= 0:
        raise ValueError("Quafu cloud shots must be a positive integer")
    _shots_to_repeat(shots)
    return shots


def _shots_to_repeat(shots: int) -> int:
    if shots % QUARK_SHOT_UNIT:
        raise ValueError(f"QuarkStudio shots must be a multiple of {QUARK_SHOT_UNIT}")
    return shots // QUARK_SHOT_UNIT


def _prepare_circuit(circuit: Circuit, *, auto_measure: bool) -> Circuit:
    if not isinstance(circuit, Circuit):
        raise TypeError("QuafuCloudBackend currently accepts a QuantumBridge Circuit")
    if circuit.num_parameters:
        raise ValueError("Quafu cloud execution requires all circuit parameters to be bound")
    prepared = circuit.copy()
    if not prepared.measurements:
        if not auto_measure:
            raise ValueError("Quafu cloud execution requires measurements")
        prepared.measure_all()
    return prepared


def _extract_counts(payload: Mapping[str, Any]) -> dict[str, int]:
    candidates: list[Any] = [payload.get("count"), payload.get("counts")]
    data = payload.get("data")
    if isinstance(data, Mapping):
        candidates.extend((data.get("count"), data.get("counts")))
    for candidate in candidates:
        if isinstance(candidate, Mapping):
            if any(not isinstance(key, str) or isinstance(value, bool)
                   or not isinstance(value, Integral) or value < 0
                   for key, value in candidate.items()):
                raise JobError("Quafu counts must map string labels to nonnegative integers")
            counts = {key: int(value) for key, value in candidate.items()}
            if counts and sum(counts.values()) > 0:
                return counts
    return {}


def _status_from_payload(payload: Mapping[str, Any]) -> JobStatus:
    status_value = payload.get("status", payload.get("state", payload.get("task_status", "running")))
    normalized = str(status_value).strip().lower().replace("_", " ")
    if normalized in {"cancelled", "canceled"}:
        return JobStatus.CANCELLED
    if normalized in {"error", "failed", "failure", "invalid"} or payload.get("error"):
        return JobStatus.ERROR
    if _extract_counts(payload):
        return JobStatus.DONE
    if normalized in {"done", "finished", "completed", "success", "succeeded"}:
        return JobStatus.DONE
    if normalized in {"queued", "pending", "waiting", "submitted"}:
        return JobStatus.QUEUED
    return JobStatus.RUNNING


def _error_detail(payload: Mapping[str, Any] | None) -> object:
    if not payload:
        return "unknown remote error"
    return payload.get("error", payload.get("message", payload.get("status", "unknown remote error")))
