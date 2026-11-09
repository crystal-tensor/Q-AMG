# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is part of QuantumBridge SDK.
# This implementation is developed for the QuantumBridge native architecture.

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional

from .exceptions import JobError, JobTimeoutError


class JobStatus(str, Enum):
    INITIALIZING = "INITIALIZING"
    QUEUED = "QUEUED"
    VALIDATING = "VALIDATING"
    RUNNING = "RUNNING"
    DONE = "DONE"
    CANCELLED = "CANCELLED"
    ERROR = "ERROR"


class _StatusAccessor(str):
    def __new__(cls, job: "Job"):
        return str.__new__(cls, job._status.value)

    def __init__(self, job: "Job") -> None:
        self._job = job

    def __call__(self) -> JobStatus:
        return self._job._status


class Job:
    def __init__(
        self,
        job_id: str,
        result: Any = None,
        backend: Any = None,
        executor: Optional[Callable[[], Any]] = None,
        status: JobStatus | str = JobStatus.INITIALIZING,
        error: Optional[BaseException] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        self._job_id = str(job_id)
        self._backend = backend
        self._result = result
        self._executor = executor
        self._status = JobStatus(status)
        self._error = error
        self._metadata = dict(metadata or {})
        self._created_at = datetime.now(timezone.utc)
        if self._executor is None:
            self._status = JobStatus.DONE if self._error is None else JobStatus.ERROR
        else:
            self.submit()

    @property
    def status(self) -> _StatusAccessor:
        return _StatusAccessor(self)

    def job_id(self) -> str:
        return self._job_id

    def backend(self):
        return self._backend

    def creation_date(self) -> datetime:
        return self._created_at

    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def submit(self) -> None:
        if self._status in {JobStatus.DONE, JobStatus.CANCELLED, JobStatus.ERROR}:
            return
        self._status = JobStatus.RUNNING
        try:
            self._result = self._executor() if self._executor is not None else self._result
        except BaseException as exc:
            self._error = exc
            self._status = JobStatus.ERROR
            return
        self._status = JobStatus.DONE

    def result(self, timeout: Optional[float] = None):
        if timeout is not None and timeout < 0:
            raise JobTimeoutError("QuantumBridge job timeout must be non-negative.")
        if self._status == JobStatus.CANCELLED:
            raise JobError("QuantumBridge job was cancelled.")
        if self._status == JobStatus.ERROR:
            raise JobError(f"QuantumBridge job failed: {self._error}")
        if self._status != JobStatus.DONE:
            self.submit()
        if self._status != JobStatus.DONE:
            raise JobTimeoutError("QuantumBridge job did not reach DONE state.")
        return self._result

    def cancel(self) -> bool:
        if self._status in {JobStatus.DONE, JobStatus.ERROR, JobStatus.CANCELLED}:
            return False
        self._status = JobStatus.CANCELLED
        return True

    def cancelled(self) -> bool:
        return self._status == JobStatus.CANCELLED

    def running(self) -> bool:
        return self._status == JobStatus.RUNNING

    def done(self) -> bool:
        return self._status == JobStatus.DONE

    def in_final_state(self) -> bool:
        return self._status in {JobStatus.DONE, JobStatus.CANCELLED, JobStatus.ERROR}

    def wait_for_final_state(self, timeout: Optional[float] = None, wait: float = 0.1, callback=None) -> JobStatus:
        if timeout is not None and timeout < 0:
            raise JobTimeoutError("QuantumBridge job timeout must be non-negative.")
        if not self.in_final_state():
            self.submit()
        if callback is not None:
            callback(self._job_id, self._status, self)
        return self._status

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self._job_id,
            "backend_name": None if self._backend is None else getattr(self._backend, "name", None),
            "status": self._status.value,
            "created_at": self._created_at.isoformat(),
            "metadata": self.metadata(),
        }
