# Copyright 2026 QuantumBridge Contributors. Licensed under Apache-2.0.
"""Explicit finance execution, shot budgets and restartable cloud task receipts."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from quantumbridge.compat.quafu import QuafuCloudBackend, QuarkStudioTransport
from quantumbridge.compat.quafu.cloud_backend import QuafuCloudJob
from quantumbridge.core import Circuit
from quantumbridge.devices import StatevectorDevice
from quantumbridge.serialization import serialize_circuit

from .kernels import labels


def _journal_digest(data):
    content = {key: value for key, value in data.items() if key != "integrity_sha256"}
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def probabilities_from_counts(counts, width, *, bit_order="q0-first"):
    """Strictly normalize provider counts; never silently invent missing outcomes."""
    if bit_order not in {"q0-first", "q0-last"}:
        raise ValueError("bit_order must be q0-first or q0-last")
    if not counts:
        raise ValueError("nonempty counts required")
    normalized = {}
    for key, value in counts.items():
        if not isinstance(key, str) or len(key) != width or set(key) - {"0", "1"}:
            raise ValueError("provider counts require fixed-width binary labels")
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError("counts must be nonnegative integers")
        normalized[key if bit_order == "q0-first" else key[::-1]] = int(value)
    total = sum(normalized.values())
    if total < 2:
        raise ValueError("at least two measured shots required")
    return normalized, [normalized.get(label, 0) / total for label in labels(width)], total


def circuit_artifact(circuit):
    measured = circuit.copy()
    if not measured.measurements:
        measured.measure_all()
    source = serialize_circuit(measured, "qasm2", metadata_policy="external").data
    return {"openqasm": source, "sha256": hashlib.sha256(source.encode()).hexdigest(),
            "num_qubits": circuit.num_qubits, "gates": len(circuit.operations),
            "two_qubit_gates": sum(len(op.controls + op.targets) == 2 for op in circuit.operations)}


def run_local(case, *, shots=1024, seed=20261023):
    if type(shots) is not int or shots < 2:
        raise ValueError("shots must be an integer >=2")
    device, results = StatevectorDevice(), {}
    for index, (name, circuit) in enumerate(case.experiments.items()):
        counts = device.counts(circuit, shots=shots, seed=seed + index)
        normalized, probabilities, total = probabilities_from_counts(counts, circuit.num_qubits)
        ideal = device.probabilities(circuit)
        results[name] = {
            "execution_mode": "local_simulator", "hardware_access": False,
            "counts": normalized, "probabilities": probabilities, "shots": total,
            "ideal_probabilities": [ideal.get(label, 0) for label in labels(circuit.num_qubits)],
            "seed": seed + index, "artifact": circuit_artifact(circuit),
            "bit_order": "q0-first",
        }
    return results


class FinanceCloudSession:
    """At most 20 explicit 1024-shot jobs, including one bit-order calibration.

    A persisted submitted task is retrieved on rerun, never automatically sent
    again. There is a small unavoidable crash window between remote acceptance
    and durable local storage because the provider has no idempotency key.
    Concurrent kernels are refused by an OS file lock. A timeout preserves the
    pending receipt. The lock requires the POSIX fcntl module on this platform.
    """

    def __init__(self, backend, *, journal, max_tasks=20, shots=1024, max_qubits=6):
        if type(max_tasks) is not int or not 1 <= max_tasks <= 20:
            raise ValueError("max_tasks must be 1..20")
        if type(shots) is not int or shots != 1024:
            raise ValueError("this approved batch permits exactly 1024 shots per task")
        if type(max_qubits) is not int or not 2 <= max_qubits <= 6:
            raise ValueError("max_qubits must be 2..6")
        self.backend = backend
        self.journal = Path(journal)
        self.max_tasks, self.shots, self.max_qubits = max_tasks, shots, max_qubits
        self.data = {"schema": "qb-finance-cloud-v1", "backend": backend.name,
                     "shots": shots, "jobs": {}, "bit_order": None}
        self._reload()

    def _reload(self):
        if self.journal.exists():
            self.data = json.loads(self.journal.read_text(encoding="utf-8"))
            seal = self.data.get("integrity_sha256", "")
            if not isinstance(seal, str) or not hmac.compare_digest(seal, _journal_digest(self.data)):
                raise ValueError("cloud journal integrity check failed; inspect existing tasks, do not resubmit")
            if (self.data.get("schema"), self.data.get("backend"), self.data.get("shots")) != (
                    "qb-finance-cloud-v1", self.backend.name, self.shots):
                raise ValueError("journal belongs to a different backend or shot contract")
            if len(self.data["jobs"]) > self.max_tasks:
                raise ValueError("existing journal exceeds this task budget")
            self._validate_journal()

    def _validate_journal(self):
        task_ids = set()
        for key, saved in self.data["jobs"].items():
            info, artifact = saved["job"], saved["artifact"]
            digest = hashlib.sha256(artifact["openqasm"].encode()).hexdigest()
            if digest != artifact["sha256"] or digest != info["openqasm_sha256"]:
                raise ValueError("cloud journal circuit hash mismatch")
            if (info["shots"] != self.shots or info["backend_name"] != self.backend.name
                    or info["compile"] is not True or info["name"] != f"qb-finance-{key}"):
                raise ValueError("cloud journal submission contract mismatch")
            if not isinstance(info["job_id"], str) or not info["job_id"] or info["job_id"] in task_ids:
                raise ValueError("duplicate or invalid cloud task ID")
            task_ids.add(info["job_id"])
            if "raw_counts" in saved:
                if info["status"] != "DONE":
                    raise ValueError("counts cannot certify an unfinished or failed task")
                _, _, total = probabilities_from_counts(saved["raw_counts"], artifact["num_qubits"])
                if total != self.shots:
                    raise ValueError("cloud journal shot count mismatch")
        order = self.data["bit_order"]
        if order is not None:
            calibration = self.data["jobs"].get("calibration/bit-order", {})
            counts = calibration.get("raw_counts", {})
            dominant = max(counts, key=counts.get) if counts else None
            if (order not in {"q0-first", "q0-last"} or dominant != ("10" if order == "q0-first" else "01")
                    or counts[dominant] / self.shots < 0.8):
                raise ValueError("cloud journal bit order disagrees with calibration evidence")

    @contextmanager
    def _journal_lock(self):
        import fcntl

        self.journal.parent.mkdir(parents=True, exist_ok=True)
        with self.journal.with_suffix(self.journal.suffix + ".lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("cloud ledger is already in use by another kernel") from exc
            try:
                self._reload()
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    @classmethod
    def from_environment(cls, backend_name="Dongling", **kwargs):
        """Only this explicit call reads credentials; no notebook embeds a token."""
        return cls(QuafuCloudBackend(backend_name), **kwargs)

    def _save(self):
        self.journal.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.journal.with_suffix(self.journal.suffix + ".tmp")
        self.data["integrity_sha256"] = _journal_digest(self.data)
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(self.data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(self.journal)

    def preflight(self, cases):
        needed = {f"{case.key}/{name}" for case in cases for name in case.experiments}
        needed.add("calibration/bit-order")
        for case in cases:
            for name, circuit in case.experiments.items():
                if circuit.num_qubits > self.max_qubits:
                    raise ValueError("finance circuit exceeds approved qubit limit")
                artifact = circuit_artifact(circuit)
                previous = self.data["jobs"].get(f"{case.key}/{name}")
                if previous and previous["artifact"]["sha256"] != artifact["sha256"]:
                    raise ValueError("changed circuit cannot reuse an old task receipt")
        if len(set(self.data["jobs"]) | needed) > self.max_tasks:
            raise ValueError("batch would exceed approved total task budget")
        return {"total_tasks_including_existing": len(set(self.data["jobs"]) | needed),
                "maximum_shots": self.max_tasks * self.shots, "max_qubits": self.max_qubits}

    def _raw_job(self, key, circuit, *, timeout, poll_interval):
        artifact = circuit_artifact(circuit)
        jobs = self.data["jobs"]
        if key in jobs:
            saved = jobs[key]
            if artifact["sha256"] != saved["artifact"]["sha256"]:
                raise ValueError("changed circuit cannot reuse an old task receipt")
            if "raw_counts" in saved:
                return saved
            info = saved["job"]
            job = QuafuCloudJob(task_id=info["job_id"], transport=self.backend._transport,
                               backend_name=self.backend.name, shots=self.shots, name=info["name"],
                               compile=True, openqasm_sha256=info["openqasm_sha256"])
        else:
            if (self.journal.parent / "diagnostics/diagnostic_task_ledger.json").exists():
                raise RuntimeError("original finance batch is frozen while diagnostic ledger exists")
            if circuit.num_qubits > self.max_qubits or len(jobs) >= self.max_tasks:
                raise ValueError("approved hardware budget exceeded")
            job = self.backend.submit(circuit, shots=self.shots, name=f"qb-finance-{key}")
            saved = jobs[key] = {"job": job.to_dict(), "artifact": artifact}
            self._save()
        result = job.result(timeout=timeout, poll_interval=poll_interval)
        saved.update(job=job.to_dict(), raw_counts=dict(result.counts))
        self._save()
        return saved

    def calibrate_bit_order(self, *, timeout=300, poll_interval=2):
        with self._journal_lock():
            return self._calibrate_bit_order_locked(timeout=timeout, poll_interval=poll_interval)

    def _calibrate_bit_order_locked(self, *, timeout, poll_interval):
        if self.data["bit_order"] is not None:
            return self.data["bit_order"]
        circuit = Circuit(2, 2, name="finance-bit-order").x(0)
        saved = self._raw_job("calibration/bit-order", circuit, timeout=timeout, poll_interval=poll_interval)
        counts, _, shots = probabilities_from_counts(saved["raw_counts"], 2)
        if shots != self.shots:
            raise ValueError("provider shots differ from requested shots")
        dominant = max(counts, key=counts.get)
        if dominant not in {"10", "01"} or counts[dominant] / shots < 0.8:
            raise ValueError("bit-order calibration inconclusive; do not decode finance jobs")
        self.data["bit_order"] = "q0-first" if dominant == "10" else "q0-last"
        self._save()
        return self.data["bit_order"]

    def run(self, case, *, timeout=300, poll_interval=2):
        with self._journal_lock():
            return self._run_locked(case, timeout=timeout, poll_interval=poll_interval)

    def _run_locked(self, case, *, timeout, poll_interval):
        self.preflight([case])
        order = self._calibrate_bit_order_locked(timeout=timeout, poll_interval=poll_interval)
        results = {}
        hardware_transport = isinstance(self.backend._transport, QuarkStudioTransport)
        for name, circuit in case.experiments.items():
            saved = self._raw_job(f"{case.key}/{name}", circuit, timeout=timeout, poll_interval=poll_interval)
            counts, probs, total = probabilities_from_counts(saved["raw_counts"], circuit.num_qubits, bit_order=order)
            if total != self.shots:
                raise ValueError("provider shots differ from requested shots; inspect task, do not resubmit")
            results[name] = {"execution_mode": "cloud_hardware" if hardware_transport else "injected_transport",
                             "hardware_access": hardware_transport,
                             "counts": counts, "raw_counts": saved["raw_counts"], "probabilities": probs,
                             "shots": total, "task_id": saved["job"]["job_id"], "backend": self.backend.name,
                             "provider_bit_order": order, "bit_order": "q0-first", "artifact": saved["artifact"]}
        return results
