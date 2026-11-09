# Copyright 2026 QuantumBridge Contributors. Licensed under Apache-2.0.
"""Four bounded preparation/measurement probes; never mitigates finance counts."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

from quantumbridge.core import Circuit

from .execution import FinanceCloudSession, circuit_artifact, probabilities_from_counts
from .compiled_diagnostics import compiled_probabilities, qlisp_probabilities
from .statistics import wilson_interval


def diagnostic_circuits():
    circuits = {}
    for width in (3, 4):
        for state in (0, 1):
            key = f"diagnostic/{width}q-{'one' if state else 'zero'}"
            circuit = Circuit(width, width, name=key)
            if state:
                for wire in range(width):
                    circuit.x(wire)
            circuits[key] = circuit
    return circuits


class FinanceDiagnosticSession(FinanceCloudSession):
    """Separate sealed ledger, serialized with the original batch's OS lock.

    An intent is persisted before submission. An ambiguous submit failure stops
    future attempts until a human reconciles the provider task list; it is never
    automatically retried. The original ledger is read-only and fingerprinted.
    """

    def __init__(self, backend, *, original_journal, journal):
        self.original = FinanceCloudSession(backend, journal=original_journal)
        if not self.original.journal.exists() or not self.original.data["jobs"]:
            raise ValueError("an existing completed original batch is required")
        if any("raw_counts" not in item for item in self.original.data["jobs"].values()):
            raise ValueError("original tasks must be resolved before diagnostics")
        if Path(journal).resolve() != (self.original.journal.parent / "diagnostics/diagnostic_task_ledger.json").resolve():
            raise ValueError("diagnostics must use the shared companion ledger location")
        self.original_sha256 = hashlib.sha256(self.original.journal.read_bytes()).hexdigest()
        self.plan = {key: circuit_artifact(circuit) for key, circuit in diagnostic_circuits().items()}
        super().__init__(backend, journal=journal, max_tasks=4)
        self.data.setdefault("original_ledger_sha256", self.original_sha256)
        self.data.setdefault("purpose", "D1 zero/one preparation plus measurement; no mitigation")
        self._check_budget()

    def _validate_journal(self):
        super()._validate_journal()
        if self.data.get("original_ledger_sha256") != self.original_sha256:
            raise ValueError("diagnostic ledger original batch fingerprint mismatch")
        for key, saved in self.data["jobs"].items():
            if key not in self.plan or saved["artifact"] != self.plan[key]:
                raise ValueError("diagnostic circuit differs from frozen four-probe plan")
        self._check_budget()

    def _check_budget(self):
        if hashlib.sha256(self.original.journal.read_bytes()).hexdigest() != self.original_sha256:
            raise ValueError("original finance ledger changed; stop diagnostics")
        original_ids = {item["job"]["job_id"] for item in self.original.data["jobs"].values()}
        probe_ids = {item["job"]["job_id"] for item in self.data["jobs"].values()}
        if original_ids & probe_ids:
            raise ValueError("duplicate task ID across original and diagnostic ledgers")
        intent = int("pending_submission" in self.data)
        if len(original_ids) + len(probe_ids) + intent > 20 or len(probe_ids) + intent > 4:
            raise ValueError("combined original plus diagnostic task budget exceeded")

    @contextmanager
    def _journal_lock(self):
        with self.original._journal_lock():
            self._reload()
            self._check_budget()
            yield

    def run_one(self, key, *, timeout=300, poll_interval=2):
        if key not in self.plan:
            raise ValueError("only the four predeclared D1 probes are authorized")
        with self._journal_lock():
            if "pending_submission" in self.data:
                raise RuntimeError("unresolved submission intent: inspect remote tasks, never resubmit")
            if key not in self.data["jobs"]:
                if len(self.data["jobs"]) >= 4 or len(self.original.data["jobs"]) + len(self.data["jobs"]) >= 20:
                    raise ValueError("combined task budget exhausted")
                self.data["pending_submission"] = {"key": key, "artifact_sha256": self.plan[key]["sha256"]}
                self._save()
                job = self.backend.submit(diagnostic_circuits()[key], shots=1024, name=f"qb-finance-{key}")
                if job.job_id() in {item["job"]["job_id"] for item in self.original.data["jobs"].values()} | {
                        item["job"]["job_id"] for item in self.data["jobs"].values()}:
                    raise ValueError("provider returned duplicate task ID; reconcile intent before proceeding")
                self.data["jobs"][key] = {"artifact": self.plan[key], "job": job.to_dict()}
                del self.data["pending_submission"]
                self._save()
            saved = self._raw_job(key, diagnostic_circuits()[key], timeout=timeout, poll_interval=poll_interval)
            _, _, total = probabilities_from_counts(saved["raw_counts"], self.plan[key]["num_qubits"])
            if total != 1024:
                raise ValueError("diagnostic returned an unexpected shot count; stop")
            return json.loads(json.dumps(saved))


def zero_one_metrics(counts, width, prepared):
    if width not in (3, 4) or prepared not in (0, 1) or type(prepared) is not int:
        raise ValueError("3/4-bit zero/one probes required")
    normalized, _, shots = probabilities_from_counts(counts, width)
    if shots != 1024:
        raise ValueError("1024-shot diagnostic contract required")
    target = str(prepared) * width
    marginal = [sum(value for label, value in normalized.items() if label[i] != str(prepared)) for i in range(width)]
    pairs = []
    for i in range(width):
        for j in range(i + 1, width):
            both = sum(value for label, value in normalized.items()
                       if label[i] != str(prepared) and label[j] != str(prepared))
            pairs.append({"raw_label_positions": [i, j], "joint_error_probability": both / shots,
                          "joint_error_wilson_95": wilson_interval(both, shots),
                          "error_covariance": both / shots - marginal[i] * marginal[j] / shots**2})
    histogram = {str(k): sum(value for label, value in normalized.items()
                             if sum(bit != str(prepared) for bit in label) == k) for k in range(width + 1)}
    return {"prepared": target, "shots": shots, "success_probability": normalized.get(target, 0) / shots,
            "success_wilson_95": wilson_interval(normalized.get(target, 0), shots),
            "marginal_errors_by_raw_label_position": [
                {"position": i, "probability": count / shots, "wilson_95": wilson_interval(count, shots)}
                for i, count in enumerate(marginal)],
            "number_of_bit_errors_histogram": histogram, "pair_errors": pairs,
            "multiple_bit_error_probability": sum(int(value) for key, value in histogram.items() if int(key) >= 2) / shots,
            "mapping_status": "AMBIGUOUS", "mitigation_applied": False,
            "limitations": "zero/one invariant under all bit permutations; preparation plus measurement, not isolated readout; IID Wilson intervals are descriptive"}


def audit_task(saved, payload):
    """Validate observed provider fields, without assuming another SDK's contract."""
    task_id = saved["job"]["job_id"]
    if str(payload.get("tid")) != task_id or payload.get("chip") != saved["job"]["backend_name"]:
        raise ValueError("provider task/backend identity mismatch")
    if payload.get("count", payload.get("counts")) != saved.get("raw_counts"):
        raise ValueError("provider counts differ from the immutable task receipt")
    if str(payload.get("status", "")).lower() != "finished" or payload.get("error"):
        raise ValueError("provider task is not successfully finished")
    submitted = payload.get("circuit")
    if not isinstance(submitted, str) or hashlib.sha256(submitted.encode()).hexdigest() != saved["artifact"]["sha256"]:
        raise ValueError("provider submitted source differs from immutable QASM")
    expected = compiled_probabilities(submitted)
    compiled_source = payload.get("transpiled")
    if not isinstance(compiled_source, str) or not compiled_source.strip():
        raise ValueError("compiled QASM is unavailable; cannot claim compilation equivalence")
    compiled = compiled_probabilities(compiled_source)
    qlisp = qlisp_probabilities(payload["qlisp"]) if payload.get("qlisp") else None

    def tvd(a, b):
        if len(a) != len(b):
            raise ValueError("compiled and submitted classical widths differ")
        return sum(abs(x - y) for x, y in zip(a, b)) / 2

    reverse_source_measurements = {physical: bit for bit, physical in expected["physical_measurement_map"].items()}
    final_map = {str(logical): compiled["physical_measurement_map"][bit]
                 for logical, bit in reverse_source_measurements.items()}
    fields = {name: {"status": "AVAILABLE" if name in payload else "ABSENT", "value": payload.get(name)}
              for name in ("compiler", "compiler_version", "options", "calibration_id", "bit_order", "measurements",
                           "shots", "created", "finished", "corrected")}
    return {"task_id": task_id, "backend": payload["chip"], "provider_status": payload["status"],
            "source_sha256": saved["artifact"]["sha256"],
            "compiled_sha256": hashlib.sha256(compiled_source.encode()).hexdigest(),
            "qlisp_sha256": hashlib.sha256(payload["qlisp"].encode()).hexdigest() if qlisp else None,
            "counts_match_immutable_receipt": True, "provider_fields": sorted(payload), "fields": fields,
            "shots_requested": saved["job"]["shots"], "shots_returned": sum(saved["raw_counts"].values()),
            "submitted_options": {"compile": True, "other_options": "provider defaults; versions unavailable"},
            "native_submitted": expected, "native_compiled": compiled,
            "logical_to_physical_at_final_measurement": final_map,
            "mapping_scope": "terminal measurements only; not full placement/routing trajectory or provider counts bit order",
            "compiled_vs_submitted_tvd": tvd(expected["probabilities"], compiled["probabilities"]),
            "qlisp_vs_compiled_tvd": tvd(qlisp["probabilities"], compiled["probabilities"]) if qlisp else None,
            "qlisp_measurement_map_matches": qlisp["physical_measurement_map"] == compiled["physical_measurement_map"] if qlisp else None,
            "readout_correction": "no correction applied locally; provider corrected field retained verbatim",
            "scope": "all-zero-input ideal distribution only; not arbitrary unitary, pulse or hardware certification"}
