# Copyright 2026 QuantumBridge Contributors. Licensed under Apache-2.0.
"""Same-circuit financial diagnostics with explicit unverified hardware status."""

from __future__ import annotations

import numpy as np
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from quantumbridge.devices import StatevectorDevice
from quantumbridge.serialization import deserialize_circuit

from .execution import FinanceCloudSession, circuit_artifact, probabilities_from_counts
from .kernels import labels
from .statistics import distribution_diagnostics, optimization_diagnostics, risk_sampling_diagnostics


def restore_submitted_circuits(case, journal):
    """Use exact submitted source across numerical environments, not rebuilt bytes.

    Hash checks remain exact. Full complex native states must also match the
    unchanged financial circuit within the existing preparation-test tolerance.
    This only reads the ledger; no provider, credential or new submission.
    """
    from copy import copy

    journal = Path(journal)
    data = json.loads(journal.read_text(encoding="utf-8"))
    session = FinanceCloudSession(SimpleNamespace(name=data["backend"]), journal=journal)
    plan = json.loads(journal.with_name("plan.json").read_text(encoding="utf-8"))
    restored = copy(case)
    restored.experiments = {}
    device = StatevectorDevice()
    for name, reference in case.experiments.items():
        saved = session.data["jobs"][f"{case.key}/{name}"]["artifact"]
        source, digest = saved["openqasm"], saved["sha256"]
        if hashlib.sha256(source.encode()).hexdigest() != digest or plan[case.key][name] != digest:
            raise ValueError("frozen source and plan hash mismatch")
        circuit = deserialize_circuit(source, "qasm2")
        if (circuit_artifact(circuit)["sha256"] != digest or circuit.num_qubits != reference.num_qubits
                or not np.allclose(device.statevector(circuit), device.statevector(reference), atol=2e-14, rtol=0)):
            raise ValueError("submitted circuit differs from the unchanged financial reference")
        restored.experiments[name] = circuit
    return restored


def analyze_case_evidence(case, results=None, *, bootstrap_samples=1000):
    if results is None:
        return {"case": case.key, "hardware_status": "NOT_EXECUTED", "hardware_results": None}
    if set(results) != set(case.experiments):
        raise ValueError("all fixed case experiments are required")
    hardware = all(row.get("hardware_access") is True and row.get("execution_mode") == "cloud_hardware"
                   for row in results.values())
    task_ids = set()
    device, diagnostics, ideal_results = StatevectorDevice(), {}, {}
    for name, circuit in case.experiments.items():
        row = results[name]
        expected_artifact = circuit_artifact(circuit)
        if row["artifact"]["sha256"] != expected_artifact["sha256"] or row["artifact"]["openqasm"] != expected_artifact["openqasm"]:
            raise ValueError("receipt does not match the frozen submitted circuit")
        normalized, probabilities, shots = probabilities_from_counts(row["counts"], circuit.num_qubits)
        if row.get("bit_order") != "q0-first" or shots != row["shots"] or shots != 1024:
            raise ValueError("receipt measurement order or shot contract mismatch")
        if not np.allclose(probabilities, row["probabilities"], atol=1e-14, rtol=0):
            raise ValueError("receipt probabilities disagree with measured counts")
        if hardware:
            task_id = row.get("task_id")
            if not isinstance(task_id, str) or not task_id or task_id in task_ids or not row.get("backend"):
                raise ValueError("unique task IDs and backend provenance required")
            task_ids.add(task_id)
            raw_normalized, _, raw_shots = probabilities_from_counts(
                row["raw_counts"], circuit.num_qubits, bit_order=row["provider_bit_order"])
            if normalized != raw_normalized or shots != raw_shots:
                raise ValueError("raw provider counts disagree with normalized receipt")
        ideal = device.probabilities(circuit)
        ideal_probabilities = [ideal.get(label, 0) for label in labels(circuit.num_qubits)]
        ideal_results[name] = {"probabilities": ideal_probabilities, "shots": 1024}
        diagnostics[name] = distribution_diagnostics(normalized, ideal_probabilities)
        diagnostics[name]["circuit_sha256"] = expected_artifact["sha256"]
        diagnostics[name]["execution_mode"] = row["execution_mode"]
        if case.key in {"allocation", "constraints"}:
            diagnostics[name]["optimization"] = optimization_diagnostics(case, normalized)
        if case.key == "risk":
            diagnostics[name]["risk_uncertainty"] = risk_sampling_diagnostics(
                case.inputs["losses"], normalized, confidence=case.inputs["confidence"],
                bootstrap_samples=bootstrap_samples)
    return {"case": case.key, "hardware_status": "INCONCLUSIVE" if hardware else "NOT_EXECUTED",
            "hardware_status_reason": "business tolerances not approved; receipt consistency is not independent hardware certification"
            if hardware else "local or injected transport is not hardware evidence",
            "reference": case.reference, "ideal_circuit_analysis": case.analyze(ideal_results),
            "measured_analysis": case.analyze(results), "experiment_diagnostics": diagnostics,
            "error_attribution": "model/reference vs ideal circuit vs finite shots vs hardware distribution; no specific noise source inferred"}
