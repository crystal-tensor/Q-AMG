"""Offline superconducting experiment preparation and bounded reference execution.

Preparing/verifying a bundle never constructs a remote backend. The optional
executor requires an explicitly supplied endpoint and an explicit job/shot cap.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from quantumbridge import Circuit
from quantumbridge.serialization import qasm3_dumps, qpy_dump, qpy_load

from .backends import CompanyBackend, make_backend
from .benchmark import CLASSICAL_METHODS, dump, run_benchmark, snapshot_sources
from .problem import build_problem
from .quantum import qaoa_circuit
from .schema import DesignSpec, PlannerSettings

ROOT = Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canary_circuits() -> dict[str, tuple[Circuit, dict[str, float]]]:
    basis = Circuit(3)
    basis.x(0)  # Asymmetric state detects an endianness reversal.
    basis.measure_all()
    plus = Circuit(1)
    plus.h(0)
    plus.measure_all()
    result = {"basis_100": (basis, {"100": 1.}), "plus": (plus, {"0": .5, "1": .5})}
    for basis_name in ["z", "x", "y"]:
        bell = Circuit(2)
        bell.h(0)
        bell.cx(0, 1)
        if basis_name == "y":
            for wire in [0, 1]:
                bell.sdg(wire)
        if basis_name in {"x", "y"}:
            for wire in [0, 1]:
                bell.h(wire)
        bell.measure_all()
        expected = {"01": .5, "10": .5} if basis_name == "y" else {"00": .5, "11": .5}
        result[f"bell_{basis_name}"] = bell, expected
    return result


def prepare_bundle(output: Path, corpus_path: Path, transfer_path: Path,
                   instances: int = 30, seeds: int = 5, shots: int = 256,
                   canary_repeats: int = 5) -> dict:
    if output.exists():
        raise FileExistsError("Choose a new bundle directory")
    if min(instances, seeds, canary_repeats) < 1 or not 32 <= shots <= 16384:
        raise ValueError("Invalid experiment counts or shot budget")
    corpus, transfer = json.loads(corpus_path.read_text()), json.loads(transfer_path.read_text())
    if transfer["corpus_sha256"] != digest(corpus_path) or transfer["encoding"] != "compact_binary":
        raise ValueError("Transfer angles must match the exact corpus and compact binary encoding")
    train_ids = {c["id"] for c in corpus["instances"] if c["split"] == "train"}
    if any(c["trained_on"] not in train_ids for c in transfer["contexts"].values()):
        raise ValueError("Transfer file includes a held-out or unknown training instance")
    cases = [c for c in corpus["instances"] if c["split"] == "heldout"][:instances]
    if len(cases) != instances:
        raise ValueError("Corpus has fewer held-out instances than requested")
    output.mkdir(parents=True)
    shutil.copy2(corpus_path, output/"corpus.json")
    shutil.copy2(transfer_path, output/"transfer.json")
    source_hashes = snapshot_sources(output)
    for name in ["run_hardware.py", "prepare_hardware.py"]:
        path = ROOT/"scripts"/name
        destination = output/"source_snapshot"/"scripts"/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        source_hashes[f"scripts/{name}"] = digest(path)
    files, jobs, logical_gates = {}, [], set()

    def save_circuit(name, circuit):
        logical_gates.update(circuit.count_ops())
        folder = output/"circuits"
        folder.mkdir(exist_ok=True)
        qpy_path, qasm_path = folder/f"{name}.qpy", folder/f"{name}.qasm"
        with qpy_path.open("wb") as stream:
            qpy_dump(circuit, stream)
        qasm_path.write_text(qasm3_dumps(circuit))
        return str(qpy_path.relative_to(output))

    for name, (circuit, expected) in canary_circuits().items():
        path = save_circuit(name, circuit)
        for repeat in range(canary_repeats):
            jobs.append({"kind": "canary", "name": name, "repeat": repeat, "seed": 2000+repeat,
                         "circuit": path, "shots": shots, "expected_probabilities": expected,
                         "max_abs_probability_error": .2})
    for case in cases:
        problem = build_problem(DesignSpec.model_validate(case["spec"]))
        qubo = problem.encode().compact_binary()
        context = transfer["contexts"][f"{problem.spec.family}:{len(problem.groups)}"]
        parameters, depth = context["parameters"], transfer["depth"]
        if len(parameters) != 2*depth or not np.isfinite(parameters).all():
            raise ValueError("Invalid transferred angles")
        circuit, gamma, beta, _ = qaoa_circuit(qubo, depth, "x")
        bound = circuit.assign_parameters({**dict(zip(gamma, parameters[:depth], strict=True)),
                                          **dict(zip(beta, parameters[depth:], strict=True))})
        path = save_circuit(case["id"], bound)
        optimum = min(problem.cost(s) for s in problem.assignments() if not problem.violations(s))
        dump(output/"problems"/f"{case['id']}.json", problem.as_dict())
        for repeat in range(seeds):
            jobs.append({"kind": "structure", "instance": case["id"], "family": problem.spec.family,
                         "seed": 1000+repeat, "circuit": path, "shots": shots,
                         "original_variables": problem.encode().n, "logical_qubits": qubo.n,
                         "parameters": parameters, "trained_on": context["trained_on"],
                         "optimum": optimum, "target_cost": optimum+.05*max(1, abs(optimum))})
    for path in output.rglob("*"):
        if path.is_file():
            files[str(path.relative_to(output))] = digest(path)
    manifest = {
        "schema_version": 1, "status": "offline_prepared_not_submitted", "jobs": jobs,
        "instances": [c["id"] for c in cases], "seeds": seeds, "shots": shots,
        "canary_repeats": canary_repeats, "depth": transfer["depth"], "iterations": 0,
        "budget": {"jobs_max": len(jobs), "shots_max": sum(job["shots"] for job in jobs),
                   "retries": 0, "timeout_s_per_job": 120},
        "source_sha256": source_hashes, "file_sha256": files,
        "hardware_requirements": {"kind": "superconducting_gate_model", "logical_qubits_max": max(j["logical_qubits"] for j in jobs if j["kind"] == "structure"),
                                  "native_gates": "query actual device; no physical topology assumed",
                                  "logical_gate_set": sorted(logical_gates),
                                  "bit_order": "qubit0_left"},
        "timing_policy": "Rebuild and compile logical circuits against actual capabilities in each request; all native compilation, client roundtrip, decoding, geometry and amortized train time counted",
        "execution_policy": "Canaries first, then randomized quantum/classical benchmark; no retry or automatic resume after ambiguous submissions",
        "canary_scope": "Bit-order and X/Y/Z basis distribution smoke checks; not independent proof of physical hardware or an entanglement certification experiment",
        "reference_scope": "QPY/QASM files contain bound logical circuits, not a calibrated pulse schedule or routed physical circuit",
        "hardware_validated": False, "quantum_acceleration_target_met": False,
    }
    dump(output/"manifest.json", manifest)
    return manifest


def verify_bundle(directory: Path, check_current_sources: bool = True) -> dict:
    manifest = json.loads((directory/"manifest.json").read_text())
    for relative, expected in manifest["file_sha256"].items():
        path = (directory/relative).resolve()
        if not path.is_relative_to(directory.resolve()) or digest(path) != expected:
            raise ValueError(f"Bundle hash mismatch or escaping path: {relative}")
    if check_current_sources:
        for relative, expected in manifest["source_sha256"].items():
            if digest(ROOT/relative) != expected:
                raise ValueError(f"Execution source changed; prepare a new versioned bundle: {relative}")
    jobs, budget = manifest["jobs"], manifest["budget"]
    if len(jobs) != budget["jobs_max"] or sum(j["shots"] for j in jobs) != budget["shots_max"]:
        raise ValueError("Job list does not reconcile to registered budget")
    for relative in {j["circuit"] for j in jobs}:
        if relative not in manifest["file_sha256"]:
            raise ValueError("Unhashed circuit in job plan")
        with (directory/relative).open("rb") as stream:
            circuits = qpy_load(stream)
        if len(circuits) != 1 or circuits[0].parameters:
            raise ValueError("Expected one fully bound logical circuit per file")
    return manifest


class SubmissionJournal:
    """Durable write-before-submit intent and provider receipt, with a hard budget.

    Lost network replies are NOT retried: reconcile client_request_id with the
    provider before planning a separate run. A new output directory is required.
    """

    def __init__(self, directory: Path, max_jobs: int, max_shots: int):
        self.directory, self.max_jobs, self.max_shots = directory, max_jobs, max_shots
        self.jobs, self.shots = 0, 0
        self.stopped_after_error = False
        self.path = directory/"submissions.jsonl"
        if self.path.exists():
            raise FileExistsError("A submission journal cannot be silently reused")

    def __call__(self, event: dict):
        event = dict(event)
        if event["event"] == "submit_intent":
            if self.stopped_after_error:
                raise RuntimeError("QPU submissions stopped after an error; reconcile provider state before a new run")
            shots = event["shots"]
            if self.jobs+1 > self.max_jobs or self.shots+shots > self.max_shots:
                raise RuntimeError("Declared QPU job/shot budget exhausted before submission")
            self.jobs += 1
            self.shots += shots
            qasm = event.pop("qasm3")
            sha = hashlib.sha256(qasm.encode()).hexdigest()
            folder = self.directory/"submitted_circuits"
            folder.mkdir(exist_ok=True)
            (folder/f"{sha}.qasm").write_text(qasm)
            event.update(qasm_sha256=sha, jobs_charged=self.jobs, shots_charged=self.shots)
        if event["event"] in {"submit_failed", "job_error"}:
            self.stopped_after_error = True
        event["recorded_unix_s"] = time.time()
        with self.path.open("a") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False)+"\n")
            stream.flush()
            os.fsync(stream.fileno())


def execute_bundle(directory: Path, output: Path, max_jobs: int, max_shots: int,
                   backend: CompanyBackend | None = None) -> dict:
    manifest = verify_bundle(directory)
    budget = manifest["budget"]
    if max_jobs < budget["jobs_max"] or max_shots < budget["shots_max"]:
        raise ValueError("Declared authorization is below this bundle's registered job/shot budget")
    if output.exists():
        raise FileExistsError("Choose a new execution directory; no automatic resubmission/resume")
    # All local evidence/budget checks precede construction or contact of the remote backend.
    remote = backend if backend is not None else make_backend("company")
    if not isinstance(remote, CompanyBackend):
        raise TypeError("Hardware bundle execution requires the explicit reference company adapter")
    output.mkdir(parents=True)
    journal = SubmissionJournal(output, budget["jobs_max"], budget["shots_max"])
    remote.event_sink = journal
    remote.timeout_s = min(remote.timeout_s, budget["timeout_s_per_job"])
    caps = remote.capabilities()
    dump(output/"capabilities.json", asdict(caps))
    dump(output/"registration.json", {"bundle_manifest_sha256": digest(directory/"manifest.json"),
                                       "authorized_caps": {"jobs": max_jobs, "shots": max_shots},
                                       "enforced_caps": budget, "source_sha256": manifest["source_sha256"]})
    if caps.is_simulator:
        raise ValueError("Service advertises a simulator; use local simulator commands instead")
    if caps.max_qubits < manifest["hardware_requirements"]["logical_qubits_max"] or caps.max_shots < manifest["shots"]:
        raise ValueError("Actual device cannot fit this registered experiment")
    canaries = []
    for job in [j for j in manifest["jobs"] if j["kind"] == "canary"]:
        row = {"job": job, "passed": False}
        try:
            with (directory/job["circuit"]).open("rb") as stream:
                circuit = qpy_load(stream)[0]
            start = time.perf_counter()
            compiled = remote.compile(circuit, job["seed"])
            batch = remote.sample(compiled, job["shots"], job["seed"])
            expected = job["expected_probabilities"]
            error = max(abs(batch.counts.get(bits, 0)/batch.shots-expected.get(bits, 0))
                        for bits in set(expected) | set(batch.counts))
            row.update(passed=error <= job["max_abs_probability_error"],
                       max_abs_probability_error=error, result=asdict(batch),
                       wall_with_compile_s=time.perf_counter()-start)
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
            canaries.append(row)
            dump(output/"canaries.json", canaries)
            raise
        canaries.append(row)
        dump(output/"canaries.json", canaries)
    if not all(r["passed"] for r in canaries):
        raise RuntimeError("Canary distributions failed; remaining QPU work was not submitted")
    settings = PlannerSettings(backend="company", depth=manifest["depth"], iterations=0, shots=manifest["shots"],
                               timeout_s=budget["timeout_s_per_job"])
    result = run_benchmark(output/"benchmark", settings, len(manifest["instances"]), manifest["seeds"],
                           corpus_path=directory/"corpus.json", methods=["qaoa_binary", *CLASSICAL_METHODS],
                           warm_start_file=directory/"transfer.json", quantum_backend=remote)
    result["job_budget_usage"] = {"attempts": journal.jobs, "shots": journal.shots}
    result["hardware_validated"] = False
    result["quantum_acceleration_target_met"] = False
    result["physical_evidence_note"] = "Provider declaration and receipts retained; independent hardware provenance and physical acceleration audit still required"
    dump(output/"execution-summary.json", result)
    return result
