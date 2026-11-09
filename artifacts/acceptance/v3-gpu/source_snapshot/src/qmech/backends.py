"""Quantum execution boundaries. A simulator is ALWAYS labelled a simulator.

Company HTTP endpoints below are our reference adapter protocol, not a guessed
vendor contract. Contract tests use an explicitly labelled mock transport.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

import httpx
from qiskit import QuantumCircuit, qasm3, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error


@dataclass
class Capabilities:
    name: str
    execution_source: str
    max_qubits: int
    max_shots: int
    is_simulator: bool
    verified: bool
    basis_gates: list[str] = field(default_factory=list)
    coupling_map: list[list[int]] | None = None
    noise_provenance: str | None = None
    complex_bytes_per_amplitude: int = 16


@dataclass
class SampleBatch:
    counts: dict[str, int]
    shots: int
    job_id: str
    execution_source: str
    timings: dict[str, float | None]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.shots <= 0 or not self.counts or sum(self.counts.values()) != self.shots:
            raise ValueError("Backend returned inconsistent shot counts")
        if any(type(v) is not int or v <= 0 or not k or set(k) - {"0", "1"}
               for k, v in self.counts.items()):
            raise ValueError("Backend returned invalid bitstrings or counts")
        if len({len(k) for k in self.counts}) != 1:
            raise ValueError("Backend returned mixed bit widths")


class CircuitBackend(Protocol):
    def capabilities(self) -> Capabilities: ...
    def compile(self, circuit: QuantumCircuit, seed: int) -> QuantumCircuit: ...
    def sample(self, circuit: QuantumCircuit, shots: int, seed: int) -> SampleBatch: ...


class AerBackend:
    def __init__(self, noisy: bool = False, max_qubits: int = 16,
                 device_profile: dict | None = None):
        self.noisy = noisy
        self.profile = device_profile
        self.max_qubits = min(max_qubits, (device_profile or {}).get("max_qubits", max_qubits))
        kwargs: dict[str, Any] = {}
        if noisy:
            p = device_profile or {"one_qubit_error": 0.001, "two_qubit_error": 0.01,
                                   "readout_error": 0.01, "name": "generic_not_device_calibrated"}
            for key in ["one_qubit_error", "two_qubit_error", "readout_error"]:
                if not 0 <= p[key] < 1:
                    raise ValueError(f"Invalid noise probability: {key}")
            model = NoiseModel()
            model.add_all_qubit_quantum_error(depolarizing_error(p["one_qubit_error"], 1),
                                              ["sx", "x"])
            model.add_all_qubit_quantum_error(depolarizing_error(p["two_qubit_error"], 2), ["cx"])
            e = p["readout_error"]
            model.add_all_qubit_readout_error(ReadoutError([[1-e, e], [e, 1-e]]))
            kwargs["noise_model"] = model
        self.simulator = AerSimulator(method="statevector", max_parallel_threads=2,
                                      max_memory_mb=4096, **kwargs)
        self.device = "GPU" if "GPU" in self.simulator.available_devices() else "CPU"
        self.simulator.set_options(device=self.device)

    def capabilities(self) -> Capabilities:
        return Capabilities(
            name="Aer noisy" if self.noisy else "Aer ideal",
            execution_source="noisy_simulator" if self.noisy else "ideal_simulator",
            max_qubits=self.max_qubits, max_shots=16384, is_simulator=True, verified=True,
            basis_gates=["rz", "sx", "x", "cx"],
            coupling_map=(self.profile or {}).get("coupling_map"),
            noise_provenance=(self.profile or {}).get("name", "generic_not_device_calibrated")
            if self.noisy else None)

    def compile(self, circuit: QuantumCircuit, seed: int) -> QuantumCircuit:
        if circuit.num_qubits > self.max_qubits:
            raise ValueError(f"Circuit needs {circuit.num_qubits} qubits; limit {self.max_qubits}")
        return transpile(circuit, basis_gates=self.capabilities().basis_gates,
                         coupling_map=self.capabilities().coupling_map,
                         optimization_level=2, seed_transpiler=seed)

    def sample(self, circuit: QuantumCircuit, shots: int, seed: int) -> SampleBatch:
        start = time.perf_counter()
        if circuit.num_qubits > self.max_qubits or not 1 <= shots <= 16384:
            raise ValueError("Circuit/shot count exceeds local backend limits")
        job = self.simulator.run(circuit, shots=shots, seed_simulator=seed)
        result = job.result()
        if not result.success:
            raise RuntimeError(f"Aer execution failed: {result.status}")
        # Qiskit counts are MSB-left; our saved structure variables are qubit0-left.
        counts = {key.replace(" ", "")[::-1]: int(value)
                  for key, value in result.get_counts().items()}
        elapsed = time.perf_counter() - start
        return SampleBatch(counts, shots, job.job_id(), self.capabilities().execution_source,
                           {"backend_roundtrip_s": elapsed, "queue_s": 0.0,
                            "upload_s": 0.0, "quantum_execution_s": None,
                            "simulator_execution_s": float(result.time_taken),
                            "readout_s": None},
                           {"seed_simulator": seed, "bit_order": "qubit0_left",
                            "device": self.device,
                            "noise_provenance": self.capabilities().noise_provenance,
                            "timing_note": "Simulator execution is not physical QPU time"})


class CompanyBackend:
    """Reference HTTP adapter; requires an explicitly supplied compatible service.

    GET /capabilities; POST /jobs {qasm3, shots, client_request_id};
    GET /jobs/{id}; GET /jobs/{id}/result; POST /jobs/{id}/cancel.
    See docs/company-protocol.md. Not verified against company hardware.
    """

    def __init__(self, base_url: str, token_env: str = "QMECH_QPU_TOKEN",
                 timeout_s: float = 120, poll_interval_s: float = .25,
                 client: httpx.Client | None = None):
        if not base_url:
            raise ValueError("Company endpoint has not been configured")
        self.timeout_s, self.poll_interval_s = timeout_s, poll_interval_s
        token = os.environ.get(token_env)
        self.client = client or httpx.Client(base_url=base_url.rstrip("/") + "/",
                                             headers={"Authorization": f"Bearer {token}"}
                                             if token else {}, timeout=min(timeout_s, 30))
        self._caps: Capabilities | None = None

    def capabilities(self) -> Capabilities:
        if self._caps is None:
            response = self.client.get("capabilities")
            response.raise_for_status()
            d = response.json()
            if d.get("bit_order") != "qubit0_left":
                raise ValueError("Adapter must translate provider counts to qubit0_left")
            simulator = d["is_simulator"]
            if type(simulator) is not bool:
                raise ValueError("is_simulator must be a JSON boolean")
            self._caps = Capabilities(
                name=d["name"], max_qubits=int(d["max_qubits"]),
                max_shots=int(d["max_shots"]), is_simulator=simulator,
                execution_source="remote_simulator" if simulator else "hardware_declared",
                verified=False, basis_gates=d["basis_gates"], coupling_map=d.get("coupling_map"),
                noise_provenance=d.get("calibration_id"))
        return self._caps

    def compile(self, circuit: QuantumCircuit, seed: int) -> QuantumCircuit:
        caps = self.capabilities()
        if circuit.num_qubits > caps.max_qubits:
            raise ValueError("Circuit exceeds advertised QPU capacity")
        return transpile(circuit, basis_gates=caps.basis_gates, coupling_map=caps.coupling_map,
                         optimization_level=2, seed_transpiler=seed)

    def submit(self, circuit: QuantumCircuit, shots: int) -> str:
        if not 1 <= shots <= self.capabilities().max_shots:
            raise ValueError("Shot count exceeds advertised backend limit")
        response = self.client.post("jobs", json={"qasm3": qasm3.dumps(circuit), "shots": shots,
                                                 "client_request_id": str(uuid4())})
        response.raise_for_status()
        return str(response.json()["job_id"])

    def status(self, job_id: str) -> dict:
        response = self.client.get(f"jobs/{job_id}")
        response.raise_for_status()
        return response.json()

    def cancel(self, job_id: str) -> None:
        response = self.client.post(f"jobs/{job_id}/cancel")
        response.raise_for_status()

    def sample(self, circuit: QuantumCircuit, shots: int, seed: int) -> SampleBatch:
        start = time.perf_counter()
        job_id = self.submit(circuit, shots)
        upload = time.perf_counter() - start
        try:
            while True:
                if time.perf_counter() - start >= self.timeout_s:
                    raise TimeoutError(f"QPU job {job_id} exceeded configured timeout")
                state = self.status(job_id)
                status = state["status"]
                if status == "completed":
                    break
                if status in {"failed", "cancelled"}:
                    raise RuntimeError(f"QPU job {job_id} ended: {status}")
                if status not in {"queued", "running"}:
                    raise ValueError(f"Unknown QPU job state: {status}")
                time.sleep(min(self.poll_interval_s, max(0, self.timeout_s -
                                                         (time.perf_counter() - start))))
            response = self.client.get(f"jobs/{job_id}/result")
            response.raise_for_status()
            d = response.json()
            if d.get("bit_order") != "qubit0_left":
                raise ValueError("Result bit order is unsupported")
            if any(len(k) != circuit.num_clbits for k in d["counts"]):
                raise ValueError("Result width does not match submitted circuit")
            timing = d.get("timings", {})
            for k in ["queue_s", "quantum_execution_s", "readout_s"]:
                if timing.get(k) is not None and timing[k] < 0:
                    raise ValueError("Invalid device timing")
            return SampleBatch(d["counts"], shots, job_id, self.capabilities().execution_source,
                               {"backend_roundtrip_s": time.perf_counter() - start,
                                "upload_s": upload, "queue_s": timing.get("queue_s"),
                                "quantum_execution_s": timing.get("quantum_execution_s"),
                                "simulator_execution_s": None, "readout_s": timing.get("readout_s")},
                               {"device": self.capabilities().name,
                                "hardware_independently_verified": False,
                                "bit_order": "qubit0_left"})
        except (TimeoutError, KeyboardInterrupt):
            try:
                self.cancel(job_id)
            except httpx.HTTPError:
                pass
            raise


def make_backend(name: str, max_qubits: int = 16,
                 company_config: str | Path | None = None,
                 device_profile: dict | None = None) -> CircuitBackend:
    if name in {"auto", "torch-gpu", "torch-cpu"}:
        from .torch_backend import TorchStatevectorBackend, available_gpu
        if name == "torch-cpu":
            return TorchStatevectorBackend("cpu", max_qubits)
        gpu = available_gpu()
        if gpu:
            return TorchStatevectorBackend(gpu, max_qubits)
        if name == "torch-gpu":
            raise ValueError("No supported local GPU; choose auto for CPU fallback")
        return AerBackend(max_qubits=max_qubits)
    if name in {"aer", "aer-noisy"}:
        if device_profile is None and name == "aer-noisy":
            profile_path = os.environ.get("QMECH_DEVICE_PROFILE")
            if profile_path:
                device_profile = json.loads(Path(profile_path).read_text())
        return AerBackend(name == "aer-noisy", max_qubits, device_profile)
    if name == "company":
        path = company_config or os.environ.get("QMECH_COMPANY_CONFIG")
        if not path:
            raise ValueError("Set QMECH_COMPANY_CONFIG to an explicit company adapter configuration")
        config = json.loads(Path(path).read_text())
        return CompanyBackend(config.get("base_url"), config.get("token_env", "QMECH_QPU_TOKEN"),
                              config.get("timeout_s", 120), config.get("poll_interval_s", .25))
    raise ValueError(f"Not a quantum backend: {name}")


def capabilities_dict(backend: CircuitBackend) -> dict:
    return asdict(backend.capabilities())

