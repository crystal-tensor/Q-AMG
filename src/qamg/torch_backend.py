"""Explicit statevector execution on CUDA / Apple Metal, with CPU Born sampling.

Executes the compiled quantum circuit, NOT a classical optimizer. The independent
QuantumBridge defines, lowers and compiles the native circuit and gate matrices.
This remains a simulator, not a quantum processing unit or quantum advantage.
"""
from __future__ import annotations

import time
from collections import Counter
from uuid import uuid4

import numpy as np
import torch
from quantumbridge import Circuit, transpile
from quantumbridge.utils.math import gate_matrix

from .backends import Capabilities, SampleBatch


def available_gpu() -> str | None:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return None


class TorchStatevectorBackend:
    def __init__(self, device: str | None = None, max_qubits: int = 16):
        self.device = device or available_gpu()
        if self.device not in {"cpu", "cuda", "mps"}:
            raise ValueError("No supported GPU; use the auto backend for explicit CPU fallback")
        if self.device != "cpu" and self.device != available_gpu():
            raise ValueError(f"Requested GPU is not available: {self.device}")
        self.max_qubits = max_qubits
        self._matrices = {}

    def capabilities(self) -> Capabilities:
        return Capabilities(name=f"QuantumBridge / Torch {self.device.upper()} statevector",
                            execution_source="ideal_simulator", max_qubits=self.max_qubits,
                            max_shots=16384, is_simulator=True, verified=True,
                            basis_gates=["rz", "sx", "x", "cx"],
                            complex_bytes_per_amplitude=8)

    def compile(self, circuit: Circuit, seed: int) -> Circuit:
        if circuit.num_qubits > self.max_qubits:
            raise ValueError("Circuit exceeds GPU simulation capacity")
        return transpile(circuit, basis_gates=self.capabilities().basis_gates,
                         optimization_level=2, seed_transpiler=seed)

    def synchronize(self) -> None:
        if self.device == "mps":
            torch.mps.synchronize()
        elif self.device == "cuda":
            torch.cuda.synchronize()

    @torch.no_grad()
    def probabilities(self, circuit: Circuit) -> np.ndarray:
        n = circuit.num_qubits
        if n > self.max_qubits or n < 1 or circuit.parameters:
            raise ValueError("Circuit must be bound and within the statevector capacity")
        # Peak intermediates are larger; cap the declared problem before any allocation.
        if 8*2**n*4 > 4*1024**3:
            raise MemoryError("State and gate intermediates exceed 4 GiB simulation budget")
        state = torch.zeros(2**n, dtype=torch.complex64, device=self.device)
        state[0] = 1
        if any(m.metadata.get("operation_index", len(circuit.operations)) < len(circuit.operations)
               for m in circuit.measurements):
            raise ValueError("Mid-circuit measurements are not supported by this simulator")
        for op in circuit.operations:
            if op.name == "barrier":
                continue
            if op.name not in {"rz", "sx", "x", "cx"}:
                raise ValueError(f"Compile to supported native gates before execution: {op.name}")
            qubits = op.controls + op.targets
            matrix = np.asarray(gate_matrix(op.name, op.params, op.metadata), dtype=np.complex64)
            key = (matrix.shape, matrix.tobytes())
            if key not in self._matrices:
                if len(self._matrices) >= 4096:
                    self._matrices.clear()
                self._matrices[key] = torch.as_tensor(matrix.copy(), device=self.device)
            unitary = self._matrices[key]
            # QB uses q0-first tensor axes and control-then-target gate operands.
            front = list(qubits)
            permutation = front+[i for i in range(n) if i not in front]
            inverse = tuple(map(int, np.argsort(permutation)))
            arranged = state.reshape([2]*n).permute(permutation).reshape(2**len(qubits), -1)
            state = (unitary@arranged).reshape([2]*n).permute(inverse).reshape(-1)
        self.synchronize()
        probabilities = torch.abs(state).square().cpu().numpy().astype(np.float64)
        norm = float(probabilities.sum())
        if not np.isfinite(probabilities).all() or abs(norm-1) > .002:
            raise ArithmeticError("Statevector norm drifted beyond the declared complex64 tolerance")
        return probabilities/norm

    def sample(self, circuit: Circuit, shots: int, seed: int) -> SampleBatch:
        start = time.perf_counter()
        if not 1 <= shots <= self.capabilities().max_shots:
            raise ValueError("Invalid shot count")
        probabilities = self.probabilities(circuit)
        evolved = time.perf_counter()
        mapping = [(measurement.wire, measurement.bit) for measurement in circuit.measurements]
        if not mapping:
            raise ValueError("Circuit must explicitly measure its output register")
        samples = np.random.default_rng(seed).choice(len(probabilities), shots, p=probabilities)
        counts = Counter()
        for sample in samples:
            bits = ["0"]*circuit.num_clbits
            for q, c in mapping:
                bits[c] = str((int(sample) >> (circuit.num_qubits-1-q)) & 1)
            counts["".join(bits)] += 1
        end = time.perf_counter()
        return SampleBatch(dict(counts), shots, f"torch-{self.device}-{uuid4().hex}", "ideal_simulator",
                           {"backend_roundtrip_s": end-start, "queue_s": 0.,
                            "upload_s": 0. if self.device == "cpu" else None,
                            "simulator_execution_s": evolved-start, "quantum_execution_s": None,
                            "readout_s": None, "born_sampling_s": end-evolved},
                           {"device": self.device, "statevector_dtype": "complex64", "quantum_sdk": "QuantumBridge",
                            "gate_matrix_preparation": "CPU; included in simulator execution wall time",
                            "state_evolution_device": self.device, "born_sampling_device": "cpu",
                            "upload_and_probability_readback": "Included in simulator wall time; not separately measured",
                            "bit_order": "qubit0_left", "seed_sampler": seed,
                            "device_synchronized_before_readout": True})
