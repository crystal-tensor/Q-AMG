# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit Aer was copied.
"""Native educational simulator helpers for Qiskit Aer compatibility."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from quantumbridge.compat.qiskit_adapter import circuit_from_ir, circuit_from_qiskit
from quantumbridge.core import Circuit, Measurement
from quantumbridge.devices import StatevectorDevice, StabilizerDevice, DensityMatrixDevice, MatrixProductStateDevice
from quantumbridge.schema.aer_results import (
    NoisySimulationResult,
    QasmSimulationResult,
    StatevectorSimulationResult,
    DensityMatrixSimulationResult,
)

from .noise_adapter import (
    apply_measurement_bitflip_noise,
    create_bitflip_noise_model,
    validate_noise_model,
)
from .warnings import (
    NATIVE_NOISE_WARNING,
    NATIVE_SIMULATOR_WARNING,
    aer_warnings,
    native_provenance,
)

SUPPORTED_NATIVE_GATES = {
    "h",
    "x",
    "y",
    "z",
    "rx",
    "ry",
    "rz",
    "phase",
    "cx",
    "cz",
    "swap",
}


def normalize_circuit_to_quantumbridge_ir(circuit_or_ir: Any, method: str = "statevector") -> Circuit:
    """Normalize a QuantumBridge Circuit, IRProgram, or basic Qiskit circuit."""

    if isinstance(circuit_or_ir, Circuit):
        circuit = circuit_or_ir.copy()
    elif hasattr(circuit_or_ir, "instructions") and hasattr(circuit_or_ir, "num_qubits"):
        circuit = circuit_from_ir(circuit_or_ir)
    elif isinstance(circuit_or_ir, dict):
        circuit = _circuit_from_ir_dict(circuit_or_ir)
    elif hasattr(circuit_or_ir, "data") and hasattr(circuit_or_ir, "num_qubits"):
        circuit = circuit_from_qiskit(circuit_or_ir)
    else:
        raise TypeError(
            "native Aer simulator expects a QuantumBridge Circuit, QuantumBridge IR, "
            "IR dictionary, or basic Qiskit QuantumCircuit"
        )
    if method not in {"statevector", "stabilizer", "density_matrix", "matrix_product_state"}:
        raise ValueError("QuantumBridge native normalization method must be statevector, stabilizer, density_matrix or matrix_product_state.")
    if method == "statevector" and circuit.num_qubits > 4:
        raise ValueError("Stage 9F native Aer simulator supports 1-4 qubit circuits")
    if method == "statevector":
        _validate_supported_operations(circuit.operations)
    circuit.metadata.setdefault("normalized_by", "quantumbridge.compat.qiskit_aer")
    return circuit


def run_statevector_simulator_native(
    circuit_or_ir: Any,
    initial_state: Iterable[complex] | None = None,
) -> StatevectorSimulationResult:
    """Run the minimal exact statevector path for small circuits."""

    circuit = normalize_circuit_to_quantumbridge_ir(circuit_or_ir)
    if initial_state is not None:
        raise ValueError("Stage 9F native statevector path does not support custom initial_state")
    state = StatevectorDevice(max_qubits=4).statevector(circuit)
    probabilities = _state_probabilities(state, circuit.num_qubits)
    return StatevectorSimulationResult(
        workflow="statevector_simulator_native",
        backend="statevector",
        mode="native_minimal",
        capability_level=3,
        production_ready=False,
        native_implementation=True,
        num_qubits=circuit.num_qubits,
        final_statevector=[complex(value) for value in state],
        probabilities=probabilities,
        raw_type="QuantumBridgeStatevector",
        metadata=_execution_metadata(circuit, "statevector"),
        warnings=aer_warnings(NATIVE_SIMULATOR_WARNING),
        provenance=native_provenance("statevector_simulator_native"),
    )


def run_qasm_simulator_native(
    circuit_or_ir: Any,
    shots: int = 1024,
    seed: int | None = None,
) -> QasmSimulationResult:
    """Run deterministic seeded qasm-style sampling over state probabilities."""

    circuit = normalize_circuit_to_quantumbridge_ir(circuit_or_ir)
    shots = _validate_shots(shots)
    probabilities = _measurement_probabilities(circuit)
    counts = _sample_counts(probabilities, shots, seed)
    empirical = {label: count / shots for label, count in counts.items()}
    return QasmSimulationResult(
        workflow="qasm_simulator_native",
        backend="qasm",
        mode="native_minimal",
        capability_level=3,
        production_ready=False,
        native_implementation=True,
        num_qubits=circuit.num_qubits,
        shots=shots,
        seed=seed,
        counts=counts,
        probabilities=empirical,
        raw_type="QuantumBridgeQasmCounts",
        metadata=_execution_metadata(circuit, "qasm"),
        warnings=aer_warnings(NATIVE_SIMULATOR_WARNING),
        provenance=native_provenance("qasm_simulator_native"),
    )


def run_noisy_qasm_simulator_native(
    circuit_or_ir: Any,
    shots: int = 1024,
    noise_model: dict[str, Any] | None = None,
    seed: int | None = None,
) -> NoisySimulationResult:
    """Run the educational qasm path with simple measurement bit-flip noise."""

    model = validate_noise_model(noise_model or create_bitflip_noise_model(0.0))
    base = run_qasm_simulator_native(circuit_or_ir, shots=shots, seed=seed)
    counts = dict(base.counts)
    if model["type"] == "measurement_bitflip":
        counts = apply_measurement_bitflip_noise(counts, model["p"], base.shots or shots, seed=seed)
    empirical = {label: count / (base.shots or shots) for label, count in counts.items()}
    return NoisySimulationResult(
        workflow="noisy_qasm_simulator_native",
        backend="qasm",
        mode="native_minimal",
        capability_level=3,
        production_ready=False,
        native_implementation=True,
        num_qubits=base.num_qubits,
        shots=base.shots,
        seed=seed,
        counts=counts,
        probabilities=empirical,
        noise_model=model,
        raw_type="QuantumBridgeNoisyQasmCounts",
        metadata={**base.metadata, "noise_applied": model["type"] == "measurement_bitflip"},
        warnings=aer_warnings(NATIVE_SIMULATOR_WARNING, NATIVE_NOISE_WARNING),
        provenance=native_provenance("noisy_qasm_simulator_native"),
    )


def execute_basic_circuit_native(
    circuit_or_ir: Any,
    backend: str = "statevector",
    shots: int = 1024,
    seed: int | None = None,
    noise_model: dict[str, Any] | None = None,
):
    backend = str(backend)
    if backend == "statevector":
        return run_statevector_simulator_native(circuit_or_ir)
    if backend == "density_matrix":
        return run_density_matrix_simulator_native(circuit_or_ir, shots=shots, seed=seed, noise_model=noise_model)
    if backend == "matrix_product_state":
        return run_matrix_product_state_simulator_native(circuit_or_ir, shots=shots, seed=seed, noise_model=noise_model)
    if backend in {"stabilizer", "stabilizer_simulator"}:
        return run_stabilizer_simulator_native(circuit_or_ir, shots=shots, seed=seed, noise_model=noise_model)
    if backend in {"qasm", "qasm_simulator"}:
        return run_qasm_simulator_native(circuit_or_ir, shots=shots, seed=seed)
    if backend in {"noisy_qasm", "noisy_qasm_simulator"}:
        return run_noisy_qasm_simulator_native(
            circuit_or_ir, shots=shots, noise_model=noise_model, seed=seed
        )
    raise ValueError("backend must be statevector, density_matrix, matrix_product_state, stabilizer, qasm, or noisy_qasm")


def run_matrix_product_state_simulator_native(circuit_or_ir, shots=1024, seed=None, noise_model=None, memory=False, **device_options):
    circuit = normalize_circuit_to_quantumbridge_ir(circuit_or_ir, method="matrix_product_state")
    device = MatrixProductStateDevice(noise_model=noise_model, **device_options)
    result = device.run(circuit) if shots is None else device.run_shots(circuit, shots=shots, seed=seed, memory=memory)
    payload = result.to_dict()
    return QasmSimulationResult(
        workflow="matrix_product_state_simulator_native", backend="matrix_product_state", mode="native_minimal",
        capability_level=3, native_implementation=True, num_qubits=circuit.num_qubits,
        shots=shots, seed=seed, counts=result.counts() or {}, probabilities=result.probabilities() or {},
        data={"matrix_product_state": payload.get("matrix_product_state")},
        noise_model=None if noise_model is None else noise_model.summary(),
        metadata={**result.metadata(), "cloud_access": False, "token_read": False, "hardware_access": False,
                  "qiskit_aer_parity_claim": False}, raw_type="QuantumBridgeMatrixProductState",
        warnings=["Native MPS execution is not full Aer/device/save-instruction parity."],
        provenance=native_provenance("matrix_product_state_simulator_native"),
    )


def run_density_matrix_simulator_native(circuit_or_ir, shots=None, seed=None, noise_model=None, initial_state=None, memory=False):
    circuit = normalize_circuit_to_quantumbridge_ir(circuit_or_ir, method="density_matrix")
    device = DensityMatrixDevice(noise_model=noise_model, initial_state=initial_state)
    analytic = device.run(circuit)
    result = analytic if shots is None else device.run_shots(circuit, shots=shots, seed=seed, memory=memory)
    return DensityMatrixSimulationResult(
        workflow="density_matrix_simulator_native", backend="density_matrix", mode="native_minimal",
        capability_level=3, native_implementation=True, num_qubits=circuit.num_qubits,
        shots=shots, seed=seed, counts=result.counts() or {}, probabilities=result.probabilities(),
        final_density_matrix=analytic.density_matrix().tolist(),
        noise_model=None if noise_model is None else noise_model.summary(),
        metadata={**result.metadata(), "cloud_access": False, "token_read": False, "hardware_access": False,
                  "qiskit_aer_parity_claim": False}, raw_type="QuantumBridgeDensityMatrix",
        warnings=["Native mixed-state execution is not full Qiskit Aer parity."],
        provenance=native_provenance("density_matrix_simulator_native"),
    )


def run_stabilizer_simulator_native(circuit_or_ir, shots=1024, seed=None, noise_model=None, memory=False):
    """Execute native tableau trajectories without the legacy four-qubit cap."""
    circuit = normalize_circuit_to_quantumbridge_ir(circuit_or_ir, method="stabilizer")
    result = StabilizerDevice(noise_model=noise_model).run_shots(circuit, shots=shots, seed=seed, memory=memory)
    return QasmSimulationResult(
        workflow="stabilizer_simulator_native", backend="stabilizer", mode="native_minimal",
        capability_level=3, native_implementation=True, num_qubits=circuit.num_qubits,
        shots=shots, seed=seed, counts=result.counts(), probabilities=result.probabilities(),
        noise_model=None if noise_model is None else noise_model.summary(),
        metadata={**result.metadata(), "cloud_access": False, "token_read": False, "hardware_access": False,
                  "qiskit_aer_parity_claim": False},
        raw_type="QuantumBridgeStabilizerCounts",
        warnings=["Native Clifford/stabilizer execution is not full Qiskit Aer parity."],
        provenance=native_provenance("stabilizer_simulator_native"),
    )


def simulator_result_to_dict(result: Any) -> dict[str, Any]:
    if hasattr(result, "to_dict"):
        return result.to_dict()
    raise TypeError("simulator_result_to_dict expects a QuantumBridge AerResult")


def _validate_supported_operations(operations: Iterable[Any]) -> None:
    for op in operations:
        if op.name not in SUPPORTED_NATIVE_GATES:
            raise ValueError(f"Stage 9F native Aer simulator does not support operation {op.name!r}")


def _validate_shots(shots: int) -> int:
    shots = int(shots)
    if shots <= 0:
        raise ValueError("shots must be positive")
    return shots


def _state_probabilities(state: np.ndarray, num_qubits: int) -> dict[str, float]:
    probabilities: dict[str, float] = {}
    for index, amplitude in enumerate(state):
        label = format(index, f"0{num_qubits}b")
        value = float(abs(amplitude) ** 2)
        if value > 1e-15:
            probabilities[label] = value
    return probabilities


def _measurement_probabilities(circuit: Circuit) -> dict[str, float]:
    state = StatevectorDevice(max_qubits=4).statevector(circuit)
    full = _state_probabilities(state, circuit.num_qubits)
    if not circuit.measurements:
        return _normalize_distribution(full)
    num_bits = circuit.num_bits or max(item.bit for item in circuit.measurements) + 1
    measured: dict[str, float] = {}
    for full_label, probability in full.items():
        bits = ["0"] * num_bits
        for item in circuit.measurements:
            bits[item.bit] = full_label[item.wire]
        label = "".join(bits)
        measured[label] = measured.get(label, 0.0) + probability
    return _normalize_distribution(measured)


def _sample_counts(probabilities: dict[str, float], shots: int, seed: int | None) -> dict[str, int]:
    labels = sorted(probabilities)
    weights = np.array([probabilities[label] for label in labels], dtype=float)
    weights = weights / weights.sum()
    rng = np.random.default_rng(seed)
    draws = rng.choice(labels, size=shots, p=weights)
    counts = {label: int(np.count_nonzero(draws == label)) for label in labels}
    return {label: count for label, count in counts.items() if count}


def _normalize_distribution(probabilities: dict[str, float]) -> dict[str, float]:
    total = float(sum(probabilities.values()))
    if total <= 0:
        raise ValueError("probability distribution is empty")
    return {key: float(value) / total for key, value in probabilities.items() if value > 1e-15}


def _execution_metadata(circuit: Circuit, backend: str) -> dict[str, Any]:
    return {
        "backend": backend,
        "operation_count": len(circuit.operations),
        "measurement_count": len(circuit.measurements),
        "supported_gates": sorted(SUPPORTED_NATIVE_GATES),
        "cloud_access": False,
        "token_read": False,
        "hardware_access": False,
        "production_simulator": False,
        "qiskit_aer_parity_claim": False,
    }


def _circuit_from_ir_dict(payload: dict[str, Any]) -> Circuit:
    from quantumbridge.ir.qb_ir import decode_instruction_metadata

    registers = dict(payload.get("registers", {}))
    qreg = dict(registers.get("quantum", {}))
    creg = dict(registers.get("classical", {}))
    circuit = Circuit(
        int(payload.get("num_qubits", qreg.get("size", 0))),
        int(payload.get("num_bits", creg.get("size", 0))),
        payload.get("name"),
        dict(payload.get("metadata", {})),
    )
    for instruction in payload.get("instructions", ()):
        op = str(instruction["op"])
        targets = tuple(int(value) for value in instruction.get("targets", ()))
        controls = tuple(int(value) for value in instruction.get("controls", ()))
        params = tuple(float(value) for value in instruction.get("params", ()))
        circuit.append("cx" if op == "cnot" else "phase" if op == "p" else op,
                       targets, controls=controls, params=params, metadata=decode_instruction_metadata(instruction.get("metadata", {})))
    for measurement in payload.get("measurements", ()):
        wires = measurement.get("wires", ())
        bits = measurement.get("bits", ())
        circuit.measurements.append(Measurement(int(wires[0]), int(bits[0]), measurement.get("kind", "computational"), dict(measurement.get("metadata", {}))))
    return circuit
