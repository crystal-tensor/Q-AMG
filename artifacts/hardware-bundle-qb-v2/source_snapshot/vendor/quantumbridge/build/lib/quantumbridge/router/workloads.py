# This file is independently implemented for QuantumBridge SDK.
"""Small benchmark workloads for the Router MVP."""

from __future__ import annotations

import math

from quantumbridge.core import Circuit


def bell_circuit() -> Circuit:
    return Circuit(2, 2, name="bell").h(0).cx(0, 1).measure(0, 0).measure(1, 1)


def ghz_circuit(num_qubits: int = 3) -> Circuit:
    circuit = Circuit(num_qubits, num_qubits, name=f"ghz_{num_qubits}")
    circuit.h(0)
    for index in range(num_qubits - 1):
        circuit.cx(index, index + 1)
    circuit.measure_all()
    return circuit


def qaoa_small_circuit() -> Circuit:
    circuit = Circuit(4, 4, name="qaoa_small")
    for qubit in range(4):
        circuit.h(qubit)
    for control, target in ((0, 1), (1, 2), (2, 3), (3, 0)):
        circuit.cx(control, target).rz(0.41, target).cx(control, target)
    for qubit in range(4):
        circuit.rx(0.77, qubit)
    circuit.measure_all()
    return circuit


def vqe_small_circuit() -> Circuit:
    circuit = Circuit(2, 2, name="vqe_small")
    circuit.ry(math.pi / 5, 0).ry(math.pi / 7, 1).cx(0, 1).rz(math.pi / 9, 1).cx(0, 1)
    circuit.measure(0, 0).measure(1, 1)
    return circuit


def workload_by_name(name: str) -> Circuit:
    key = str(name).strip().lower().replace("-", "_")
    if key == "bell":
        return bell_circuit()
    if key in {"ghz", "ghz_3"}:
        return ghz_circuit(3)
    if key == "ghz_4":
        return ghz_circuit(4)
    if key in {"qaoa", "qaoa_small"}:
        return qaoa_small_circuit()
    if key in {"vqe", "vqe_small"}:
        return vqe_small_circuit()
    raise ValueError(f"unknown Router workload: {name}")


def demo_workloads() -> list[Circuit]:
    return [bell_circuit(), ghz_circuit(3), qaoa_small_circuit(), vqe_small_circuit()]
