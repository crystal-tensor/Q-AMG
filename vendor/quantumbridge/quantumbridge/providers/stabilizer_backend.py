# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This implementation is developed for the QuantumBridge native architecture.

from quantumbridge.devices import StabilizerDevice

from .backend import _SimulatorBackend


class StabilizerBackend(_SimulatorBackend):
    """Native Backend/Job execution for Clifford circuits and classical feedback."""

    def __init__(self, name="stabilizer", max_qubits=None, noise_model=None, max_branches=100000, **options):
        device = StabilizerDevice(max_qubits=max_qubits, noise_model=noise_model, max_branches=max_branches)
        defaults = {"shots": 1024, **options}
        super().__init__(name=name, device=device, max_qubits=max_qubits,
                         basis_gates=("id", "x", "y", "z", "h", "s", "sdg", "sx", "sxdg", "cx", "cy", "cz", "swap", "dcx", "ecr", "iswap", "reset"),
                         **defaults)
