# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This implementation is developed for the QuantumBridge native architecture.

from quantumbridge.devices import DensityMatrixDevice
from .backend import _DEFAULT_BASIS_GATES, _SimulatorBackend


class DensityMatrixBackend(_SimulatorBackend):
    """Native mixed-state Backend/Job execution with arbitrary CPTP Kraus channels."""

    def __init__(self, name="density_matrix", max_qubits=None, noise_model=None, initial_state=None,
                 max_branches=4096, max_memory_mb=1024, **options):
        device = DensityMatrixDevice(max_qubits, noise_model, initial_state, max_branches, max_memory_mb)
        super().__init__(name=name, device=device, max_qubits=max_qubits,
                         basis_gates=(*_DEFAULT_BASIS_GATES, "reset", "kraus"), **options)

    def _configured_device(self, options):
        device = super()._configured_device(options)
        return device.with_initial_state(options.get("initial_state")) if "initial_state" in options else device
