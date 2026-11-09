# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# Independently implemented for QuantumBridge SDK.

from quantumbridge.devices.matrix_product_state_device import MatrixProductStateDevice
from .backend import _DEFAULT_BASIS_GATES, _SimulatorBackend


class MatrixProductStateBackend(_SimulatorBackend):
    """Backend/Job access to native MPS simulation without dense state expansion."""

    def __init__(self, name="matrix_product_state", max_qubits=None, max_bond_dim=None, cutoff=1e-16,
                 max_memory_mb=1024, max_branches=4096, noise_model=None, initial_state=None, **options):
        device = MatrixProductStateDevice(max_qubits, max_bond_dim, cutoff, max_memory_mb,
                                          max_branches, noise_model, initial_state)
        super().__init__(name=name, device=device, max_qubits=max_qubits,
                         basis_gates=(*_DEFAULT_BASIS_GATES, "reset", "kraus"), **options)

    def _configured_device(self, options):
        device = super()._configured_device(options)
        overrides = {key: options.get(key) for key in ("max_bond_dim", "cutoff", "max_memory_mb", "max_branches", "initial_state") if key in options}
        return device._configured(**overrides) if overrides else device
