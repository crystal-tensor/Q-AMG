# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is part of QuantumBridge SDK.
# This implementation is developed for the QuantumBridge native architecture.

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Optional

from .backend import Backend
from .stabilizer_backend import StabilizerBackend
from .density_matrix_backend import DensityMatrixBackend
from .matrix_product_state_backend import MatrixProductStateBackend
from .exceptions import QiskitBackendNotFoundError


class Provider:
    """Local provider registry with Qiskit-style backend discovery."""

    def __init__(self, backends: Optional[Iterable[Backend]] = None):
        default_backends = backends or (
            Backend(name="statevector", max_qubits=None),
            Backend(name="qasm_simulator", max_qubits=None, shots=1024),
            StabilizerBackend(),
            DensityMatrixBackend(),
            MatrixProductStateBackend(),
        )
        self._backends: dict[str, Backend] = {}
        for backend in default_backends:
            self.add_backend(backend)

    def add_backend(self, backend: Backend) -> Backend:
        if not isinstance(backend, Backend):
            raise TypeError("QuantumBridge Provider.add_backend expects a Backend.")
        backend.set_provider(self)
        self._backends[backend.name] = backend
        return backend

    def backends(self, name: Optional[str] = None, filters: Optional[Callable[[Backend], bool]] = None, **kwargs) -> list[Backend]:
        candidates = list(self._backends.values())
        if name is not None:
            candidates = [backend for backend in candidates if backend.name == name]
        for key, value in kwargs.items():
            candidates = [backend for backend in candidates if getattr(backend, key, None) == value]
        if filters is not None:
            candidates = [backend for backend in candidates if filters(backend)]
        return candidates

    def get_backend(self, name: str = "statevector", **kwargs) -> Backend:
        matches = self.backends(name=name, **kwargs)
        if len(matches) != 1:
            raise QiskitBackendNotFoundError(f"No QuantumBridge backend matches name={name!r}.")
        return matches[0]

    def backend_names(self) -> list[str]:
        return sorted(self._backends)

    def __iter__(self):
        return iter(self.backends())
