# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is part of QuantumBridge SDK.
# This implementation is developed for the QuantumBridge native architecture.

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import count
from numbers import Real
from typing import Any, Optional, Union

from quantumbridge.core import Circuit, Parameter
from quantumbridge.devices import ShotSampler, StatevectorDevice

from .job import Job
from .options import Options


_JOB_COUNTER = count(1)


class Backend:
    """Local BackendV2-like execution target for QuantumBridge circuits."""

    def __init__(
        self,
        name: str = "quantumbridge_statevector",
        device: Optional[StatevectorDevice] = None,
        max_qubits: Optional[int] = None,
        basis_gates: Optional[Sequence[str]] = None,
        simulator: bool = True,
        **options: Any,
    ) -> None:
        self.name = name
        self.device = device or StatevectorDevice(max_qubits=max_qubits)
        self.max_qubits = max_qubits
        self.basis_gates = tuple(basis_gates or _DEFAULT_BASIS_GATES)
        self.simulator = bool(simulator)
        option_fields = {"shots": None, "seed_simulator": None, "memory": False}
        option_fields.update(options)
        self.options = Options(**option_fields)
        self._provider = None
        self._version = "0.1"

    @property
    def version(self) -> str:
        return self._version

    @property
    def num_qubits(self) -> Optional[int]:
        return self.max_qubits

    @property
    def provider(self):
        return self._provider

    def set_provider(self, provider) -> "Backend":
        self._provider = provider
        return self

    def set_options(self, **fields: Any) -> None:
        self.options.update_options(**fields)

    def configuration(self) -> dict[str, Any]:
        return {
            "backend_name": self.name,
            "backend_version": self.version,
            "n_qubits": self.max_qubits,
            "basis_gates": list(self.basis_gates),
            "simulator": self.simulator,
            "local": True,
            "conditional": bool(getattr(self.device, "supports_classical_control", False)),
            "memory": bool(self.options.get("memory", False)),
        }

    def properties(self) -> dict[str, Any]:
        return {
            "backend_name": self.name,
            "qubits": [] if self.max_qubits is None else [{} for _ in range(self.max_qubits)],
            "gates": [{"gate": gate, "parameters": []} for gate in self.basis_gates],
            "general": [{"name": "native", "value": True}],
        }

    def status(self) -> dict[str, Any]:
        return {
            "backend_name": self.name,
            "backend_version": self.version,
            "operational": True,
            "pending_jobs": 0,
            "status_msg": "local QuantumBridge backend ready",
        }

    def run(
        self,
        circuits,
        shots: Optional[int] = None,
        seed_simulator: Optional[int] = None,
        parameter_values: Optional[Union[Mapping[Union[Parameter, str], Real], Sequence[Mapping[Union[Parameter, str], Real]]]] = None,
        observable=None,
        **run_options: Any,
    ) -> Job:
        options = self.options.copy()
        options.update_options(**run_options)
        if shots is not None:
            options.shots = shots
        if seed_simulator is not None:
            options.seed_simulator = seed_simulator
        job_index = next(_JOB_COUNTER)
        job_id = f"{self.name}-local-{job_index}"

        def execute():
            return self._execute(circuits, options, parameter_values=parameter_values, observable=observable)

        return Job(
            job_id=job_id,
            backend=self,
            executor=execute,
            metadata={
                "backend_name": self.name,
                "options": options.to_dict(),
                "num_circuits": len(_as_circuit_sequence(circuits)),
            },
        )

    def _execute(
        self,
        circuits,
        options: Options,
        parameter_values: Optional[Union[Mapping[Union[Parameter, str], Real], Sequence[Mapping[Union[Parameter, str], Real]]]] = None,
        observable=None,
    ):
        circuit_sequence = _as_circuit_sequence(circuits)
        parameter_sets = _parameter_sets_for_circuits(parameter_values, len(circuit_sequence))
        device = self.device
        if "execution_limits" in options:
            from copy import copy
            device = copy(device)
            device.execution_limits = options.get("execution_limits")
        results = []
        for circuit, parameters in zip(circuit_sequence, parameter_sets):
            self._validate_circuit(circuit)
            shots = options.get("shots")
            if shots is None:
                result = device.run(circuit, parameters=parameters, observable=observable)
            else:
                result = ShotSampler(shots, seed=options.get("seed_simulator"), base_device=device, memory=options.get("memory", False)).run(circuit, parameters=parameters)
                result.metadata_data.setdefault("backend_name", self.name)
            results.append(result)
        return results[0] if isinstance(circuits, Circuit) else results

    def _validate_circuit(self, circuit: Circuit) -> None:
        if not isinstance(circuit, Circuit):
            raise TypeError("QuantumBridge Backend.run expects Circuit objects.")
        if self.max_qubits is not None and circuit.num_qubits > self.max_qubits:
            raise ValueError("QuantumBridge circuit exceeds backend qubit limit.")


class BackendV2(Backend):
    """Alias class for Qiskit BackendV2-style discovery."""


class _SimulatorBackend(BackendV2):
    def _configured_device(self, options):
        device = self.device.with_noise_model(options.get("noise_model")) if "noise_model" in options else self.device
        if "execution_limits" in options:
            from copy import copy
            device = copy(device)
            device.execution_limits = options.get("execution_limits")
        return device

    def _execute(self, circuits, options, parameter_values=None, observable=None):
        sequence = _as_circuit_sequence(circuits)
        parameter_sets = _parameter_sets_for_circuits(parameter_values, len(sequence))
        device = self._configured_device(options)
        results = []
        for circuit, parameters in zip(sequence, parameter_sets):
            self._validate_circuit(circuit)
            if observable is not None or options.get("shots") is None:
                result = device.run(circuit, parameters=parameters, observable=observable)
            else:
                result = device.run_shots(circuit, shots=options.shots, seed=options.get("seed_simulator"),
                                          parameters=parameters, memory=options.get("memory", False))
            result.metadata_data["backend_name"] = self.name
            results.append(result)
        return results[0] if isinstance(circuits, Circuit) else results


def _as_circuit_sequence(circuits) -> list[Circuit]:
    if isinstance(circuits, Circuit):
        return [circuits]
    if isinstance(circuits, Sequence) and all(isinstance(circuit, Circuit) for circuit in circuits):
        return list(circuits)
    raise TypeError("QuantumBridge Backend.run expects a Circuit or sequence of Circuit objects.")


def _parameter_sets_for_circuits(parameters, count_: int) -> list[Mapping[Union[Parameter, str], Real]]:
    if parameters is None:
        return [{} for _ in range(count_)]
    if isinstance(parameters, Mapping):
        return [dict(parameters) for _ in range(count_)]
    if isinstance(parameters, Sequence) and all(isinstance(item, Mapping) for item in parameters):
        if len(parameters) != count_:
            raise ValueError("QuantumBridge backend parameter_values length must match circuit count.")
        return [dict(item) for item in parameters]
    raise TypeError("QuantumBridge backend parameter_values must be a mapping or sequence of mappings.")


_DEFAULT_BASIS_GATES = (
    "id",
    "x",
    "y",
    "z",
    "h",
    "sx",
    "sxdg",
    "s",
    "sdg",
    "t",
    "tdg",
    "rx",
    "ry",
    "rz",
    "phase",
    "cx",
    "cz",
    "swap",
    "ccx",
    "crx",
    "cry",
    "crz",
    "cp",
    "rxx",
    "ryy",
    "rzz",
    "unitary",
)
