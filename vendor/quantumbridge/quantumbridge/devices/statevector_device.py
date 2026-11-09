# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is part of QuantumBridge SDK.
# This implementation is developed for the QuantumBridge native architecture.
# Design source: docs/mvp/simulator_design_v0.1.md and docs/mvp/api_contract_v0.1.md.

from __future__ import annotations

from numbers import Real
from typing import Mapping, Optional, Union

import numpy as np

from quantumbridge.core.parameters import Parameter
from quantumbridge.results import Result
from quantumbridge.execution.device_derivatives import NativeDeviceDerivatives
from quantumbridge.execution.resource_context import meter_native_samples
from quantumbridge.utils.math import apply_unitary, expectation_observable, gate_matrix, probabilities_from_state, variance_observable


class StatevectorDevice(NativeDeviceDerivatives):
    """Exact pure-state simulator for QuantumBridge MVP circuits."""

    def __init__(self, max_qubits: Optional[int] = None, execution_limits=None, max_memory_mb=1024) -> None:
        self.max_qubits = max_qubits
        self.execution_limits = execution_limits
        self.max_memory_mb = max_memory_mb
        if max_memory_mb is not None and (isinstance(max_memory_mb, bool) or not isinstance(max_memory_mb, Real) or not np.isfinite(max_memory_mb) or max_memory_mb <= 0):
            raise ValueError("QuantumBridge max_memory_mb must be positive or None.")

    def _guard_density(self, width):
        if self.max_memory_mb is not None and 16 * (1 << (2 * width)) * 4 > self.max_memory_mb * 1024 ** 2:
            raise MemoryError("QuantumBridge dynamic dense output exceeds max_memory_mb.")

    def _initial(self, width):
        if self.max_memory_mb is not None and 16 * (1 << width) * 8 > self.max_memory_mb * 1024 ** 2:
            raise MemoryError("QuantumBridge dynamic statevector exceeds max_memory_mb.")
        from quantumbridge.execution.resource_context import current_resources
        if current_resources() is None:state = np.zeros(1 << width, dtype=complex)
        else:
            from quantumbridge.execution import mps_arrays as arrays
            state = arrays.filled("statevector_initial", (1 << width,), complex, 0)
        state[0] = 1
        return state

    def _dynamic_branches(self, circuit, parameters=None):
        from quantumbridge.execution.control_flow_engine import dynamic_branches
        return dynamic_branches(self, circuit, parameters)

    def _bound_circuit(self, circuit, parameters: Optional[Mapping[Union[Parameter, str], Real]]):
        return (circuit.bind(parameters) if parameters else circuit).lowered()

    def statevector(self, circuit, parameters: Optional[Mapping[Union[Parameter, str], Real]] = None) -> np.ndarray:
        circuit = self._bound_circuit(circuit, parameters)
        if circuit.has_dynamic_operations:
            raise ValueError("QuantumBridge StatevectorDevice requires unitary operations and terminal measurements; use StabilizerDevice for dynamic Clifford circuits.")
        if self.max_qubits is not None and circuit.num_qubits > self.max_qubits:
            raise ValueError("QuantumBridge circuit exceeds this device qubit limit.")
        state = self._initial(circuit.num_qubits)
        for op in circuit.operations:
            if op.metadata.get("directive") and op.name in {"barrier", "delay"}:
                continue
            matrix = gate_matrix(op.name, op.params, op.metadata)
            wires = op.controls + op.targets
            if op.metadata.get("requires_zero_input"):
                from .execution import validate_preparation_input
                validate_preparation_input(state,wires,circuit.num_qubits,"statevector")
            state = apply_unitary(state, matrix, wires, circuit.num_qubits)
        from quantumbridge.execution.resource_context import current_resources
        if current_resources() is None:return np.exp(1j * float(circuit.global_phase)) * state
        from quantumbridge.execution import mps_arrays as arrays
        output = arrays.owned("statevector_global_phase", state.shape, complex)
        np.multiply(state, np.exp(1j*float(circuit.global_phase)), out=output)
        return output

    def probabilities(self, circuit, wires=None, parameters: Optional[Mapping[Union[Parameter, str], Real]] = None) -> dict[str, float]:
        if circuit.has_dynamic_operations:
            probabilities = {}
            for state, _, _, weight in self._dynamic_branches(circuit, parameters):
                for label, probability in probabilities_from_state(state, circuit.num_qubits, wires).items():
                    probabilities[label] = probabilities.get(label, 0.0) + weight * probability
            return probabilities
        state = self.statevector(circuit, parameters)
        return probabilities_from_state(state, circuit.num_qubits, wires)

    def expectation(self, circuit, observable, parameters: Optional[Mapping[Union[Parameter, str], Real]] = None) -> float:
        if circuit.has_dynamic_operations:
            return float(sum(weight * expectation_observable(state, circuit.num_qubits, observable).real for state, _, _, weight in self._dynamic_branches(circuit, parameters)))
        state = self.statevector(circuit, parameters)
        value = expectation_observable(state, circuit.num_qubits, observable)
        return float(np.real_if_close(value))

    def variance(self, circuit, observable, parameters: Optional[Mapping[Union[Parameter, str], Real]] = None) -> float:
        if circuit.has_dynamic_operations:
            from quantumbridge.utils.math import observable_matrix
            self._guard_density(circuit.num_qubits)
            matrix = observable_matrix(observable, circuit.num_qubits)
            branches = self._dynamic_branches(circuit, parameters)
            first = sum(weight * np.vdot(state, matrix @ state).real for state, _, _, weight in branches)
            second = sum(weight * np.vdot(state, matrix @ matrix @ state).real for state, _, _, weight in branches)
            return max(0.0, float(second - first ** 2))
        state = self.statevector(circuit, parameters)
        return variance_observable(state, circuit.num_qubits, observable)

    def density_matrix(self, circuit, parameters: Optional[Mapping[Union[Parameter, str], Real]] = None) -> np.ndarray:
        self._guard_density(circuit.num_qubits)
        from quantumbridge.execution.resource_context import current_resources
        scope=current_resources()
        if scope is not None:
            from quantumbridge.execution.array_reservations import array_layout
            from quantumbridge.information.density_matrix import DensityMatrix
            from quantumbridge.execution import mps_arrays as arrays
            shape=(1<<circuit.num_qubits,)*2
            scope.ledger.preflight(memory_bytes=array_layout(shape,complex)[2])
            if not circuit.has_dynamic_operations:
                return DensityMatrix.from_statevector(self.statevector(circuit,parameters)).data
            result=arrays.filled("statevector_ensemble_density",shape,complex,0)
            for state,_,_,weight in self._dynamic_branches(circuit,parameters):
                current=DensityMatrix.from_statevector(state).data
                np.multiply(current,weight,out=current)
                np.add(result,current,out=result)
                current=None
            return result
        if circuit.has_dynamic_operations:
            return sum(weight * np.outer(state, state.conj()) for state, _, _, weight in self._dynamic_branches(circuit, parameters))
        state = self.statevector(circuit, parameters)
        return np.outer(state, np.conjugate(state))

    @meter_native_samples
    def sample(
        self,
        circuit,
        wires=None,
        shots: int = 1,
        seed: int | None = None,
        parameters: Optional[Mapping[Union[Parameter, str], Real]] = None,
    ) -> list[str]:
        if circuit.has_dynamic_operations:
            from quantumbridge.execution.control_flow_engine import DynamicExecutionEngine
            return DynamicExecutionEngine(self, circuit, parameters).quantum_samples(wires, shots, seed)
        from quantumbridge.information import Statevector

        state = Statevector(self.statevector(circuit, parameters))
        return state.sample_memory(shots, qargs=wires, seed=seed)

    def counts(
        self,
        circuit,
        wires=None,
        shots: int = 1024,
        seed: int | None = None,
        parameters: Optional[Mapping[Union[Parameter, str], Real]] = None,
    ) -> dict[str, int]:
        counts: dict[str, int] = {}
        for bitstring in self.sample(circuit, wires=wires, shots=shots, seed=seed, parameters=parameters):
            counts[bitstring] = counts.get(bitstring, 0) + 1
        return counts

    def run(self, circuit, parameters: Optional[Mapping[Union[Parameter, str], Real]] = None, observable=None) -> Result:
        if circuit.has_dynamic_operations:
            branches = self._dynamic_branches(circuit, parameters)
            if observable is not None:
                return Result(expectation_data=float(sum(weight * expectation_observable(state, circuit.num_qubits, observable).real for state, _, _, weight in branches)),
                    metadata_data={"device": "StatevectorDevice", "simulation": "statevector_branch_ensemble", **getattr(branches, "metadata", {})})
            probabilities = {}
            for state, bits, _, weight in branches:
                values = {"".join(map(str, bits)): 1} if circuit.has_measurements else probabilities_from_state(state, circuit.num_qubits)
                for label, probability in values.items(): probabilities[label] = probabilities.get(label, 0.0) + weight * probability
            self._guard_density(circuit.num_qubits)
            density = sum(weight * np.outer(state, state.conj()) for state, _, _, weight in branches)
            return Result(density=density, probabilities_data=probabilities,
                expectation_data=None if observable is None else self.expectation(circuit, observable, parameters),
                metadata_data={"device": "StatevectorDevice", "simulation": "statevector_branch_ensemble", **getattr(branches, "metadata", {})})
        state = self.statevector(circuit, parameters)
        probs = probabilities_from_state(state, circuit.num_qubits)
        expectation = None if observable is None else float(np.real_if_close(expectation_observable(state, circuit.num_qubits, observable)))
        return Result(
            state=state,
            probabilities_data=probs,
            expectation_data=expectation,
            metadata_data={
                "device": "StatevectorDevice",
                "qubit_order": "q0-first bitstrings",
                "num_qubits": circuit.num_qubits,
            },
        )

    def run_shots(self, circuit, shots=1024, seed=None, parameters=None, memory=False):
        circuit = circuit.lowered()
        if circuit.has_dynamic_operations:
            from quantumbridge.execution.control_flow_engine import dynamic_shots
            return dynamic_shots(self, circuit, shots, seed, parameters, memory)
        from types import SimpleNamespace
        from .sampler_device import ShotSampler

        return ShotSampler(shots, seed, base_device=SimpleNamespace(probabilities=self.probabilities), memory=memory).run(circuit, parameters)
