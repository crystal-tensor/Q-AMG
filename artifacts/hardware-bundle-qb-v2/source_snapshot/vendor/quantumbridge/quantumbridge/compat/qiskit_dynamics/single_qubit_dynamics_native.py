# This file is independently implemented for QuantumBridge SDK.
# No source code from IBM, Qiskit, or Qiskit Dynamics was copied.
"""Educational offline single-qubit dynamics workflows."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from quantumbridge.schema.dynamics_results import (
    DephasingMetadataResult,
    SingleQubitDynamicsResult,
)

from .hamiltonian_model_native import build_single_qubit_hamiltonian
from .time_evolution_native import evolve_single_qubit_state_native, serialize_state, serialize_states
from .warnings import dynamics_warnings, native_provenance


def default_times(stop: float = 6.283185307179586, count: int = 41) -> list[float]:
    if count <= 1:
        return [0.0]
    return [float(stop) * index / float(count - 1) for index in range(count)]


def run_z_precession_native(times: Iterable[float] | None = None, frequency: float = 1.0) -> SingleQubitDynamicsResult:
    time_values = list(times) if times is not None else default_times()
    hamiltonian = build_single_qubit_hamiltonian(axis="z", frequency=frequency)
    initial_state = np.array([1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)], dtype=complex)
    evolution = evolve_single_qubit_state_native(initial_state, hamiltonian, time_values)
    return _dynamics_result(
        workflow="z_precession_native",
        hamiltonian=hamiltonian,
        evolution=evolution,
        initial_state=initial_state,
        metadata={"educational": True, "offline_only": True, "workflow_family": "z_precession"},
    )


def run_rabi_drive_dynamics_native(
    times: Iterable[float] | None = None,
    drive_amplitude: float = 1.0,
) -> SingleQubitDynamicsResult:
    time_values = list(times) if times is not None else default_times()
    hamiltonian = build_single_qubit_hamiltonian(axis="x", frequency=drive_amplitude)
    initial_state = np.array([1.0, 0.0], dtype=complex)
    evolution = evolve_single_qubit_state_native(initial_state, hamiltonian, time_values)
    return _dynamics_result(
        workflow="rabi_drive_dynamics_native",
        hamiltonian=hamiltonian,
        evolution=evolution,
        initial_state=initial_state,
        metadata={"educational": True, "offline_only": True, "workflow_family": "rabi_drive"},
    )


def run_dephasing_metadata_simulation_native(
    times: Iterable[float] | None = None,
    gamma: float = 0.0,
) -> DephasingMetadataResult:
    base = run_z_precession_native(times=times)
    time_values = base.time_points
    damping = [math.exp(-float(gamma) * time) for time in time_values]
    damped_x = [value * damping[index] for index, value in enumerate(base.expectation_values["x"])]
    damped_y = [value * damping[index] for index, value in enumerate(base.expectation_values["y"])]
    expectations = {"x": damped_x, "y": damped_y, "z": list(base.expectation_values["z"])}
    return DephasingMetadataResult(
        workflow="dephasing_metadata_simulation_native",
        mode="native_minimal",
        capability_level=3,
        native_implementation=True,
        production_ready=False,
        hardware_calibration=False,
        input_summary={"gamma": float(gamma), "time_count": len(time_values)},
        data={"damping": damping, "base_workflow": base.workflow},
        time_points=time_values,
        expectation_values=expectations,
        final_state=base.final_state,
        raw_type="NativeDephasingMetadata",
        metadata={"educational": True, "offline_only": True, "full_lindblad_solver": False},
        warnings=dynamics_warnings("Dephasing path is metadata-level damping, not a full Lindblad solver."),
        provenance=native_provenance("dephasing_metadata_simulation_native"),
    )


def dynamics_result_to_dict(result) -> dict[str, object]:
    if hasattr(result, "to_dict"):
        return result.to_dict()
    return {"raw_type": type(result).__name__, "repr": repr(result)}


def _dynamics_result(
    workflow: str,
    hamiltonian: dict[str, object],
    evolution: dict[str, object],
    initial_state: np.ndarray,
    metadata: dict[str, object],
) -> SingleQubitDynamicsResult:
    return SingleQubitDynamicsResult(
        workflow=workflow,
        mode="native_minimal",
        capability_level=3,
        native_implementation=True,
        production_ready=False,
        hardware_calibration=False,
        input_summary={
            "axis": hamiltonian["axis"],
            "frequency": hamiltonian["frequency"],
            "initial_state": serialize_state(initial_state),
            "time_count": len(evolution["times"]),
        },
        data={
            "hamiltonian": hamiltonian["matrix_serialized"],
            "states": serialize_states(evolution["states"]),
        },
        time_points=evolution["times"],
        expectation_values=evolution["expectation_values"],
        final_state=list(evolution["final_state"]),
        raw_type="NativeSingleQubitDynamics",
        metadata=metadata,
        warnings=dynamics_warnings(),
        provenance=native_provenance(workflow),
    )
