# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is part of QuantumBridge SDK.
# This implementation is developed for the QuantumBridge native architecture.

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
import math

from .coupling import CouplingMap


@dataclass(frozen=True)
class InstructionProperties:
    duration: float | None = None
    error: float | None = None

    def __post_init__(self):
        if self.duration is not None and (isinstance(self.duration, bool) or not isinstance(self.duration, Real) or not math.isfinite(self.duration) or self.duration < 0):
            raise ValueError("Instruction duration must be finite non-negative seconds.")
        if self.error is not None and (isinstance(self.error, bool) or not isinstance(self.error, Real) or not math.isfinite(self.error) or not 0 <= self.error <= 1):
            raise ValueError("Instruction error must be in [0,1].")


class Target:
    num_qubits: int
    basis_gates: tuple[str, ...]
    coupling_map: CouplingMap | None = None
    dt: float | None = None

    def __init__(self, num_qubits: int, basis_gates=(), coupling_map=None, dt: float | None = None,
                 durations=None, pulse_alignment=1, acquire_alignment=1, granularity=1,
                 conditional=True, measure=True, reset=True, allow_channels=False, feedback_latency=None):
        if isinstance(num_qubits, bool) or not isinstance(num_qubits, Integral) or num_qubits < 0: raise ValueError("Target qubit count must be a non-negative integer.")
        if dt is not None and (isinstance(dt, bool) or not isinstance(dt, Real) or not math.isfinite(dt) or dt <= 0): raise ValueError("Target dt must be finite positive seconds.")
        for value in (pulse_alignment, acquire_alignment, granularity):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1: raise ValueError("Target timing alignments must be positive integers.")
        object.__setattr__(self, "num_qubits", int(num_qubits))
        object.__setattr__(self, "basis_gates", tuple(str(gate) for gate in basis_gates))
        if coupling_map is None or isinstance(coupling_map, CouplingMap):
            normalized_coupling = coupling_map
        else:
            normalized_coupling = CouplingMap(coupling_map)
        object.__setattr__(self, "coupling_map", normalized_coupling)
        object.__setattr__(self, "dt", None if dt is None else float(dt))
        self.pulse_alignment, self.acquire_alignment, self.granularity = pulse_alignment, acquire_alignment, granularity
        if any(not isinstance(value, bool) for value in (conditional, measure, reset, allow_channels)): raise TypeError("Target capabilities must be booleans.")
        self.conditional, self.measure, self.reset, self.allow_channels = conditional, measure, reset, allow_channels
        if feedback_latency is not None: InstructionProperties(feedback_latency)
        self.feedback_latency = feedback_latency
        self.instructions = {name: {None: InstructionProperties()} for name in self.basis_gates}
        self.durations = dict(durations or {})
        for value in self.durations.values(): InstructionProperties(value)
        if self.num_qubits and normalized_coupling is not None and any(max(edge) >= self.num_qubits for edge in normalized_coupling.edges):
            raise ValueError("Coupling edge exceeds target capacity.")

    def add_instruction(self, instruction, properties=None, name=None):
        name = name or (instruction if isinstance(instruction, str) else instruction.name)
        if name in self.instructions: raise ValueError("Target instruction name already exists.")
        table = {None: InstructionProperties()} if properties is None else dict(properties)
        normalized = {}
        for wires, value in table.items():
            wires = None if wires is None else tuple(wires)
            if wires is not None and (len(set(wires)) != len(wires) or any(isinstance(wire, bool) or not isinstance(wire, Integral) or wire < 0 or self.num_qubits and wire >= self.num_qubits for wire in wires)):
                raise ValueError("Invalid target instruction qargs.")
            if value is not None and not isinstance(value, InstructionProperties): raise TypeError("Target requires native InstructionProperties.")
            normalized[wires] = value or InstructionProperties()
        self.instructions[name] = normalized
        self.basis_gates = tuple(self.instructions)

    def instruction_supported(self, name, qargs=None):
        if qargs is None: return name in self.instructions or name in {"measure", "reset"} and getattr(self, name)
        wires = tuple(qargs)
        if any(isinstance(wire, bool) or not isinstance(wire, Integral) or wire < 0 or self.num_qubits and wire >= self.num_qubits for wire in wires) or len(set(wires)) != len(wires): return False
        if name in {"barrier", "delay"}: return True
        if name in {"measure", "reset"}:
            table = self.instructions.get(name)
            return getattr(self, name) and len(wires) == 1 and (table is None or None in table or wires in table)
        if name == "kraus": return self.allow_channels
        one = {"id", "identity", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "sx", "sxdg", "rx", "ry", "rz", "phase", "p", "u", "u1", "u2", "u3"}
        two = {"cx", "cy", "cz", "ch", "crx", "cry", "crz", "cp", "cphase", "swap", "rxx", "ryy", "rzz", "iswap", "ecr", "dcx"}
        arity = 1 if name in one else 2 if name in two else 3 if name in {"ccx", "cswap"} else None
        if arity is not None and len(wires) != arity: return False
        table = self.instructions.get(name)
        if table is None or None not in table and wires not in table: return False
        if len(wires) == 2 and self.coupling_map is not None:
            if name in {"cx", "cy", "ch", "crx", "cry", "crz", "cp", "cphase"}: return self.coupling_map.has_edge(*wires)
            return self.coupling_map.is_connected(*wires)
        return True

    def duration(self, name, wires):
        if name == "barrier": return 0.
        if not self.instruction_supported(name, wires): raise ValueError("Instruction is not available on target qargs.")
        table = self.instructions.get(name, {})
        properties = table.get(tuple(wires), table.get(None))
        seconds = None if properties is None else properties.duration
        seconds = self.durations.get((name, tuple(wires)), self.durations.get(name, seconds))
        if seconds is None: raise ValueError(f"Missing instruction duration for {name} on {tuple(wires)}.")
        value = seconds if self.dt is None else seconds / self.dt
        if self.dt is not None and not math.isclose(value / self.granularity, round(value / self.granularity), abs_tol=1e-8):
            raise ValueError("Instruction duration violates target timing granularity.")
        return float(round(value)) if self.dt is not None else float(value)

    @classmethod
    def from_device_spec(cls, spec):
        data = spec.to_dict() if hasattr(spec, "to_dict") else dict(spec)
        capabilities = data.get("capabilities", {})
        return cls(data.get("num_qubits", data.get("n_qubits", 0)), data.get("basis_gates", data.get("gates", ())),
                   data.get("coupling_map", data.get("coupling", ())), dt=data.get("dt"), durations=data.get("durations"),
                   conditional=bool(capabilities.get("conditional", False)), measure=bool(capabilities.get("measure", True)),
                   reset=bool(capabilities.get("reset", False)))

    @classmethod
    def from_backend(cls, backend) -> "Target":
        config = backend.configuration() if hasattr(backend, "configuration") else {}
        num_qubits = getattr(backend, "max_qubits", config.get("n_qubits", config.get("num_qubits", 1)))
        basis_gates = getattr(backend, "basis_gates", config.get("basis_gates", ()))
        coupling = config.get("coupling_map")
        return cls(num_qubits=num_qubits, basis_gates=basis_gates, coupling_map=coupling)

    def supports_operation(self, name: str) -> bool:
        return self.instruction_supported(str(name))

    def is_connected(self, left: int, right: int) -> bool:
        if self.coupling_map is None:
            return True
        return self.coupling_map.is_connected(left, right)

    def to_dict(self) -> dict:
        return {
            "num_qubits": self.num_qubits,
            "basis_gates": list(self.basis_gates),
            "coupling_map": None if self.coupling_map is None else [list(edge) for edge in self.coupling_map.edges],
            "dt": self.dt,
            "instruction_properties": [{"name": name, "qargs": None if wires is None else list(wires),
                                        "duration": props.duration, "error": props.error}
                                       for name, table in self.instructions.items() for wires, props in table.items()],
            "conditional": self.conditional, "measure": self.measure, "reset": self.reset,
            "allow_channels": self.allow_channels, "pulse_alignment": self.pulse_alignment,
            "acquire_alignment": self.acquire_alignment, "granularity": self.granularity,
            "feedback_latency": self.feedback_latency,
        }


NativeTarget = Target
