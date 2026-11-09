# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is part of QuantumBridge SDK.
# This implementation is developed for the QuantumBridge native architecture.

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .analysis import DepthAnalysisPass, GateCountAnalysisPass, TwoQubitCountAnalysisPass
from .config import PassManagerConfig
from .coupling import CouplingMap
from .decomposition import GateDecompositionPass
from .optimization import CircuitOptimizationPass, RemoveIdentityPass, TwoQubitGateReductionPass
from .pass_manager import PassManager
from .routing import BasicRoutingPass, SwapInsertionPass
from .target import Target
from .limits import CompilationLimits
from .layout import Layout, apply_layout, select_initial_layout
from .pass_base import TransformationPass, AnalysisPass, PassResult
from .scheduling import SchedulingPass
from quantumbridge.core.circuit_metadata import execution_metadata, execution_copy


class _LayoutPass(TransformationPass):
    handles_dynamic = True
    def __init__(self, target, initial_layout): self.target, self.initial_layout = target, initial_layout
    def run(self, circuit):
        layout = select_initial_layout(circuit, self.target) if self.initial_layout == "topology" else Layout.trivial(circuit.num_qubits) if self.initial_layout is None else self.initial_layout
        out = apply_layout(circuit, layout, self.target)
        return PassResult(out, {"initial_layout": execution_metadata(out)["initial_layout"]}, True)


class _ValidateTargetPass(AnalysisPass):
    handles_dynamic = True
    def __init__(self, target): self.target = target
    def run(self, circuit):
        from quantumbridge.core import Measurement
        from quantumbridge.core.classical import UnsupportedFeature
        def walk(body):
            for item in body.iter_instructions():
                node = item.metadata.get("control_node")
                if node is not None:
                    if not self.target.conditional: raise UnsupportedFeature("target_control_flow", "Target does not declare classical control support.")
                    for child in node.bodies: walk(child)
                    continue
                if any(key in item.metadata for key in ("condition", "gate_condition", "control_flow_stack")) and not self.target.conditional:
                    raise UnsupportedFeature("target_conditional")
                name, wires = ("measure", (item.wire,)) if isinstance(item, Measurement) else (item.name, item.wires)
                if not self.target.instruction_supported(name, wires):
                    raise UnsupportedFeature("target_instruction", f"Target does not support {name} on {wires}.")
        walk(circuit)
        initial = execution_metadata(circuit).get("initial_layout", dict(enumerate(range(circuit.num_qubits))))
        physical = execution_metadata(circuit).get("routing", {}).get("final_layout", dict(enumerate(range(circuit.num_qubits))))
        return PassResult(circuit, {"target_validated": True,
                                   "logical_final_layout": {logical: physical[wire] for logical, wire in initial.items()}}, False)


_DEFAULT_BASIS = (
    "id",
    "x",
    "y",
    "z",
    "h",
    "s",
    "sdg",
    "t",
    "tdg",
    "rx",
    "ry",
    "rz",
    "phase",
    "p",
    "u",
    "u1",
    "u2",
    "u3",
    "cx",
    "cy",
    "cz",
    "ch",
    "swap",
    "ccx",
)


def generate_preset_pass_manager(
    optimization_level: int = 1,
    *,
    basis_gates: Sequence[str] | None = None,
    coupling_map: CouplingMap | Sequence[tuple[int, int]] | None = None,
    target: Target | None = None,
    initial_layout=None,
    restore_layout=True,
    scheduling_method=None,
    seed_transpiler=0,
    limits=None,
) -> PassManager:
    """Build a small Qiskit-style preset pass manager.

    The native pipeline is intentionally compact, but it performs real analysis,
    basis decomposition, optional routing diagnostics/swap insertion, and
    optimization passes.
    """

    if isinstance(optimization_level, bool) or not isinstance(optimization_level, int) or optimization_level < 0 or optimization_level > 3:
        raise ValueError("QuantumBridge optimization_level must be an integer in [0, 3].")
    resolved_target = _resolve_target(target, basis_gates, coupling_map)
    limits = limits or CompilationLimits()
    passes = [_LayoutPass(resolved_target, initial_layout)]
    passes.append(GateDecompositionPass(resolved_target.basis_gates,limits=limits))
    if resolved_target.coupling_map is not None:
        passes.append(BasicRoutingPass(resolved_target.coupling_map))
        passes.append(SwapInsertionPass(resolved_target.coupling_map, restore_layout, resolved_target, limits))
        passes.append(GateDecompositionPass(resolved_target.basis_gates,limits=limits))
    if optimization_level >= 1:
        passes.append(RemoveIdentityPass())
    if optimization_level >= 2:
        passes.append(CircuitOptimizationPass())
    if optimization_level >= 3:
        passes.append(TwoQubitGateReductionPass())
        passes.append(CircuitOptimizationPass())
    passes.append(_ValidateTargetPass(resolved_target))
    if scheduling_method is not None: passes.append(SchedulingPass(resolved_target, scheduling_method, limits=limits))
    passes.extend([DepthAnalysisPass(), GateCountAnalysisPass(), TwoQubitCountAnalysisPass()])
    manager = PassManager(passes, limits=limits, seed=seed_transpiler)
    manager.metadata = {
        "optimization_level": optimization_level,
        "basis_gates": list(resolved_target.basis_gates),
        "target": resolved_target.to_dict(),
    }
    return manager


def transpile(
    circuits,
    *,
    basis_gates: Sequence[str] | None = None,
    coupling_map: CouplingMap | Sequence[tuple[int, int]] | None = None,
    optimization_level: int = 1,
    target: Target | None = None,
    return_report: bool = False,
    initial_layout=None,
    restore_layout=True,
    scheduling_method=None,
    seed_transpiler=0,
    limits=None,
):
    """Transpile one circuit or a sequence of circuits with the native pipeline."""

    single = not isinstance(circuits, (list, tuple))
    items = [circuits] if single else list(circuits)
    manager = generate_preset_pass_manager(
        optimization_level=optimization_level,
        basis_gates=basis_gates,
        coupling_map=coupling_map,
        target=target,
        initial_layout=initial_layout,
        restore_layout=restore_layout,
        scheduling_method=scheduling_method,
        seed_transpiler=seed_transpiler,
        limits=limits,
    )
    outputs = []
    reports = []
    for circuit in items:
        result = manager.run(circuit)
        out = execution_copy(result.circuit)
        execution_metadata(out).setdefault("transpile", {})
        execution_metadata(out)["transpile"].update(
            {
                "optimization_level": optimization_level,
                "basis_gates": list(manager.metadata["basis_gates"]),
                "analyses": result.analyses,
                "changed": result.changed,
            }
        )
        outputs.append(out)
        reports.append(result.analyses.get("compiler_report", result.analyses))
    payload = outputs[0] if single else outputs
    if return_report:
        return payload, reports[0] if single else reports
    return payload


native_transpile = transpile


def _resolve_target(target: Target | None, basis_gates, coupling_map) -> Target:
    if target is not None:
        if basis_gates is not None or coupling_map is not None:
            raise ValueError("QuantumBridge transpile target cannot be combined with basis_gates or coupling_map.")
        return target
    normalized_coupling = None
    if coupling_map is not None:
        normalized_coupling = coupling_map if isinstance(coupling_map, CouplingMap) else CouplingMap(coupling_map)
    return Target(num_qubits=0, basis_gates=tuple(basis_gates or _DEFAULT_BASIS), coupling_map=normalized_coupling, allow_channels=True)


def pass_manager_config_from_options(**options: Any) -> PassManagerConfig:
    return PassManagerConfig(
        basis_gates=tuple(options.get("basis_gates") or _DEFAULT_BASIS),
        optimization_level=int(options.get("optimization_level", 1)),
        coupling_map=options.get("coupling_map"),
    )
