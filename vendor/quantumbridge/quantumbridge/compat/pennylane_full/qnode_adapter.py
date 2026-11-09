# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
"""PennyLane QNode inventory and passthrough scaffold.

Design source: docs/compat/strategy/pennylane_full_coverage_strategy.md.
"""

from quantumbridge.ecosystem.registry import EcosystemAdapter
from quantumbridge.compat.contracts import CapabilityLevel
from quantumbridge.compat.pennylane_full.result_adapter import wrap_qnode_result as _result_wrap_qnode_result
from quantumbridge.compat.pennylane_full.warnings import adapter_metadata, unsupported

ADAPTER = EcosystemAdapter("pennylane", "pennylane", ("pennylane",), "pennylane-full", "pennylane_qnode")

dependency_available = ADAPTER.dependency_available
get_upstream_version = ADAPTER.get_upstream_version
list_public_api_inventory = ADAPTER.list_public_api_inventory
passthrough_class = ADAPTER.passthrough_class
passthrough_function = ADAPTER.passthrough_function
wrap_result = ADAPTER.wrap_result
to_quantumbridge_schema = ADAPTER.to_quantumbridge_schema
provenance_metadata = ADAPTER.provenance_metadata
warn_unsupported = ADAPTER.warn_unsupported


def describe_qnode(qnode) -> dict:
    return {
        "name": getattr(qnode, "__name__", type(qnode).__name__),
        "callable": callable(qnode),
        "device": repr(getattr(qnode, "device", None)),
        "interface": getattr(qnode, "interface", None),
        "diff_method": getattr(qnode, "diff_method", None),
        "metadata_only": True,
        **adapter_metadata(CapabilityLevel.SCHEMA_ADAPTER),
    }


def run_qnode_passthrough(qnode, *args, **kwargs):
    return _result_wrap_qnode_result(qnode(*args, **kwargs), metadata={"adapter": "qnode_adapter"})


def wrap_qnode_result(raw):
    return _result_wrap_qnode_result(raw, metadata={"adapter": "qnode_adapter"})


def qnode_to_workflow_metadata(qnode) -> dict:
    device = getattr(qnode, "device", None)
    return {
        "name": getattr(qnode, "__name__", type(qnode).__name__),
        "callable": callable(qnode),
        "device": repr(device),
        "device_name": getattr(device, "name", None),
        "shots": _shots_to_metadata(getattr(device, "shots", None)),
        "interface": getattr(qnode, "interface", None),
        "diff_method": getattr(qnode, "diff_method", None),
        "metadata_only": True,
        **adapter_metadata(CapabilityLevel.SCHEMA_ADAPTER),
    }


def _shots_to_metadata(shots):
    if shots is None:
        return None
    total = getattr(shots, "total_shots", None)
    shot_vector = getattr(shots, "shot_vector", None)
    if total is None and shot_vector is None:
        return None if repr(shots) == "Shots(total=None)" else repr(shots)
    return {
        "total_shots": total,
        "shot_vector": repr(shot_vector) if shot_vector else None,
    }


def _unwrap_constructed_tape(constructed):
    if hasattr(constructed, "operations"):
        return constructed
    if isinstance(constructed, tuple) and constructed:
        return _unwrap_constructed_tape(constructed[0])
    if isinstance(constructed, list) and constructed:
        return _unwrap_constructed_tape(constructed[0])
    return constructed


def _get_qnode_tape(qnode, sample_args=None, sample_kwargs=None):
    if sample_args is not None:
        construct = getattr(qnode, "construct", None)
        if construct is not None:
            tape = _unwrap_constructed_tape(construct(tuple(sample_args), dict(sample_kwargs or {})))
            if tape is not None:
                return tape
        try:
            qnode(*tuple(sample_args), **dict(sample_kwargs or {}))
        except Exception:
            return None
    for attr in ("qtape", "_tape", "tape"):
        tape = getattr(qnode, attr, None)
        if tape is not None:
            return tape
    return None


def qnode_to_quantumbridge_ir(qnode, sample_args=None, sample_kwargs=None):
    try:
        tape = _get_qnode_tape(qnode, sample_args=sample_args, sample_kwargs=sample_kwargs)
        if tape is None:
            return unsupported(
                "QNode tape is not introspectable without sample_args; pass sample_args for dynamic QNodes.",
                CapabilityLevel.INVENTORY,
            ).to_dict()
        from .tape_adapter import tape_to_quantumbridge_ir

        ir = tape_to_quantumbridge_ir(tape)
        if isinstance(ir, dict):
            ir.setdefault("qnode", qnode_to_workflow_metadata(qnode))
        return ir
    except Exception as exc:
        return unsupported(f"QNode could not be converted to QuantumBridge IR: {type(exc).__name__}: {exc}").to_dict()
