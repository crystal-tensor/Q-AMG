# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
"""PennyLane measurement metadata adapter."""

from __future__ import annotations

from quantumbridge.compat.contracts import CapabilityLevel

from .measurement_adapter import ADAPTER
from .result_adapter import wrap_measurement_result as _wrap_measurement_result
from .warnings import adapter_metadata, unsupported


MEASUREMENT_NAMES = (
    "expval",
    "probs",
    "sample",
    "counts",
    "state",
    "density_matrix",
    "var",
    "vn_entropy",
    "mutual_info",
)

dependency_available = ADAPTER.dependency_available
get_upstream_version = ADAPTER.get_upstream_version
list_public_api_inventory = ADAPTER.list_public_api_inventory


def list_measurements() -> list[str]:
    return list(MEASUREMENT_NAMES)


def describe_measurement(name: str) -> dict:
    supported = name in MEASUREMENT_NAMES
    return {
        "name": name,
        "measurement_type": name,
        "supported": supported,
        "metadata_only": True,
        "unsupported_reason": None if supported else f"{name!r} is not in the Stage 8C measurement subset.",
        **adapter_metadata(CapabilityLevel.SCHEMA_ADAPTER if supported else CapabilityLevel.INVENTORY),
    }


def _wire_to_python(wire):
    try:
        return int(wire)
    except (TypeError, ValueError):
        return str(wire)


def _wires_to_list(wires) -> list:
    if wires is None:
        return []
    return [_wire_to_python(wire) for wire in wires]


def _measurement_name(obj) -> str:
    cls_name = type(obj).__name__
    class_map = {
        "ExpectationMP": "expval",
        "ProbabilityMP": "probs",
        "SampleMP": "sample",
        "CountsMP": "counts",
        "StateMP": "state",
        "DensityMatrixMP": "density_matrix",
        "VarianceMP": "var",
        "VnEntropyMP": "vn_entropy",
        "MutualInfoMP": "mutual_info",
    }
    if cls_name in class_map:
        return class_map[cls_name]
    if cls_name.endswith("MP"):
        return cls_name[:-2].lower()
    return getattr(obj, "name", cls_name).lower()


def measurement_to_metadata(obj) -> dict:
    measurement_type = _measurement_name(obj)
    observable = getattr(obj, "obs", None)
    wires = _wires_to_list(getattr(obj, "wires", []))
    supported = measurement_type in MEASUREMENT_NAMES
    return {
        "measurement_type": measurement_type,
        "observable": None if observable is None else getattr(observable, "name", type(observable).__name__),
        "wires": wires,
        "shots": getattr(obj, "shots", None),
        "return_type": type(obj).__name__,
        "raw_type": type(obj).__name__,
        "supported": supported,
        "unsupported_reason": None if supported else f"PennyLane measurement {measurement_type!r} is not in the Stage 8C subset.",
        **adapter_metadata(CapabilityLevel.SCHEMA_ADAPTER if supported else CapabilityLevel.INVENTORY),
    }


def wrap_measurement_result(raw):
    return _wrap_measurement_result(raw, metadata={"adapter": "measurements_adapter"})


def measurement_to_quantumbridge_result_fragment(obj) -> dict:
    metadata = measurement_to_metadata(obj)
    if metadata.get("supported") is False:
        return unsupported(metadata["unsupported_reason"]).to_dict()
    return {
        "schema_version": "0.1",
        "ecosystem": "pennylane",
        "measurement": metadata,
        "capability_level": int(CapabilityLevel.SCHEMA_ADAPTER),
        "warnings": metadata["warnings"],
        "provenance": metadata["provenance"],
    }
