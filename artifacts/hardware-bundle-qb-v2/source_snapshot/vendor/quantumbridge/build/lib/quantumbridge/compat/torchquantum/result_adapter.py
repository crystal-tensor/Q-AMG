# This file is independently implemented for QuantumBridge SDK.
# No source code from TorchQuantum or PyTorch was copied.
"""Result wrapping helpers for TorchQuantum-style compatibility."""

from __future__ import annotations

from typing import Any

from quantumbridge.schema.torchquantum_results import TorchQuantumCompatibilityResult

from .warnings import torchquantum_warnings, native_provenance


def wrap_torchquantum_result(raw: Any, workflow: str = "torchquantum_result_wrapper") -> TorchQuantumCompatibilityResult:
    if isinstance(raw, TorchQuantumCompatibilityResult):
        return raw
    return TorchQuantumCompatibilityResult(
        workflow=workflow,
        mode="native_minimal",
        capability_level=2,
        production_ready=False,
        native_implementation=True,
        raw_type=type(raw).__name__,
        metadata={"value": raw if isinstance(raw, (str, int, float, bool, type(None))) else repr(raw)[:200]},
        warnings=torchquantum_warnings(),
        provenance=native_provenance(workflow),
    )


def to_quantumbridge_schema(result: Any) -> dict[str, Any]:
    return wrap_torchquantum_result(result).to_dict()
