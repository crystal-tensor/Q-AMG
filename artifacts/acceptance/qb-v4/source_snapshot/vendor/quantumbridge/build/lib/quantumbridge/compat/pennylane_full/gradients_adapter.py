# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
"""PennyLane gradients inventory, passthrough, and native gradient transforms.

Design source: docs/compat/strategy/pennylane_full_coverage_strategy.md.
"""

from quantumbridge.ecosystem.registry import EcosystemAdapter
from quantumbridge.transforms import finite_difference, metric_tensor, qng_step, spsa_gradient

ADAPTER = EcosystemAdapter("pennylane", "pennylane", ("pennylane.gradients",), "pennylane-full", "pennylane_gradients")
NATIVE_GRADIENTS = {
    "finite_diff": finite_difference,
    "finite_difference": finite_difference,
    "spsa": spsa_gradient,
    "spsa_gradient": spsa_gradient,
    "metric_tensor": metric_tensor,
    "qng": qng_step,
    "qng_step": qng_step,
}

dependency_available = ADAPTER.dependency_available
get_upstream_version = ADAPTER.get_upstream_version
list_public_api_inventory = ADAPTER.list_public_api_inventory
passthrough_class = ADAPTER.passthrough_class
passthrough_function = ADAPTER.passthrough_function
wrap_result = ADAPTER.wrap_result
to_quantumbridge_schema = ADAPTER.to_quantumbridge_schema
provenance_metadata = ADAPTER.provenance_metadata
warn_unsupported = ADAPTER.warn_unsupported


def list_native_gradients() -> list[str]:
    return ["finite_difference", "spsa_gradient", "metric_tensor", "qng_step"]


def describe_gradient(name: str) -> dict:
    supported = name in NATIVE_GRADIENTS
    return {
        "name": name,
        "supported": supported,
        "native": supported,
        "unsupported_reason": None if supported else f"{name!r} is not in the current native PennyLane gradient subset.",
        **provenance_metadata(),
    }


def get_gradient_transform(name: str):
    if name in NATIVE_GRADIENTS:
        return NATIVE_GRADIENTS[name]
    return passthrough_function(name)
