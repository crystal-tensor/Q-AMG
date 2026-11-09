# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
"""Qiskit Aer noise public API inventory, passthrough, and native noise helpers.

Design source: docs/compat/strategy/qiskit_aer_compatibility_strategy.md.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quantumbridge.ecosystem.registry import EcosystemAdapter

ADAPTER = EcosystemAdapter("qiskit-aer", "qiskit-aer", ("qiskit_aer.noise",), "qiskit-aer", "qiskit_aer_noise")

dependency_available = ADAPTER.dependency_available
get_upstream_version = ADAPTER.get_upstream_version
list_public_api_inventory = ADAPTER.list_public_api_inventory
passthrough_class = ADAPTER.passthrough_class
passthrough_function = ADAPTER.passthrough_function
wrap_result = ADAPTER.wrap_result
to_quantumbridge_schema = ADAPTER.to_quantumbridge_schema
provenance_metadata = ADAPTER.provenance_metadata
warn_unsupported = ADAPTER.warn_unsupported


def create_bitflip_noise_model(p: float = 0.0) -> dict[str, Any]:
    """Create a small educational measurement bit-flip noise descriptor."""

    p = float(p)
    _validate_probability(p)
    return {
        "type": "measurement_bitflip",
        "p": p,
        "educational_only": True,
        "qiskit_aer_noise_model_parity": False,
        "production_ready": False,
    }


def create_depolarizing_metadata(p: float = 0.0) -> dict[str, Any]:
    """Create educational depolarizing metadata without claiming Aer parity."""

    p = float(p)
    _validate_probability(p)
    return {
        "type": "depolarizing_metadata",
        "p": p,
        "applied_as": "metadata_only",
        "educational_only": True,
        "qiskit_aer_noise_model_parity": False,
        "production_ready": False,
    }


def validate_noise_model(noise_model: dict[str, Any] | None) -> dict[str, Any]:
    if noise_model is None:
        return create_bitflip_noise_model(0.0)
    data = dict(noise_model)
    kind = str(data.get("type", "measurement_bitflip"))
    if kind not in {"measurement_bitflip", "depolarizing_metadata"}:
        raise ValueError("native Aer noise model type must be measurement_bitflip or depolarizing_metadata")
    p = float(data.get("p", 0.0))
    _validate_probability(p)
    data["p"] = p
    data.setdefault("educational_only", True)
    data.setdefault("qiskit_aer_noise_model_parity", False)
    data.setdefault("production_ready", False)
    return data


def apply_measurement_bitflip_noise(
    counts: dict[str, int],
    p: float,
    shots: int,
    seed: int | None = None,
) -> dict[str, int]:
    """Apply independent bit flips to sampled bitstrings for education demos."""

    p = float(p)
    _validate_probability(p)
    shots = int(shots)
    if shots <= 0:
        raise ValueError("shots must be positive")
    normalized = {str(key): int(value) for key, value in counts.items() if int(value) > 0}
    if p == 0.0:
        return dict(normalized)
    if sum(normalized.values()) != shots:
        raise ValueError("counts total must match shots")
    rng = np.random.default_rng(seed)
    noisy: dict[str, int] = {}
    for bitstring, count in sorted(normalized.items()):
        for _ in range(count):
            bits = ["1" if (rng.random() < p) ^ (bit == "1") else "0" for bit in bitstring]
            label = "".join(bits)
            noisy[label] = noisy.get(label, 0) + 1
    return noisy


def _validate_probability(p: float) -> None:
    if not 0.0 <= p <= 1.0:
        raise ValueError("noise probability p must be between 0 and 1")
