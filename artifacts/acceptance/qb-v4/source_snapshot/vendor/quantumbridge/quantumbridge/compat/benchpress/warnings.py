# This file is independently implemented for QuantumBridge SDK.
# No source code from IBM, Qiskit, or Benchpress was copied.
"""Warnings and provenance for Stage 10B benchmarking compatibility."""

BENCHPRESS_COMPATIBILITY_WARNING = (
    "QuantumBridge Benchpress compatibility support is a clean-room educational "
    "benchmarking adapter. It is not a full Benchpress replacement and does not "
    "provide production benchmark parity."
)

BENCHMARK_RESULT_WARNING = (
    "QuantumBridge benchmark results are deterministic local validation results "
    "for small educational workloads, not official performance rankings."
)

UPSTREAM_BENCHPRESS_WARNING = (
    "Upstream passthrough requires optional Benchpress ecosystem packages."
)


def default_benchmark_warnings() -> list[str]:
    return [BENCHPRESS_COMPATIBILITY_WARNING, BENCHMARK_RESULT_WARNING]


def default_benchmark_provenance() -> dict[str, object]:
    return {
        "adapter_package": "benchpress",
        "clean_room": True,
        "copied_upstream_source": False,
        "copied_upstream_text": False,
        "official_endorsement": False,
        "official_benchmark": False,
        "production_benchmark_parity": False,
        "cloud_access": False,
        "token_access": False,
        "hardware_access": False,
    }
