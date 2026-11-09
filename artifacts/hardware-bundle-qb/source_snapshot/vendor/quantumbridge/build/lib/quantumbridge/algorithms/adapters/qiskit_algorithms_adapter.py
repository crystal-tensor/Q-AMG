# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
# Design source: docs/roadmap/p2_upstream_integration_strategy_v0.1.md.

from __future__ import annotations

from quantumbridge.results import Result


class QiskitAlgorithmsAdapter:
    mode = "Upstream Passthrough"

    def _require(self):
        try:
            import qiskit_algorithms
        except Exception as exc:
            raise ImportError("QuantumBridge Qiskit Algorithms adapter requires optional dependency 'qiskit-algorithms'.") from exc
        return qiskit_algorithms

    def status(self, algorithm_name: str) -> dict:
        self._require()
        return {
            "algorithm": algorithm_name,
            "status": "adapter-available",
            "mode": self.mode,
            "dependency": "qiskit-algorithms",
        }

    def wrap_result(self, upstream_result, algorithm_name: str) -> Result:
        return Result(
            metadata_data={
                "algorithm": algorithm_name,
                "upstream_result_type": type(upstream_result).__name__,
                "provenance": {"mode": self.mode, "dependency": "qiskit-algorithms"},
            }
        )

    def vqe(self, *args, **kwargs):
        self._require()
        return self.status("VQE")

    def qaoa(self, *args, **kwargs):
        self._require()
        return self.status("QAOA")

    def numpy_minimum_eigensolver(self, *args, **kwargs):
        self._require()
        return self.status("NumPyMinimumEigensolver")

    def numpy_eigensolver(self, *args, **kwargs):
        self._require()
        return self.status("NumPyEigensolver")

    def sampling_vqe(self, *args, **kwargs):
        self._require()
        return self.status("SamplingVQE")

    def vqd(self, *args, **kwargs):
        self._require()
        return self.status("VQD")

    def adapt_vqe(self, *args, **kwargs):
        self._require()
        return self.status("AdaptVQE")

    def grover(self, *args, **kwargs):
        self._require()
        return self.status("Grover")

    def amplitude_estimation(self, *args, **kwargs):
        self._require()
        return self.status("AmplitudeEstimation")

    def optimizer(self, name: str = "optimizer", *args, **kwargs):
        self._require()
        return self.status(name)

    def gradient(self, name: str = "gradient", *args, **kwargs):
        self._require()
        return self.status(name)
