# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is part of QuantumBridge SDK.
# This implementation is developed for the QuantumBridge native architecture.

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real
from typing import Optional, Union

from quantumbridge.core import Circuit, Parameter
from quantumbridge.primitives.estimator import Estimator
from quantumbridge.primitives.sampler import Sampler
from quantumbridge.results import Result


class BackendSampler(Sampler):
    """Sampler V2 facade backed by a local QuantumBridge Backend."""

    def __init__(self, backend=None, options: Optional[Mapping[str, object]] = None, **run_options) -> None:
        backend = _default_backend() if backend is None else backend
        options_dict = _merged_options(options, run_options)
        shots = options_dict.get("shots", _backend_option(backend, "shots", 1024))
        seed = options_dict.get("seed_simulator", options_dict.get("seed", _backend_option(backend, "seed_simulator", None)))
        super().__init__(shots=1024 if shots is None else shots, seed=None if seed is None else int(seed))
        self.backend = backend
        self.options = options_dict

    def _run_legacy(
        self,
        circuit: Circuit,
        shots: Optional[int] = None,
        parameters: Optional[Mapping[Union[Parameter, str], Real]] = None,
    ) -> Result:
        run_options = {key: value for key, value in self.options.items() if key not in {"shots", "seed", "seed_simulator"}}
        run_options.setdefault("memory", True)
        job = self.backend.run(
            circuit,
            shots=self.shots if shots is None else shots,
            seed_simulator=self.seed,
            parameter_values=parameters,
            **run_options,
        )
        result = job.result()
        result.metadata_data.setdefault("primitive_backend", getattr(self.backend, "name", None))
        return result


class BackendEstimator(Estimator):
    """Estimator V2 facade backed by a local QuantumBridge Backend."""

    def __init__(self, backend=None, options: Optional[Mapping[str, object]] = None, default_precision: Optional[float] = None, **run_options):
        backend = _default_backend() if backend is None else backend
        super().__init__(device=getattr(backend, "device", None), default_precision=default_precision)
        self.backend = backend
        self.options = _merged_options(options, run_options)

    def _run_legacy(
        self,
        circuit: Circuit,
        observable,
        parameters: Optional[Mapping[Union[Parameter, str], Real]] = None,
    ) -> Result:
        job = self.backend.run(
            circuit,
            parameter_values=parameters,
            observable=observable,
            **{key: value for key, value in self.options.items() if key not in {"shots", "seed", "seed_simulator"}},
        )
        result = job.result()
        result.metadata_data.setdefault("primitive_backend", getattr(self.backend, "name", None))
        return result


BackendSamplerV2 = BackendSampler
BackendEstimatorV2 = BackendEstimator


def _default_backend():
    from quantumbridge.providers import Backend

    return Backend()


def _backend_option(backend, name: str, default):
    options = getattr(backend, "options", None)
    if options is None:
        return default
    getter = getattr(options, "get", None)
    if getter is None:
        return default
    return getter(name, default)


def _merged_options(options: Optional[Mapping[str, object]], run_options: Mapping[str, object]) -> dict[str, object]:
    merged = dict(options or {})
    merged.update(dict(run_options))
    return merged
