# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is part of QuantumBridge SDK.
# This implementation is developed for the QuantumBridge native architecture.

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any, Optional, Union

import numpy as np

from quantumbridge.core import Circuit, Parameter
from quantumbridge.devices import StatevectorDevice
from quantumbridge.primitives.legacy_containers import DataBin, PrimitiveResult, PubResult
from quantumbridge.results import Result
from .v2 import StatevectorEstimator as StatevectorEstimator


class Estimator:
    def __init__(self, device: Optional[StatevectorDevice] = None, default_precision: Optional[float] = None):
        self.device = device or StatevectorDevice()
        self.default_precision = default_precision

    def run(
        self,
        pubs,
        observable=None,
        precision: Optional[float] = None,
        parameter_values: Optional[Union[Mapping[Union[Parameter, str], Real], Sequence[Mapping[Union[Parameter, str], Real]]]] = None,
    ):
        if observable is not None and isinstance(pubs, Circuit):
            return self._run_legacy(pubs, observable, parameters=_single_parameter_mapping(parameter_values))
        return self.run_pubs(pubs, precision=precision, parameter_values=parameter_values)

    def run_pubs(
        self,
        pubs,
        precision: Optional[float] = None,
        parameter_values: Optional[Union[Mapping[Union[Parameter, str], Real], Sequence[Mapping[Union[Parameter, str], Real]]]] = None,
    ) -> PrimitiveResult:
        pub_sequence = _as_pub_sequence(pubs)
        pub_results: list[PubResult] = []
        for index, pub in enumerate(pub_sequence):
            circuit, observables, pub_parameters, pub_precision = _parse_estimator_pub(pub)
            effective_precision = _resolve_precision(pub_precision, precision, self.default_precision)
            parameter_sets = _resolve_parameter_sets(circuit, pub_parameters, parameter_values)
            observable_values, observable_shape = _flatten_observables(observables)
            ev_batches = []
            std_batches = []
            mps_statistics = []
            for params in parameter_sets:
                evaluations = [self._run_legacy(circuit, observable, parameters=params) for observable in observable_values]
                evs = [float(result.expectation_value()) for result in evaluations]
                statistics = [{key: result.metadata()[key] for key in ("max_bond_observed", "max_discarded_weight", "max_bond_dim", "cutoff") if key in result.metadata()}
                              for result in evaluations if result.metadata().get("simulation") == "matrix_product_state"]
                if statistics:
                    mps_statistics.append(statistics)
                shaped_evs = _restore_shape(list(evs), observable_shape)
                shaped_stds = _restore_shape([0.0 for _ in evs], observable_shape)
                ev_batches.append(shaped_evs)
                std_batches.append(shaped_stds)
            data = DataBin(
                evs=ev_batches[0] if len(ev_batches) == 1 else ev_batches,
                stds=std_batches[0] if len(std_batches) == 1 else std_batches,
                parameter_values=parameter_sets[0] if len(parameter_sets) == 1 else parameter_sets,
                num_observables=len(observable_values),
                observables=observables,
                observable_shape=observable_shape,
            )
            pub_results.append(
                PubResult(
                    data=data,
                    metadata={
                        "primitive": "EstimatorV2",
                        "pub_index": index,
                        "precision": effective_precision,
                        "num_parameter_sets": len(parameter_sets),
                        "num_observables": len(observable_values),
                        "circuit_metadata": dict(getattr(circuit, "metadata", {}) or {}),
                        "simulation": evaluations[0].metadata().get("simulation", getattr(self.device, "simulation", "exact_statevector")) if evaluations else getattr(self.device, "simulation", "exact_statevector"),
                        **({"mps_truncation": mps_statistics} if mps_statistics else {}),
                    },
                )
            )
        return PrimitiveResult(
            pub_results,
            metadata={
                "primitive": "EstimatorV2",
                "num_pubs": len(pub_results),
                "default_precision": self.default_precision,
                "native": True,
            },
        )

    def _run_legacy(
        self,
        circuit: Circuit,
        observable,
        parameters: Optional[Mapping[Union[Parameter, str], Real]] = None,
    ) -> Result:
        value = self.device.expectation(circuit, observable, parameters=parameters)
        return Result(expectation_data=value, metadata_data={"primitive": "Estimator"})


def _as_pub_sequence(pubs) -> Sequence[Any]:
    if not isinstance(pubs, (list, tuple)):
        raise TypeError("QuantumBridge Estimator V2 expects a sequence of pubs or legacy Circuit plus observable.")
    if isinstance(pubs, tuple) and pubs and isinstance(pubs[0], Circuit):
        return (pubs,)
    return pubs


def _parse_estimator_pub(pub) -> tuple[Circuit, Any, Optional[Any], Optional[float]]:
    if isinstance(pub, Mapping):
        circuit = pub.get("circuit")
        observable = pub.get("observable", pub.get("observables"))
        if not isinstance(circuit, Circuit):
            raise TypeError("QuantumBridge estimator pub mapping requires a Circuit under 'circuit'.")
        if observable is None:
            raise TypeError("QuantumBridge estimator pub mapping requires 'observable' or 'observables'.")
        return circuit, observable, pub.get("parameters", pub.get("parameter_values")), pub.get("precision")
    if isinstance(pub, tuple) and len(pub) >= 2:
        circuit = pub[0]
        if not isinstance(circuit, Circuit):
            raise TypeError("QuantumBridge estimator pub tuple must start with a Circuit.")
        observable = pub[1]
        parameters = pub[2] if len(pub) >= 3 else None
        precision = pub[3] if len(pub) >= 4 else None
        return circuit, observable, parameters, precision
    raise TypeError("Unsupported QuantumBridge estimator pub.")


def _flatten_observables(observables) -> tuple[list[Any], Any]:
    if isinstance(observables, np.ndarray):
        return [observables], ()
    if isinstance(observables, (list, tuple)):
        if not observables:
            raise ValueError("QuantumBridge estimator pubs require at least one observable.")
        flat: list[Any] = []
        shape = _collect_observables(observables, flat)
        return flat, shape
    return [observables], ()


def _collect_observables(value, flat: list[Any]):
    if isinstance(value, np.ndarray):
        flat.append(value)
        return ()
    if isinstance(value, (list, tuple)):
        shape = []
        for item in value:
            shape.append(_collect_observables(item, flat))
        return tuple(shape)
    flat.append(value)
    return ()


def _restore_shape(values: list[float], shape):
    if shape == ():
        return values.pop(0)
    return [_restore_shape(values, item) for item in shape]


def _resolve_parameter_sets(circuit: Circuit, pub_parameters, run_parameters) -> list[Mapping[Union[Parameter, str], Real]]:
    parameters = pub_parameters if pub_parameters is not None else run_parameters
    if parameters is None:
        return [{}]
    if isinstance(parameters, Mapping):
        return [dict(parameters)]
    if isinstance(parameters, Sequence) and not isinstance(parameters, (str, bytes)):
        if not parameters:
            return [{}]
        if all(isinstance(item, Mapping) for item in parameters):
            return [dict(item) for item in parameters]
        return _parameter_array_to_mappings(circuit, parameters)
    try:
        return _parameter_array_to_mappings(circuit, parameters)
    except Exception as exc:
        raise TypeError("QuantumBridge estimator parameters must be a mapping, sequence of mappings, or numeric array.") from exc


LegacyEstimator = Estimator


def _parameter_array_to_mappings(circuit: Circuit, values) -> list[Mapping[Union[Parameter, str], Real]]:
    params = tuple(circuit.parameters)
    arr = np.asarray(values, dtype=object)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[-1] != len(params):
        raise ValueError(
            "QuantumBridge estimator parameter value rows must match circuit.num_parameters "
            f"({arr.shape[-1]} values for {len(params)} parameters)."
        )
    return [dict(zip(params, row.tolist())) for row in arr.reshape(-1, len(params))]


def _resolve_precision(pub_precision: Optional[float], run_precision: Optional[float], default_precision: Optional[float]) -> Optional[float]:
    precision = pub_precision if pub_precision is not None else run_precision if run_precision is not None else default_precision
    if precision is not None and (isinstance(precision, bool) or not isinstance(precision, Real) or not np.isfinite(precision) or precision < 0):
        raise ValueError("QuantumBridge estimator precision must be finite and non-negative when provided.")
    return None if precision is None else float(precision)


def _single_parameter_mapping(parameters):
    if parameters is None or isinstance(parameters, Mapping):
        return parameters
    raise TypeError("Legacy QuantumBridge Estimator.run(Circuit, observable) accepts at most one parameter mapping.")
