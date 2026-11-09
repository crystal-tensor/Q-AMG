# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is part of QuantumBridge SDK.
# This implementation is developed for the QuantumBridge native architecture.

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any, Optional, Union

from quantumbridge.core import Circuit, Parameter
from quantumbridge.devices import ShotSampler
from quantumbridge.primitives.legacy_containers import BitArray, DataBin, PrimitiveResult, PubResult
from quantumbridge.results import Result
from .v2 import StatevectorSampler as StatevectorSampler


class Sampler:
    def __init__(self, shots: int = 1024, seed: Optional[int] = None, noise_model=None):
        self.shots = shots
        self.seed = seed
        self.noise_model = noise_model

    def run(
        self,
        pubs,
        shots: Optional[int] = None,
        parameter_values: Optional[Union[Mapping[Union[Parameter, str], Real], Sequence[Mapping[Union[Parameter, str], Real]]]] = None,
    ):
        if isinstance(pubs, Circuit):
            return self._run_legacy(pubs, shots=shots, parameters=_single_parameter_mapping(parameter_values))
        return self.run_pubs(pubs, shots=shots, parameter_values=parameter_values)

    def run_pubs(
        self,
        pubs,
        shots: Optional[int] = None,
        parameter_values: Optional[Union[Mapping[Union[Parameter, str], Real], Sequence[Mapping[Union[Parameter, str], Real]]]] = None,
    ) -> PrimitiveResult:
        pub_sequence = _as_pub_sequence(pubs)
        pub_results: list[PubResult] = []
        for index, pub in enumerate(pub_sequence):
            circuit, pub_parameters, pub_shots = _parse_sampler_pub(pub)
            effective_shots = _resolve_shots(pub_shots, shots, self.shots)
            parameter_sets = _resolve_parameter_sets(circuit, pub_parameters, parameter_values)
            counts_batch = []
            quasi_batch = []
            samples_batch = []
            execution_metadata = []
            for params in parameter_sets:
                result = self._run_legacy(circuit, shots=effective_shots, parameters=params)
                if result.metadata().get("dynamic_control_flow"):
                    execution_metadata.append({key: value for key, value in result.metadata().items() if key not in {"memory", "classical_memory"}})
                counts = result.counts() or {}
                num_bits = result.metadata_data.get("num_bits", circuit.num_qubits)
                memory = result.memory()
                samples = BitArray.from_counts(counts, num_bits=num_bits) if memory is None else BitArray(tuple(memory), num_bits)
                counts_batch.append(counts)
                quasi_batch.append({label: count / effective_shots for label, count in counts.items()})
                samples_batch.append(samples)
            data = DataBin(
                counts=counts_batch[0] if len(counts_batch) == 1 else counts_batch,
                quasi_dists=quasi_batch[0] if len(quasi_batch) == 1 else quasi_batch,
                samples=samples_batch[0] if len(samples_batch) == 1 else samples_batch,
                meas=samples_batch[0] if len(samples_batch) == 1 else samples_batch,
                num_shots=effective_shots,
                parameter_values=parameter_sets[0] if len(parameter_sets) == 1 else parameter_sets,
            )
            pub_results.append(
                PubResult(
                    data=data,
                    metadata={
                        "primitive": "SamplerV2",
                        "pub_index": index,
                        "shots": effective_shots,
                        "seed": self.seed,
                        "num_parameter_sets": len(parameter_sets),
                        "circuit_metadata": dict(getattr(circuit, "metadata", {}) or {}),
                        **({"execution_metadata": execution_metadata} if execution_metadata else {}),
                    },
                )
            )
        return PrimitiveResult(
            pub_results,
            metadata={
                "primitive": "SamplerV2",
                "num_pubs": len(pub_results),
                "default_shots": self.shots,
                "native": True,
            },
        )

    def _run_legacy(
        self,
        circuit: Circuit,
        shots: Optional[int] = None,
        parameters: Optional[Mapping[Union[Parameter, str], Real]] = None,
    ) -> Result:
        return ShotSampler(_resolve_shots(None, shots, self.shots), self.seed, noise_model=self.noise_model).run(circuit, parameters=parameters)


def _as_pub_sequence(pubs) -> Sequence[Any]:
    if not isinstance(pubs, (list, tuple)):
        raise TypeError("QuantumBridge Sampler V2 expects a sequence of pubs or a legacy Circuit.")
    if isinstance(pubs, tuple) and pubs and isinstance(pubs[0], Circuit):
        return (pubs,)
    return pubs


def _parse_sampler_pub(pub) -> tuple[Circuit, Optional[Any], Optional[int]]:
    if isinstance(pub, Circuit):
        return pub, None, None
    if isinstance(pub, Mapping):
        circuit = pub.get("circuit")
        if circuit is None:
            circuit = pub.get("quantum_circuit")
        if not isinstance(circuit, Circuit):
            raise TypeError("QuantumBridge sampler pub mapping requires a Circuit under 'circuit'.")
        return circuit, pub.get("parameters", pub.get("parameter_values")), pub.get("shots")
    if isinstance(pub, tuple) and pub:
        circuit = pub[0]
        if not isinstance(circuit, Circuit):
            raise TypeError("QuantumBridge sampler pub tuple must start with a Circuit.")
        parameters = pub[1] if len(pub) >= 2 else None
        shots = pub[2] if len(pub) >= 3 else None
        return circuit, parameters, shots
    raise TypeError("Unsupported QuantumBridge sampler pub.")


def _resolve_shots(pub_shots: Optional[int], run_shots: Optional[int], default_shots: int) -> int:
    shots = pub_shots if pub_shots is not None else run_shots if run_shots is not None else default_shots
    if not isinstance(shots, int) or shots <= 0:
        raise ValueError("QuantumBridge sampler shots must be a positive integer.")
    return shots


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
        raise TypeError("QuantumBridge sampler parameters must be a mapping, sequence of mappings, or numeric array.") from exc


LegacySampler = Sampler


def _parameter_array_to_mappings(circuit: Circuit, values) -> list[Mapping[Union[Parameter, str], Real]]:
    import numpy as np

    params = tuple(circuit.parameters)
    arr = np.asarray(values, dtype=object)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[-1] != len(params):
        raise ValueError(
            "QuantumBridge sampler parameter value rows must match circuit.num_parameters "
            f"({arr.shape[-1]} values for {len(params)} parameters)."
        )
    return [dict(zip(params, row.tolist())) for row in arr.reshape(-1, len(params))]


def _single_parameter_mapping(parameters):
    if parameters is None or isinstance(parameters, Mapping):
        return parameters
    raise TypeError("Legacy QuantumBridge Sampler.run(Circuit) accepts at most one parameter mapping.")
