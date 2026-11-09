# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is part of QuantumBridge SDK.
# This implementation is developed for the QuantumBridge native architecture.
# Design source: docs/mvp/simulator_design_v0.1.md and docs/mvp/api_contract_v0.1.md.

from __future__ import annotations

from numbers import Real
from typing import Mapping, Optional, Union

import numpy as np

from quantumbridge.core.parameters import Parameter
from quantumbridge.results import Result
from .statevector_device import StatevectorDevice


class ShotSampler:
    """Seeded sampler over exact MVP statevector probabilities."""

    def __init__(self, shots: int, seed: Optional[int] = None, base_device: Optional[StatevectorDevice] = None, noise_model=None, memory: bool = False) -> None:
        if not isinstance(shots, int) or shots <= 0:
            raise ValueError("QuantumBridge sampler shots must be a positive integer.")
        self.shots = shots
        self.seed = seed
        self.base_device = base_device or StatevectorDevice()
        self.noise_model = noise_model
        self.memory = bool(memory)

    def run(self, circuit, parameters: Optional[Mapping[Union[Parameter, str], Real]] = None) -> Result:
        circuit = circuit.lowered()
        device = self.base_device
        if self.noise_model is not None:
            from quantumbridge.noise.execution import _device_with_noise

            device = _device_with_noise(device, self.noise_model)
        if hasattr(device, "run_shots"):
            result = device.run_shots(circuit, shots=self.shots, seed=self.seed, parameters=parameters, memory=self.memory)
            if self.noise_model is not None:
                result.metadata_data["noise_model"] = self.noise_model.summary()
            return result
        probabilities = device.probabilities(circuit, parameters=parameters)
        labels = sorted(probabilities)
        weights = np.array([probabilities[label] for label in labels], dtype=float)
        weights = weights / weights.sum()
        rng = np.random.default_rng(self.seed)
        draws = rng.choice(labels, size=self.shots, p=weights)
        num_bits = circuit.num_qubits
        if circuit.measurements:
            num_bits = circuit.num_bits
            recorded = []
            for draw in draws:
                bits = ["0"] * num_bits
                for measurement in circuit.measurements:
                    bits[measurement.bit] = draw[measurement.wire]
                recorded.append("".join(bits))
            draws = np.asarray(recorded)
            labels = sorted(set(recorded))
        counts = {label: int(np.count_nonzero(draws == label)) for label in labels}
        counts = {label: count for label, count in counts.items() if count}
        empirical = {label: count / self.shots for label, count in counts.items()}
        result = Result(
            probabilities_data=empirical,
            counts_data=counts,
            metadata_data={
                "device": "ShotSampler",
                "shots": self.shots,
                "num_bits": num_bits,
                "seed": self.seed,
                "source": "StatevectorDevice probabilities",
                "qubit_order": "q0-first bitstrings",
                "noise_model": None if self.noise_model is None else self.noise_model.summary(),
            },
        )
        if self.memory:
            result.metadata_data["memory"] = draws.tolist()
        return result
