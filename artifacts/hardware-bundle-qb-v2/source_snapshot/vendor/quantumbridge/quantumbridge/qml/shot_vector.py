# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.

from dataclasses import dataclass
from numbers import Integral
import numpy as np


@dataclass(frozen=True)
class ShotSpec:
    partitions: tuple
    is_vector: bool = False

    @classmethod
    def parse(cls, shots):
        if isinstance(shots, cls): return shots
        def positive(value):
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError("QuantumBridge shots and copies must be positive integers.")
            return int(value)
        if shots is None: return cls((None,))
        if isinstance(shots, Integral): return cls((positive(shots),))
        if not isinstance(shots, (list, tuple)) or not shots: raise ValueError("Invalid QuantumBridge shot vector.")
        partitions = []
        for item in shots:
            if isinstance(item, tuple) and len(item) == 2:
                count, copies = map(positive, item)
                if copies > 10000: raise ValueError("Shot copies exceed the partition budget.")
                partitions.extend([count] * copies)
            else: partitions.append(positive(item))
            if len(partitions) > 10000: raise ValueError("Shot vector exceeds the partition budget.")
        return cls(tuple(partitions), True)

    def seeds(self, seed):
        if not self.is_vector: return (seed,)
        return tuple(int(child.generate_state(1)[0]) for child in np.random.SeedSequence(seed).spawn(len(self.partitions)))


ShotVector = ShotSpec
