# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is part of QuantumBridge SDK.
# This implementation is developed for the QuantumBridge native architecture.


def to_torch(value, dtype=None):
    try:
        import torch
    except Exception as exc:
        raise ImportError("QuantumBridge Torch interface requires the optional torch package.") from exc
    if dtype is None:
        return torch.tensor(value)
    return torch.tensor(value, dtype=dtype)

