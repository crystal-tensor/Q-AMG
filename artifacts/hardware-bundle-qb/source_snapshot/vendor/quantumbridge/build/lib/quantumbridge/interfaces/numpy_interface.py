# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is part of QuantumBridge SDK.
# This implementation is developed for the QuantumBridge native architecture.

import numpy as np


def to_numpy(value, dtype=float):
    return np.asarray(value, dtype=dtype)

