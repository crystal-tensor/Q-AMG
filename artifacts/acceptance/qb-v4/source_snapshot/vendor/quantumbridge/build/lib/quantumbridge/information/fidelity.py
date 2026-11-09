# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is part of QuantumBridge SDK.
# This implementation is developed for the QuantumBridge native architecture.

import numpy as np


def state_fidelity(a, b) -> float:
    from quantumbridge.information.metrics import state_fidelity_general

    return state_fidelity_general(a, b)
