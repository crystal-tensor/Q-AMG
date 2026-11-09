# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is part of QuantumBridge SDK.
# This implementation is developed for the QuantumBridge native architecture.

"""Native circuit-library builders for reusable QuantumBridge circuits."""

from .arithmetic import integer_comparator, weighted_adder
from .data_preparation import amplitude_encoding, basis_state, initialize
from .standard_gates import standard_gate, standard_gate_definition, standard_gate_object
from quantumbridge.core.standard_gates import (
    RXGate, RYGate, RZGate, PhaseGate, UGate, HGate, XGate, YGate, ZGate,
    CXGate, CYGate, CZGate, CHGate, SwapGate, CSwapGate,
    CRXGate, CRYGate, CRZGate, CPhaseGate, CCXGate, CCZGate, CUGate, UnitaryGate,
    IGate, SGate, SdgGate, TGate, TdgGate, SXGate, SXdgGate, RXXGate, RYYGate, RZZGate,
)
from .templates import efficient_su2, qft, real_amplitudes, two_local, zz_feature_map

__all__ = [
    "RXGate", "RYGate", "RZGate", "PhaseGate", "UGate", "HGate", "XGate", "YGate", "ZGate",
    "CXGate", "CYGate", "CZGate", "CHGate", "SwapGate", "CSwapGate",
    "CRXGate", "CRYGate", "CRZGate", "CPhaseGate", "CCXGate", "CCZGate", "CUGate", "UnitaryGate",
    "IGate", "SGate", "SdgGate", "TGate", "TdgGate", "SXGate", "SXdgGate", "RXXGate", "RYYGate", "RZZGate",
    "amplitude_encoding",
    "basis_state",
    "efficient_su2",
    "initialize",
    "integer_comparator",
    "qft",
    "real_amplitudes",
    "standard_gate",
    "standard_gate_definition",
    "standard_gate_object",
    "two_local",
    "weighted_adder",
    "zz_feature_map",
]
