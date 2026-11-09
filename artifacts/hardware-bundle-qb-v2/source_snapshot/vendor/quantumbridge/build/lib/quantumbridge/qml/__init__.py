# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is part of QuantumBridge SDK.
# This implementation is developed for the QuantumBridge native architecture.

from .embeddings import amplitude_embedding, angle_embedding, basis_embedding
from .subroutines import iqp_embedding, qft, trotter_product
from .measurements import ClassicalShadowResult, MeasurementRequest
from .observables import Hadamard, Hamiltonian, Identity, PauliX, PauliY, PauliZ, SparseHamiltonian
from .qnode import QNode
from .shot_vector import ShotSpec, ShotVector
from .tape import Tape
from .templates import basic_entangler, basic_entangler_layers, hardware_efficient_ansatz, strongly_entangling_layers

NativeQNode = QNode

__all__ = [
    "Hadamard",
    "Hamiltonian",
    "Identity",
    "ClassicalShadowResult",
    "MeasurementRequest",
    "PauliX",
    "PauliY",
    "PauliZ",
    "QNode",
    "NativeQNode",
    "ShotSpec",
    "ShotVector",
    "SparseHamiltonian",
    "Tape",
    "amplitude_embedding",
    "angle_embedding",
    "basis_embedding",
    "iqp_embedding",
    "qft",
    "trotter_product",
    "basic_entangler",
    "basic_entangler_layers",
    "hardware_efficient_ansatz",
    "strongly_entangling_layers",
]
