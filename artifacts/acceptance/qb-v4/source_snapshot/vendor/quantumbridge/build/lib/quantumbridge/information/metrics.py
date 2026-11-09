# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is part of QuantumBridge SDK.
# This implementation is developed for the QuantumBridge native architecture.

from __future__ import annotations

import numpy as np


def _as_matrix(obj) -> np.ndarray:
    return np.asarray(obj.data if hasattr(obj, "data") else obj, dtype=complex)


def _matrix_sqrt_psd(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh((matrix + matrix.conj().T) / 2)
    values = np.clip(values.real, 0.0, None)
    return vectors @ np.diag(np.sqrt(values)) @ vectors.conj().T


def purity(state) -> float:
    matrix = _as_matrix(state)
    if matrix.ndim == 1:
        return float(np.real_if_close(np.vdot(matrix, matrix) ** 2))
    return float(np.real_if_close(np.trace(matrix @ matrix)))


def trace_distance(a, b) -> float:
    delta = _as_matrix(a) - _as_matrix(b)
    singular_values = np.linalg.svd(delta, compute_uv=False)
    return float(0.5 * np.sum(singular_values))


def hellinger_fidelity(a: dict[str, float], b: dict[str, float]) -> float:
    keys = set(a) | set(b)
    value = sum(np.sqrt(float(a.get(key, 0.0)) * float(b.get(key, 0.0))) for key in keys)
    return float(value * value)


def hellinger_distance(a: dict[str, float], b: dict[str, float]) -> float:
    fidelity = np.clip(hellinger_fidelity(a, b), 0.0, 1.0)
    return float(np.sqrt(1.0 - np.sqrt(fidelity)))


def _as_choi(channel):
    from quantumbridge.information.channels import _QuantumChannel, Choi, Kraus
    from quantumbridge.information.operator import Operator

    if isinstance(channel, _QuantumChannel):
        return Choi(channel)
    operator = channel if isinstance(channel, Operator) else Operator(channel)
    return Choi(Kraus(operator))


def process_fidelity(
    channel,
    target=None,
    require_cp: bool = True,
    require_tp: bool = True,
) -> float:
    source = _as_choi(channel)
    if source._input_dim != source._output_dim:
        raise ValueError("QuantumBridge process_fidelity requires a square channel.")
    if require_cp and not source.is_cp():
        raise ValueError("QuantumBridge process_fidelity requires a completely positive channel.")
    if require_tp and not source.is_tp():
        raise ValueError("QuantumBridge process_fidelity requires a trace-preserving channel.")

    if target is None:
        from quantumbridge.information.operator import Operator

        target = Operator(np.eye(source._input_dim, dtype=complex))
    reference = _as_choi(target)
    if source.dim != reference.dim:
        raise ValueError("QuantumBridge process_fidelity requires equal channel dimensions.")
    if require_cp and not reference.is_cp():
        raise ValueError("QuantumBridge process_fidelity target must be completely positive.")
    if require_tp and not reference.is_tp():
        raise ValueError("QuantumBridge process_fidelity target must be trace preserving.")

    dim = source._input_dim
    reference_values = np.linalg.eigvalsh((reference.data + reference.data.conj().T) / 2)
    if np.count_nonzero(reference_values > 1e-10) == 1:
        value = np.real_if_close(np.trace(source.data @ reference.data) / (dim * dim))
        return float(np.clip(np.real(value), 0.0, 1.0))
    return state_fidelity_general(source.data / dim, reference.data / dim)


def average_gate_fidelity(
    channel,
    target=None,
    require_cp: bool = True,
    require_tp: bool = False,
) -> float:
    source = _as_choi(channel)
    dim = source._input_dim
    fidelity = process_fidelity(
        source,
        target,
        require_cp=require_cp,
        require_tp=require_tp,
    )
    return float((dim * fidelity + 1) / (dim + 1))


def gate_error(
    channel,
    target=None,
    require_cp: bool = True,
    require_tp: bool = False,
) -> float:
    return float(
        1
        - average_gate_fidelity(
            channel,
            target,
            require_cp=require_cp,
            require_tp=require_tp,
        )
    )


def state_fidelity_general(a, b) -> float:
    from quantumbridge.information.density_matrix import DensityMatrix
    from quantumbridge.information.statevector import Statevector

    if isinstance(a, Statevector) and isinstance(b, Statevector):
        return float(abs(np.vdot(a.data, b.data)) ** 2)
    a_matrix = _as_matrix(a)
    b_matrix = _as_matrix(b)
    if a_matrix.ndim == 1 and b_matrix.ndim == 1:
        return float(abs(np.vdot(a_matrix, b_matrix)) ** 2)
    rho = DensityMatrix.from_statevector(a_matrix).data if a_matrix.ndim == 1 else a_matrix
    sigma = DensityMatrix.from_statevector(b_matrix).data if b_matrix.ndim == 1 else b_matrix
    root = _matrix_sqrt_psd(rho)
    inner = root @ sigma @ root
    return float(np.real_if_close(np.trace(_matrix_sqrt_psd(inner)) ** 2))
