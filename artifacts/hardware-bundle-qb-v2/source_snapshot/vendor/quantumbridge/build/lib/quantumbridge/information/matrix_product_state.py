# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# Independently implemented from tensor contraction and Schmidt decomposition.

from __future__ import annotations

from numbers import Integral, Real

import numpy as np

from quantumbridge.utils.math import gate_matrix, pauli_matrix
from quantumbridge.execution import mps_arrays as arrays


def mps_routing(wires, width):
    """Gather operands in matrix order and restore the physical q0-first chain."""
    selected = tuple(wires)
    if not selected or len(set(selected)) != len(selected) or any(
            type(wire) is not int or not 0 <= wire < width for wire in selected):
        raise ValueError("MPS routing requires distinct valid physical wires.")
    labels, swaps, start = list(range(width)), [], min(selected)
    for offset, wire in enumerate(selected):
        position = labels.index(wire)
        while position > start+offset:
            labels[position-1], labels[position] = labels[position], labels[position-1]
            swaps.append(position-1)
            position -= 1
    return start, tuple(swaps)


class MatrixProductState:
    """Open-boundary qubit MPS in q0-first order with an orthogonality center.

    cutoff bounds discarded squared Schmidt weight per split, not the final
    simulation error. max_bond_dim may force additional, reported truncation.
    """

    def __init__(self, tensors, *, max_bond_dim=None, cutoff=1e-16, max_memory_mb=1024):
        if max_bond_dim is not None and (isinstance(max_bond_dim, bool) or not isinstance(max_bond_dim, Integral) or max_bond_dim < 1):
            raise ValueError("QuantumBridge max_bond_dim must be a positive integer or None.")
        if isinstance(cutoff, bool) or not isinstance(cutoff, Real) or not np.isfinite(cutoff) or not 0 <= cutoff < 1:
            raise ValueError("QuantumBridge MPS cutoff must be in [0, 1).")
        if max_memory_mb is not None and (isinstance(max_memory_mb, bool) or not isinstance(max_memory_mb, Real) or not np.isfinite(max_memory_mb) or max_memory_mb <= 0):
            raise ValueError("QuantumBridge max_memory_mb must be positive or None.")
        self.max_bond_dim = None if max_bond_dim is None else int(max_bond_dim)
        self.cutoff = float(cutoff)
        self.max_memory_mb = None if max_memory_mb is None else float(max_memory_mb)
        self.discarded_weight = 0.0
        self.truncations = 0
        self._tensors = [arrays.numeric(tensor) for tensor in tensors]
        if not self._tensors:
            raise ValueError("QuantumBridge MPS needs at least one qubit tensor.")
        previous = 1
        for tensor in self._tensors:
            if tensor.ndim != 3 or tensor.shape[0] != previous or tensor.shape[1] != 2 or min(tensor.shape) < 1 or not arrays.finite(tensor):
                raise ValueError("QuantumBridge MPS tensors require finite connected (left, 2, right) shapes.")
            previous = tensor.shape[2]
        if previous != 1:
            raise ValueError("QuantumBridge MPS boundary bonds must have dimension one.")
        self._guard(sum(tensor.size for tensor in self._tensors) * 3)
        self._tensors = [arrays.copied(tensor) for tensor in self._tensors]
        self.peak_bond_dimension = max(self.bond_dimensions, default=1)
        self._center = len(self._tensors) - 1
        # A left-to-right QR sweep canonicalizes arbitrary supplied tensors.
        for index in range(len(self._tensors) - 1):
            tensor = self._tensors[index]
            q, r = arrays.decomposed(arrays.reshaped(tensor, (tensor.shape[0]*2, tensor.shape[2])), "qr")
            updated = arrays.contracted(r, self._tensors[index + 1])
            self._tensors[index] = arrays.reshaped(q, (tensor.shape[0], 2, q.shape[1]))
            self._tensors[index + 1] = updated
        if not np.isclose(arrays.norm_squared(self._tensors[-1]), 1, atol=1e-10, rtol=0):
            raise ValueError("QuantumBridge MPS tensors must describe a normalized state.")
        if max_bond_dim is not None and max(self.bond_dimensions, default=1) > max_bond_dim:
            raise ValueError("Supplied MPS bonds exceed max_bond_dim; explicitly compress the initial state first.")

    @classmethod
    def zero(cls, num_qubits, **options):
        if isinstance(num_qubits, bool) or not isinstance(num_qubits, Integral) or num_qubits < 1:
            raise ValueError("QuantumBridge MPS width must be a positive integer.")
        def tensor():
            result = arrays.owned("mps_zero_input", (1, 2, 1), complex)
            result.fill(0)
            result[0, 0, 0] = 1
            return result
        return cls([tensor() for _ in range(num_qubits)], **options)

    @classmethod
    def from_statevector(cls, state, **options):
        data = arrays.numeric(state if isinstance(state, np.ndarray) else getattr(state, "data", state))
        if data.ndim != 1 or data.size < 2 or data.size & (data.size - 1) or not arrays.finite(data) or not np.isclose(arrays.norm_squared(data), 1, atol=1e-10, rtol=0):
            raise ValueError("QuantumBridge MPS initial vector must be finite, normalized and have qubit dimension.")
        result = cls.zero(data.size.bit_length() - 1, **options)
        result._guard(data.size * 8)
        result._split(0, result.num_qubits, arrays.reshaped(data, (1, data.size, 1)))
        return result

    @property
    def num_qubits(self):
        return len(self._tensors)

    @property
    def tensors(self):
        return tuple(arrays.copied(tensor) for tensor in self._tensors)

    @property
    def tensor_bytes(self):
        return sum(tensor.nbytes for tensor in self._tensors)

    @property
    def bond_dimensions(self):
        return tuple(tensor.shape[2] for tensor in self._tensors[:-1])

    def copy(self):
        result = object.__new__(MatrixProductState)
        result.__dict__ = dict(self.__dict__)
        result._tensors = arrays.copied_tensors(self._tensors)
        return result

    def _guard(self, elements):
        if self.max_memory_mb is not None and 16 * elements > self.max_memory_mb * 1024 ** 2:
            raise MemoryError("QuantumBridge MPS contraction/output exceeds max_memory_mb.")

    def _wires(self, wires):
        selected = tuple(range(self.num_qubits) if wires is None else wires)
        if len(set(selected)) != len(selected) or any(isinstance(wire, bool) or not isinstance(wire, Integral) or not 0 <= wire < self.num_qubits for wire in selected):
            raise ValueError("QuantumBridge MPS wires must be distinct valid integer wires.")
        return selected

    def _move_center(self, target):
        while self._center < target:
            index = self._center
            tensor = self._tensors[index]
            q, r = arrays.decomposed(arrays.reshaped(tensor, (tensor.shape[0]*2, tensor.shape[2])), "qr")
            updated = arrays.contracted(r, self._tensors[index + 1])
            self._tensors[index] = arrays.reshaped(q, (tensor.shape[0], 2, q.shape[1]))
            self._tensors[index + 1] = updated
            self._center += 1
        while self._center > target:
            index = self._center
            tensor = self._tensors[index]
            q, r = arrays.decomposed(arrays.reshaped(tensor, (tensor.shape[0], 2*tensor.shape[2])).T, "qr")
            updated = arrays.contracted(self._tensors[index - 1], r.T)
            self._tensors[index] = arrays.reshaped(q.T, (q.shape[1], 2, tensor.shape[2]))
            self._tensors[index - 1] = updated
            self._center -= 1

    def _split(self, start, width, block, *, _tangent_factors=None):
        left, right = block.shape[0], block.shape[-1]
        tensors = list(self._tensors)
        peak, discarded_weight, truncations = self.peak_bond_dimension, self.discarded_weight, self.truncations
        for offset in range(width - 1):
            matrix = arrays.reshaped(block, (left*2, block.size//(left*2)))
            self._guard(matrix.size * 8 + sum(tensor.size for tensor in self._tensors))
            u, singular, vh = arrays.decomposed(matrix, "svd")
            squared = arrays.owned("mps_schmidt_squared", singular.shape, singular.dtype)
            np.multiply(singular, singular, out=squared)
            total = float(np.sum(squared))
            tails = arrays.owned("mps_schmidt_tails", singular.shape, singular.dtype)
            np.cumsum(squared[::-1], out=tails)
            tails = tails[::-1]
            rank = len(singular)
            if _tangent_factors is None:
                for candidate in range(1, len(singular)):
                    if tails[candidate] <= self.cutoff * total:
                        rank = candidate
                        break
                if self.max_bond_dim is not None:
                    rank = min(rank, self.max_bond_dim)
            else:
                # Tangents must not inherit an approximate simulation cutoff.
                # Only numerical-null Schmidt values may be removed; their
                # discarded weight is still recorded in the existing receipt.
                floor = 8 * np.finfo(float).eps * max(matrix.shape) * singular[0]
                rank = max(1, sum(float(value) > floor for value in singular))
                if self.max_bond_dim is not None and rank > self.max_bond_dim:
                    raise MemoryError("MPS tangent numerical rank exceeds max_bond_dim; refusing derivative truncation.")
            peak = max(peak, rank)
            discarded = float(np.sum(squared[rank:]) / total) if total else 0.0
            if discarded:
                discarded_weight += discarded
                truncations += 1
            retained = float(np.sum(squared[:rank]))
            if _tangent_factors is not None:
                _tangent_factors.append(float(np.sqrt(retained)))
            tensors[start + offset] = arrays.reshaped(u[:, :rank], (left, 2, rank))
            block = arrays.owned("mps_schmidt_next_block", (rank, vh.shape[1]), vh.dtype)
            np.multiply(singular[:rank, None], vh[:rank], out=block)
            np.divide(block, np.sqrt(retained), out=block)
            left = rank
            del matrix, u, singular, vh, squared, tails
        tensors[start + width - 1] = arrays.reshaped(block, (left, 2, right))
        self._tensors, self._center = tensors, start + width - 1
        self.peak_bond_dimension, self.discarded_weight, self.truncations = peak, discarded_weight, truncations

    def _block_gate(self, matrix, start, width, *, _tangent_factors=None):
        self._move_center(start)
        left = self._tensors[start].shape[0]
        right = self._tensors[start + width - 1].shape[2]
        self._guard(left * (1 << width) * right * 10 + sum(tensor.size for tensor in self._tensors))
        block = self._tensors[start]
        for tensor in self._tensors[start + 1:start + width]:
            block = arrays.contracted(block, tensor)
        reshaped = arrays.reshaped(block, (left, 1 << width, right))
        block = arrays.einsum("ij,ajb->aib", reshaped.shape, matrix, reshaped)
        if _tangent_factors is not None:
            scale = arrays.component_scale(block)
            _tangent_factors.append(scale)
            if scale == 0:
                return False
            block = arrays.rescaled(block, scale)
        norm = arrays.norm_squared(block)
        if norm <= 0:
            raise ValueError("QuantumBridge cannot normalize a zero MPS branch.")
        if _tangent_factors is None:
            self._split(start, width, arrays.divided(block, np.sqrt(norm)))
        else:
            _tangent_factors.append(float(np.sqrt(norm)))
            self._split(start, width, arrays.divided(block, np.sqrt(norm)),
                        _tangent_factors=_tangent_factors)
        return True

    def apply(self, matrix, wires):
        """Apply a supplied local matrix, normalizing the resulting pure branch."""
        original = (self._tensors, self._center, self.peak_bond_dimension, self.discarded_weight, self.truncations)
        try:
            self._tensors = arrays.adopt_for_mutation(self._tensors)
            return self._apply(matrix, wires)
        except BaseException:
            self._tensors, self._center, self.peak_bond_dimension, self.discarded_weight, self.truncations = original
            raise

    def _apply(self, matrix, wires, *, _tangent_factors=None):
        selected = self._wires(wires)
        if not selected:
            raise ValueError("QuantumBridge MPS matrix application requires wires.")
        matrix = arrays.numeric(matrix)
        dimension = 1 << len(selected)
        if matrix.shape != (dimension, dimension) or not arrays.finite(matrix):
            raise ValueError("QuantumBridge MPS matrix dimensions must match its wires.")
        def block_gate(matrix, start, width):
            if _tangent_factors is None:
                return self._block_gate(matrix, start, width)
            return self._block_gate(matrix, start, width, _tangent_factors=_tangent_factors)
        if len(selected) == 1:
            block_gate(matrix, selected[0], 1)
            return self
        # Gather logical wires in the matrix's order, then restore physical order.
        start, swaps = mps_routing(tuple(int(wire) for wire in selected), self.num_qubits)
        for position in swaps:
            block_gate(gate_matrix("swap"), position, 2)
        if block_gate(matrix, start, len(selected)) is False:
            return self
        for position in reversed(swaps):
            block_gate(gate_matrix("swap"), position, 2)
        return self

    def statevector(self):
        self._guard((1 << self.num_qubits) * 4)
        block = self._tensors[0]
        for tensor in self._tensors[1:]:
            self._guard(block.size * tensor.shape[2] * 4)
            block = arrays.contracted(block, tensor)
        return arrays.copied(arrays.reshaped(block, (block.size,)), category="mps_statevector_output")

    def reduced_density_matrix(self, wires=None):
        selected = self._wires(wires)
        env = arrays.filled("mps_density_initial", (1, 1, 1, 1), complex, 1)
        kept = 0
        for index, tensor in enumerate(self._tensors):
            next_kept = kept + int(index in selected)
            self._guard((tensor.shape[0] ** 2 + tensor.shape[2] ** 2) * (1 << (2 * next_kept)) * 8)
            conjugate = arrays.conjugated(tensor)
            if index in selected:
                env = arrays.einsum("abij,asr,btv->rvisjt",
                    (tensor.shape[2], tensor.shape[2], 1 << kept, 2, 1 << kept, 2), env, tensor, conjugate)
                env = arrays.reshaped(env, (tensor.shape[2], tensor.shape[2], 1 << next_kept, 1 << next_kept))
            else:
                env = arrays.einsum("abij,asr,bst->rtij",
                    (tensor.shape[2], tensor.shape[2], 1 << kept, 1 << kept), env, tensor, conjugate)
            kept = next_kept
        matrix = env[0, 0]
        order = [sorted(selected).index(wire) for wire in selected]
        axes = order + [axis + len(order) for axis in order]
        permuted = arrays.reshaped(matrix, (2,) * (2 * kept)).transpose(axes)
        return arrays.copied(arrays.reshaped(permuted, (1 << kept, 1 << kept)), category="mps_density_output")

    def probabilities(self, wires=None):
        selected = self._wires(wires)
        env, kept = arrays.filled("mps_probability_initial", (1, 1, 1), complex, 1), 0
        for index, tensor in enumerate(self._tensors):
            next_kept = kept + int(index in selected)
            self._guard((tensor.shape[0] ** 2 + tensor.shape[2] ** 2) * (1 << next_kept) * 8)
            conjugate = arrays.conjugated(tensor)
            if index in selected:
                env = arrays.einsum("abi,asr,bst->rtis",
                    (tensor.shape[2], tensor.shape[2], 1 << kept, 2), env, tensor, conjugate)
                env = arrays.reshaped(env, (tensor.shape[2], tensor.shape[2], 1 << next_kept))
            else:
                env = arrays.einsum("abi,asr,bst->rti",
                    (tensor.shape[2], tensor.shape[2], 1 << kept), env, tensor, conjugate)
            kept = next_kept
        order = [sorted(selected).index(wire) for wire in selected]
        values = arrays.reshaped(arrays.reshaped(env[0, 0].real, (2,) * kept).transpose(order), (1 << kept,))
        return {format(index, f"0{kept}b") if kept else "": max(0.0, float(value)) for index, value in enumerate(values)}

    def expectation_pauli(self, label):
        if not isinstance(label, str) or len(label) != self.num_qubits or set(label) - set("IXYZ"):
            raise ValueError("QuantumBridge MPS Pauli labels must match the state width.")
        env = arrays.filled("mps_pauli_initial", (1, 1), complex, 1)
        for tensor, char in zip(self._tensors, label):
            self._guard((tensor.shape[0] ** 2 + tensor.shape[2] ** 2) * 8)
            env = arrays.einsum("ab,asr,st,btv->rv", (tensor.shape[2], tensor.shape[2]),
                env, arrays.conjugated(tensor), pauli_matrix(char), tensor)
        return complex(env[0, 0])

    def project(self, wire, outcome):
        self._wires((wire,))
        if outcome not in (0, 1):
            raise ValueError("QuantumBridge MPS measurement outcomes must be zero or one.")
        self._tensors = arrays.adopt_for_mutation(self._tensors)
        self._move_center(wire)
        tensor = self._tensors[wire]
        probability = float(np.clip(arrays.norm_squared(tensor[:, outcome, :]), 0, 1))
        if probability:
            tensor[:, 1 - outcome, :] = 0
            tensor /= np.sqrt(probability)
        return probability

    def measure(self, wire, rng):
        self._wires((wire,))
        self._move_center(wire)
        p0 = float(np.clip(arrays.norm_squared(self._tensors[wire][:, 0, :]), 0, 1))
        outcome = int(rng.random() >= p0)
        self.project(wire, outcome)
        return outcome

    def to_dict(self, *, serialization_limits=None):
        from quantumbridge.schema.serialization import check_numeric_export
        check_numeric_export(self._tensors, limits=serialization_limits)
        return {"format": "quantumbridge.mps.v1", "tensors": [
            {"real": tensor.real.tolist(), "imag": tensor.imag.tolist()} for tensor in self._tensors
        ], "discarded_weight": self.discarded_weight, "truncations": self.truncations,
                "max_bond_dim": self.max_bond_dim, "cutoff": self.cutoff,
                "peak_bond_dimension": self.peak_bond_dimension}

    @classmethod
    def from_dict(cls, payload, *, max_memory_mb=1024, serialization_limits=None):
        from quantumbridge.schema.serialization import validate_payload_structure
        validate_payload_structure(payload, limits=serialization_limits)
        if payload.get("format") != "quantumbridge.mps.v1":
            raise ValueError("Unsupported QuantumBridge MPS serialization format.")
        from quantumbridge.ir.qb_ir import matrix_from_ir

        result = cls([matrix_from_ir(value) for value in payload["tensors"]], max_memory_mb=max_memory_mb,
                     max_bond_dim=payload.get("max_bond_dim"), cutoff=payload.get("cutoff", 1e-16))
        weight, count = payload.get("discarded_weight", 0.0), payload.get("truncations", 0)
        if not isinstance(weight, Real) or not np.isfinite(weight) or weight < 0 or not isinstance(count, Integral) or count < 0:
            raise ValueError("Invalid QuantumBridge MPS truncation metadata.")
        result.discarded_weight, result.truncations = float(weight), int(count)
        peak = payload.get("peak_bond_dimension", result.peak_bond_dimension)
        if isinstance(peak, bool) or not isinstance(peak, Integral) or peak < result.peak_bond_dimension:
            raise ValueError("Invalid QuantumBridge MPS peak bond metadata.")
        result.peak_bond_dimension = int(peak)
        return result
