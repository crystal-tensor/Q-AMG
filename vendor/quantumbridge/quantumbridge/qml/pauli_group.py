# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
"""Native joint Pauli measurements, with an auditable shared sample buffer."""

from dataclasses import dataclass
from contextlib import nullcontext
from numbers import Integral
from uuid import uuid4

import numpy as np

from quantumbridge.utils.math import PauliString


@dataclass(frozen=True)
class PauliGroup:
    words: tuple
    basis: tuple
    indices: tuple
    coefficients: tuple
    allocation: object = None
    group_index: int = 0
    max_sample_values: int = 2000000

    def validate(self, width):
        if not self.words or len(self.words) != len(self.indices):
            raise ValueError("Joint Pauli groups require indexed nonempty words.")
        basis = dict(self.basis)
        if len(basis) != len(self.basis) or any(
            isinstance(wire, bool) or not isinstance(wire, Integral) or not 0 <= wire < width or label not in {"X", "Y", "Z"}
            for wire, label in self.basis
        ):
            raise ValueError("Joint Pauli basis must use distinct in-register X/Y/Z wires.")
        occupied = set()
        for word in self.words:
            if not word or len(dict(word)) != len(word) or any(basis.get(wire) != label for wire, label in word):
                raise ValueError("Joint Pauli words must share their declared local basis.")
            occupied.update(wire for wire, _ in word)
        if occupied != set(basis):
            raise ValueError("Joint Pauli basis must match its measured words.")
        if not self.coefficients or any(len(row) != len(self.words) or np.iscomplexobj(row) or not np.all(np.isfinite(row)) for row in self.coefficients):
            raise ValueError("Joint Pauli coefficient rows must be finite and match the words.")


@dataclass(frozen=True)
class PauliGroupResult:
    group: PauliGroup
    means: np.ndarray
    covariance: np.ndarray
    sample_covariance: object
    bitstrings: object
    eigenvalues: object
    shots: object
    seed: object
    execution_id: str
    device_calls: int
    resources: object = None

    def metadata(self):
        from quantumbridge.execution.resource_context import current_resources
        resources = current_resources()
        size, rows = len(self.group.words), len(self.group.coefficients)
        workspace = nullcontext() if resources is None else resources.ledger.reserve(
            "qwc_metadata_workspace", memory_bytes=8*(rows*size+size*size+8*rows))
        with workspace:
            return self._metadata(resources, size, rows)

    def _metadata(self, resources, size, rows):
        if resources is not None:
            # Lists duplicate numeric payload; bitstrings remain shared references.
            count = size+size*size+(0 if self.sample_covariance is None else size*size)
            count += 0 if self.eigenvalues is None else self.eigenvalues.size
            ticket = resources.retain("qwc_metadata_numerical_payload", 8*(count+4*rows))
            resources.ledger.unavailable("python_container_allocator_memory",
                "QWC list/scalar payload excludes Python objects and allocator overhead.")
        rows = np.asarray(self.group.coefficients)
        variances = np.maximum(0., np.sum((rows @ self.covariance) * rows, axis=1))
        if self.shots is None:
            errors = [0.] * len(rows)
        elif self.sample_covariance is None:
            errors = [None if np.any(row) else 0. for row in rows]
        else:
            errors = np.sqrt(np.maximum(0., np.sum((rows @ self.sample_covariance) * rows, axis=1)) / self.shots).tolist()
        result = {"execution_id": self.execution_id, "basis": self.group.basis,
                "words": self.group.words, "source_term_indices": self.group.indices,
                "source_coefficients": self.group.coefficients, "shots": self.shots, "seed": self.seed,
                "sample_count": 0 if self.bitstrings is None else len(self.bitstrings),
                "bitstrings": self.bitstrings,
                "eigenvalues": None if self.eigenvalues is None else self.eigenvalues.tolist(),
                "means": self.means.tolist(), "covariance": self.covariance.tolist(),
                "source_expectation_contributions": (rows @ self.means).tolist(),
                "source_group_variances": variances.tolist(), "source_group_standard_errors": errors,
                "sample_covariance": None if self.sample_covariance is None else self.sample_covariance.tolist(),
                "covariance_semantics": "exact observable covariance" if self.shots is None else "empirical population covariance; sample_covariance uses ddof=1",
                "device_calls": self.device_calls, "joint_sample_calls": int(self.shots is not None),
                "resources":self.resources,
                "provenance": "native analytic Pauli expectations" if self.shots is None else "one native Device.sample call; all word eigenvalues derive from this bitstring buffer"}
        if resources is not None:
            ticket.observe(8*(count+4*len(rows)))
        return result


def measure_pauli_group(device, circuit, group, shots, seed):
    group.validate(circuit.num_qubits)
    size = len(group.words)
    if shots is not None and (isinstance(shots, bool) or not isinstance(shots, Integral) or shots < 1):
        raise ValueError("Joint Pauli shots must be a positive integer.")
    from quantumbridge.execution.resource_context import current_resources
    scope = current_resources()
    result_bytes = 8*(size+size*size)
    if shots is not None:
        result_bytes += shots*len(group.basis)+8*shots*size+(8*size*size if shots > 1 else 0)
    ticket = None if scope is None else scope.retain("qwc_retained_arrays_and_bitstrings", result_bytes)
    # This scope holds result arrays, not the sequential decoding workspace.
    temporary_bytes = 0 if shots is None else shots*(len(group.basis)+24*size+16)+8*size*size
    workspace = nullcontext() if scope is None else scope.ledger.reserve("qwc_statistics_workspace", memory_bytes=temporary_bytes)
    with workspace:
        result = _measure_pauli_group(device, circuit, group, shots, seed)
    if ticket is not None:
        actual = sum(value.nbytes for value in (result.means, result.covariance, result.sample_covariance, result.eigenvalues) if value is not None)
        actual += 0 if result.bitstrings is None else sum(len(word) for word in result.bitstrings)
        ticket.observe(actual)
    return result


def _measure_pauli_group(device, circuit, group, shots, seed):
    size = len(group.words)
    resources = None
    def capture():
        nonlocal resources
        current = getattr(device,"last_execution_metadata",None)
        if not current or getattr(device,"simulation",None) != "matrix_product_state":
            return
        if resources is None:
            resources = dict(current)
        else:
            for key in ("max_bond_observed","max_discarded_weight","max_tensor_bytes","max_truncations"):
                resources[key] = max(resources[key],current[key])
            resources["max_bond_dimensions_observed"] = [max(a,b) for a,b in zip(resources["max_bond_dimensions_observed"],current["max_bond_dimensions_observed"])]
    def expectation(word):
        value = device.expectation(circuit,PauliString(word))
        capture()
        return value
    if shots is None:
        if size * size > group.max_sample_values:
            raise MemoryError("Joint Pauli analytic covariance exceeds the sample-value budget.")
        means = np.asarray([expectation(word) for word in group.words])
        covariance, calls = np.empty((size, size)), size
        for i, first in enumerate(group.words):
            for j in range(i + 1):
                product = dict(first)
                for wire, label in group.words[j]:
                    if wire in product:
                        del product[wire]
                    else:
                        product[wire] = label
                second = 1. if not product else expectation(sorted(product.items()))
                calls += bool(product)
                covariance[i, j] = covariance[j, i] = second - means[i] * means[j]
        memory = eigenvalues = sample_covariance = None
    else:
        if isinstance(shots, bool) or not isinstance(shots, Integral) or shots < 1:
            raise ValueError("Joint Pauli shots must be a positive integer.")
        if shots * (len(group.basis) + size) + size * size > group.max_sample_values:
            raise MemoryError("Joint Pauli samples exceed the sample-value budget.")
        rotated = circuit.copy()
        for wire, label in group.basis:
            if label == "X":
                rotated.h(wire)
            elif label == "Y":
                rotated.sdg(wire).h(wire)
        wires = tuple(wire for wire, _ in group.basis)
        memory = tuple(device.sample(rotated, wires=wires, shots=shots, seed=seed))
        capture()
        if len(memory) != shots or any(not isinstance(label, str) or len(label) != len(wires) or set(label) - {"0", "1"} for label in memory):
            raise ValueError("Native Device returned an invalid joint bitstring buffer.")
        positions = {wire: index for index, wire in enumerate(wires)}
        bits = np.asarray([[int(bit) for bit in label] for label in memory], dtype=np.int8)
        eigenvalues = np.column_stack([1 - 2 * (bits[:, [positions[wire] for wire, _ in word]].sum(axis=1) % 2)
                                      for word in group.words])
        means = np.mean(eigenvalues, axis=0)
        centered = eigenvalues - means
        covariance = centered.T @ centered / shots
        sample_covariance = None if shots == 1 else covariance * shots / (shots - 1)
        calls = 1
    for value in (means, covariance, sample_covariance, eigenvalues):
        if value is not None:
            value.setflags(write=False)
    return PauliGroupResult(group, means, covariance, sample_covariance, memory, eigenvalues,
                            shots, seed, str(uuid4()), calls,resources)
