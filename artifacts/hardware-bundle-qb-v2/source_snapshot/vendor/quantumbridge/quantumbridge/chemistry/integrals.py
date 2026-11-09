# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
# Design source: docs/roadmap/p2_upstream_integration_strategy_v0.1.md.

from dataclasses import dataclass


@dataclass
class ElectronicIntegrals:
    one_body: object = None
    two_body: object = None
    overlap: object = None
    kinetic: object = None
    nuclear_attraction: object = None
    nuclear_repulsion_energy: float = 0.
    basis_functions: tuple = ()
    provenance: object = None


def molecular_integrals(molecule, *, limits=None):
    import numpy as np
    from .limits import QChemLimits
    from .basis import build_basis
    from .gaussian_integrals import overlap, kinetic, nuclear_attraction, electron_repulsion
    limits = QChemLimits() if limits is None else limits
    basis = build_basis(molecule, limits=limits)
    n = len(basis)
    limits.check("max_integral_tensor_bytes", 8 * (n ** 4 + 4 * n ** 2))
    coordinates, charges = molecule.coordinates_bohr, molecule.nuclear_charges
    nuclear = 0.
    for a in range(len(charges)):
        for b in range(a):
            distance = np.linalg.norm(coordinates[a] - coordinates[b])
            if distance < 1e-8:
                raise ValueError("Nuclear separation is singular or too small.")
            nuclear += charges[a] * charges[b] / distance
    s, t, v = (np.empty((n, n)) for _ in range(3))
    eri = np.empty((n, n, n, n))
    for p in range(n):
        for q in range(n):
            s[p, q], t[p, q] = overlap(basis[p], basis[q]), kinetic(basis[p], basis[q])
            v[p, q] = sum(nuclear_attraction(basis[p], basis[q], center, charge) for center, charge in zip(coordinates, charges))
            for r in range(n):
                for u in range(n):
                    eri[p, q, r, u] = electron_repulsion(basis[p], basis[q], basis[r], basis[u])
    if any(not np.all(np.isfinite(value)) for value in (s, t, v, eri)):
        raise ArithmeticError("Molecular integrals contain nonfinite values.")
    return ElectronicIntegrals(t + v, eri, s, t, v, float(nuclear), basis,
        {"implementation": "quantumbridge_native", "units": "atomic", "eri_convention": "chemist (pq|rs)",
         "basis": "STO-3G", "basis_data": "MolSSI BSE sto-3g v1 H numerical parameters", "runtime_network": False})
