# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
"""Independent closed-shell Roothaan-Hall iteration on native AO integrals."""
from dataclasses import dataclass
from numbers import Integral
import numpy as np
from .integrals import molecular_integrals
from .limits import QChemLimits


class SCFConvergenceError(RuntimeError):
    def __init__(self, history):
        super().__init__("Native RHF did not converge within the iteration budget.")
        self.history = tuple(history)


@dataclass(frozen=True)
class SCFResult:
    integrals: object
    coefficients: object
    orbital_energies: object
    occupations: object
    density: object
    fock: object
    electronic_energy: float
    total_energy: float
    history: tuple
    molecule: object
    converged: bool = True


def _fock(h, eri, density):
    return h + np.einsum("rs,pqrs->pq", density, eri) - .5 * np.einsum("rs,prqs->pq", density, eri)


def run_restricted_hartree_fock(molecule, *, max_iterations=100, energy_tolerance=1e-11,
                               density_tolerance=1e-9, damping=0., limits=None):
    limits = QChemLimits() if limits is None else limits
    if not isinstance(limits, QChemLimits):
        raise TypeError("Chemistry limits require QChemLimits.")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, Integral) or max_iterations < 1:
        raise ValueError("SCF max_iterations must be a positive integer.")
    limits.check("max_scf_iterations", max_iterations)
    if any(isinstance(x, bool) or not np.isfinite(x) or x <= 0 for x in (energy_tolerance, density_tolerance)) or not np.isfinite(damping) or not 0 <= damping < 1:
        raise ValueError("Invalid SCF tolerances or damping.")
    integrals = molecular_integrals(molecule, limits=limits)
    n = len(integrals.overlap)
    if molecule.multiplicity != 1 or molecule.n_electrons < 2 or molecule.n_electrons % 2 or molecule.n_electrons > 2 * n:
        raise NotImplementedError("Native RHF requires a nonempty closed shell within the basis capacity.")
    eigenvalues, eigenvectors = np.linalg.eigh(integrals.overlap)
    if min(eigenvalues) < 1e-8:
        raise ValueError("AO overlap is linearly dependent at the native threshold.")
    orthogonalizer = (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T
    density, history, previous_energy = np.zeros((n, n)), [], None
    occupied = molecule.n_electrons // 2
    h, eri = integrals.one_body, integrals.two_body
    for iteration in range(1, max_iterations + 1):
        fock = _fock(h, eri, density)
        orbital_energies, vectors = np.linalg.eigh(orthogonalizer.T @ fock @ orthogonalizer)
        coefficients = orthogonalizer @ vectors
        candidate = 2 * coefficients[:, :occupied] @ coefficients[:, :occupied].T
        updated = (1 - damping) * candidate + damping * density
        fock_new = _fock(h, eri, updated)
        electronic = .5 * float(np.sum(updated * (h + fock_new)))
        total = electronic + integrals.nuclear_repulsion_energy
        change = float(np.linalg.norm(updated - density))
        residual = float(np.linalg.norm(fock_new @ updated @ integrals.overlap - integrals.overlap @ updated @ fock_new))
        history.append({"iteration": iteration, "total_energy": total, "density_change": change, "commutator_residual": residual})
        if previous_energy is not None and abs(total - previous_energy) < energy_tolerance and change < density_tolerance and residual < 1e-7:
            occupations = np.array([2.] * occupied + [0.] * (n - occupied))
            # Return the occupied orbital projector, not a fractionally populated mixing buffer.
            fock_new = _fock(h, eri, candidate)
            electronic = .5 * float(np.sum(candidate * (h + fock_new)))
            total = electronic + integrals.nuclear_repulsion_energy
            history[-1]["total_energy"] = total
            return SCFResult(integrals, coefficients, orbital_energies, occupations, candidate, fock_new,
                             electronic, total, tuple(history), molecule)
        density, previous_energy = updated, total
    raise SCFConvergenceError(history)
