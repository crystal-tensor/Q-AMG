# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
"""Native geometry -> RHF orbitals -> spin fermions -> qubit Hamiltonian."""
from dataclasses import dataclass
from collections.abc import Mapping
import numpy as np
from .fermion import FermionicOp
from .integrals import ElectronicIntegrals
from .limits import QChemLimits
from .mappings import JordanWignerMapper
from .scf import SCFResult, run_restricted_hartree_fock


def electron_integrals(molecule, scf_result=None, *, limits=None):
    limits = QChemLimits() if limits is None else limits
    if not isinstance(limits, QChemLimits):
        raise TypeError("Chemistry limits require QChemLimits.")
    result = run_restricted_hartree_fock(molecule, limits=limits) if scf_result is None else scf_result
    if not isinstance(result, SCFResult) or not result.converged or result.molecule != molecule:
        raise ValueError("Electronic integrals require a converged RHF result for this molecule.")
    c, ao = np.asarray(result.coefficients), result.integrals
    n = len(ao.one_body)
    limits.check("max_atoms", len(molecule.atoms))
    limits.check("max_basis_functions", n)
    limits.check("max_spin_orbitals", 2 * n)
    limits.check("max_integral_tensor_bytes", 8 * (n ** 4 + 4 * n ** 2))
    if c.shape != (n, n) or not np.all(np.isfinite(c)) or not np.allclose(c.T @ ao.overlap @ c, np.eye(n), atol=1e-8, rtol=0):
        raise ValueError("RHF orbital coefficients are not finite AO-orthonormal orbitals.")
    one = c.T @ ao.one_body @ c
    two = np.einsum("up,vq,wr,xs,uvwx->pqrs", c, c, c, c, ao.two_body, optimize=True)
    return ElectronicIntegrals(one_body=one, two_body=two, nuclear_repulsion_energy=ao.nuclear_repulsion_energy,
        provenance={"implementation": "quantumbridge_native", "basis": "RHF molecular orbitals", "units": "Hartree",
                    "eri_convention": "chemist (pq|rs)", "spin_order": "interleaved spatial alpha,beta"})


def build_fermionic_hamiltonian(molecule, scf_result, *, limits=None):
    limits = QChemLimits() if limits is None else limits
    if not isinstance(limits, QChemLimits):
        raise TypeError("Chemistry limits require QChemLimits.")
    integrals = electron_integrals(molecule, scf_result, limits=limits)
    n, terms = len(integrals.one_body), [("", integrals.nuclear_repulsion_energy)]
    limits.check("max_spin_orbitals", 2 * n)
    # Chemist (pq|rs): 1/2 (pq|rs) a+p,sigma a+r,tau a_s,tau a_q,sigma.
    for p in range(n):
        for q in range(n):
            for sigma in (0, 1):
                terms.append((f"+_{2*p+sigma} -_{2*q+sigma}", integrals.one_body[p, q]))
            for r in range(n):
                for s in range(n):
                    for sigma in (0, 1):
                        for tau in (0, 1):
                            terms.append((f"+_{2*p+sigma} +_{2*r+tau} -_{2*s+tau} -_{2*q+sigma}", .5 * integrals.two_body[p, q, r, s]))
                            limits.check("max_hamiltonian_terms", len(terms))
    return FermionicOp(terms, register_length=2 * n).normal_ordered(max_terms=limits.max_hamiltonian_terms)


@dataclass(frozen=True)
class MolecularHamiltonian:
    molecule: object
    hamiltonian: object
    fermionic_hamiltonian: FermionicOp
    scf_result: SCFResult
    metadata: dict

    @property
    def num_qubits(self):
        return self.hamiltonian.num_qubits

    @property
    def n_electrons(self):
        return self.molecule.n_electrons

    def exact_ground_energy(self):
        """Lowest energy in the requested electron-number sector; nuclear included."""
        matrix = self.hamiltonian.matrix()
        if not np.allclose(matrix, matrix.conj().T, atol=1e-10, rtol=0):
            raise ValueError("Molecular Hamiltonian is not Hermitian.")
        sector = [state for state in range(len(matrix)) if state.bit_count() == self.n_electrons]
        return float(np.linalg.eigvalsh(matrix[np.ix_(sector, sector)])[0])

    def to_dict(self):
        return {"num_qubits": self.num_qubits, "n_electrons": self.n_electrons,
                "pauli_terms": [{"coefficient": float(value), "paulis": list(pauli.terms)} for value, pauli in self.hamiltonian.terms],
                "metadata": dict(self.metadata)}


def molecular_hamiltonian_native(molecule, mapping="jordan_wigner", *, limits=None, scf_options=None):
    limits = QChemLimits() if limits is None else limits
    if mapping != "jordan_wigner":
        raise NotImplementedError("Native molecular workflow currently supports Jordan-Wigner mapping.")
    if scf_options is not None and not isinstance(scf_options, Mapping):
        raise TypeError("SCF options must be a mapping.")
    result = run_restricted_hartree_fock(molecule, limits=limits, **({} if scf_options is None else scf_options))
    fermion = build_fermionic_hamiltonian(molecule, result, limits=limits)
    qubit = JordanWignerMapper().map(fermion, max_terms=limits.max_hamiltonian_terms)
    qubit.to_quantumbridge()  # Reject a non-Hermitian coefficient sum, never discard its imaginary part.
    return MolecularHamiltonian(molecule, qubit, fermion, result,
        {"implementation": "quantumbridge_native", "mapping": mapping, "basis": "STO-3G",
         "geometry_bohr": molecule.coordinates_bohr.tolist(), "input_unit": molecule.unit,
         "spin_orbitals": qubit.num_qubits, "spin_order": "spatial-0 alpha,beta; spatial-1 alpha,beta; ...",
         "qubit_order": "mode zero first", "eri_convention": "chemist (pq|rs)",
         "nuclear_repulsion_energy": result.integrals.nuclear_repulsion_energy, "nuclear_included_once": True,
         "rhf_total_energy": result.total_energy, "scf_iterations": len(result.history),
         "warnings": ["Native hydrogen STO-3G s-shell subset; not full qchem or differentiable HF."],
         "upstream_required": False, "runtime_network": False})
