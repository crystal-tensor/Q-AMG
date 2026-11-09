# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
"""Numerical H STO-3G data, not an upstream implementation or runtime download.

MolSSI BSE version 1, revision 2018-06-19, reference hehre1969a:
https://www.basissetexchange.org/api/basis/sto-3g/format/json/?elements=1
See LICENSES/BSE_DATA_NOTICE.txt for source attribution/license boundary.
"""
from .gaussian_integrals import ContractedSGaussian
from .limits import QChemLimits
from .molecule import Molecule


STO3G_H_EXPONENTS = (3.425250914, .6239137298, .1688554040)
STO3G_H_COEFFICIENTS = (.1543289673, .5353281423, .4446345422)


def build_basis(molecule, *, limits=None):
    if not isinstance(molecule, Molecule):
        raise TypeError("Native basis construction requires Molecule.")
    limits = QChemLimits() if limits is None else limits
    if not isinstance(limits, QChemLimits):
        raise TypeError("Chemistry limits require QChemLimits.")
    limits.check("max_atoms", len(molecule.atoms))
    limits.check("max_basis_functions", len(molecule.atoms))
    limits.check("max_spin_orbitals", 2 * len(molecule.atoms))
    if molecule.basis not in {"sto-3g", "sto3g"} or any(atom != "H" for atom in molecule.atoms):
        raise NotImplementedError("Native basis currently supports hydrogen STO-3G s shells only.")
    return tuple(ContractedSGaussian(center, STO3G_H_EXPONENTS, STO3G_H_COEFFICIENTS) for center in molecule.coordinates_bohr)
