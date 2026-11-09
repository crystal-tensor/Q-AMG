# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
# Design source: docs/roadmap/p2_upstream_integration_strategy_v0.1.md.

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
from numbers import Integral
import numpy as np


BOHR_TO_ANGSTROM = 0.529177210903  # CODATA 2018, used consistently for this API.
_SYMBOLS = "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og".split()
_NUMBERS = {symbol: index + 1 for index, symbol in enumerate(_SYMBOLS)}


@dataclass(frozen=True)
class Molecule:
    atoms: tuple[str, ...]
    coordinates: tuple[tuple[float, float, float], ...]
    charge: int = 0
    multiplicity: int = 1
    spin: int | None = None
    basis: str = "sto-3g"
    unit: str = "angstrom"

    def __init__(
        self,
        atoms: Iterable[str] = None,
        coordinates: Iterable[Iterable[float]] = None,
        charge: int = 0,
        multiplicity: int = 1,
        spin: int | None = None,
        basis: str = "sto-3g",
        unit: str = "angstrom",
        *, symbols=None,
    ):
        if symbols is not None:
            if atoms is not None:
                raise ValueError("Specify atoms or symbols, not both.")
            atoms = symbols
        if atoms is None or coordinates is None:
            raise ValueError("Molecule needs atomic symbols and coordinates.")
        if np.iscomplexobj(coordinates):
            raise ValueError("Molecular coordinates must be real.")
        atoms_tuple = tuple(str(atom) for atom in atoms)
        coords_tuple = tuple(tuple(float(value) for value in coord) for coord in coordinates)
        if len(atoms_tuple) != len(coords_tuple) or not atoms_tuple:
            raise ValueError("QuantumBridge Molecule requires matching non-empty atoms and coordinates.")
        if any(len(coord) != 3 for coord in coords_tuple):
            raise ValueError("QuantumBridge Molecule coordinates must be 3D.")
        if any(atom not in _NUMBERS for atom in atoms_tuple) or not np.all(np.isfinite(coords_tuple)):
            raise ValueError("Molecule requires recognized elements and finite coordinates.")
        if any(isinstance(v, bool) or not isinstance(v, Integral) for v in (charge, multiplicity)) or multiplicity < 1:
            raise ValueError("Molecular charge and positive multiplicity must be integers.")
        electrons = sum(_NUMBERS[atom] for atom in atoms_tuple) - charge
        unpaired = multiplicity - 1
        if electrons < unpaired or (electrons - unpaired) % 2:
            raise ValueError("Molecular electron count and multiplicity are inconsistent.")
        if spin is not None and (isinstance(spin, bool) or not isinstance(spin, Integral) or spin != unpaired):
            raise ValueError("Molecular spin must equal multiplicity minus one.")
        units = {"angstrom": "angstrom", "ang": "angstrom", "bohr": "bohr", "a0": "bohr"}
        if str(unit).lower() not in units or not isinstance(basis, str) or not basis.strip():
            raise ValueError("Molecule requires explicit Angstrom/Bohr units and a basis name.")
        object.__setattr__(self, "atoms", atoms_tuple)
        object.__setattr__(self, "coordinates", coords_tuple)
        object.__setattr__(self, "charge", int(charge))
        object.__setattr__(self, "multiplicity", int(multiplicity))
        object.__setattr__(self, "spin", None if spin is None else int(spin))
        object.__setattr__(self, "basis", basis.strip().lower())
        object.__setattr__(self, "unit", units[str(unit).lower()])

    @property
    def symbols(self):
        return self.atoms

    @property
    def atomic_numbers(self):
        return tuple(_NUMBERS[atom] for atom in self.atoms)

    @property
    def nuclear_charges(self):
        return self.atomic_numbers

    @property
    def n_electrons(self):
        return sum(self.atomic_numbers) - self.charge

    @property
    def coordinates_bohr(self):
        return np.asarray(self.coordinates) / (BOHR_TO_ANGSTROM if self.unit == "angstrom" else 1.)

    def to_xyz(self) -> str:
        lines = [str(len(self.atoms)), f"charge={self.charge} multiplicity={self.multiplicity} basis={self.basis} unit={self.unit}"]
        for atom, coord in zip(self.atoms, self.coordinates):
            lines.append(f"{atom} {coord[0]:.12g} {coord[1]:.12g} {coord[2]:.12g}")
        return "\n".join(lines) + "\n"

    @classmethod
    def from_xyz(cls, text: str, **kwargs) -> "Molecule":
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) < 2:
            raise ValueError("QuantumBridge XYZ text requires atom count and at least one atom line.")
        count = int(lines[0])
        atom_lines = lines[2 : 2 + count]
        if count < 1 or len(atom_lines) != count:
            raise ValueError("XYZ atom count does not match coordinate records.")
        atoms = []
        coords = []
        for line in atom_lines:
            parts = line.split()
            if len(parts) != 4:
                raise ValueError("QuantumBridge XYZ atom lines must contain symbol and three coordinates.")
            atoms.append(parts[0])
            coords.append(tuple(float(value) for value in parts[1:]))
        return cls(atoms, coords, **kwargs)

    def bond_scan(self, atom_a: int, atom_b: int, distances: Iterable[float]) -> list["Molecule"]:
        if len(self.atoms) != 2 or {atom_a, atom_b} != {0, 1}:
            raise ValueError("QuantumBridge minimal bond_scan currently supports two-atom molecules.")
        out = []
        for distance in distances:
            coords = [(0.0, 0.0, 0.0), (0.0, 0.0, float(distance))]
            out.append(Molecule(self.atoms, coords, self.charge, self.multiplicity, self.spin, self.basis, self.unit))
        return out
