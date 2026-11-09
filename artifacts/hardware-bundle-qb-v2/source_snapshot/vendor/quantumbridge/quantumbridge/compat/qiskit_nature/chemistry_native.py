# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit Nature, PySCF, or OpenFermion was copied.
"""Executable educational Qiskit Nature compatibility workflows.

The native workflows use small deterministic Hamiltonians for clean-room
testing and examples. They are not production electronic-structure
calculations and do not implement materials band-gap workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any

import numpy as np

from quantumbridge.chemistry import Molecule, QubitHamiltonian
from quantumbridge.schema.chemistry_results import (
    H2WorkflowResult,
    LiHWorkflowResult,
    UpstreamNatureResult,
)

QISKIT_NATURE_WARNING = (
    "QuantumBridge Qiskit Nature support is an optional passthrough/schema bridge "
    "plus minimal educational native chemistry workflows. It is not a full "
    "Qiskit Nature replacement and is not production quantum chemistry software."
)
NATIVE_CHEMISTRY_WARNING = (
    "Native H2/LiH workflows are deterministic educational examples for small "
    "Hamiltonians, not production electronic-structure calculations."
)
UPSTREAM_WARNING = (
    "Upstream passthrough requires optional qiskit-nature, PySCF, or OpenFermion packages."
)
MATERIAL_WARNING = "Materials band-gap workflows are not implemented in this stage."


@dataclass(frozen=True)
class MinimalMolecularProblem:
    molecule: Molecule
    formula: str
    bond_length: float
    hamiltonian: QubitHamiltonian
    nuclear_repulsion_energy: float
    particle_count: int
    workflow: str
    reference_note: str

    @property
    def qubit_count(self) -> int:
        return 1 + max((wire for _, pauli in self.hamiltonian.terms for wire, _ in pauli.terms), default=0)


def build_h2_problem(bond_length: float = 0.735, basis: str = "sto-3g") -> MinimalMolecularProblem:
    """Build a deterministic minimal H2 problem."""

    molecule = Molecule(["H", "H"], [(0.0, 0.0, 0.0), (0.0, 0.0, float(bond_length))], basis=basis)
    return MinimalMolecularProblem(
        molecule=molecule,
        formula="H2",
        bond_length=float(bond_length),
        hamiltonian=_minimal_h2_qubit_hamiltonian(),
        nuclear_repulsion_energy=0.7151043390810812,
        particle_count=2,
        workflow="h2_native_exact",
        reference_note="Small two-qubit educational Hamiltonian used for deterministic adapter tests.",
    )


def build_lih_problem(bond_length: float = 1.6, basis: str = "sto-3g") -> MinimalMolecularProblem:
    """Build a deterministic minimal LiH problem."""

    molecule = Molecule(["Li", "H"], [(0.0, 0.0, 0.0), (0.0, 0.0, float(bond_length))], basis=basis)
    return MinimalMolecularProblem(
        molecule=molecule,
        formula="LiH",
        bond_length=float(bond_length),
        hamiltonian=_minimal_lih_qubit_hamiltonian(),
        nuclear_repulsion_energy=0.995317634356,
        particle_count=4,
        workflow="lih_native_exact",
        reference_note="Small four-qubit educational Hamiltonian used for deterministic adapter tests.",
    )


def run_h2_native(bond_length: float = 0.735, basis: str = "sto-3g") -> H2WorkflowResult:
    """Run the native educational H2 exact-diagonalization workflow."""

    return _run_native_problem(build_h2_problem(bond_length=bond_length, basis=basis), H2WorkflowResult)


def run_lih_native(bond_length: float = 1.6, basis: str = "sto-3g") -> LiHWorkflowResult:
    """Run the native educational LiH exact-diagonalization workflow."""

    return _run_native_problem(build_lih_problem(bond_length=bond_length, basis=basis), LiHWorkflowResult)


def run_h2_upstream_passthrough(bond_length: float = 0.735, basis: str = "sto3g") -> UpstreamNatureResult:
    """Run a local qiskit-nature/PySCF H2 passthrough smoke workflow when installed."""

    return _run_upstream_driver_smoke(
        formula="H2",
        atom=f"H 0 0 0; H 0 0 {float(bond_length)}",
        bond_length=float(bond_length),
        basis=basis,
        max_cycle=None,
    )


def run_lih_upstream_passthrough(bond_length: float = 1.6, basis: str = "sto3g") -> UpstreamNatureResult:
    """Run a local qiskit-nature/PySCF LiH passthrough smoke workflow when installed."""

    return _run_upstream_driver_smoke(
        formula="LiH",
        atom=f"Li 0 0 0; H 0 0 {float(bond_length)}",
        bond_length=float(bond_length),
        basis=basis,
        max_cycle=1,
    )


def chemistry_hamiltonian_to_vqe_problem_metadata(problem: MinimalMolecularProblem) -> dict[str, Any]:
    """Return metadata that can be handed to Stage 9C VQE examples."""

    return {
        "workflow": problem.workflow,
        "molecule": problem.formula,
        "qubit_count": problem.qubit_count,
        "particle_count": problem.particle_count,
        "pauli_terms": _pauli_terms_to_dicts(problem.hamiltonian),
        "warning": NATIVE_CHEMISTRY_WARNING,
    }


def _run_native_problem(problem: MinimalMolecularProblem, result_class):
    matrix = problem.hamiltonian.matrix(problem.qubit_count)
    eigenvalues = [float(np.real(value)) for value in np.linalg.eigvalsh(matrix)]
    electronic_energy = min(eigenvalues)
    total_energy = electronic_energy + float(problem.nuclear_repulsion_energy)
    return result_class(
        workflow=problem.workflow,
        molecule=problem.formula,
        mode="native_minimal",
        capability_level=3,
        bond_length=problem.bond_length,
        basis=problem.molecule.basis,
        driver="quantumbridge-native-minimal",
        mapper="pauli-hamiltonian",
        qubit_count=problem.qubit_count,
        particle_count=problem.particle_count,
        pauli_terms=_pauli_terms_to_dicts(problem.hamiltonian),
        ground_state_energy=total_energy,
        electronic_energy=electronic_energy,
        nuclear_repulsion_energy=float(problem.nuclear_repulsion_energy),
        eigenvalues=eigenvalues,
        raw_type="MinimalMolecularProblem",
        production_ready=False,
        native_implementation=True,
        metadata={
            "reference_note": problem.reference_note,
            "materials_band_gap": False,
            "cloud_access": False,
            "token_read": False,
            "hardware_access": False,
        },
        warnings=[QISKIT_NATURE_WARNING, NATIVE_CHEMISTRY_WARNING, MATERIAL_WARNING],
        provenance={
            "adapter": "quantumbridge.compat.qiskit_nature.chemistry_native",
            "official_endorsement": False,
            "source_code_copied": False,
            "upstream_source_copied": False,
            "ibm_branding_copied": False,
            "stage": "9D",
        },
    )


def _run_upstream_driver_smoke(
    *, formula: str, atom: str, bond_length: float, basis: str, max_cycle: int | None
) -> UpstreamNatureResult:
    try:
        qiskit_nature = import_module("qiskit_nature")
        import_module("pyscf")
        drivers = import_module("qiskit_nature.second_q.drivers")
    except Exception as exc:
        return UpstreamNatureResult(
            workflow=f"{formula.lower()}_upstream_passthrough",
            molecule=formula,
            mode="upstream_passthrough",
            capability_level=1,
            bond_length=bond_length,
            basis=basis,
            driver="PySCFDriver",
            mapper=None,
            ground_state_energy=None,
            raw_type=type(exc).__name__,
            upstream_package="qiskit-nature",
            upstream_version=None,
            production_ready=False,
            native_implementation=False,
            warnings=[QISKIT_NATURE_WARNING, UPSTREAM_WARNING, MATERIAL_WARNING],
            provenance=_upstream_provenance(),
            unsupported_reason=f"optional upstream dependency unavailable: {type(exc).__name__}: {exc}",
        )
    kwargs: dict[str, Any] = {"atom": atom, "basis": basis}
    if max_cycle is not None:
        kwargs["max_cycle"] = int(max_cycle)
    problem = drivers.PySCFDriver(**kwargs).run()
    particle_count = problem.num_particles
    if isinstance(particle_count, tuple):
        particle_total: int | tuple[int, int] = int(sum(particle_count))
    else:
        particle_total = int(particle_count)
    return UpstreamNatureResult(
        workflow=f"{formula.lower()}_upstream_passthrough",
        molecule=formula,
        mode="upstream_passthrough",
        capability_level=1,
        bond_length=bond_length,
        basis=basis,
        driver="PySCFDriver",
        mapper=None,
        particle_count=particle_total,
        ground_state_energy=float(getattr(problem, "reference_energy", 0.0) or 0.0),
        nuclear_repulsion_energy=float(getattr(problem, "nuclear_repulsion_energy", 0.0) or 0.0),
        raw_type=type(problem).__name__,
        upstream_package="qiskit-nature",
        upstream_version=getattr(qiskit_nature, "__version__", None),
        production_ready=False,
        native_implementation=False,
        metadata={
            "num_spatial_orbitals": getattr(problem, "num_spatial_orbitals", None),
            "materials_band_gap": False,
            "cloud_access": False,
            "token_read": False,
            "hardware_access": False,
        },
        warnings=[QISKIT_NATURE_WARNING, UPSTREAM_WARNING, MATERIAL_WARNING],
        provenance=_upstream_provenance(),
    )


def _minimal_h2_qubit_hamiltonian() -> QubitHamiltonian:
    return QubitHamiltonian.from_pauli_terms(
        [
            (-1.052373245772859, ()),
            (0.39793742484318045, ((0, "Z"),)),
            (-0.39793742484318045, ((1, "Z"),)),
            (-0.01128010425623538, ((0, "Z"), (1, "Z"))),
            (0.18093119978423156, ((0, "X"), (1, "X"))),
        ]
    )


def _minimal_lih_qubit_hamiltonian() -> QubitHamiltonian:
    return QubitHamiltonian.from_pauli_terms(
        [
            (-7.49894690201071, ()),
            (0.218291, ((0, "Z"),)),
            (-0.125411, ((1, "Z"),)),
            (0.170597, ((2, "Z"),)),
            (-0.091214, ((3, "Z"),)),
            (0.045322, ((0, "Z"), (1, "Z"))),
            (-0.032118, ((1, "Z"), (2, "Z"))),
            (0.028451, ((2, "Z"), (3, "Z"))),
            (0.061707, ((0, "X"), (1, "X"))),
            (0.044103, ((2, "X"), (3, "X"))),
        ]
    )


def _pauli_terms_to_dicts(hamiltonian: QubitHamiltonian) -> list[dict[str, Any]]:
    rows = []
    for coefficient, pauli in hamiltonian.terms:
        rows.append(
            {
                "coefficient": float(coefficient),
                "paulis": [{"wire": int(wire), "op": str(label)} for wire, label in pauli.terms],
            }
        )
    return rows


def _upstream_provenance() -> dict[str, Any]:
    return {
        "adapter": "quantumbridge.compat.qiskit_nature.chemistry_native",
        "official_endorsement": False,
        "source_code_copied": False,
        "upstream_source_copied": False,
        "ibm_branding_copied": False,
        "cloud_access": False,
        "token_storage": False,
        "stage": "9D",
    }
