# Copyright 2026 QuantumBridge Contributors. Apache-2.0.
"""Four independent native scientific case studies with inspectable baselines."""

import numpy as np

from quantumbridge.core import Circuit
from quantumbridge.devices import StatevectorDevice
from quantumbridge.chemistry import Molecule, molecular_hamiltonian_native
from quantumbridge.chemistry.excitations import apply_fermionic_excitation
from quantumbridge.chemistry.fermion import FermionicOp
from quantumbridge.chemistry.mappings import JordanWignerMapper
from quantumbridge.finance.kernels import labels, prepare_real_state
from quantumbridge.utils.math import PauliString

from .common import ApplicationCase, optimization_case


def chemistry_case():
    """Geometry-derived H2/STO-3G, RHF, a real variational circuit and Pauli sampling."""
    problem = molecular_hamiltonian_native(
        Molecule(["H", "H"], [(0, 0, 0), (0, 0, 0.735)])
    )
    matrix = problem.hamiltonian.matrix()
    device = StatevectorDevice()

    def ansatz(theta):
        circuit = Circuit(4, name="h2-double-excitation").x(0).x(1)
        return apply_fermionic_excitation(circuit, theta, (0, 1, 2, 3))

    trace = []
    for theta in np.linspace(-0.6, 0.6, 61):
        vector = device.statevector(ansatz(theta))
        trace.append(
            {
                "theta": float(theta),
                "energy_hartree": float(np.vdot(vector, matrix @ vector).real),
            }
        )
    best = min(trace, key=lambda row: row["energy_hartree"])
    center = best["theta"]
    for theta in np.linspace(center - 0.02, center + 0.02, 41):
        vector = device.statevector(ansatz(theta))
        trace.append(
            {
                "theta": float(theta),
                "energy_hartree": float(np.vdot(vector, matrix @ vector).real),
            }
        )
    best = min(trace, key=lambda row: row["energy_hartree"])
    state = ansatz(best["theta"])
    circuits, estimands, coefficients, constant, table = {}, {}, {}, 0.0, []
    for coefficient, pauli in problem.hamiltonian.terms:
        label = "".join(dict(pauli.terms).get(q, "I") for q in range(4))
        table.append({"pauli": label, "coefficient_hartree": float(coefficient)})
        if not pauli.terms:
            constant += coefficient
            continue
        probe = state.copy()
        for wire, axis in pauli.terms:
            if axis == "X":
                probe.h(wire)
            elif axis == "Y":
                probe.phase(-np.pi / 2, wire)
                probe.h(wire)
        circuits[label] = probe
        estimands[label] = [
            (-1.0) ** sum(int(bits[wire]) for wire, axis in pauli.terms)
            for bits in labels(4)
        ]
        coefficients[label] = float(coefficient)
    scan = []
    for distance in (0.5, 0.65, 0.735, 0.9, 1.2):
        other = molecular_hamiltonian_native(
            Molecule(["H", "H"], [(0, 0, 0), (0, 0, distance)])
        )
        scan.append(
            {
                "distance_angstrom": distance,
                "rhf_hartree": other.scf_result.total_energy,
                "exact_sector_hartree": other.exact_ground_energy(),
            }
        )
    return ApplicationCase(
        "chemistry",
        "science",
        "H2: geometry, RHF, variational energy and Pauli measurements",
        {
            "geometry_angstrom": [[0, 0, 0], [0, 0, 0.735]],
            "basis": "hydrogen STO-3G s-shells",
            "electrons": 2,
            "qubits": 4,
            "units": "Hartree; nuclear repulsion included once",
        },
        {
            "scf_history": list(problem.scf_result.history),
            "pauli_terms": table,
            "variational_trace": trace,
            "bond_scan": scan,
        },
        {
            "exact_ground_energy_hartree": problem.exact_ground_energy(),
            "rhf_energy_hartree": problem.scf_result.total_energy,
        },
        circuits,
        estimands,
        {
            "optimized_theta": best["theta"],
            "variational_energy_hartree": best["energy_hartree"],
            "variational_gap_hartree": best["energy_hartree"]
            - problem.exact_ground_energy(),
            "pauli_energy_constant": float(constant),
            "pauli_energy_terms": coefficients,
        },
    )


def biomedicine_case():
    """A four-residue hydrophobic square-lattice research model, not a drug predictor."""
    candidates = []
    turns = {0: 1, 1: 0, 2: -1}
    steps = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    for bits in labels(4):
        codes = [int(bits[:2], 2), int(bits[2:], 2)]
        valid_codes = all(code in turns for code in codes)
        points, heading = [(0, 0), (1, 0)], 0
        if valid_codes:
            for code in codes:
                heading = (heading + turns[code]) % 4
                dx, dy = steps[heading]
                points.append((points[-1][0] + dx, points[-1][1] + dy))
        overlaps = len(points) - len(set(points))
        contact = int(
            valid_codes
            and abs(points[0][0] - points[-1][0]) + abs(points[0][1] - points[-1][1])
            == 1
        )
        violation = sum(code not in turns for code in codes) + overlaps
        candidates.append(
            {
                "bitstring": bits,
                "turn_codes": codes,
                "coordinates": points,
                "contact_0_3": contact,
                "violations": violation,
                "feasible": violation == 0,
                "objective": -contact,
                "penalized_cost": -contact + 5 * violation,
            }
        )
    return optimization_case(
        "biomedicine",
        "science",
        "Four-residue lattice conformation and contact energy",
        {
            "sequence": "HHHH",
            "residues": 4,
            "first_bond": "fixed along +x",
            "turn_encoding": {
                "00": "left",
                "01": "straight",
                "10": "right",
                "11": "invalid",
            },
            "contact_energy": -1,
            "penalty": 5,
            "data_source": "explicit coarse-grained synthetic model, not atomic coordinates or drug data",
        },
        candidates,
    )


def hubbard_hamiltonian(interaction, hopping=1.0):
    terms = []
    for spin in (0, 1):
        terms.extend(
            [(f"+_{spin} -_{2 + spin}", -hopping), (f"+_{2 + spin} -_{spin}", -hopping)]
        )
    for up, down in ((0, 1), (2, 3)):
        terms.append((f"+_{up} -_{up} +_{down} -_{down}", interaction))
    return JordanWignerMapper().map(FermionicOp(terms, register_length=4))


def materials_case():
    sectors = {
        number: [i for i in range(16) if i.bit_count() == number]
        for number in (1, 2, 3)
    }
    sweep, ground = [], None
    for interaction in (0.0, 1.0, 2.0, 4.0, 6.0, 8.0):
        matrix = hubbard_hamiltonian(interaction).matrix()
        energies = {
            number: float(np.linalg.eigvalsh(matrix[np.ix_(indices, indices)])[0])
            for number, indices in sectors.items()
        }
        values, vectors = np.linalg.eigh(matrix[np.ix_(sectors[2], sectors[2])])
        vector = np.zeros(16)
        vector[sectors[2]] = vectors[:, 0].real
        occupation = [
            int(bits[0]) * int(bits[1]) + int(bits[2]) * int(bits[3])
            for bits in labels(4)
        ]
        double = float(np.asarray(occupation) @ abs(vector) ** 2)
        sweep.append(
            {
                "U_over_t": interaction,
                "energy_N2": energies[2],
                "finite_cluster_charge_gap": energies[3]
                + energies[1]
                - 2 * energies[2],
                "double_occupation": double,
            }
        )
        if interaction == 4.0:
            ground = vector
    return ApplicationCase(
        "materials",
        "science",
        "Two-site Fermi-Hubbard spectrum and double occupation",
        {
            "sites": 2,
            "spin_modes": 4,
            "particle_sector": 2,
            "hopping_t": 1.0,
            "measured_U": 4.0,
            "boundary": "open two-site cluster",
        },
        {
            "interaction_sweep": sweep,
            "mode_order": ["site0 up", "site0 down", "site1 up", "site1 down"],
            "preparation_amplitudes": ground.tolist(),
            "state_preparation_source": "classical exact sector eigensolver, not a quantum ground-state solver",
        },
        {
            "U4_energy_analytic": (4 - np.sqrt(4**2 + 16)) / 2,
            "U4_double_occupation_analytic": (1 - 4 / np.sqrt(4**2 + 16)) / 2,
        },
        {"occupation": prepare_real_state(ground, name="hubbard-U4-state")},
        {
            "occupation": [
                int(bits[0]) * int(bits[1]) + int(bits[2]) * int(bits[3])
                for bits in labels(4)
            ]
        },
    )


def dynamics_circuit(time, steps):
    circuit = Circuit(3, name="three-spin-trotter").x(0)
    for _ in range(steps):
        for left, right in ((0, 1), (1, 2)):
            circuit.rzz(2 * time / steps, left, right)
        for wire in range(3):
            circuit.rx(2 * 0.7 * time / steps, wire)
    return circuit


def physics_case():
    matrix = np.zeros((8, 8), complex)
    for pair in ((0, 1), (1, 2)):
        matrix += PauliString([(q, "Z") for q in pair]).matrix(3)
    for wire in range(3):
        matrix += 0.7 * PauliString([(wire, "X")]).matrix(3)
    energies, vectors = np.linalg.eigh(matrix)
    initial = np.eye(8)[:, 4]
    magnetization = np.array([(-1.0) ** int(bits[0]) for bits in labels(3)])
    correlation = np.array(
        [(-1.0) ** (int(bits[0]) + int(bits[2])) for bits in labels(3)]
    )
    table = []
    device = StatevectorDevice()
    for time in np.linspace(0, 1.5, 13):
        exact = vectors @ (np.exp(-1j * time * energies) * (vectors.conj().T @ initial))
        for steps in (1, 4, 12):
            approximation = device.statevector(dynamics_circuit(time, steps))
            table.append(
                {
                    "time": float(time),
                    "trotter_steps": steps,
                    "infidelity": max(
                        0.0, float(1 - abs(np.vdot(exact, approximation)) ** 2)
                    ),
                    "magnetization_z0": float(magnetization @ abs(approximation) ** 2),
                    "exact_z0": float(magnetization @ abs(exact) ** 2),
                    "correlation_z0z2": float(correlation @ abs(approximation) ** 2),
                }
            )
    return ApplicationCase(
        "physics",
        "science",
        "Three-spin Ising dynamics and Trotter error",
        {
            "J": 1.0,
            "transverse_field": 0.7,
            "initial_state": "100",
            "time_unit": "hbar/J",
            "measured_time": 1.5,
            "measured_steps": 12,
        },
        {"time_evolution": table, "spectrum": energies.tolist()},
        {
            "final_exact_z0": table[-1]["exact_z0"],
            "final_trotter_infidelity": table[-1]["infidelity"],
        },
        {
            "magnetization": dynamics_circuit(1.5, 12),
            "correlation": dynamics_circuit(1.5, 12),
        },
        {"magnetization": magnetization.tolist(), "correlation": correlation.tolist()},
    )


BUILDERS = {
    "chemistry": chemistry_case,
    "biomedicine": biomedicine_case,
    "materials": materials_case,
    "physics": physics_case,
}


def build_science_case(direction):
    try:
        builder = BUILDERS[direction]
    except KeyError as error:
        raise ValueError("Unknown scientific direction.") from error
    return builder()
