import numpy as np
import pytest
from quantumbridge import StatevectorDevice

from qamg.backends import QBBackend
from qamg.problem import build_problem
from qamg.quantum import QuantumStructurePrior, qaoa_circuit
from qamg.schema import DesignSpec, PlannerSettings


def test_xy_mixer_preserves_catalogue_choices():
    p = build_problem(DesignSpec(groups=3))
    c, gamma, beta, _ = qaoa_circuit(p.encode(), 2, "xy")
    c = c.remove_final_measurements(inplace=False).assign_parameters(
        {**dict(zip(gamma, [.31, .87])), **dict(zip(beta, [.23, .61]))})
    probabilities = StatevectorDevice().probabilities(c)
    assert sum(probabilities.values()) == pytest.approx(1)
    for bits, probability in probabilities.items():
        if probability > 1e-10:
            assert p.selection(bits) is not None


def test_measured_candidates_have_real_job_provenance():
    p = build_problem(DesignSpec(groups=3))
    result = QuantumStructurePrior(QBBackend()).plan(
        p, PlannerSettings(iterations=6, training_shots=64, shots=128), 41)
    assert result.execution_source == "ideal_simulator"
    assert result.resources["shots"] == sum(t["shots"] for t in result.trace)
    assert "OPENQASM 3" in result.circuit_qasm3
    assert result.best is not None
    witness = next(t for t in result.trace if t["job_id"] == result.best.first_job_id)
    assert result.best.bits in witness["counts"]
    assert result.timings["quantum_execution_s"] is None
    assert result.resources["capabilities"]["is_simulator"] is True


def test_noise_is_explicit_and_reproducible():
    p = build_problem(DesignSpec(groups=3))
    settings = PlannerSettings(iterations=0, shots=64)
    r = QuantumStructurePrior(QBBackend(noisy=True)).plan(p, settings, 42)
    assert r.execution_source == "noisy_simulator"
    assert r.resources["capabilities"]["noise_provenance"] == "generic_not_device_calibrated"
    assert sum(r.trace[0]["counts"].values()) == 64


def test_capacity_and_warm_start_checked():
    p = build_problem(DesignSpec(groups=5))
    with pytest.raises(ValueError, match="capacity"):
        QuantumStructurePrior(QBBackend(max_qubits=1)).plan(p, PlannerSettings(), 42)
    with pytest.raises(ValueError, match="shape"):
        QuantumStructurePrior(QBBackend()).plan(p, PlannerSettings(), 42,
                                                initial_parameters=[np.nan]*4)


def test_compact_measurements_decode_without_classical_solving():
    p = build_problem(DesignSpec(groups=3))
    r = QuantumStructurePrior(QBBackend()).plan(
        p, PlannerSettings(encoding="compact_binary", iterations=0, shots=128), 42)
    assert r.resources["logical_qubits"] == 3
    assert r.resources["original_qubo_variables"] == 6
    for batch in r.trace:
        assert all(len(k) == 3 for k in batch["raw_measurement_counts"])
        for raw, count in batch["raw_measurement_counts"].items():
            assert batch["counts"][p.bits(tuple(map(int, raw)))] == count

