import pytest

from qamg.backends import QBBackend
from qamg.decomposition import condition_problem
from qamg.problem import build_problem
from qamg.quantum import QuantumStructurePrior
from qamg.schema import DesignSpec, PlannerSettings


def test_conditioning_preserves_original_constraints_and_objective():
    p = build_problem(DesignSpec(groups=5))
    fixed = (0, 0, 0, 0, 0)
    active = [0, 2, 4]
    sub, offset = condition_problem(p, active, fixed)
    for choices in sub.assignments():
        state = list(fixed)
        for i, c in zip(active, choices, strict=True):
            state[i] = c
        assert bool(p.violations(tuple(state))) == bool(sub.violations(choices))
        assert p.cost(tuple(state)) == pytest.approx(sub.cost(choices)+offset)


def test_small_backend_updates_every_original_group_with_measured_blocks():
    p = build_problem(DesignSpec(groups=5))
    result = QuantumStructurePrior(QBBackend(max_qubits=2)).plan(
        p, PlannerSettings(encoding="compact_binary", iterations=0, shots=128,
                           decomposition_passes=1), 42)
    assert result.resources["logical_qubits"] <= 2
    assert result.resources["original_qubo_variables"] == 10
    assert result.resources["groups_covered_by_quantum_updates"] == list(range(5))
    assert p.feasible(result.best.bits)
    assert len(result.trace) == 3
    assert all(t["subplan"]["execution_source"] == "ideal_simulator" for t in result.trace)
    assert result.best.bits in result.trace[-1]["counts"]
