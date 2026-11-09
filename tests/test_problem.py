from itertools import product

import numpy as np
import pytest

from qamg.classical import solve_classical
from qamg.problem import build_problem
from qamg.schema import DesignSpec, UnaryRule


@pytest.mark.parametrize("family", ["drive", "clamp"])
def test_original_qubo_ising_agree_for_every_bitstring(family):
    problem = build_problem(DesignSpec(family=family, groups=3))
    problem.unary.append(UnaryRule(group=2, option=0, reason="geometry_feedback"))
    qubo = problem.encode()
    offset, h, jz = qubo.ising()
    for bits in product([0, 1], repeat=qubo.n):
        z = 1-2*np.array(bits)
        ising_energy = offset+h@z+sum(c*z[i]*z[j] for (i, j), c in jz.items())
        text = "".join(map(str, bits))
        assert qubo.energy(bits) == pytest.approx(problem.original_energy(text))
        assert ising_energy == pytest.approx(qubo.energy(bits))


@pytest.mark.parametrize("family", ["drive", "clamp"])
def test_penalties_and_strong_baselines(family):
    problem = build_problem(DesignSpec(family=family, groups=5, load="heavy"))
    exact = solve_classical(problem, "exact")
    assert exact.best is not None
    for method in ["cpsat", "cpsat_qubo"]:
        result = solve_classical(problem, method)
        assert result.best.energy == pytest.approx(exact.best.energy)
        assert problem.feasible(result.best.bits)
    qubo = problem.encode()
    for bits in product([0, 1], repeat=qubo.n):
        text = "".join(map(str, bits))
        if not problem.feasible(text):
            assert qubo.energy(text) > exact.best.energy


def test_bit_order_and_invalid_selection():
    p = build_problem(DesignSpec(groups=3))
    assert p.bits((0, 1, 0)) == "100110"
    assert p.selection("100110") == (0, 1, 0)
    assert p.selection("110010") is None
    with pytest.raises(ValueError):
        p.selection("101")


@pytest.mark.parametrize("family", ["drive", "clamp"])
def test_compact_encoding_preserves_every_design_energy(family):
    p = build_problem(DesignSpec(family=family))
    full, compact = p.encode(), p.encode().compact_binary()
    assert compact.n*2 == full.n
    for choices in p.assignments():
        assert compact.energy(choices) == pytest.approx(full.energy(p.bits(choices)))

