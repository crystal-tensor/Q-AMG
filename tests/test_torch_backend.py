import numpy as np
import pytest
from quantumbridge import Circuit, StatevectorDevice

from qamg.problem import build_problem
from qamg.quantum import QuantumStructurePrior
from qamg.schema import DesignSpec, PlannerSettings
from qamg.torch_backend import TorchStatevectorBackend, available_gpu


@pytest.mark.parametrize("requested", ["cpu", "gpu"])
def test_qb_gpu_evolution_matches_qb_cpu_reference(requested):
    device = available_gpu() if requested == "gpu" else "cpu"
    if device is None:
        pytest.skip("No supported GPU on this machine")
    q = Circuit(3, 3)
    q.rx(.3, 0)
    q.h(1)
    q.cx(1, 2)
    q.rz(.7, 2)
    q.ry(.4, 1)
    q.swap(0, 2)
    for wire in range(3):
        q.measure(wire, wire)
    backend = TorchStatevectorBackend(device)
    compiled = backend.compile(q, 41)
    actual = backend.probabilities(compiled)
    reference = StatevectorDevice().statevector(compiled)
    np.testing.assert_allclose(actual, np.abs(reference)**2, atol=2e-6)


def test_classical_register_permutation_is_not_endian_guessing():
    q = Circuit(3, 3)
    q.x(0)
    for wire in range(3):
        q.measure(wire, 2-wire)
    b = TorchStatevectorBackend("cpu")
    batch = b.sample(b.compile(q, 42), 64, 42)
    assert batch.counts == {"001": 64}


def test_gpu_measured_structure_trace_is_explicitly_a_simulation():
    device = available_gpu()
    if device is None:
        pytest.skip("No supported GPU on this machine")
    p = build_problem(DesignSpec(groups=3))
    r = QuantumStructurePrior(TorchStatevectorBackend(device)).plan(
        p, PlannerSettings(encoding="compact_binary", iterations=0, depth=1, shots=128), 42)
    assert r.best is not None
    assert r.execution_source == "ideal_simulator"
    assert r.resources["capabilities"]["is_simulator"] is True
    assert r.trace[0]["metadata"]["state_evolution_device"] == device
    assert r.trace[0]["metadata"]["born_sampling_device"] == "cpu"
    assert r.timings["quantum_execution_s"] is None
    assert r.resources["statevector_bytes_lower_bound"] == 8*2**3

