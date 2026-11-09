import subprocess
import sys

import numpy as np
from quantumbridge import Circuit

from qamg.torch_backend import TorchStatevectorBackend


def test_noncommuting_qb_gates_match_independent_numpy_matrix_oracle():
    a, b = .31, -.73
    h = np.array([[1, 1], [1, -1]])/np.sqrt(2)
    ry = np.array([[np.cos(a/2), -np.sin(a/2)], [np.sin(a/2), np.cos(a/2)]])
    rz = np.diag([np.exp(-1j*b/2), np.exp(1j*b/2)])
    expected = np.abs(rz @ h @ ry @ np.array([1., 0.]))**2
    q = Circuit(1).ry(a, 0).h(0).rz(b, 0).measure_all()
    backend = TorchStatevectorBackend("cpu")
    np.testing.assert_allclose(backend.probabilities(backend.compile(q, 42)), expected, atol=1e-6)


def test_active_quantum_modules_run_with_foreign_sdk_imports_blocked():
    code = """
import sys, importlib.abc
class BlockForeign(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'qiskit','qiskit_aer','pennylane'}:
            raise RuntimeError('Foreign SDK runtime import: '+fullname)
sys.meta_path.insert(0, BlockForeign())
from qamg.backends import QBBackend
from qamg.quantum import QuantumStructurePrior
from qamg.problem import build_problem
from qamg.schema import DesignSpec,PlannerSettings
from qamg import hardware, runtime
p=build_problem(DesignSpec(groups=3))
r=QuantumStructurePrior(QBBackend()).plan(p,PlannerSettings(encoding='compact_binary',iterations=0,depth=1,shots=64),42)
assert r.best is not None and r.resources['quantum_sdk']=='QuantumBridge'
assert not any(k.split('.')[0] in {'qiskit','qiskit_aer','pennylane'} for k in sys.modules)
print('QB native circuit, compiler, sample and serialization passed without foreign SDKs')
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False, timeout=60)
    assert result.returncode == 0, result.stderr
