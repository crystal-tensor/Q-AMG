# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
"""Chemistry result schema adapter."""

from quantumbridge.compat.qiskit_nature.nature_adapter import ADAPTER
from quantumbridge.compat.qiskit_nature.chemistry_native import QISKIT_NATURE_WARNING
from quantumbridge.schema import ChemistryResult


def wrap_chemistry_result(obj, metadata=None) -> ChemistryResult:
    payload = dict(metadata or {})
    payload.setdefault("warning", QISKIT_NATURE_WARNING)
    return ADAPTER.make_result(ChemistryResult, obj, metadata=payload)
