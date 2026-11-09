# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit Nature, PySCF, or OpenFermion was copied.
"""Chemistry result schemas for executable compatibility slices."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar


@dataclass
class ChemistryResult:
    """Serializable chemistry workflow result envelope for Stage 9D."""

    workflow: str
    molecule: str
    mode: str
    capability_level: int
    bond_length: float | None = None
    basis: str = "sto-3g"
    driver: str | None = None
    mapper: str | None = None
    qubit_count: int | None = None
    particle_count: int | tuple[int, int] | None = None
    pauli_terms: list[dict[str, Any]] = field(default_factory=list)
    ground_state_energy: float | None = None
    electronic_energy: float | None = None
    nuclear_repulsion_energy: float | None = None
    eigenvalues: list[float] = field(default_factory=list)
    raw_type: str | None = None
    upstream_package: str | None = None
    upstream_version: str | None = None
    quantumbridge_version: str | None = None
    production_ready: bool = False
    native_implementation: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    unsupported_reason: str | None = None
    schema_version: str = "0.1"
    ecosystem: str = "quantumbridge_native_chemistry"

    ECOSYSTEM: ClassVar[str | None] = None

    def __post_init__(self) -> None:
        if self.ECOSYSTEM is not None:
            self.ecosystem = self.ECOSYSTEM
        self.validate()

    def validate(self) -> bool:
        if self.mode not in {"native_minimal", "upstream_passthrough", "comparison"}:
            raise ValueError("mode must be native_minimal, upstream_passthrough, or comparison")
        if self.capability_level not in {0, 1, 2, 3, 4}:
            raise ValueError("capability_level must be between 0 and 4")
        if not self.workflow or not self.molecule:
            raise ValueError("workflow and molecule must be non-empty")
        if self.ground_state_energy is None and self.unsupported_reason is None:
            raise ValueError("executable chemistry results require ground_state_energy or unsupported_reason")
        if not isinstance(self.pauli_terms, list):
            raise TypeError("pauli_terms must be a list")
        if not isinstance(self.eigenvalues, list):
            raise TypeError("eigenvalues must be a list")
        if not isinstance(self.metadata, dict) or not isinstance(self.provenance, dict):
            raise TypeError("metadata and provenance must be dictionaries")
        if not isinstance(self.warnings, list):
            raise TypeError("warnings must be a list")
        return True

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ChemistryResult":
        data = dict(payload)
        data["pauli_terms"] = [dict(row) for row in data.get("pauli_terms", ())]
        data["eigenvalues"] = [float(value) for value in data.get("eigenvalues", ())]
        data["metadata"] = dict(data.get("metadata", {}))
        data["warnings"] = [str(value) for value in data.get("warnings", ())]
        data["provenance"] = dict(data.get("provenance", {}))
        return cls(**data)


class MolecularProblemResult(ChemistryResult):
    ECOSYSTEM = "quantumbridge_native_chemistry"


class QubitHamiltonianResult(ChemistryResult):
    ECOSYSTEM = "quantumbridge_native_chemistry"


class ExactDiagonalizationResult(ChemistryResult):
    ECOSYSTEM = "quantumbridge_native_chemistry"


class H2WorkflowResult(ChemistryResult):
    ECOSYSTEM = "quantumbridge_native_chemistry"


class LiHWorkflowResult(ChemistryResult):
    ECOSYSTEM = "quantumbridge_native_chemistry"


class UpstreamNatureResult(ChemistryResult):
    ECOSYSTEM = "qiskit_nature"


class ChemistryComparisonResult(ChemistryResult):
    ECOSYSTEM = "qiskit_nature_comparison"
