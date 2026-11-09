# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit Optimization was copied.
"""Native educational binary QuadraticProgram helpers.

The implementation here is intentionally small and deterministic. It supports
binary variables, linear/quadratic objectives, linear constraints, exact
brute-force solving for small problems, and metadata useful for compatibility
tests. It is not a production optimizer or a full Qiskit Optimization clone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Iterable, Mapping, Sequence

from quantumbridge.compat.qiskit_optimization.quadratic_program_result import NativeOptimizationResult
from quantumbridge.compat.qiskit_optimization.warnings import (
    NATIVE_QUADRATIC_PROGRAM_WARNING,
    OPTIMIZATION_ADAPTER_WARNING,
)

SUPPORTED_CONSTRAINT_SENSES = {"==", "<=", ">="}
SUPPORTED_OBJECTIVE_SENSES = {"minimize", "maximize"}


@dataclass(frozen=True)
class NativeBinaryVariable:
    """Binary decision variable."""

    name: str


@dataclass(frozen=True)
class NativeLinearConstraint:
    """Linear constraint over binary variables."""

    coefficients: dict[str, float]
    sense: str
    rhs: float
    name: str | None = None

    def validate(self, variable_names: set[str]) -> bool:
        if self.sense not in SUPPORTED_CONSTRAINT_SENSES:
            raise ValueError("constraint sense must be one of ==, <=, >=")
        missing = set(self.coefficients) - variable_names
        if missing:
            raise ValueError(f"constraint references unknown variables: {sorted(missing)}")
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "coefficients": dict(self.coefficients),
            "sense": self.sense,
            "rhs": self.rhs,
            "name": self.name,
        }


@dataclass(frozen=True)
class NativeQuadraticObjective:
    """Linear plus quadratic objective."""

    sense: str = "minimize"
    constant: float = 0.0
    linear: dict[str, float] = field(default_factory=dict)
    quadratic: dict[tuple[str, str], float] = field(default_factory=dict)

    def validate(self, variable_names: set[str]) -> bool:
        if self.sense not in SUPPORTED_OBJECTIVE_SENSES:
            raise ValueError("objective sense must be minimize or maximize")
        missing = set(self.linear) - variable_names
        missing.update({name for pair in self.quadratic for name in pair if name not in variable_names})
        if missing:
            raise ValueError(f"objective references unknown variables: {sorted(missing)}")
        return True


@dataclass
class NativeQuadraticProgram:
    """Minimal binary QuadraticProgram data model."""

    name: str | None = None
    variables: list[NativeBinaryVariable] = field(default_factory=list)
    objective: NativeQuadraticObjective = field(default_factory=NativeQuadraticObjective)
    constraints: list[NativeLinearConstraint] = field(default_factory=list)
    warnings: list[str] = field(
        default_factory=lambda: [OPTIMIZATION_ADAPTER_WARNING, NATIVE_QUADRATIC_PROGRAM_WARNING]
    )
    provenance: dict[str, Any] = field(
        default_factory=lambda: {
            "adapter_package": "qiskit-optimization",
            "official_endorsement": False,
            "native_subset": "binary quadratic program exact enumeration",
            "copied_upstream_source": False,
            "cloud_access": False,
            "token_access": False,
            "hardware_access": False,
        }
    )

    @property
    def variable_names(self) -> tuple[str, ...]:
        return tuple(variable.name for variable in self.variables)

    def validate(self) -> bool:
        names = self.variable_names
        if len(set(names)) != len(names):
            raise ValueError("variable names must be unique")
        variable_set = set(names)
        self.objective.validate(variable_set)
        for constraint in self.constraints:
            constraint.validate(variable_set)
        return True


def create_quadratic_program_native(name: str | None = None) -> NativeQuadraticProgram:
    """Create an empty native binary QuadraticProgram."""

    return NativeQuadraticProgram(name=name)


def add_binary_var(problem: NativeQuadraticProgram, name: str) -> NativeQuadraticProgram:
    """Add a binary variable to a native problem."""

    _require_problem(problem)
    normalized = _normalize_name(name)
    if normalized in problem.variable_names:
        raise ValueError(f"duplicate binary variable: {normalized}")
    problem.variables.append(NativeBinaryVariable(normalized))
    problem.validate()
    return problem


def add_binary_vars(problem: NativeQuadraticProgram, names: Iterable[str]) -> NativeQuadraticProgram:
    """Add multiple binary variables."""

    for name in names:
        add_binary_var(problem, name)
    return problem


def set_minimize(
    problem: NativeQuadraticProgram,
    linear: Mapping[str, float] | None = None,
    quadratic: Mapping[tuple[str, str], float] | None = None,
    constant: float = 0.0,
) -> NativeQuadraticProgram:
    """Set a minimization objective."""

    return _set_objective(problem, "minimize", linear, quadratic, constant)


def set_maximize(
    problem: NativeQuadraticProgram,
    linear: Mapping[str, float] | None = None,
    quadratic: Mapping[tuple[str, str], float] | None = None,
    constant: float = 0.0,
) -> NativeQuadraticProgram:
    """Set a maximization objective."""

    return _set_objective(problem, "maximize", linear, quadratic, constant)


def add_linear_constraint(
    problem: NativeQuadraticProgram,
    coefficients: Mapping[str, float],
    sense: str,
    rhs: float,
    name: str | None = None,
) -> NativeQuadraticProgram:
    """Add a linear constraint."""

    _require_problem(problem)
    normalized = _normalize_linear(coefficients)
    constraint = NativeLinearConstraint(
        coefficients=normalized,
        sense=_normalize_sense(sense),
        rhs=float(rhs),
        name=str(name) if name is not None else None,
    )
    constraint.validate(set(problem.variable_names))
    problem.constraints.append(constraint)
    problem.validate()
    return problem


def evaluate_objective(
    problem: NativeQuadraticProgram,
    assignment: Mapping[str, int] | Sequence[int],
) -> float:
    """Evaluate the objective in its declared sense."""

    _require_problem(problem)
    row = _assignment_dict(problem, assignment)
    value = float(problem.objective.constant)
    for name, coefficient in problem.objective.linear.items():
        value += coefficient * row[name]
    for (left, right), coefficient in problem.objective.quadratic.items():
        value += coefficient * row[left] * row[right]
    return value


def is_feasible(problem: NativeQuadraticProgram, assignment: Mapping[str, int] | Sequence[int]) -> bool:
    """Return whether an assignment satisfies all linear constraints."""

    _require_problem(problem)
    row = _assignment_dict(problem, assignment)
    for constraint in problem.constraints:
        lhs = sum(coefficient * row[name] for name, coefficient in constraint.coefficients.items())
        if constraint.sense == "==" and abs(lhs - constraint.rhs) > 1e-9:
            return False
        if constraint.sense == "<=" and lhs > constraint.rhs + 1e-9:
            return False
        if constraint.sense == ">=" and lhs < constraint.rhs - 1e-9:
            return False
    return True


def enumerate_binary_assignments(problem: NativeQuadraticProgram) -> tuple[dict[str, int], ...]:
    """Enumerate all binary assignments in variable order."""

    _require_problem(problem)
    names = problem.variable_names
    return tuple(dict(zip(names, values)) for values in product((0, 1), repeat=len(names)))


def solve_quadratic_program_bruteforce_native(
    problem: NativeQuadraticProgram,
    *,
    include_metadata: bool = True,
) -> NativeOptimizationResult:
    """Solve a small binary quadratic program by complete enumeration."""

    _require_problem(problem)
    problem.validate()
    samples: list[dict[str, Any]] = []
    feasible_rows: list[dict[str, int]] = []
    for assignment in enumerate_binary_assignments(problem):
        objective_value = evaluate_objective(problem, assignment)
        feasible = is_feasible(problem, assignment)
        bitstring = "".join(str(assignment[name]) for name in problem.variable_names)
        row = {
            "assignment": dict(assignment),
            "bitstring": bitstring,
            "objective_value": objective_value,
            "feasible": feasible,
        }
        samples.append(row)
        if feasible:
            feasible_rows.append(dict(assignment))

    feasible_samples = [sample for sample in samples if sample["feasible"]]
    reverse = problem.objective.sense == "maximize"
    feasible_samples.sort(key=lambda sample: (sample["objective_value"], sample["bitstring"]), reverse=reverse)
    if feasible_samples:
        best = feasible_samples[0]
        assignment = dict(best["assignment"])
        objective_value: float | None = float(best["objective_value"])
        feasible = True
    else:
        assignment = {}
        objective_value = None
        feasible = False

    qubo = quadratic_program_to_qubo_metadata(problem) if include_metadata else {}
    ising = quadratic_program_to_ising_metadata(problem) if include_metadata else {}
    result = NativeOptimizationResult(
        problem_name=problem.name or "native_quadratic_program",
        objective_sense=problem.objective.sense,
        objective_value=objective_value,
        assignment=assignment,
        feasible=feasible,
        method="bruteforce-exact",
        path="quantumbridge-native",
        samples=tuple(samples),
        feasible_assignments=tuple(feasible_rows),
        constraints=tuple(constraint.to_dict() for constraint in problem.constraints),
        qubo_metadata=qubo,
        ising_metadata=ising,
        warnings=tuple(problem.warnings),
        provenance=dict(problem.provenance),
        metadata={
            "num_variables": len(problem.variables),
            "num_constraints": len(problem.constraints),
            "solver": "deterministic full enumeration",
            "production_ready": False,
        },
        unsupported_reason=None if feasible else "no feasible binary assignment",
    )
    result.validate()
    return result


def quadratic_program_to_qubo_metadata(
    problem: NativeQuadraticProgram,
    penalty: float | None = None,
) -> dict[str, Any]:
    """Return QUBO-style metadata for the objective and constraints."""

    _require_problem(problem)
    problem.validate()
    sign = -1.0 if problem.objective.sense == "maximize" else 1.0
    linear = {name: sign * value for name, value in problem.objective.linear.items()}
    quadratic = {pair: sign * value for pair, value in problem.objective.quadratic.items()}
    offset = sign * problem.objective.constant
    warnings: list[str] = []
    penalty_applied = False

    if penalty is not None:
        penalty_value = float(penalty)
        for constraint in problem.constraints:
            if constraint.sense != "==":
                warnings.append("penalty conversion only applies equality constraints in this minimal subset")
                continue
            penalty_applied = True
            offset += penalty_value * constraint.rhs * constraint.rhs
            for name, coefficient in constraint.coefficients.items():
                linear[name] = linear.get(name, 0.0) + penalty_value * (coefficient * coefficient - 2.0 * coefficient * constraint.rhs)
            items = list(constraint.coefficients.items())
            for left_index, (left_name, left_coefficient) in enumerate(items):
                for right_name, right_coefficient in items[left_index + 1 :]:
                    pair = _normalize_pair(left_name, right_name)
                    quadratic[pair] = quadratic.get(pair, 0.0) + 2.0 * penalty_value * left_coefficient * right_coefficient

    return {
        "format": "qubo-metadata-v0.1",
        "variables": list(problem.variable_names),
        "objective_sense": problem.objective.sense,
        "minimization_offset": offset,
        "linear": dict(sorted(linear.items())),
        "quadratic": _quadratic_entries(quadratic),
        "constraints": [constraint.to_dict() for constraint in problem.constraints],
        "penalty": penalty,
        "penalty_applied": penalty_applied,
        "warnings": warnings,
        "production_ready": False,
    }


def quadratic_program_to_ising_metadata(
    problem: NativeQuadraticProgram,
    penalty: float | None = None,
) -> dict[str, Any]:
    """Return Ising-style metadata using x = (1 - z) / 2 for QUBO terms."""

    qubo = quadratic_program_to_qubo_metadata(problem, penalty=penalty)
    linear = {str(name): float(value) for name, value in qubo["linear"].items()}
    quadratic: dict[tuple[str, str], float] = {
        (str(entry["left"]), str(entry["right"])): float(entry["coefficient"])
        for entry in qubo["quadratic"]
    }
    offset = float(qubo["minimization_offset"])
    h: dict[str, float] = {}
    j: dict[tuple[str, str], float] = {}

    for name, coefficient in linear.items():
        offset += coefficient / 2.0
        h[name] = h.get(name, 0.0) - coefficient / 2.0
    for (left, right), coefficient in quadratic.items():
        if left == right:
            offset += coefficient / 2.0
            h[left] = h.get(left, 0.0) - coefficient / 2.0
            continue
        offset += coefficient / 4.0
        h[left] = h.get(left, 0.0) - coefficient / 4.0
        h[right] = h.get(right, 0.0) - coefficient / 4.0
        pair = _normalize_pair(left, right)
        j[pair] = j.get(pair, 0.0) + coefficient / 4.0

    return {
        "format": "ising-metadata-v0.1",
        "variables": list(problem.variable_names),
        "source_qubo_format": qubo["format"],
        "offset": offset,
        "h": dict(sorted(h.items())),
        "j": _quadratic_entries(j),
        "warnings": list(qubo.get("warnings", ())),
        "production_ready": False,
    }


def native_quadratic_program_to_dict(problem: NativeQuadraticProgram) -> dict[str, Any]:
    """Serialize a native problem."""

    _require_problem(problem)
    problem.validate()
    return {
        "name": problem.name,
        "variables": [{"name": variable.name, "type": "binary"} for variable in problem.variables],
        "objective": {
            "sense": problem.objective.sense,
            "constant": problem.objective.constant,
            "linear": dict(problem.objective.linear),
            "quadratic": _quadratic_entries(problem.objective.quadratic),
        },
        "constraints": [constraint.to_dict() for constraint in problem.constraints],
        "warnings": list(problem.warnings),
        "provenance": dict(problem.provenance),
    }


def native_quadratic_program_from_dict(data: Mapping[str, Any]) -> NativeQuadraticProgram:
    """Deserialize a native problem."""

    payload = dict(data)
    problem = create_quadratic_program_native(payload.get("name"))
    for variable in payload.get("variables", ()):
        add_binary_var(problem, variable["name"] if isinstance(variable, Mapping) else str(variable))
    objective = dict(payload.get("objective", {}))
    quadratic = {
        _normalize_pair(entry["left"], entry["right"]): float(entry["coefficient"])
        for entry in objective.get("quadratic", ())
    }
    _set_objective(
        problem,
        objective.get("sense", "minimize"),
        objective.get("linear", {}),
        quadratic,
        objective.get("constant", 0.0),
    )
    for constraint in payload.get("constraints", ()):
        add_linear_constraint(
            problem,
            constraint.get("coefficients", {}),
            constraint.get("sense", "=="),
            constraint.get("rhs", 0.0),
            constraint.get("name"),
        )
    problem.warnings = [str(value) for value in payload.get("warnings", problem.warnings)]
    problem.provenance = dict(payload.get("provenance", problem.provenance))
    problem.validate()
    return problem


def _set_objective(
    problem: NativeQuadraticProgram,
    sense: str,
    linear: Mapping[str, float] | None,
    quadratic: Mapping[tuple[str, str], float] | None,
    constant: float,
) -> NativeQuadraticProgram:
    _require_problem(problem)
    objective = NativeQuadraticObjective(
        sense=str(sense),
        constant=float(constant),
        linear=_normalize_linear(linear or {}),
        quadratic=_normalize_quadratic(quadratic or {}),
    )
    objective.validate(set(problem.variable_names))
    problem.objective = objective
    problem.validate()
    return problem


def _normalize_name(name: str) -> str:
    normalized = str(name).strip()
    if not normalized:
        raise ValueError("variable name must be non-empty")
    return normalized


def _normalize_sense(sense: str) -> str:
    normalized = str(sense).strip()
    aliases = {"=": "==", "EQ": "==", "LE": "<=", "GE": ">="}
    normalized = aliases.get(normalized.upper(), normalized)
    if normalized not in SUPPORTED_CONSTRAINT_SENSES:
        raise ValueError("constraint sense must be one of ==, <=, >=")
    return normalized


def _normalize_linear(values: Mapping[str, float]) -> dict[str, float]:
    return {str(name): float(value) for name, value in values.items() if float(value) != 0.0}


def _normalize_quadratic(values: Mapping[tuple[str, str], float]) -> dict[tuple[str, str], float]:
    normalized: dict[tuple[str, str], float] = {}
    for pair, value in values.items():
        if len(pair) != 2:
            raise ValueError("quadratic objective keys must be variable pairs")
        coefficient = float(value)
        if coefficient == 0.0:
            continue
        key = _normalize_pair(pair[0], pair[1])
        normalized[key] = normalized.get(key, 0.0) + coefficient
    return normalized


def _normalize_pair(left: str, right: str) -> tuple[str, str]:
    first = _normalize_name(left)
    second = _normalize_name(right)
    return tuple(sorted((first, second)))


def _assignment_dict(problem: NativeQuadraticProgram, assignment: Mapping[str, int] | Sequence[int]) -> dict[str, int]:
    names = problem.variable_names
    if isinstance(assignment, Mapping):
        row = {str(name): int(value) for name, value in assignment.items()}
    else:
        if len(assignment) != len(names):
            raise ValueError("assignment length must match variable count")
        row = dict(zip(names, (int(value) for value in assignment)))
    missing = set(names) - set(row)
    extra = set(row) - set(names)
    if missing or extra:
        raise ValueError(f"assignment keys mismatch; missing={sorted(missing)} extra={sorted(extra)}")
    for name, value in row.items():
        if value not in {0, 1}:
            raise ValueError(f"assignment for {name} must be binary")
    return row


def _quadratic_entries(values: Mapping[tuple[str, str], float]) -> list[dict[str, Any]]:
    return [
        {"left": left, "right": right, "coefficient": coefficient}
        for (left, right), coefficient in sorted(values.items())
    ]


def _require_problem(problem: NativeQuadraticProgram) -> None:
    if not isinstance(problem, NativeQuadraticProgram):
        raise TypeError("problem must be a NativeQuadraticProgram")
