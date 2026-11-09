# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
"""Qiskit Optimization QuadraticProgram inventory, passthrough, and native subset.

Design source: docs/compat/strategy/qiskit_optimization_compatibility_strategy.md.
"""

from __future__ import annotations

from importlib import import_module, metadata
from typing import Any

from quantumbridge.ecosystem.registry import EcosystemAdapter
from quantumbridge.compat.qiskit_optimization.quadratic_program_native import (
    NativeQuadraticProgram,
    add_binary_var,
    add_binary_vars,
    add_linear_constraint,
    create_quadratic_program_native,
    evaluate_objective,
    enumerate_binary_assignments,
    is_feasible,
    native_quadratic_program_from_dict,
    native_quadratic_program_to_dict,
    quadratic_program_to_ising_metadata,
    quadratic_program_to_qubo_metadata,
    set_maximize,
    set_minimize,
    solve_quadratic_program_bruteforce_native,
)
from quantumbridge.compat.qiskit_optimization.quadratic_program_result import NativeOptimizationResult
from quantumbridge.compat.qiskit_optimization.warnings import UPSTREAM_PASSTHROUGH_WARNING

ADAPTER = EcosystemAdapter("qiskit-optimization", "qiskit-optimization", ("qiskit_optimization", "qiskit_optimization.problems"), "qiskit-optimization", "qiskit_optimization_quadratic_program")

dependency_available = ADAPTER.dependency_available
get_upstream_version = ADAPTER.get_upstream_version
list_public_api_inventory = ADAPTER.list_public_api_inventory
passthrough_class = ADAPTER.passthrough_class
passthrough_function = ADAPTER.passthrough_function
wrap_result = ADAPTER.wrap_result
to_quantumbridge_schema = ADAPTER.to_quantumbridge_schema
provenance_metadata = ADAPTER.provenance_metadata
warn_unsupported = ADAPTER.warn_unsupported


def validate_optimization_dependencies() -> dict[str, Any]:
    """Return dependency status for optional upstream optimization packages."""

    modules = ("qiskit_optimization", "qiskit_algorithms")
    packages = {"qiskit_optimization": "qiskit-optimization", "qiskit_algorithms": "qiskit-algorithms"}
    report: dict[str, Any] = {"available": True, "modules": {}, "warnings": [UPSTREAM_PASSTHROUGH_WARNING]}
    for module_name in modules:
        package_name = packages[module_name]
        try:
            import_module(module_name)
            version = metadata.version(package_name)
            report["modules"][module_name] = {
                "available": True,
                "package": package_name,
                "version": version,
            }
        except Exception as exc:  # pragma: no cover - exact exception depends on env
            report["available"] = False
            report["modules"][module_name] = {
                "available": False,
                "package": package_name,
                "version": None,
                "error": str(exc),
            }
    return report


def create_upstream_quadratic_program(name: str | None = None):
    """Create an upstream qiskit_optimization.QuadraticProgram if installed."""

    QuadraticProgram = _require_upstream_quadratic_program_class()
    return QuadraticProgram(name or "quantumbridge_native_quadratic_program")


def native_to_upstream_quadratic_program(problem: NativeQuadraticProgram):
    """Convert the native minimal model to an upstream QuadraticProgram."""

    problem.validate()
    upstream = create_upstream_quadratic_program(problem.name)
    for variable in problem.variables:
        upstream.binary_var(variable.name)
    quadratic = dict(problem.objective.quadratic)
    if problem.objective.sense == "minimize":
        upstream.minimize(
            constant=problem.objective.constant,
            linear=dict(problem.objective.linear),
            quadratic=quadratic,
        )
    else:
        upstream.maximize(
            constant=problem.objective.constant,
            linear=dict(problem.objective.linear),
            quadratic=quadratic,
        )
    for constraint in problem.constraints:
        upstream.linear_constraint(
            linear=dict(constraint.coefficients),
            sense=constraint.sense,
            rhs=constraint.rhs,
            name=constraint.name,
        )
    return upstream


def upstream_to_native_quadratic_program(qp) -> NativeQuadraticProgram:
    """Best-effort conversion from an upstream QuadraticProgram to native form."""

    problem = create_quadratic_program_native(getattr(qp, "name", None))
    for variable in getattr(qp, "variables", ()):
        add_binary_var(problem, getattr(variable, "name", str(variable)))

    objective = getattr(qp, "objective", None)
    sense = _objective_sense_name(getattr(objective, "sense", None))
    linear = _linear_coefficients(getattr(objective, "linear", None), problem.variable_names)
    quadratic = _quadratic_coefficients(getattr(objective, "quadratic", None), problem.variable_names)
    constant = float(getattr(objective, "constant", 0.0) or 0.0)
    if sense == "maximize":
        set_maximize(problem, linear=linear, quadratic=quadratic, constant=constant)
    else:
        set_minimize(problem, linear=linear, quadratic=quadratic, constant=constant)

    for constraint in getattr(qp, "linear_constraints", ()):
        add_linear_constraint(
            problem,
            _linear_coefficients(getattr(constraint, "linear", None), problem.variable_names),
            _constraint_sense_name(getattr(constraint, "sense", None)),
            float(getattr(constraint, "rhs", 0.0) or 0.0),
            getattr(constraint, "name", None),
        )
    problem.validate()
    return problem


def solve_quadratic_program_exact_upstream(problem: NativeQuadraticProgram) -> NativeOptimizationResult:
    """Solve through optional upstream MinimumEigenOptimizer if available."""

    _require_upstream_optimizer_modules()
    from qiskit_algorithms import NumPyMinimumEigensolver
    from qiskit_optimization.algorithms import MinimumEigenOptimizer

    upstream_problem = native_to_upstream_quadratic_program(problem)
    raw_result = MinimumEigenOptimizer(NumPyMinimumEigensolver()).solve(upstream_problem)
    return wrap_upstream_optimization_result(raw_result, problem)


def wrap_upstream_optimization_result(raw, problem: NativeQuadraticProgram | None = None) -> NativeOptimizationResult:
    """Wrap an upstream optimization result in the native serializable shape."""

    assignment = _raw_assignment(raw, problem)
    objective_value = getattr(raw, "fval", None)
    if objective_value is None:
        objective_value = getattr(raw, "objective_value", None)
    if problem is not None:
        feasible = is_feasible(problem, assignment) if assignment else False
        objective_sense = problem.objective.sense
        constraints = tuple(constraint.to_dict() for constraint in problem.constraints)
        qubo_metadata = quadratic_program_to_qubo_metadata(problem)
        ising_metadata = quadratic_program_to_ising_metadata(problem)
        num_variables = len(problem.variables)
        provenance = dict(problem.provenance)
    else:
        feasible = bool(assignment)
        objective_sense = "minimize"
        constraints = ()
        qubo_metadata = {}
        ising_metadata = {}
        num_variables = len(assignment)
        provenance = {
            "adapter_package": "qiskit-optimization",
            "official_endorsement": False,
            "copied_upstream_source": False,
        }
    result = NativeOptimizationResult(
        problem_name=(problem.name if problem else None) or "upstream_quadratic_program",
        objective_sense=objective_sense,
        objective_value=float(objective_value) if objective_value is not None else None,
        assignment=assignment,
        feasible=feasible,
        method="upstream-minimum-eigen-optimizer",
        path="qiskit-optimization-upstream",
        constraints=constraints,
        qubo_metadata=qubo_metadata,
        ising_metadata=ising_metadata,
        upstream_package="qiskit-optimization",
        upstream_version=_metadata_version("qiskit-optimization"),
        warnings=(UPSTREAM_PASSTHROUGH_WARNING,),
        provenance=provenance,
        metadata={
            "raw_type": type(raw).__name__,
            "num_variables": num_variables,
            "production_ready": False,
        },
        unsupported_reason=None if feasible else "upstream result did not expose a feasible assignment",
    )
    result.validate()
    return result


def _require_upstream_quadratic_program_class():
    try:
        from qiskit_optimization import QuadraticProgram
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        raise ImportError("qiskit-optimization is required for upstream QuadraticProgram passthrough") from exc
    return QuadraticProgram


def _require_upstream_optimizer_modules() -> None:
    missing = []
    for module_name, package_name in (
        ("qiskit_optimization", "qiskit-optimization"),
        ("qiskit_algorithms", "qiskit-algorithms"),
    ):
        try:
            import_module(module_name)
        except Exception:
            missing.append(package_name)
    if missing:
        raise ImportError(f"optional upstream packages are required: {', '.join(missing)}")


def _metadata_version(package_name: str) -> str | None:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def _objective_sense_name(sense: Any) -> str:
    text = str(getattr(sense, "name", sense)).lower()
    return "maximize" if "max" in text else "minimize"


def _constraint_sense_name(sense: Any) -> str:
    text = str(getattr(sense, "name", sense)).upper()
    if "GE" in text or ">=" in text:
        return ">="
    if "LE" in text or "<=" in text:
        return "<="
    return "=="


def _linear_coefficients(linear: Any, variable_names: tuple[str, ...]) -> dict[str, float]:
    if linear is None:
        return {}
    if hasattr(linear, "to_dict"):
        raw = linear.to_dict()
        return {str(key): float(value) for key, value in raw.items()}
    if hasattr(linear, "to_array"):
        values = linear.to_array()
        return {
            name: float(values[index])
            for index, name in enumerate(variable_names)
            if float(values[index]) != 0.0
        }
    if isinstance(linear, dict):
        return {str(key): float(value) for key, value in linear.items()}
    return {}


def _quadratic_coefficients(quadratic: Any, variable_names: tuple[str, ...]) -> dict[tuple[str, str], float]:
    if quadratic is None:
        return {}
    if hasattr(quadratic, "to_dict"):
        raw = quadratic.to_dict()
        return {
            (str(left), str(right)): float(value)
            for (left, right), value in raw.items()
        }
    if hasattr(quadratic, "to_array"):
        values = quadratic.to_array()
        rows = {}
        for left_index, left_name in enumerate(variable_names):
            for right_index, right_name in enumerate(variable_names):
                value = float(values[left_index][right_index])
                if value != 0.0:
                    rows[(left_name, right_name)] = value
        return rows
    if isinstance(quadratic, dict):
        return {(str(left), str(right)): float(value) for (left, right), value in quadratic.items()}
    return {}


def _raw_assignment(raw: Any, problem: NativeQuadraticProgram | None) -> dict[str, int]:
    names = problem.variable_names if problem is not None else ()
    if hasattr(raw, "x") and raw.x is not None and names:
        return {name: int(round(float(raw.x[index]))) for index, name in enumerate(names)}
    if hasattr(raw, "variables_dict") and raw.variables_dict is not None:
        return {str(key): int(round(float(value))) for key, value in raw.variables_dict.items()}
    return {}


__all__ = [
    "ADAPTER",
    "NativeQuadraticProgram",
    "NativeOptimizationResult",
    "add_binary_var",
    "add_binary_vars",
    "add_linear_constraint",
    "create_quadratic_program_native",
    "create_upstream_quadratic_program",
    "dependency_available",
    "enumerate_binary_assignments",
    "evaluate_objective",
    "get_upstream_version",
    "is_feasible",
    "list_public_api_inventory",
    "native_quadratic_program_from_dict",
    "native_quadratic_program_to_dict",
    "native_to_upstream_quadratic_program",
    "passthrough_class",
    "passthrough_function",
    "provenance_metadata",
    "quadratic_program_to_ising_metadata",
    "quadratic_program_to_qubo_metadata",
    "set_maximize",
    "set_minimize",
    "solve_quadratic_program_bruteforce_native",
    "solve_quadratic_program_exact_upstream",
    "to_quantumbridge_schema",
    "upstream_to_native_quadratic_program",
    "validate_optimization_dependencies",
    "warn_unsupported",
    "wrap_result",
    "wrap_upstream_optimization_result",
]
