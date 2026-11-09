# This file is independently implemented for QuantumBridge SDK.
# No source code from IBM, Qiskit, or Benchpress was copied.
"""Default local benchmark suites built from existing executable slices."""

from __future__ import annotations

import math
from typing import Any

from quantumbridge.core import Circuit

from .benchmark_case import BenchmarkCase, create_benchmark_case
from .metrics import energy_delta, l1_distance, max_probability_delta, objective_delta


def circuit_basic_suite() -> list[BenchmarkCase]:
    return [
        create_benchmark_case(
            "H circuit statevector",
            "circuit_basic",
            {"circuit": "single-qubit H", "runner": "Stage 9F statevector"},
            {"probabilities": {"0": 0.5, "1": 0.5}},
            runner=_run_h_statevector_case,
            tags=["circuit", "statevector"],
        ),
        create_benchmark_case(
            "Bell qasm counts",
            "circuit_basic",
            {"circuit": "Bell measured", "shots": 128},
            {"allowed_counts": ["00", "11"]},
            runner=_run_bell_counts_case,
            tags=["circuit", "counts"],
        ),
        create_benchmark_case(
            "Parameterized RX RY statevector",
            "circuit_basic",
            {"circuit": "RX/RY", "parameters": [math.pi / 4, math.pi / 6]},
            {"probability_sum": 1.0},
            runner=_run_parameterized_case,
            tags=["circuit", "parameters"],
        ),
    ]


def simulator_suite() -> list[BenchmarkCase]:
    return [
        create_benchmark_case("Stage 9F statevector simulator", "simulator", runner=_run_aer_statevector_case, tags=["aer", "statevector"]),
        create_benchmark_case("Stage 9F qasm counts", "simulator", runner=_run_aer_qasm_case, tags=["aer", "counts"]),
        create_benchmark_case("Stage 9F noisy counts", "simulator", runner=_run_aer_noisy_case, tags=["aer", "noise"]),
    ]


def algorithms_suite() -> list[BenchmarkCase]:
    return [
        create_benchmark_case("Stage 9C VQE native", "algorithms", runner=_run_vqe_case, tags=["vqe"]),
        create_benchmark_case("Stage 9C QAOA native", "algorithms", runner=_run_qaoa_case, tags=["qaoa"]),
        create_benchmark_case("Stage 9C Grover native", "algorithms", runner=_run_grover_case, tags=["grover"]),
    ]


def finance_optimization_suite() -> list[BenchmarkCase]:
    return [
        create_benchmark_case("Stage 9A portfolio native", "finance_optimization", runner=_run_portfolio_case, tags=["finance"]),
        create_benchmark_case("Stage 9B quadratic program native", "finance_optimization", runner=_run_quadratic_program_case, tags=["optimization"]),
    ]


def chemistry_suite() -> list[BenchmarkCase]:
    return [
        create_benchmark_case("Stage 9D H2 native", "chemistry", runner=_run_h2_case, tags=["chemistry", "h2"]),
        create_benchmark_case("Stage 9D LiH native", "chemistry", runner=_run_lih_case, tags=["chemistry", "lih"]),
    ]


def qml_suite() -> list[BenchmarkCase]:
    return [
        create_benchmark_case("Stage 9E quantum kernel native", "qml", runner=_run_quantum_kernel_case, tags=["qml", "kernel"]),
        create_benchmark_case("Stage 9E QNN classifier native", "qml", runner=_run_qnn_case, tags=["qml", "qnn"]),
        create_benchmark_case("Stage 9K TorchQuantum-like classifier", "qml", runner=_run_torchquantum_case, tags=["qml", "torchquantum"]),
    ]


def mitigation_suite() -> list[BenchmarkCase]:
    return [
        create_benchmark_case("Stage 9G ZNE native", "mitigation", runner=_run_zne_case, tags=["mitiq", "zne"]),
        create_benchmark_case("Stage 9G readout mitigation native", "mitigation", runner=_run_readout_case, tags=["mitiq", "readout"]),
    ]


def backend_suite() -> list[BenchmarkCase]:
    return [
        create_benchmark_case("Stage 10A QOS-UQCI mock runtime", "backend", runner=_run_qos_uqci_case, tags=["qos-uqci"]),
        create_benchmark_case("Stage 10A Quafu mock backend", "backend", runner=_run_quafu_case, tags=["quafu"]),
    ]


def bridge_suite() -> list[BenchmarkCase]:
    return [
        create_benchmark_case(
            "Stage 9H Qiskit to PennyLane bridge",
            "bridge",
            runner=_run_qiskit_to_pennylane_case,
            tags=["bridge"],
            metadata={"required_dependencies": ["qiskit"]},
        ),
        create_benchmark_case(
            "Stage 9H PennyLane to Qiskit bridge",
            "bridge",
            runner=_run_pennylane_to_qiskit_case,
            tags=["bridge"],
            metadata={"required_dependencies": ["qiskit"]},
        ),
        create_benchmark_case("Stage 9J MQT-like mapping", "bridge", runner=_run_mqt_mapping_case, tags=["mqt", "mapping"]),
    ]


def full_smoke_suite() -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    for suite in (
        circuit_basic_suite,
        simulator_suite,
        algorithms_suite,
        finance_optimization_suite,
        chemistry_suite,
        qml_suite,
        mitigation_suite,
        backend_suite,
        bridge_suite,
    ):
        cases.extend(suite())
    return cases


def suite_by_name(name: str) -> list[BenchmarkCase]:
    suites = {
        "circuit_basic": circuit_basic_suite,
        "simulator": simulator_suite,
        "algorithms": algorithms_suite,
        "finance_optimization": finance_optimization_suite,
        "chemistry": chemistry_suite,
        "qml": qml_suite,
        "mitigation": mitigation_suite,
        "backend": backend_suite,
        "bridge": bridge_suite,
        "full_smoke": full_smoke_suite,
    }
    try:
        return suites[name]()
    except KeyError as exc:
        raise ValueError(f"unknown benchmark suite: {name}") from exc


def _run_h_statevector_case(seed: int | None = None) -> dict[str, Any]:
    from quantumbridge.compat.qiskit_aer.simulator_native import run_statevector_simulator_native

    circuit = Circuit(1, name="bench_h").h(0)
    result = run_statevector_simulator_native(circuit)
    expected = {"0": 0.5, "1": 0.5}
    delta = max_probability_delta(result.probabilities, expected)
    return _payload(result, passed=delta < 1e-9, metrics={"max_probability_delta": delta}, expected_summary=expected)


def _run_bell_counts_case(seed: int | None = None) -> dict[str, Any]:
    from quantumbridge.compat.qiskit_aer.simulator_native import run_qasm_simulator_native

    circuit = Circuit(2, 2, name="bench_bell").h(0).cx(0, 1).measure(0, 0).measure(1, 1)
    result = run_qasm_simulator_native(circuit, shots=128, seed=seed)
    passed = set(result.counts).issubset({"00", "11"}) and sum(result.counts.values()) == 128
    return _payload(result, passed=passed, metrics={"shot_count": sum(result.counts.values())})


def _run_parameterized_case(seed: int | None = None) -> dict[str, Any]:
    from quantumbridge.compat.qiskit_aer.simulator_native import run_statevector_simulator_native

    circuit = Circuit(1, name="bench_parameterized").rx(math.pi / 4, 0).ry(math.pi / 6, 0)
    result = run_statevector_simulator_native(circuit)
    total = sum(result.probabilities.values())
    return _payload(result, passed=abs(total - 1.0) < 1e-9, metrics={"probability_sum": total})


def _run_aer_statevector_case(seed: int | None = None) -> dict[str, Any]:
    from quantumbridge.compat.qiskit_aer.examples import run_statevector_example

    output = run_statevector_example()
    return _payload(output, passed=bool(output["native"].probabilities))


def _run_aer_qasm_case(seed: int | None = None) -> dict[str, Any]:
    from quantumbridge.compat.qiskit_aer.examples import run_qasm_counts_example

    output = run_qasm_counts_example(shots=128, seed=7 if seed is None else seed)
    return _payload(output, passed=bool(output["native"].counts), metrics={"shot_count": sum(output["native"].counts.values())})


def _run_aer_noisy_case(seed: int | None = None) -> dict[str, Any]:
    from quantumbridge.compat.qiskit_aer.examples import run_noisy_counts_example

    output = run_noisy_counts_example(shots=128, seed=7 if seed is None else seed)
    return _payload(output, passed=bool(output["native"].counts), metrics={"nonzero_states": len(output["native"].counts)})


def _run_vqe_case(seed: int | None = None) -> dict[str, Any]:
    from quantumbridge.compat.qiskit_algorithms.algorithms_native import run_vqe_native

    result = run_vqe_native()
    delta = energy_delta(result.eigenvalue, -2.0)
    return _payload(result, passed=delta is not None and delta < 1e-9, metrics={"energy_delta": delta}, expected_summary={"eigenvalue": -2.0})


def _run_qaoa_case(seed: int | None = None) -> dict[str, Any]:
    from quantumbridge.compat.qiskit_algorithms.algorithms_native import run_qaoa_native_maxcut

    result = run_qaoa_native_maxcut(((0, 1), (1, 2), (2, 0)), num_nodes=3)
    return _payload(result, passed=result.objective_value is not None and result.objective_value >= 2.0, metrics={"objective_value": result.objective_value})


def _run_grover_case(seed: int | None = None) -> dict[str, Any]:
    from quantumbridge.compat.qiskit_algorithms.algorithms_native import run_grover_native

    result = run_grover_native(["11"], num_qubits=2)
    probability = result.probabilities.get("11", 0.0)
    return _payload(result, passed=probability >= 0.99, metrics={"marked_probability": probability})


def _run_portfolio_case(seed: int | None = None) -> dict[str, Any]:
    from quantumbridge.compat.qiskit_finance.portfolio_optimization_adapter import run_portfolio_optimization_native

    result = run_portfolio_optimization_native()
    feasible = result.budget is None or sum(result.selection) == result.budget
    return _payload(result, passed=feasible, metrics={"objective_value": result.objective_value})


def _run_quadratic_program_case(seed: int | None = None) -> dict[str, Any]:
    from quantumbridge.compat.qiskit_optimization.quadratic_program_native import (
        add_binary_vars,
        add_linear_constraint,
        create_quadratic_program_native,
        set_minimize,
        solve_quadratic_program_bruteforce_native,
    )

    problem = create_quadratic_program_native("bench_quadratic_program")
    add_binary_vars(problem, ["x", "y"])
    set_minimize(problem, linear={"x": -1.0, "y": -1.0}, quadratic={("x", "y"): 2.0})
    add_linear_constraint(problem, {"x": 1.0, "y": 1.0}, "<=", 1.0, name="budget")
    result = solve_quadratic_program_bruteforce_native(problem)
    delta = objective_delta(result.objective_value, -1.0)
    return _payload(result, passed=result.feasible and delta == 0.0, metrics={"objective_delta": delta})


def _run_h2_case(seed: int | None = None) -> dict[str, Any]:
    from quantumbridge.compat.qiskit_nature.chemistry_native import run_h2_native

    result = run_h2_native()
    return _payload(result, passed=result.ground_state_energy < -1.0, metrics={"ground_state_energy": result.ground_state_energy})


def _run_lih_case(seed: int | None = None) -> dict[str, Any]:
    from quantumbridge.compat.qiskit_nature.chemistry_native import run_lih_native

    result = run_lih_native()
    return _payload(result, passed=result.ground_state_energy < -7.0, metrics={"ground_state_energy": result.ground_state_energy})


def _run_quantum_kernel_case(seed: int | None = None) -> dict[str, Any]:
    from quantumbridge.compat.qiskit_machine_learning.quantum_kernel_native import run_quantum_kernel_native

    result = run_quantum_kernel_native()
    return _payload(result, passed=bool(result.kernel_matrix), metrics={"rows": len(result.kernel_matrix)})


def _run_qnn_case(seed: int | None = None) -> dict[str, Any]:
    from quantumbridge.compat.qiskit_machine_learning.qnn_classifier_native import run_qnn_classifier_native

    result = run_qnn_classifier_native()
    return _payload(result, passed=result.accuracy >= 0.5, metrics={"accuracy": result.accuracy})


def _run_torchquantum_case(seed: int | None = None) -> dict[str, Any]:
    from quantumbridge.compat.torchquantum.training_native import run_torchquantum_like_classifier_native

    result = run_torchquantum_like_classifier_native()
    return _payload(result, passed=result.accuracy is not None and result.accuracy >= 0.5, metrics={"accuracy": result.accuracy})


def _run_zne_case(seed: int | None = None) -> dict[str, Any]:
    from quantumbridge.compat.mitiq.examples import run_zne_example

    output = run_zne_example(shots=128, seed=13 if seed is None else seed)
    native = output["native"]
    return _payload(
        native,
        passed=native.mitigated_expectation_value is not None,
        metrics={"mitigated_expectation": native.mitigated_expectation_value},
    )


def _run_readout_case(seed: int | None = None) -> dict[str, Any]:
    from quantumbridge.compat.mitiq.examples import run_readout_mitigation_example

    output = run_readout_mitigation_example(shots=128, seed=21 if seed is None else seed)
    native = output["native"]
    noisy_total = sum(native.noisy_counts.values()) or 1
    noisy_probabilities = {key: value / noisy_total for key, value in native.noisy_counts.items()}
    return _payload(
        native,
        passed=bool(native.mitigated_probabilities),
        metrics={"l1_noisy_vs_mitigated": l1_distance(noisy_probabilities, native.mitigated_probabilities)},
    )


def _run_qos_uqci_case(seed: int | None = None) -> dict[str, Any]:
    from quantumbridge.compat.qos_uqci.examples import run_qos_uqci_mock_runtime_example

    output = run_qos_uqci_mock_runtime_example(shots=128, seed=21 if seed is None else seed)
    native = output["result"]
    return _payload(native, passed=bool(native.counts), metrics={"shot_count": sum(native.counts.values())})


def _run_quafu_case(seed: int | None = None) -> dict[str, Any]:
    from quantumbridge.compat.quafu.examples import run_quafu_mock_backend_example

    output = run_quafu_mock_backend_example(shots=128, seed=23 if seed is None else seed)
    native = output["result"]
    return _payload(native, passed=bool(native.counts), metrics={"shot_count": sum(native.counts.values())})


def _run_qiskit_to_pennylane_case(seed: int | None = None) -> dict[str, Any]:
    from quantumbridge.compat.pennylane_qiskit.examples import run_qiskit_to_pennylane_example

    output = run_qiskit_to_pennylane_example()
    native = output["native"]
    return _payload(native, passed=bool(native.counts), metrics={"shot_count": sum(native.counts.values())})


def _run_pennylane_to_qiskit_case(seed: int | None = None) -> dict[str, Any]:
    from quantumbridge.compat.pennylane_qiskit.examples import run_pennylane_to_qiskit_example

    output = run_pennylane_to_qiskit_example()
    native = output["native"]
    return _payload(native, passed=bool(native.counts), metrics={"shot_count": sum(native.counts.values())})


def _run_mqt_mapping_case(seed: int | None = None) -> dict[str, Any]:
    from quantumbridge.compat.mqt.examples import run_mqt_qmap_like_example

    output = run_mqt_qmap_like_example(shots=128, seed=13 if seed is None else seed)
    comparison = output["comparison"]
    l1_value = comparison.comparison.get("l1_probability_distance")
    return _payload(
        comparison,
        passed=bool(comparison.comparison.get("comparable")) and l1_value == 0.0,
        metrics={"l1_distance": l1_value},
    )


def _payload(raw: Any, *, passed: bool, metrics: dict[str, Any] | None = None, expected_summary: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "passed": bool(passed),
        "metrics": dict(metrics or {}),
        "output_summary": _plain(raw),
        "expected_summary": dict(expected_summary or {}),
    }


def _plain(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)
