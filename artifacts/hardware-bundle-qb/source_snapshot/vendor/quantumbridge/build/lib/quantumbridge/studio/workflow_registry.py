# This file is independently implemented for QuantumBridge SDK.
# No source code from IBM, Qiskit, PennyLane, Benchpress, or third-party projects was copied.
"""Workflow registry for the QuantumBridge Studio local backend API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from quantumbridge.core import Circuit

from .api_models import StudioInputField, StudioInputSchema, StudioWorkflowDetail, StudioWorkflowSummary
from .provenance_service import base_studio_provenance
from .warning_service import workflow_boundary_warnings

WorkflowRunner = Callable[[dict[str, Any]], Any]


@dataclass(frozen=True)
class WorkflowSpec:
    workflow_id: str
    title: str
    project_id: str
    category: str
    description: str
    runner: WorkflowRunner
    input_fields: tuple[StudioInputField, ...] = ()
    defaults: dict[str, Any] = field(default_factory=dict)
    result_schema: str = "QuantumBridgeResult"
    examples: tuple[dict[str, Any], ...] = ()
    studio_ready: str = "partial"
    executable: bool = True
    local_only: bool = True

    def input_schema(self) -> StudioInputSchema:
        return StudioInputSchema(
            workflow_id=self.workflow_id,
            fields=list(self.input_fields),
            defaults=dict(self.defaults),
            warnings=workflow_boundary_warnings(self.workflow_id, self.project_id),
            provenance=base_studio_provenance(self.workflow_id, self.project_id),
        )

    def summary(self) -> StudioWorkflowSummary:
        return StudioWorkflowSummary(
            workflow_id=self.workflow_id,
            title=self.title,
            project_id=self.project_id,
            category=self.category,
            executable=self.executable,
            studio_ready=self.studio_ready,
            warnings=workflow_boundary_warnings(self.workflow_id, self.project_id),
            provenance=base_studio_provenance(self.workflow_id, self.project_id),
        )

    def detail(self) -> StudioWorkflowDetail:
        return StudioWorkflowDetail(
            workflow_id=self.workflow_id,
            title=self.title,
            project_id=self.project_id,
            category=self.category,
            executable=self.executable,
            studio_ready=self.studio_ready,
            description=self.description,
            input_schema=self.input_schema(),
            output_schema={"type": "object", "result_schema": self.result_schema},
            result_schema=self.result_schema,
            examples=list(self.examples),
            warnings=workflow_boundary_warnings(self.workflow_id, self.project_id),
            provenance=base_studio_provenance(self.workflow_id, self.project_id),
        )


def list_workflows() -> list[StudioWorkflowSummary]:
    return [spec.summary() for spec in _registry().values()]


def get_workflow(workflow_id: str) -> StudioWorkflowDetail:
    return _require_workflow(workflow_id).detail()


def get_workflow_spec(workflow_id: str) -> WorkflowSpec:
    return _require_workflow(workflow_id)


def search_workflows(query: str) -> list[StudioWorkflowSummary]:
    needle = str(query).lower()
    return [
        spec.summary()
        for spec in _registry().values()
        if needle in spec.workflow_id.lower()
        or needle in spec.title.lower()
        or needle in spec.project_id.lower()
        or needle in spec.category.lower()
    ]


def filter_workflows(
    project_id: str | None = None,
    category: str | None = None,
    executable: bool | None = True,
    backend: str | None = None,
) -> list[StudioWorkflowSummary]:
    specs = list(_registry().values())
    if project_id is not None:
        specs = [spec for spec in specs if spec.project_id == project_id]
    if category is not None:
        specs = [spec for spec in specs if spec.category == category]
    if executable is not None:
        specs = [spec for spec in specs if spec.executable is bool(executable)]
    if backend is not None:
        needle = str(backend).lower()
        specs = [spec for spec in specs if needle in spec.workflow_id.lower() or needle in spec.project_id.lower()]
    return [spec.summary() for spec in specs]


def get_workflow_input_schema(workflow_id: str) -> StudioInputSchema:
    return _require_workflow(workflow_id).input_schema()


def validate_workflow_request(workflow_id: str, inputs: dict[str, Any] | None) -> tuple[bool, list[str]]:
    from .workflow_inputs import validate_inputs_against_schema

    return validate_inputs_against_schema(get_workflow_input_schema(workflow_id), inputs or {})


def get_workflow_examples(workflow_id: str) -> list[dict[str, Any]]:
    return list(_require_workflow(workflow_id).examples)


def execute_registered_workflow(workflow_id: str, inputs: dict[str, Any] | None = None) -> Any:
    spec = _require_workflow(workflow_id)
    payload = {**spec.defaults, **dict(inputs or {})}
    return spec.runner(payload)


def _require_workflow(workflow_id: str) -> WorkflowSpec:
    try:
        return _registry()[workflow_id]
    except KeyError as exc:
        raise KeyError(f"unknown Studio workflow_id: {workflow_id}") from exc


def _registry() -> dict[str, WorkflowSpec]:
    global _WORKFLOW_REGISTRY
    try:
        return _WORKFLOW_REGISTRY
    except NameError:
        _WORKFLOW_REGISTRY = _build_registry()
        return _WORKFLOW_REGISTRY


def _field(name: str, field_type: str, default: Any = None, description: str = "", required: bool = False, enum: list[Any] | None = None) -> StudioInputField:
    return StudioInputField(name=name, field_type=field_type, default=default, description=description, required=required, enum=list(enum or []))


def _build_registry() -> dict[str, WorkflowSpec]:
    specs = [
        _spec("finance.portfolio_optimization_native", "Portfolio Optimization", "qiskit-finance", "finance", _run_portfolio, fields=(_field("budget", "integer", 2),), result_schema="PortfolioOptimizationResult"),
        _spec("optimization.quadratic_program_native", "Quadratic Program", "qiskit-optimization", "optimization", _run_quadratic_program, result_schema="NativeOptimizationResult"),
        _spec("algorithms.vqe_native", "VQE Native", "qiskit-algorithms", "algorithms", lambda inputs: _run_vqe(inputs), result_schema="VQEResult"),
        _spec("algorithms.qaoa_native", "QAOA MaxCut Native", "qiskit-algorithms", "algorithms", _run_qaoa, fields=(_field("p", "integer", 1),), result_schema="QAOAResult"),
        _spec("algorithms.grover_native", "Grover Native", "qiskit-algorithms", "algorithms", _run_grover, fields=(_field("marked_bitstrings", "list", ["11"]),), result_schema="GroverResult"),
        _spec("nature.h2_native", "H2 Chemistry Native", "qiskit-nature", "chemistry", _run_h2, fields=(_field("bond_length", "float", 0.735),), result_schema="H2WorkflowResult"),
        _spec("nature.lih_native", "LiH Chemistry Native", "qiskit-nature", "chemistry", _run_lih, fields=(_field("bond_length", "float", 1.6),), result_schema="LiHWorkflowResult"),
        _spec("ml.quantum_kernel_native", "Quantum Kernel Native", "qiskit-machine-learning", "machine-learning", _run_quantum_kernel, result_schema="QuantumKernelResult"),
        _spec("ml.kernel_classifier_native", "Kernel Classifier Native", "qiskit-machine-learning", "machine-learning", _run_kernel_classifier, result_schema="KernelClassifierResult"),
        _spec("ml.qnn_classifier_native", "QNN Classifier Native", "qiskit-machine-learning", "machine-learning", _run_qnn_classifier, result_schema="QNNClassifierResult"),
        _spec("aer.statevector_native", "Statevector Simulator Native", "qiskit-aer", "simulator", _run_statevector, result_schema="StatevectorSimulationResult"),
        _spec("aer.qasm_counts_native", "QASM Counts Native", "qiskit-aer", "simulator", _run_qasm, fields=(_field("shots", "integer", 128), _field("seed", "integer", 7)), result_schema="QasmSimulationResult"),
        _spec("aer.noisy_counts_native", "Noisy Counts Native", "qiskit-aer", "simulator", _run_noisy_qasm, fields=(_field("shots", "integer", 128), _field("seed", "integer", 7)), result_schema="NoisySimulationResult"),
        _spec("mitiq.zne_native", "ZNE Native", "mitiq", "mitigation", _run_zne, fields=(_field("shots", "integer", 256), _field("seed", "integer", 13)), result_schema="ZNEResult"),
        _spec("mitiq.readout_mitigation_native", "Readout Mitigation Native", "mitiq", "mitigation", _run_readout, fields=(_field("shots", "integer", 256), _field("seed", "integer", 21)), result_schema="ReadoutMitigationResult"),
        _spec("pennylane_qiskit.qiskit_to_pennylane", "Qiskit to PennyLane Bridge", "pennylane-qiskit", "bridge", _run_qiskit_to_pennylane, result_schema="QiskitToPennyLaneResult"),
        _spec("pennylane_qiskit.pennylane_to_qiskit", "PennyLane to Qiskit Bridge", "pennylane-qiskit", "bridge", _run_pennylane_to_qiskit, result_schema="PennyLaneToQiskitResult"),
        _spec("experiments.rabi_native", "Rabi Experiment Native", "qiskit-experiments", "experiments", _run_rabi, result_schema="RabiExperimentResult"),
        _spec("experiments.t1_native", "T1 Experiment Native", "qiskit-experiments", "experiments", _run_t1, result_schema="T1ExperimentResult"),
        _spec("experiments.ramsey_native", "Ramsey Experiment Native", "qiskit-experiments", "experiments", _run_ramsey, result_schema="RamseyExperimentResult"),
        _spec("dynamics.z_precession_native", "Z Precession Dynamics Native", "qiskit-dynamics", "dynamics", _run_z_precession, result_schema="SingleQubitDynamicsResult"),
        _spec("dynamics.rabi_drive_native", "Rabi Drive Dynamics Native", "qiskit-dynamics", "dynamics", _run_rabi_drive, result_schema="SingleQubitDynamicsResult"),
        _spec("mqt.core_like_circuit", "MQT Core-like Circuit", "mqt", "compiler", _run_mqt_core, result_schema="MQTCoreLikeCircuit"),
        _spec("mqt.ddsim_like_simulation", "DDSIM-like Simulation", "mqt", "simulator", _run_mqt_ddsim, result_schema="MQTDDLikeResult"),
        _spec("mqt.qmap_like_routing", "QMAP-like Routing", "mqt", "routing", _run_mqt_qmap, result_schema="MQTQMAPLikeResult"),
        _spec("torchquantum.layer_native", "TorchQuantum-like Layer", "torchquantum", "qml", _run_tq_layer, result_schema="TorchQuantumLayerResult"),
        _spec("torchquantum.batch_forward_native", "TorchQuantum-like Batch Forward", "torchquantum", "qml", _run_tq_batch, result_schema="TorchQuantumBatchResult"),
        _spec("torchquantum.classifier_native", "TorchQuantum-like Classifier", "torchquantum", "qml", _run_tq_classifier, result_schema="TorchQuantumClassifierResult"),
        _spec("qos_uqci.bell_job_spec", "QOS-UQCI Bell Job Spec", "qos-uqci", "backend", _run_qos_job, result_schema="QOSUQCIJobSpec"),
        _spec("qos_uqci.mock_runtime", "QOS-UQCI Mock Runtime", "qos-uqci", "backend", _run_qos_mock, result_schema="BackendExecutionResult"),
        _spec("quafu.bell_payload", "Quafu Bell Payload", "quafu", "backend", _run_quafu_payload, result_schema="QuafuPayload"),
        _spec("quafu.mock_backend", "Quafu Mock Backend", "quafu", "backend", _run_quafu_mock, result_schema="BackendExecutionResult"),
        _spec("benchpress.basic_suite", "Benchpress Basic Suite", "benchpress", "benchmark", _run_benchpress_basic, result_schema="BenchmarkSuiteResult"),
        _spec("benchpress.algorithms_suite", "Benchpress Algorithms Suite", "benchpress", "benchmark", _run_benchpress_algorithms, result_schema="BenchmarkSuiteResult"),
        _spec("benchpress.backend_suite", "Benchpress Backend Suite", "benchpress", "benchmark", _run_benchpress_backend, result_schema="BenchmarkSuiteResult"),
        _spec("benchpress.full_smoke_suite", "Benchpress Full Smoke Suite", "benchpress", "benchmark", _run_benchpress_full, result_schema="BenchmarkSuiteResult"),
    ]
    return {spec.workflow_id: spec for spec in specs}


def _spec(workflow_id: str, title: str, project_id: str, category: str, runner: WorkflowRunner, fields: tuple[StudioInputField, ...] = (), result_schema: str = "QuantumBridgeResult") -> WorkflowSpec:
    defaults = {field.name: field.default for field in fields if field.default is not None}
    return WorkflowSpec(
        workflow_id=workflow_id,
        title=title,
        project_id=project_id,
        category=category,
        description=f"Local-only QuantumBridge Studio workflow for {title}.",
        runner=runner,
        input_fields=fields,
        defaults=defaults,
        result_schema=result_schema,
        examples=({"inputs": defaults},),
        studio_ready="true",
    )


def _bell_circuit(measured: bool = True) -> Circuit:
    circuit = Circuit(2, 2 if measured else 0, name="studio_bell")
    circuit.h(0).cx(0, 1)
    if measured:
        circuit.measure(0, 0).measure(1, 1)
    return circuit


def _run_portfolio(inputs: dict[str, Any]):
    from quantumbridge.compat.qiskit_finance.portfolio_optimization_adapter import run_portfolio_optimization_native

    custom_keys = {"expected_returns", "covariances", "risk_factor", "budget"}
    if custom_keys.issubset(inputs):
        return run_portfolio_optimization_native(
            expected_returns=inputs["expected_returns"],
            covariances=inputs["covariances"],
            risk_factor=float(inputs["risk_factor"]),
            budget=int(inputs["budget"]),
        )
    return run_portfolio_optimization_native()


def _run_quadratic_program(inputs: dict[str, Any]):
    from quantumbridge.compat.qiskit_optimization.quadratic_program_native import add_binary_vars, add_linear_constraint, create_quadratic_program_native, set_minimize, solve_quadratic_program_bruteforce_native

    problem = create_quadratic_program_native("studio_binary_program")
    add_binary_vars(problem, ["x0", "x1"])
    set_minimize(problem, linear={"x0": -1.0, "x1": -0.75}, quadratic={("x0", "x1"): 0.25})
    add_linear_constraint(problem, {"x0": 1.0, "x1": 1.0}, "<=", 1.0, name="budget")
    return solve_quadratic_program_bruteforce_native(problem)


def _run_vqe(inputs: dict[str, Any]):
    from quantumbridge.compat.qiskit_algorithms.algorithms_native import run_vqe_native

    return run_vqe_native()


def _run_qaoa(inputs: dict[str, Any]):
    from quantumbridge.compat.qiskit_algorithms.algorithms_native import run_qaoa_native_maxcut

    return run_qaoa_native_maxcut(((0, 1), (1, 2), (2, 0)), num_nodes=3, p=int(inputs.get("p", 1)))


def _run_grover(inputs: dict[str, Any]):
    from quantumbridge.compat.qiskit_algorithms.algorithms_native import run_grover_native

    return run_grover_native(inputs.get("marked_bitstrings", ["11"]), num_qubits=2)


def _run_h2(inputs: dict[str, Any]):
    from quantumbridge.compat.qiskit_nature.chemistry_native import run_h2_native

    return run_h2_native(bond_length=float(inputs.get("bond_length", 0.735)))


def _run_lih(inputs: dict[str, Any]):
    from quantumbridge.compat.qiskit_nature.chemistry_native import run_lih_native

    return run_lih_native(bond_length=float(inputs.get("bond_length", 1.6)))


def _run_quantum_kernel(inputs: dict[str, Any]):
    from quantumbridge.compat.qiskit_machine_learning.quantum_kernel_native import run_quantum_kernel_native

    return run_quantum_kernel_native()


def _run_kernel_classifier(inputs: dict[str, Any]):
    from quantumbridge.compat.qiskit_machine_learning.kernel_classifier_native import run_kernel_classifier_native

    return run_kernel_classifier_native()


def _run_qnn_classifier(inputs: dict[str, Any]):
    from quantumbridge.compat.qiskit_machine_learning.qnn_classifier_native import run_qnn_classifier_native

    return run_qnn_classifier_native()


def _run_statevector(inputs: dict[str, Any]):
    from quantumbridge.compat.qiskit_aer.simulator_native import run_statevector_simulator_native

    return run_statevector_simulator_native(_bell_circuit(measured=False))


def _run_qasm(inputs: dict[str, Any]):
    from quantumbridge.compat.qiskit_aer.simulator_native import run_qasm_simulator_native

    return run_qasm_simulator_native(_bell_circuit(), shots=int(inputs.get("shots", 128)), seed=inputs.get("seed"))


def _run_noisy_qasm(inputs: dict[str, Any]):
    from quantumbridge.compat.qiskit_aer.simulator_native import run_noisy_qasm_simulator_native

    return run_noisy_qasm_simulator_native(_bell_circuit(), shots=int(inputs.get("shots", 128)), seed=inputs.get("seed"))


def _run_zne(inputs: dict[str, Any]):
    from quantumbridge.compat.mitiq.zne_native import run_zne_native

    return run_zne_native(_bell_circuit(measured=True), shots=int(inputs.get("shots", 256)), seed=inputs.get("seed"))


def _run_readout(inputs: dict[str, Any]):
    from quantumbridge.compat.mitiq.readout_mitigation_native import run_readout_mitigation_native

    return run_readout_mitigation_native(_bell_circuit(), shots=int(inputs.get("shots", 256)), seed=inputs.get("seed"))


def _run_qiskit_to_pennylane(inputs: dict[str, Any]):
    from quantumbridge.compat.pennylane_qiskit.examples import run_qiskit_to_pennylane_example

    return run_qiskit_to_pennylane_example()["native"]


def _run_pennylane_to_qiskit(inputs: dict[str, Any]):
    from quantumbridge.compat.pennylane_qiskit.examples import run_pennylane_to_qiskit_example

    return run_pennylane_to_qiskit_example()["native"]


def _run_rabi(inputs: dict[str, Any]):
    from quantumbridge.compat.qiskit_experiments.rabi_experiment_native import run_rabi_experiment_native

    return run_rabi_experiment_native(seed=7)


def _run_t1(inputs: dict[str, Any]):
    from quantumbridge.compat.qiskit_experiments.t1_experiment_native import run_t1_experiment_native

    return run_t1_experiment_native(seed=7)


def _run_ramsey(inputs: dict[str, Any]):
    from quantumbridge.compat.qiskit_experiments.ramsey_experiment_native import run_ramsey_experiment_native

    return run_ramsey_experiment_native(seed=7)


def _run_z_precession(inputs: dict[str, Any]):
    from quantumbridge.compat.qiskit_dynamics.single_qubit_dynamics_native import run_z_precession_native

    return run_z_precession_native()


def _run_rabi_drive(inputs: dict[str, Any]):
    from quantumbridge.compat.qiskit_dynamics.single_qubit_dynamics_native import run_rabi_drive_dynamics_native

    return run_rabi_drive_dynamics_native()


def _run_mqt_core(inputs: dict[str, Any]):
    from quantumbridge.compat.mqt.examples import run_mqt_core_like_example

    return run_mqt_core_like_example()


def _run_mqt_ddsim(inputs: dict[str, Any]):
    from quantumbridge.compat.mqt.examples import run_mqt_ddsim_like_example

    return run_mqt_ddsim_like_example()


def _run_mqt_qmap(inputs: dict[str, Any]):
    from quantumbridge.compat.mqt.examples import run_mqt_qmap_like_example

    return run_mqt_qmap_like_example()


def _run_tq_layer(inputs: dict[str, Any]):
    from quantumbridge.compat.torchquantum.examples import run_torchquantum_like_layer_example

    return run_torchquantum_like_layer_example()


def _run_tq_batch(inputs: dict[str, Any]):
    from quantumbridge.compat.torchquantum.examples import run_torchquantum_like_batch_forward_example

    return run_torchquantum_like_batch_forward_example()


def _run_tq_classifier(inputs: dict[str, Any]):
    from quantumbridge.compat.torchquantum.examples import run_torchquantum_like_classifier_example

    return run_torchquantum_like_classifier_example()


def _run_qos_job(inputs: dict[str, Any]):
    from quantumbridge.compat.qos_uqci.examples import run_qos_uqci_bell_job_example

    return run_qos_uqci_bell_job_example()


def _run_qos_mock(inputs: dict[str, Any]):
    from quantumbridge.compat.qos_uqci.examples import run_qos_uqci_mock_runtime_example

    return run_qos_uqci_mock_runtime_example()["result"]


def _run_quafu_payload(inputs: dict[str, Any]):
    from quantumbridge.compat.quafu.examples import run_quafu_bell_payload_example

    return run_quafu_bell_payload_example()


def _run_quafu_mock(inputs: dict[str, Any]):
    from quantumbridge.compat.quafu.examples import run_quafu_mock_backend_example

    return run_quafu_mock_backend_example()["result"]


def _run_benchpress_basic(inputs: dict[str, Any]):
    from quantumbridge.compat.benchpress.benchmark_runner import run_benchmark_suite
    from quantumbridge.compat.benchpress.benchmark_suites import circuit_basic_suite

    return run_benchmark_suite(circuit_basic_suite(), suite_name="studio_benchpress_basic")


def _run_benchpress_algorithms(inputs: dict[str, Any]):
    from quantumbridge.compat.benchpress.benchmark_runner import run_benchmark_suite
    from quantumbridge.compat.benchpress.benchmark_suites import algorithms_suite

    return run_benchmark_suite(algorithms_suite(), suite_name="studio_benchpress_algorithms")


def _run_benchpress_backend(inputs: dict[str, Any]):
    from quantumbridge.compat.benchpress.benchmark_runner import run_benchmark_suite
    from quantumbridge.compat.benchpress.benchmark_suites import backend_suite

    return run_benchmark_suite(backend_suite(), suite_name="studio_benchpress_backend")


def _run_benchpress_full(inputs: dict[str, Any]):
    from quantumbridge.compat.benchpress.benchmark_runner import run_benchmark_suite
    from quantumbridge.compat.benchpress.benchmark_suites import full_smoke_suite

    return run_benchmark_suite(full_smoke_suite(), suite_name="studio_benchpress_full_smoke")
