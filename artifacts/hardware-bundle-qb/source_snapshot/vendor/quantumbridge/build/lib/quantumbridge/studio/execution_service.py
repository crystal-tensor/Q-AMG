# This file is independently implemented for QuantumBridge SDK.
"""Local workflow execution service for QuantumBridge Studio."""

from __future__ import annotations

from itertools import count
from typing import Any

from .api_models import StudioError, StudioExecutionRequest, StudioExecutionResult, to_json_safe
from .provenance_service import base_studio_provenance, collect_result_provenance
from .result_store import DEFAULT_RESULT_STORE
from .warning_service import collect_result_warnings, workflow_boundary_warnings
from .workflow_inputs import coerce_input_types, validate_inputs_against_schema
from .workflow_registry import execute_registered_workflow, get_workflow_input_schema, get_workflow_spec

_COUNTER = count(1)


def execute_workflow(workflow_id: str, inputs: dict[str, Any] | None = None, seed: int | None = None) -> StudioExecutionResult:
    request = StudioExecutionRequest(workflow_id=workflow_id, inputs=dict(inputs or {}), seed=seed)
    return execute_workflow_by_request(request)


def execute_workflow_by_request(request: StudioExecutionRequest | dict[str, Any]) -> StudioExecutionResult:
    req = request if isinstance(request, StudioExecutionRequest) else StudioExecutionRequest.from_dict(request)
    req.validate()
    schema = get_workflow_input_schema(req.workflow_id)
    valid, errors = validate_inputs_against_schema(schema, req.inputs)
    if not valid:
        return _store(_error_result(req.workflow_id, "input_validation_failed", "; ".join(errors)))
    if req.dry_run:
        return _store(_dry_run_result(req.workflow_id, req.inputs))
    try:
        spec = get_workflow_spec(req.workflow_id)
        payload = coerce_input_types(schema, req.inputs)
        if req.seed is not None:
            payload.setdefault("seed", req.seed)
        raw = execute_registered_workflow(req.workflow_id, payload)
        result = StudioExecutionResult(
            execution_id=_next_execution_id(),
            workflow_id=req.workflow_id,
            status="succeeded",
            result=to_json_safe(raw),
            warnings=workflow_boundary_warnings(req.workflow_id, spec.project_id) + collect_result_warnings(raw),
            provenance={
                **base_studio_provenance(req.workflow_id, spec.project_id),
                "result_provenance": collect_result_provenance(raw),
            },
        )
        return _store(result)
    except Exception as exc:  # noqa: BLE001 - local API returns structured errors.
        return _store(_error_result(req.workflow_id, type(exc).__name__, str(exc)))


def execute_batch_workflows(requests: list[StudioExecutionRequest | dict[str, Any]]) -> list[StudioExecutionResult]:
    return [execute_workflow_by_request(request) for request in requests]


def dry_run_workflow(workflow_id: str, inputs: dict[str, Any] | None = None) -> StudioExecutionResult:
    return execute_workflow_by_request(StudioExecutionRequest(workflow_id=workflow_id, inputs=dict(inputs or {}), dry_run=True))


def get_execution_status(execution_id: str) -> dict[str, str]:
    result = DEFAULT_RESULT_STORE.get_result(execution_id)
    return {"execution_id": execution_id, "workflow_id": result.workflow_id, "status": result.status}


def get_execution_result(execution_id: str) -> StudioExecutionResult:
    return DEFAULT_RESULT_STORE.get_result(execution_id)


def _store(result: StudioExecutionResult) -> StudioExecutionResult:
    return DEFAULT_RESULT_STORE.save_result(result)


def _next_execution_id() -> str:
    return f"studio-exec-{next(_COUNTER):06d}"


def _error_result(workflow_id: str, code: str, message: str) -> StudioExecutionResult:
    return StudioExecutionResult(
        execution_id=_next_execution_id(),
        workflow_id=workflow_id,
        status="failed",
        result={},
        error=StudioError(message=message, code=code),
        warnings=workflow_boundary_warnings(workflow_id),
        provenance=base_studio_provenance(workflow_id),
        unsupported_reason=message,
    )


def _dry_run_result(workflow_id: str, inputs: dict[str, Any] | None) -> StudioExecutionResult:
    spec = get_workflow_spec(workflow_id)
    return StudioExecutionResult(
        execution_id=_next_execution_id(),
        workflow_id=workflow_id,
        status="dry_run",
        result={"validated_inputs": dict(inputs or {}), "would_execute": True},
        warnings=workflow_boundary_warnings(workflow_id, spec.project_id),
        provenance=base_studio_provenance(workflow_id, spec.project_id),
    )
