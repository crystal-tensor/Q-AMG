# This file is independently implemented for QuantumBridge SDK.
"""Export helpers for QuantumBridge Studio local API."""

from __future__ import annotations

import json
from typing import Any

from .api_models import StudioExportResult, to_json_safe
from .catalog_service import list_ecosystem_projects
from .execution_service import execute_workflow
from .workflow_inputs import get_default_inputs
from .workflow_registry import get_workflow, list_workflows


def export_result_json(result: Any) -> StudioExportResult:
    content = json.dumps(to_json_safe(result), indent=2, sort_keys=True)
    return StudioExportResult(export_type="json", content=content)


def export_result_markdown(result: Any) -> StudioExportResult:
    payload = to_json_safe(result)
    workflow_id = payload.get("workflow_id") if isinstance(payload, dict) else None
    execution_id = payload.get("execution_id") if isinstance(payload, dict) else None
    content = f"# QuantumBridge Studio Result\\n\\n- execution_id: `{execution_id}`\\n- workflow_id: `{workflow_id}`\\n\\n```json\\n{json.dumps(payload, indent=2, sort_keys=True)}\\n```\\n"
    return StudioExportResult(export_type="markdown", content=content, workflow_id=workflow_id, execution_id=execution_id)


def export_workflow_python_snippet(workflow_id: str, inputs: dict[str, Any] | None = None) -> StudioExportResult:
    payload = inputs if inputs is not None else get_default_inputs(workflow_id)
    content = (
        "from quantumbridge.studio.execution_service import execute_workflow\\n\\n"
        f"result = execute_workflow({workflow_id!r}, inputs={payload!r})\\n"
        "print(result.to_json())\\n"
    )
    return StudioExportResult(export_type="python", content=content, workflow_id=workflow_id)


def export_workflow_notebook_stub(workflow_id: str, inputs: dict[str, Any] | None = None) -> StudioExportResult:
    snippet = export_workflow_python_snippet(workflow_id, inputs).content
    notebook = {
        "metadata": {"quantumbridge": "studio-notebook-stub-v0.1"},
        "cells": [
            {"cell_type": "markdown", "source": [f"# QuantumBridge Studio workflow: {workflow_id}\\n"]},
            {"cell_type": "code", "source": snippet.splitlines(True), "outputs": []},
        ],
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return StudioExportResult(export_type="notebook_stub", content=json.dumps(notebook, indent=2), workflow_id=workflow_id)


def export_workflow_sample(workflow_id: str, format: str = "json", inputs: dict[str, Any] | None = None) -> StudioExportResult:
    """Export a local workflow sample without cloud, token, or hardware access."""

    normalized = _normalize_export_format(format)
    payload = inputs if inputs is not None else get_default_inputs(workflow_id)
    if normalized == "python":
        return export_workflow_python_snippet(workflow_id, payload)
    if normalized == "notebook_stub":
        return export_workflow_notebook_stub(workflow_id, payload)
    result = execute_workflow(workflow_id, payload)
    if normalized == "json":
        exported = export_result_json(result)
    elif normalized == "markdown":
        exported = export_result_markdown(result)
    else:
        raise ValueError("format must be json, markdown, python, or notebook")
    exported.workflow_id = workflow_id
    return exported


def export_catalog_json() -> StudioExportResult:
    return StudioExportResult(export_type="json", content=json.dumps([item.to_dict() for item in list_ecosystem_projects()], indent=2, sort_keys=True))


def export_workflow_registry_json() -> StudioExportResult:
    return StudioExportResult(export_type="json", content=json.dumps([item.to_dict() for item in list_workflows()], indent=2, sort_keys=True))


def _normalize_export_format(format: str) -> str:
    value = str(format).strip().lower().replace("-", "_")
    if value == "notebook":
        return "notebook_stub"
    if value in {"json", "markdown", "python", "notebook_stub"}:
        return value
    raise ValueError("format must be json, markdown, python, or notebook")
