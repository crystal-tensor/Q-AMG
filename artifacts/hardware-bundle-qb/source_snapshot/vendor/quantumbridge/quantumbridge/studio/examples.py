# This file is independently implemented for QuantumBridge SDK.
"""Convenience examples for QuantumBridge Studio local backend API."""

from __future__ import annotations

from .benchmark_service import run_benchmark_suite
from .catalog_service import list_ecosystem_projects
from .execution_service import execute_workflow
from .export_service import export_catalog_json, export_result_json, export_result_markdown, export_workflow_python_snippet
from .local_router import route_request
from .workflow_registry import list_workflows


def run_catalog_api_example() -> dict[str, object]:
    projects = list_ecosystem_projects()
    workflows = list_workflows()
    return {
        "project_count": len(projects),
        "workflow_count": len(workflows),
        "first_project": projects[0].to_dict() if projects else None,
        "warnings": projects[0].warnings if projects else [],
        "provenance": projects[0].provenance if projects else {},
        "cloud_access": False,
        "token_access": False,
        "hardware_access": False,
        "frontend_ui_implementation": False,
    }


def run_workflow_execution_example() -> dict[str, object]:
    result = execute_workflow("aer.qasm_counts_native", inputs={"shots": 64, "seed": 5})
    return {
        "execution": result,
        "json_export": export_result_json(result).content,
        "markdown_export": export_result_markdown(result).content,
        "warnings": result.warnings,
        "provenance": result.provenance,
    }


def run_benchmark_api_example() -> dict[str, object]:
    report = run_benchmark_suite("circuit_basic")
    return {
        "benchmark": report,
        "warnings": report.warnings,
        "provenance": report.provenance,
        "cloud_access": False,
        "token_access": False,
        "hardware_access": False,
    }


def run_export_api_example() -> dict[str, object]:
    snippet = export_workflow_python_snippet("aer.qasm_counts_native", {"shots": 64, "seed": 5})
    return {
        "catalog_json": export_catalog_json().content,
        "python_snippet": snippet.content,
        "router_workflows": route_request("/workflows").to_dict(),
        "warnings": snippet.warnings,
        "provenance": snippet.provenance,
    }
