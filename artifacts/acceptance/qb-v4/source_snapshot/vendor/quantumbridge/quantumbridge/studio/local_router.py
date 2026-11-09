# This file is independently implemented for QuantumBridge SDK.
"""REST-like local router for QuantumBridge Studio services.

This module does not start an HTTP server and does not open ports. It simply
maps method/path/body triples to local service calls so future UI or FastAPI
adapters can reuse the same contracts.
"""

from __future__ import annotations

from typing import Any

from .api_models import StudioAPIResponse, StudioError, to_json_safe
from .benchmark_service import list_benchmark_suites, run_benchmark_suite
from .catalog_service import get_ecosystem_project, list_ecosystem_projects
from .execution_service import execute_workflow, get_execution_result
from .export_service import export_catalog_json, export_workflow_registry_json
from .provenance_service import base_studio_provenance
from .workflow_registry import get_workflow, get_workflow_input_schema, list_workflows


def list_routes() -> list[dict[str, str]]:
    return [
        {"method": "GET", "path": "/catalog/projects"},
        {"method": "GET", "path": "/catalog/projects/{project_id}"},
        {"method": "GET", "path": "/workflows"},
        {"method": "GET", "path": "/workflows/{workflow_id}"},
        {"method": "GET", "path": "/workflows/{workflow_id}/schema"},
        {"method": "POST", "path": "/execute/{workflow_id}"},
        {"method": "GET", "path": "/results/{execution_id}"},
        {"method": "GET", "path": "/benchmarks"},
        {"method": "POST", "path": "/benchmarks/{suite_id}"},
        {"method": "GET", "path": "/export/catalog"},
        {"method": "GET", "path": "/export/workflows"},
    ]


def route_request(path: str, method: str = "GET", body: dict[str, Any] | None = None) -> StudioAPIResponse:
    method = method.upper()
    path = "/" + str(path).strip("/")
    parts = [part for part in path.strip("/").split("/") if part]
    try:
        if method == "GET" and path == "/catalog/projects":
            return _ok([item.to_dict() for item in list_ecosystem_projects()])
        if method == "GET" and len(parts) == 3 and parts[:2] == ["catalog", "projects"]:
            return _ok(get_ecosystem_project(parts[2]).to_dict())
        if method == "GET" and path == "/workflows":
            return _ok([item.to_dict() for item in list_workflows()])
        if method == "GET" and len(parts) == 2 and parts[0] == "workflows":
            return _ok(get_workflow(parts[1]).to_dict())
        if method == "GET" and len(parts) == 3 and parts[0] == "workflows" and parts[2] == "schema":
            return _ok(get_workflow_input_schema(parts[1]).to_dict())
        if method == "POST" and len(parts) == 2 and parts[0] == "execute":
            payload = dict(body or {})
            return _ok(execute_workflow(parts[1], inputs=payload.get("inputs", payload), seed=payload.get("seed")).to_dict())
        if method == "GET" and len(parts) == 2 and parts[0] == "results":
            return _ok(get_execution_result(parts[1]).to_dict())
        if method == "GET" and path == "/benchmarks":
            return _ok({"suites": list_benchmark_suites()})
        if method == "POST" and len(parts) == 2 and parts[0] == "benchmarks":
            return _ok(run_benchmark_suite(parts[1]).to_dict())
        if method == "GET" and path == "/export/catalog":
            return _ok(export_catalog_json().to_dict())
        if method == "GET" and path == "/export/workflows":
            return _ok(export_workflow_registry_json().to_dict())
        return StudioAPIResponse(success=False, error=StudioError(message=f"route not found: {method} {path}", code="route_not_found"))
    except Exception as exc:  # noqa: BLE001 - router returns structured errors.
        return StudioAPIResponse(success=False, error=StudioError(message=str(exc), code=type(exc).__name__))


def handle_catalog_routes(path: str, method: str = "GET", body: dict[str, Any] | None = None) -> StudioAPIResponse:
    return route_request(path, method, body)


def handle_workflow_routes(path: str, method: str = "GET", body: dict[str, Any] | None = None) -> StudioAPIResponse:
    return route_request(path, method, body)


def handle_execution_routes(path: str, method: str = "GET", body: dict[str, Any] | None = None) -> StudioAPIResponse:
    return route_request(path, method, body)


def handle_benchmark_routes(path: str, method: str = "GET", body: dict[str, Any] | None = None) -> StudioAPIResponse:
    return route_request(path, method, body)


def handle_export_routes(path: str, method: str = "GET", body: dict[str, Any] | None = None) -> StudioAPIResponse:
    return route_request(path, method, body)


def _ok(data: Any) -> StudioAPIResponse:
    return StudioAPIResponse(success=True, data=to_json_safe(data), provenance=base_studio_provenance())
