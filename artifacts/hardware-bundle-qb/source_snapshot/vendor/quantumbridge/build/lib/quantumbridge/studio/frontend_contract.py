# This file is independently implemented for QuantumBridge SDK.
# No source code, prose, UI, or branding from IBM, Qiskit, PennyLane, or third-party projects was copied.
"""Frontend seed-data contract for QuantumBridge Studio.

The Python Studio backend remains the canonical source. This module builds the
static frontend seed bundle from local backend services, validates the bundle,
and writes the JSON/JS files consumed by the Stage 10D frontend prototype.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from .api_models import SCHEMA_VERSION, StudioExportResult, to_json_safe
from .benchmark_service import run_benchmark_suite_by_id
from .catalog_service import get_project_status_summary, list_ecosystem_projects
from .execution_service import execute_workflow
from .export_service import (
    export_result_json,
    export_result_markdown,
    export_workflow_notebook_stub,
    export_workflow_python_snippet,
)
from .workflow_inputs import get_default_inputs
from .workflow_registry import get_workflow, list_workflows

FRONTEND_SCHEMA_VERSION = "quantumbridge-studio-frontend-seed-v0.1"
DEFAULT_FRONTEND_SEED_DIR = Path("studio/src/data")
SAMPLE_RESULT_WORKFLOWS = (
    "aer.statevector_native",
    "aer.qasm_counts_native",
    "aer.noisy_counts_native",
    "finance.portfolio_optimization_native",
    "optimization.quadratic_program_native",
    "algorithms.qaoa_native",
    "algorithms.grover_native",
    "nature.h2_native",
    "ml.quantum_kernel_native",
    "mitiq.zne_native",
    "pennylane_qiskit.qiskit_to_pennylane",
    "pennylane_qiskit.pennylane_to_qiskit",
    "mqt.core_like_circuit",
    "mqt.ddsim_like_simulation",
    "torchquantum.layer_native",
    "qos_uqci.mock_runtime",
    "quafu.mock_backend",
    "benchpress.basic_suite",
)
SEED_FILE_NAMES = {
    "catalog": "sampleCatalog.json",
    "workflows": "sampleWorkflows.json",
    "workflow_details": "sampleWorkflowDetails.json",
    "results": "sampleResults.json",
    "benchmark": "sampleBenchmarkReport.json",
    "exports": "sampleExports.json",
    "schema": "studioSchemaVersion.json",
}
FORBIDDEN_SEED_PATTERNS = (
    re.compile(r"https?://", re.IGNORECASE),
    re.compile(r"api[_-]?key|password|bearer\s+[a-z0-9._-]+|secret[_-]?value", re.IGNORECASE),
    re.compile(r"carbon-components|@carbon|ibm-logo|qiskit-logo|pennylane-logo", re.IGNORECASE),
)


def build_frontend_seed_bundle() -> dict[str, Any]:
    """Build a JSON-safe frontend seed bundle from local backend services."""

    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    projects = [item.to_dict() for item in list_ecosystem_projects()]
    workflow_summaries = [item.to_dict() for item in list_workflows()]
    workflow_details = [_workflow_detail_payload(item["workflow_id"]) for item in workflow_summaries]
    sample_results = _build_sample_results()
    benchmark_report = run_benchmark_suite_by_id("circuit_basic").to_dict()
    export_samples = _build_export_samples(sample_results[0] if sample_results else None)
    schema_payload = _envelope(
        "schema",
        {
            "frontend_schema_version": FRONTEND_SCHEMA_VERSION,
            "backend_schema_version": SCHEMA_VERSION,
            "files": dict(SEED_FILE_NAMES),
            "workflow_count": len(workflow_summaries),
            "workflow_detail_count": len(workflow_details),
            "sample_result_count": len(sample_results),
            "export_formats": sorted(export_samples),
        },
        generated_at,
    )

    bundle = {
        "catalog": _envelope(
            "catalog",
            {
                "projects": projects,
                "summary": {
                    **get_project_status_summary(),
                    "executable_workflow_count": sum(len(item.get("executable_workflows", [])) for item in projects),
                },
            },
            generated_at,
        ),
        "workflows": _envelope(
            "workflows",
            {
                "workflows": workflow_summaries,
                "details": workflow_details,
                "workflow_count": len(workflow_summaries),
            },
            generated_at,
        ),
        "workflow_details": _envelope(
            "workflow_details",
            {
                "details": workflow_details,
                "workflow_detail_count": len(workflow_details),
            },
            generated_at,
        ),
        "results": _envelope(
            "results",
            {
                "results": sample_results,
                "result_count": len(sample_results),
                "sampled_workflows": [item.get("workflow_id") for item in sample_results],
            },
            generated_at,
        ),
        "benchmark": _envelope(
            "benchmark",
            {
                "benchmark": benchmark_report,
                "suite_id": benchmark_report.get("suite_id"),
            },
            generated_at,
        ),
        "exports": _envelope(
            "exports",
            {
                "exports": export_samples,
                "formats": sorted(export_samples),
            },
            generated_at,
        ),
        "schema": schema_payload,
    }
    validate_frontend_seed_bundle(bundle)
    return to_json_safe(bundle)


def validate_frontend_seed_bundle(bundle: dict[str, Any]) -> bool:
    """Validate the complete frontend seed bundle."""

    required = {"catalog", "workflows", "workflow_details", "results", "benchmark", "exports", "schema"}
    missing = required.difference(bundle)
    if missing:
        raise ValueError(f"frontend seed bundle missing sections: {sorted(missing)}")
    for key in required:
        _validate_envelope(key, bundle[key])
    validate_frontend_workflow_coverage(bundle)
    validate_frontend_result_coverage(bundle)
    validate_no_branding_or_secrets_in_seed(bundle)
    return True


def validate_frontend_workflow_coverage(bundle: dict[str, Any]) -> bool:
    """Ensure frontend seed workflows exactly cover the backend registry."""

    registry_ids = {item.workflow_id for item in list_workflows()}
    workflow_ids = {item.get("workflow_id") for item in bundle["workflows"].get("workflows", [])}
    detail_ids = {item.get("workflow_id") for item in bundle["workflow_details"].get("details", [])}
    embedded_detail_ids = {item.get("workflow_id") for item in bundle["workflows"].get("details", [])}
    if workflow_ids != registry_ids:
        raise ValueError("sampleWorkflows does not match backend workflow registry")
    if detail_ids != registry_ids or embedded_detail_ids != registry_ids:
        raise ValueError("sampleWorkflowDetails does not cover every backend workflow")
    for detail in bundle["workflow_details"].get("details", []):
        if "input_schema" not in detail or "default_inputs" not in detail:
            raise ValueError(f"workflow detail missing input schema/defaults: {detail.get('workflow_id')}")
        if "warnings" not in detail or "provenance" not in detail:
            raise ValueError(f"workflow detail missing warnings/provenance: {detail.get('workflow_id')}")
    return True


def validate_frontend_result_coverage(bundle: dict[str, Any]) -> bool:
    """Ensure representative executable sample results and exports exist."""

    results = bundle["results"].get("results", [])
    if len(results) < 10:
        raise ValueError("frontend seed bundle requires at least 10 sample results")
    succeeded = [item for item in results if item.get("status") == "succeeded"]
    if len(succeeded) < 10:
        raise ValueError("frontend seed bundle requires at least 10 succeeded sample results")
    if not bundle["benchmark"].get("benchmark"):
        raise ValueError("frontend seed bundle requires a benchmark report")
    exports = bundle["exports"].get("exports", {})
    for fmt in ("json", "markdown", "python", "notebook_stub"):
        if fmt not in exports:
            raise ValueError(f"frontend seed bundle missing export sample: {fmt}")
    return True


def validate_no_branding_or_secrets_in_seed(bundle: dict[str, Any]) -> bool:
    """Scan seed JSON for external endpoints, secret-like strings, and branding assets."""

    text = json.dumps(to_json_safe(bundle), sort_keys=True)
    for pattern in FORBIDDEN_SEED_PATTERNS:
        match = pattern.search(text)
        if match:
            raise ValueError(f"forbidden seed pattern found: {match.group(0)}")
    return True


def write_frontend_seed_bundle(output_dir: str | Path = DEFAULT_FRONTEND_SEED_DIR) -> list[Path]:
    """Write all frontend seed files and return their paths."""

    bundle = build_frontend_seed_bundle()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for section, filename in SEED_FILE_NAMES.items():
        path = out / filename
        path.write_text(json.dumps(bundle[section], indent=2, sort_keys=True) + "\n", encoding="utf-8")
        written.append(path)
    js_path = out / "seedData.js"
    js_path.write_text(_render_seed_js(bundle), encoding="utf-8")
    written.append(js_path)
    return written


def load_frontend_seed_bundle(input_dir: str | Path = DEFAULT_FRONTEND_SEED_DIR) -> dict[str, Any]:
    """Load the seed bundle written for the frontend."""

    base = Path(input_dir)
    return {section: json.loads((base / filename).read_text(encoding="utf-8")) for section, filename in SEED_FILE_NAMES.items()}


def _workflow_detail_payload(workflow_id: str) -> dict[str, Any]:
    detail = get_workflow(workflow_id).to_dict()
    detail["default_inputs"] = get_default_inputs(workflow_id)
    detail["unsupported_limitations"] = _unsupported_limitations(detail)
    detail["clean_room_notice"] = CLEAN_ROOM_NOTICE
    return detail


def _build_sample_results() -> list[dict[str, Any]]:
    results = []
    for workflow_id in SAMPLE_RESULT_WORKFLOWS:
        execution = execute_workflow(workflow_id, get_default_inputs(workflow_id))
        results.append(execution.to_dict())
    return results


def _build_export_samples(sample_result: dict[str, Any] | None) -> dict[str, Any]:
    workflow_id = sample_result.get("workflow_id") if sample_result else "aer.statevector_native"
    inputs = get_default_inputs(str(workflow_id))
    result_for_export = sample_result or execute_workflow(str(workflow_id), inputs).to_dict()
    exports: dict[str, StudioExportResult] = {
        "json": export_result_json(result_for_export),
        "markdown": export_result_markdown(result_for_export),
        "python": export_workflow_python_snippet(str(workflow_id), inputs),
        "notebook_stub": export_workflow_notebook_stub(str(workflow_id), inputs),
    }
    return {key: value.to_dict() for key, value in exports.items()}


def _envelope(section: str, payload: dict[str, Any], generated_at: str) -> dict[str, Any]:
    return {
        "schema_version": FRONTEND_SCHEMA_VERSION,
        "backend_schema_version": SCHEMA_VERSION,
        "section": section,
        "generated_at": generated_at,
        "source": "quantumbridge.studio.backend",
        "quantumbridge_version": _quantumbridge_version(),
        "warnings": list(COMMON_WARNINGS),
        "provenance": dict(COMMON_PROVENANCE),
        "clean_room_notice": CLEAN_ROOM_NOTICE,
        **payload,
    }


def _validate_envelope(section: str, payload: dict[str, Any]) -> None:
    for key in ("schema_version", "generated_at", "source", "quantumbridge_version", "warnings", "provenance", "clean_room_notice"):
        if key not in payload:
            raise ValueError(f"{section} seed envelope missing {key}")
    if payload["source"] != "quantumbridge.studio.backend":
        raise ValueError(f"{section} seed source must be quantumbridge.studio.backend")


def _render_seed_js(bundle: dict[str, Any]) -> str:
    exports = {
        "sampleCatalog": bundle["catalog"],
        "sampleWorkflows": bundle["workflows"],
        "sampleWorkflowDetails": bundle["workflow_details"],
        "sampleResults": bundle["results"],
        "sampleBenchmarkReport": bundle["benchmark"],
        "sampleExports": bundle["exports"],
        "studioSchemaVersion": bundle["schema"],
    }
    lines = [
        "// This file is generated from QuantumBridge-owned Stage 10C/10E backend API outputs.",
        "// Local prototype seed data only: no secrets, credentials, cloud calls, or hardware access.",
    ]
    for name, value in exports.items():
        lines.append(f"export const {name} = {json.dumps(value, indent=2, sort_keys=True)};")
    return "\n\n".join(lines) + "\n"


def _quantumbridge_version() -> str:
    try:
        return metadata.version("quantumbridge-sdk")
    except metadata.PackageNotFoundError:
        return "local"


def _unsupported_limitations(detail: dict[str, Any]) -> list[str]:
    limitations = ["local prototype surface", "not production UI", "no cloud/token/hardware route"]
    unsupported = detail.get("unsupported_reason")
    if unsupported:
        limitations.append(str(unsupported))
    return limitations


CLEAN_ROOM_NOTICE = (
    "QuantumBridge Studio seed data is generated from QuantumBridge local backend services. "
    "Third-party project names identify compatibility targets only; no official endorsement, "
    "copied upstream UI, copied prose, cloud access, token access, hardware access, or production readiness is claimed."
)
COMMON_WARNINGS = (
    "Local prototype seed data only.",
    "No production UI claim.",
    "No cloud execution, credential reading, or hardware access.",
    "Compatibility names are inventory identifiers, not endorsement claims.",
)
COMMON_PROVENANCE = {
    "source": "quantumbridge.studio.backend",
    "copied_upstream_source": False,
    "copied_upstream_ui": False,
    "official_endorsement": False,
    "production_ready": False,
    "cloud_access": False,
    "token_access": False,
    "hardware_access": False,
}
