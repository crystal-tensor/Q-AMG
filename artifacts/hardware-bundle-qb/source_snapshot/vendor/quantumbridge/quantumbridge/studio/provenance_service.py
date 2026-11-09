# This file is independently implemented for QuantumBridge SDK.
"""Provenance helpers for QuantumBridge Studio local API results."""

from __future__ import annotations

from typing import Any

from .api_models import StudioProvenance


def base_studio_provenance(workflow_id: str | None = None, project_id: str | None = None) -> dict[str, Any]:
    return StudioProvenance(
        workflow_id=workflow_id,
        project_id=project_id,
        metadata={
            "api_layer": "quantumbridge.studio",
            "local_router": True,
            "frontend_ui_implementation": False,
        },
    ).to_dict()


def collect_result_provenance(result: Any) -> dict[str, Any]:
    if hasattr(result, "provenance"):
        return dict(getattr(result, "provenance") or {})
    if isinstance(result, dict):
        value = result.get("provenance")
        if isinstance(value, dict):
            return dict(value)
    return {}


def build_clean_room_notice(project_id: str | None = None) -> dict[str, Any]:
    return {
        "project_id": project_id,
        "copied_upstream_source": False,
        "copied_upstream_ui": False,
        "copied_upstream_prose": False,
        "official_endorsement": False,
        "production_parity_claim": False,
    }


def build_safety_and_boundary_notice(workflow_id: str | None = None) -> dict[str, Any]:
    return {
        "workflow_id": workflow_id,
        "local_only": True,
        "cloud_access": False,
        "token_access": False,
        "hardware_access": False,
        "production_api_server": False,
        "frontend_ui_implementation": False,
    }
