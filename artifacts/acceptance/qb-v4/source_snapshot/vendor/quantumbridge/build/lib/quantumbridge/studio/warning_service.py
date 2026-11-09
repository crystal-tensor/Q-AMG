# This file is independently implemented for QuantumBridge SDK.
"""Warning aggregation helpers for QuantumBridge Studio."""

from __future__ import annotations

from typing import Any

from .warnings import studio_warnings


def collect_result_warnings(result: Any) -> list[str]:
    if hasattr(result, "warnings"):
        return [str(value) for value in (getattr(result, "warnings") or [])]
    if isinstance(result, dict):
        value = result.get("warnings")
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value]
    return []


def summarize_warning_levels(warnings: list[str]) -> dict[str, int]:
    return {"info": len(warnings), "warning": 0, "error": 0}


def build_clean_room_notice(project_id: str | None = None) -> str:
    label = f" for {project_id}" if project_id else ""
    return f"QuantumBridge Studio uses clean-room local metadata{label}; no third-party source, UI, prose, or branding is copied."


def build_safety_and_boundary_notice(workflow_id: str | None = None) -> str:
    label = f" {workflow_id}" if workflow_id else ""
    return f"Studio workflow{label} is local-only: no cloud, token, credential, or hardware access."


def workflow_boundary_warnings(workflow_id: str | None = None, project_id: str | None = None) -> list[str]:
    return studio_warnings(build_clean_room_notice(project_id), build_safety_and_boundary_notice(workflow_id))
