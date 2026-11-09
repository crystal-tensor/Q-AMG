# This file is independently implemented for QuantumBridge SDK.
"""Catalog service for QuantumBridge Studio local backend API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .api_models import StudioCatalogItem
from .provenance_service import base_studio_provenance
from .warning_service import workflow_boundary_warnings
from .workflow_registry import filter_workflows

CATALOG_PATH = Path("docs/compat/ecosystem/ibm_quantum_ecosystem_project_catalog_seed.json")


def list_ecosystem_projects() -> list[StudioCatalogItem]:
    return [_item_from_raw(raw) for raw in _load_raw_projects()]


def get_ecosystem_project(project_id: str) -> StudioCatalogItem:
    for item in list_ecosystem_projects():
        if item.project_id == project_id:
            return item
    raise KeyError(f"unknown ecosystem project_id: {project_id}")


def search_ecosystem_projects(query: str) -> list[StudioCatalogItem]:
    needle = str(query).lower()
    return [
        item
        for item in list_ecosystem_projects()
        if needle in item.project_id.lower() or needle in item.title.lower() or needle in item.category.lower()
    ]


def filter_ecosystem_projects(
    category: str | None = None,
    tags: list[str] | None = None,
    capability_level: int | None = None,
    studio_ready: str | None = None,
) -> list[StudioCatalogItem]:
    items = list_ecosystem_projects()
    if category is not None:
        items = [item for item in items if item.category == category]
    if studio_ready is not None:
        items = [item for item in items if item.studio_ready == studio_ready]
    if capability_level is not None:
        items = [item for item in items if item.capability_level >= int(capability_level)]
    if tags:
        wanted = {tag.lower() for tag in tags}
        items = [item for item in items if wanted.intersection({item.category.lower(), item.project_id.lower()})]
    return items


def list_project_workflows(project_id: str) -> list[dict[str, Any]]:
    return [item.to_dict() for item in filter_workflows(project_id=project_id)]


def get_project_status_summary() -> dict[str, Any]:
    items = list_ecosystem_projects()
    return {
        "project_count": len(items),
        "studio_ready": {value: sum(1 for item in items if item.studio_ready == value) for value in sorted({item.studio_ready for item in items})},
        "cloud_access": False,
        "token_access": False,
        "hardware_access": False,
        "official_endorsement": False,
    }


def get_clean_room_boundary_notice(project_id: str) -> dict[str, Any]:
    item = get_ecosystem_project(project_id)
    return {
        "project_id": item.project_id,
        "notice": "QuantumBridge Studio catalog metadata is clean-room, local-only, and not official upstream UI or prose.",
        "official_endorsement": False,
        "copied_upstream_source": False,
        "copied_upstream_ui": False,
        "production_parity_claim": False,
    }


def _load_raw_projects() -> list[dict[str, Any]]:
    if CATALOG_PATH.exists():
        data = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        return [dict(item) for item in data.get("projects", [])]
    return [
        {
            "project_id": "quantumbridge-studio",
            "upstream_name": "QuantumBridge Studio",
            "category": "studio",
            "current_status": "local_backend_api_slice",
            "executable_workflows": [],
            "studio_ready": "partial",
            "limitations": ["fallback catalog only"],
        }
    ]


def _item_from_raw(raw: dict[str, Any]) -> StudioCatalogItem:
    project_id = str(raw.get("project_id", raw.get("upstream_name", "unknown"))).lower().replace(" ", "-")
    workflows = [str(item) for item in raw.get("executable_workflows", [])]
    status = str(raw.get("current_status", "inventory"))
    capability = 3 if "executable" in status or workflows else 1
    return StudioCatalogItem(
        project_id=project_id,
        title=str(raw.get("upstream_name", project_id)),
        category=str(raw.get("category", "unknown")),
        capability_level=capability,
        executable_workflows=workflows,
        studio_ready=str(raw.get("studio_ready", "partial")),
        warnings=workflow_boundary_warnings(project_id=project_id) + [str(value) for value in raw.get("limitations", [])],
        provenance=base_studio_provenance(project_id=project_id),
        unsupported_reason=None if workflows else "no executable workflows recorded in catalog",
    )
