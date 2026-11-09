# This file is independently implemented for QuantumBridge SDK.
# No source code or website content from IBM, Qiskit, PennyLane, or third-party projects was copied.
"""Clean-room ecosystem project catalog schema.

The catalog identifies upstream projects as compatibility targets while keeping
QuantumBridge independent. It does not scrape websites, copy project card text,
or imply upstream endorsement.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


VALID_CLEAN_ROOM_STATUS = {"planned", "inventory", "passthrough", "executable", "advisory"}


@dataclass(frozen=True)
class EcosystemProject:
    project_id: str
    display_name: str
    upstream_name: str
    category: str
    tags: tuple[str, ...] = ()
    upstream_homepage: str | None = None
    upstream_repository: str | None = None
    upstream_license: str | None = None
    upstream_owner: str | None = None
    install_extra: str | None = None
    package_name: str | None = None
    dependency_available: bool | None = None
    adapter_module: str | None = None
    capability_level: int = 0
    executable_workflows: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()
    tests: tuple[str, ...] = ()
    docs: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)
    official_endorsement: bool = False
    clean_room_status: str = "planned"
    ui_ready: bool = False
    priority: str = "P2"
    notes: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "tags", tuple(str(tag) for tag in self.tags))
        for attr in ("executable_workflows", "examples", "tests", "docs", "warnings"):
            object.__setattr__(self, attr, tuple(str(value) for value in getattr(self, attr)))
        self.validate()

    def validate(self) -> bool:
        if not self.project_id or not self.display_name or not self.upstream_name:
            raise ValueError("project_id, display_name, and upstream_name must be non-empty")
        if self.capability_level not in {0, 1, 2, 3, 4}:
            raise ValueError("capability_level must be between 0 and 4")
        if self.clean_room_status not in VALID_CLEAN_ROOM_STATUS:
            raise ValueError("clean_room_status is invalid")
        if self.official_endorsement is not False:
            raise ValueError("ecosystem catalog entries must not imply official endorsement")
        return True

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EcosystemProject":
        data = dict(payload)
        for attr in ("tags", "executable_workflows", "examples", "tests", "docs", "warnings"):
            data[attr] = tuple(data.get(attr, ()))
        data["provenance"] = dict(data.get("provenance", {}))
        return cls(**data)


@dataclass(frozen=True)
class EcosystemCatalog:
    projects: tuple[EcosystemProject, ...]
    catalog_version: str = "0.1"
    source_mode: str = "manual-clean-room"
    official_endorsement: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "projects", tuple(self.projects))
        self.validate()

    def validate(self) -> bool:
        if self.official_endorsement is not False:
            raise ValueError("catalog must not imply official endorsement")
        ids = [project.project_id for project in self.projects]
        if len(ids) != len(set(ids)):
            raise ValueError("project_id values must be unique")
        for project in self.projects:
            project.validate()
        return True

    def filter_by_category(self, category: str) -> tuple[EcosystemProject, ...]:
        return tuple(project for project in self.projects if project.category == category)

    def search(self, query: str) -> tuple[EcosystemProject, ...]:
        normalized = query.strip().lower()
        return tuple(
            project
            for project in self.projects
            if normalized in project.display_name.lower()
            or normalized in project.upstream_name.lower()
            or any(normalized in tag.lower() for tag in project.tags)
        )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "catalog_version": self.catalog_version,
            "source_mode": self.source_mode,
            "official_endorsement": self.official_endorsement,
            "projects": [project.to_dict() for project in self.projects],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EcosystemCatalog":
        return cls(
            projects=tuple(EcosystemProject.from_dict(row) for row in payload.get("projects", ())),
            catalog_version=str(payload.get("catalog_version", "0.1")),
            source_mode=str(payload.get("source_mode", "manual-clean-room")),
            official_endorsement=bool(payload.get("official_endorsement", False)),
        )


def make_catalog(projects: Iterable[EcosystemProject]) -> EcosystemCatalog:
    return EcosystemCatalog(tuple(projects))
