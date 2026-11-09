# This file is independently implemented for QuantumBridge SDK.
# No source code from IBM, Qiskit, PennyLane, Benchpress, or third-party projects was copied.
"""Lightweight data models for the QuantumBridge Studio local backend API."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Mapping

SCHEMA_VERSION = "quantumbridge-studio-api-v0.1"


def to_json_safe(value: Any) -> Any:
    """Convert common QuantumBridge result objects into JSON-safe values."""

    if hasattr(value, "to_dict") and callable(value.to_dict):
        return to_json_safe(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_json_safe(item) for item in value]
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


class StudioSerializable:
    """Mixin used by simple dataclass models in this module."""

    def validate(self) -> bool:
        return True

    def to_dict(self) -> dict[str, Any]:
        return to_json_safe(asdict(self))

    def to_json(self) -> str:
        self.validate()
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]):
        names = {field.name for field in fields(cls)}
        return cls(**{name: data[name] for name in names if name in data})


@dataclass
class StudioWarning(StudioSerializable):
    message: str = ""
    level: str = "info"
    code: str = "studio_notice"

    def validate(self) -> bool:
        if self.level not in {"info", "warning", "error"}:
            raise ValueError("StudioWarning.level must be info, warning, or error")
        return True


@dataclass
class StudioProvenance(StudioSerializable):
    source: str = "quantumbridge"
    workflow_id: str | None = None
    project_id: str | None = None
    copied_upstream_source: bool = False
    official_endorsement: bool = False
    production_ready: bool = False
    cloud_access: bool = False
    token_access: bool = False
    hardware_access: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> bool:
        if self.copied_upstream_source:
            raise ValueError("Studio provenance cannot claim copied upstream source")
        if self.official_endorsement:
            raise ValueError("Studio provenance cannot claim official endorsement")
        if self.production_ready:
            raise ValueError("Studio executable slice is not production-ready")
        if self.cloud_access or self.token_access or self.hardware_access:
            raise ValueError("Studio local backend cannot access cloud, tokens, or hardware")
        return True


@dataclass
class StudioError(StudioSerializable):
    message: str = ""
    code: str = "studio_error"
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class StudioAPIResponse(StudioSerializable):
    success: bool = True
    data: Any = field(default_factory=dict)
    error: StudioError | dict[str, Any] | None = None
    warnings: list[StudioWarning | dict[str, Any] | str] = field(default_factory=list)
    provenance: StudioProvenance | dict[str, Any] = field(default_factory=StudioProvenance)
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> bool:
        if not self.success and self.error is None:
            raise ValueError("unsuccessful StudioAPIResponse requires an error")
        return True


@dataclass
class StudioCatalogItem(StudioSerializable):
    project_id: str = ""
    title: str = ""
    category: str = ""
    capability_level: int = 0
    executable_workflows: list[str] = field(default_factory=list)
    studio_ready: str = "partial"
    warnings: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    unsupported_reason: str | None = None
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> bool:
        if not self.project_id:
            raise ValueError("StudioCatalogItem.project_id is required")
        return True


@dataclass
class StudioInputField(StudioSerializable):
    name: str = ""
    field_type: str = "string"
    required: bool = False
    default: Any = None
    description: str = ""
    enum: list[Any] = field(default_factory=list)
    minimum: float | None = None
    maximum: float | None = None

    def validate(self) -> bool:
        if self.field_type not in {
            "integer",
            "float",
            "string",
            "boolean",
            "enum",
            "list",
            "dict",
            "matrix",
            "bitstring",
            "circuit_ir",
            "optional",
        }:
            raise ValueError(f"unsupported Studio input field type: {self.field_type}")
        if not self.name:
            raise ValueError("StudioInputField.name is required")
        return True


@dataclass
class StudioInputSchema(StudioSerializable):
    workflow_id: str = ""
    fields: list[StudioInputField | dict[str, Any]] = field(default_factory=list)
    defaults: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    unsupported_reason: str | None = None
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> bool:
        if not self.workflow_id:
            raise ValueError("StudioInputSchema.workflow_id is required")
        for item in self.fields:
            if isinstance(item, StudioInputField):
                item.validate()
            else:
                StudioInputField.from_dict(item).validate()
        return True


@dataclass
class StudioWorkflowSummary(StudioSerializable):
    workflow_id: str = ""
    title: str = ""
    project_id: str = ""
    category: str = ""
    executable: bool = True
    studio_ready: str = "partial"
    local_only: bool = True
    cloud_access: bool = False
    token_access: bool = False
    hardware_access: bool = False
    production_ready: bool = False
    warnings: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> bool:
        if not self.workflow_id:
            raise ValueError("StudioWorkflowSummary.workflow_id is required")
        if self.cloud_access or self.token_access or self.hardware_access or self.production_ready:
            raise ValueError("Studio workflow summaries must be local-only and non-production")
        return True


@dataclass
class StudioWorkflowDetail(StudioWorkflowSummary):
    description: str = ""
    input_schema: StudioInputSchema | dict[str, Any] = field(default_factory=StudioInputSchema)
    output_schema: dict[str, Any] = field(default_factory=dict)
    result_schema: str = ""
    examples: list[dict[str, Any]] = field(default_factory=list)
    unsupported_reason: str | None = None


@dataclass
class StudioExecutionRequest(StudioSerializable):
    workflow_id: str = ""
    inputs: dict[str, Any] = field(default_factory=dict)
    seed: int | None = None
    dry_run: bool = False
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> bool:
        if not self.workflow_id:
            raise ValueError("StudioExecutionRequest.workflow_id is required")
        return True


@dataclass
class StudioExecutionResult(StudioSerializable):
    execution_id: str = ""
    workflow_id: str = ""
    status: str = "succeeded"
    result: Any = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    error: StudioError | dict[str, Any] | None = None
    unsupported_reason: str | None = None
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> bool:
        if not self.execution_id:
            raise ValueError("StudioExecutionResult.execution_id is required")
        if self.status not in {"succeeded", "failed", "unsupported", "dry_run"}:
            raise ValueError("unsupported Studio execution status")
        if self.status in {"failed", "unsupported"} and self.error is None and not self.unsupported_reason:
            raise ValueError("failed or unsupported execution requires an error or unsupported_reason")
        return True


@dataclass
class StudioExportResult(StudioSerializable):
    export_type: str = "json"
    content: str = ""
    workflow_id: str | None = None
    execution_id: str | None = None
    warnings: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    unsupported_reason: str | None = None
    schema_version: str = SCHEMA_VERSION


@dataclass
class StudioBenchmarkReport(StudioSerializable):
    suite_id: str = ""
    result: Any = field(default_factory=dict)
    report_json: str = ""
    report_markdown: str = ""
    warnings: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    unsupported_reason: str | None = None
    schema_version: str = SCHEMA_VERSION
