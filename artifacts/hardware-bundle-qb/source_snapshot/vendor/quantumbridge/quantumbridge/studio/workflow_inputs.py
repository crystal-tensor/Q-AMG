# This file is independently implemented for QuantumBridge SDK.
"""Input schema helpers for Studio workflows."""

from __future__ import annotations

from typing import Any

from .api_models import StudioInputField, StudioInputSchema
from .workflow_registry import get_workflow_input_schema


def build_input_schema_for_workflow(workflow_id: str) -> StudioInputSchema:
    return get_workflow_input_schema(workflow_id)


def get_default_inputs(workflow_id: str) -> dict[str, Any]:
    return dict(build_input_schema_for_workflow(workflow_id).defaults)


def coerce_input_types(schema: StudioInputSchema, inputs: dict[str, Any] | None) -> dict[str, Any]:
    payload = {**dict(schema.defaults), **dict(inputs or {})}
    coerced: dict[str, Any] = {}
    for raw_field in schema.fields:
        field = raw_field if isinstance(raw_field, StudioInputField) else StudioInputField.from_dict(raw_field)
        if field.name not in payload:
            continue
        value = payload[field.name]
        if value is None:
            coerced[field.name] = None
        elif field.field_type == "integer":
            coerced[field.name] = int(value)
        elif field.field_type == "float":
            coerced[field.name] = float(value)
        elif field.field_type == "boolean":
            coerced[field.name] = bool(value)
        elif field.field_type == "string":
            coerced[field.name] = str(value)
        elif field.field_type == "enum":
            if field.enum and value not in field.enum:
                raise ValueError(f"{field.name} must be one of {field.enum}")
            coerced[field.name] = value
        elif field.field_type in {"list", "matrix"}:
            coerced[field.name] = list(value)
        elif field.field_type in {"dict", "circuit_ir"}:
            coerced[field.name] = dict(value)
        else:
            coerced[field.name] = value
    for key, value in payload.items():
        coerced.setdefault(key, value)
    return coerced


def validate_inputs_against_schema(schema: StudioInputSchema, inputs: dict[str, Any] | None) -> tuple[bool, list[str]]:
    errors: list[str] = []
    payload = dict(inputs or {})
    for raw_field in schema.fields:
        field = raw_field if isinstance(raw_field, StudioInputField) else StudioInputField.from_dict(raw_field)
        try:
            field.validate()
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if field.required and field.name not in payload and field.name not in schema.defaults:
            errors.append(f"{field.name} is required")
    try:
        coerce_input_types(schema, payload)
    except (TypeError, ValueError) as exc:
        errors.append(str(exc))
    return (not errors, errors)
