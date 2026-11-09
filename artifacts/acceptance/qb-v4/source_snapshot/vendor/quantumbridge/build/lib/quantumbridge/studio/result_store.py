# This file is independently implemented for QuantumBridge SDK.
"""In-memory result store for QuantumBridge Studio."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .api_models import StudioExecutionResult, to_json_safe


class InMemoryResultStore:
    def __init__(self) -> None:
        self._results: dict[str, StudioExecutionResult] = {}

    def save_result(self, result: StudioExecutionResult) -> StudioExecutionResult:
        result.validate()
        self._results[result.execution_id] = result
        return result

    def get_result(self, execution_id: str) -> StudioExecutionResult:
        try:
            return self._results[execution_id]
        except KeyError as exc:
            raise KeyError(f"unknown execution_id: {execution_id}") from exc

    def list_results(self, limit: int | None = None) -> list[StudioExecutionResult]:
        values = list(self._results.values())
        return values if limit is None else values[-int(limit):]

    def clear_results(self) -> None:
        self._results.clear()

    def export_result(self, execution_id: str, format: str = "json", path: str | Path | None = None) -> str:
        result = self.get_result(execution_id)
        if format == "json":
            content = json.dumps(to_json_safe(result.to_dict()), indent=2, sort_keys=True)
        elif format == "markdown":
            content = f"# Studio Execution Result\\n\\n- execution_id: `{result.execution_id}`\\n- workflow_id: `{result.workflow_id}`\\n- status: `{result.status}`\\n"
        else:
            raise ValueError("format must be json or markdown")
        if path is not None:
            Path(path).write_text(content, encoding="utf-8")
        return content


DEFAULT_RESULT_STORE = InMemoryResultStore()


def save_result(result: StudioExecutionResult) -> StudioExecutionResult:
    return DEFAULT_RESULT_STORE.save_result(result)


def get_result(execution_id: str) -> StudioExecutionResult:
    return DEFAULT_RESULT_STORE.get_result(execution_id)


def list_results(limit: int | None = None) -> list[StudioExecutionResult]:
    return DEFAULT_RESULT_STORE.list_results(limit)


def clear_results() -> None:
    DEFAULT_RESULT_STORE.clear_results()


def export_result(execution_id: str, format: str = "json") -> str:
    return DEFAULT_RESULT_STORE.export_result(execution_id, format=format)
