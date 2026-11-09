# This file is independently implemented for QuantumBridge SDK.
# No source code from IBM, Qiskit, or Benchpress was copied.
"""JSON and Markdown reporting for local benchmarks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_benchmark_report_json(result: Any, path: str | Path) -> Path:
    payload = result.to_dict() if hasattr(result, "to_dict") else dict(result)
    target = Path(path)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return target


def write_benchmark_report_markdown(result: Any, path: str | Path) -> Path:
    payload = result.to_dict() if hasattr(result, "to_dict") else dict(result)
    target = Path(path)
    lines = [
        "# QuantumBridge Benchmark Report",
        "",
        f"- workflow: {payload.get('workflow')}",
        f"- status: {payload.get('status')}",
        f"- passed: {payload.get('passed')}",
        f"- elapsed_seconds: {payload.get('elapsed_seconds')}",
        "",
        "## Metrics",
        "",
    ]
    for key, value in dict(payload.get("metrics", {})).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Boundaries", "", "- not official Benchpress output", "- not production performance ranking", "- no cloud/token/hardware access"])
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target
