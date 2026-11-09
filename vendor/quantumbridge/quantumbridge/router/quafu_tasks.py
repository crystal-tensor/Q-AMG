# This file is independently implemented for QuantumBridge SDK.
# No source code from Quafu or vendor SDKs was copied.
"""Quafu task-list evidence adapter for Router PoCs.

The public Quafu task page loads task rows from `/api/tasks` after login. This
module consumes that authorized JSON response and turns it into a queue snapshot
that the Router can use without touching tokens or submitting/cancelling jobs.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Iterable, Mapping

from .core import BackendState


ACTIVE_STATUSES = {"Submitted", "Transpiled", "Pending"}
TERMINAL_STATUSES = {"Finished", "Failed", "Canceled"}
KNOWN_QUAFU_CHIPS = ["Baihua", "Dongling", "Haituo", "Yunmeng", "Miaofeng", "Yudu", "Hongluo"]


@dataclass
class QuafuChipQueue:
    chip_name: str
    total_tasks: int = 0
    active_tasks: int = 0
    terminal_tasks: int = 0
    status_counts: dict[str, int] = field(default_factory=dict)
    latest_submit_time: str | None = None
    oldest_active_submit_time: str | None = None
    task_names_seen: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QuafuTaskEvidence:
    total_reported: int
    rows_seen: int
    chips_seen: list[str]
    active_statuses: list[str]
    terminal_statuses: list[str]
    chip_queues: list[QuafuChipQueue]
    queue_imbalance_ratio: float
    router_need_score: float
    evidence_codes: list[str]
    warnings: list[str]
    provenance: dict[str, Any]
    schema_version: str = "quantumbridge-quafu-task-evidence-v0.1"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["chip_queues"] = [queue.to_dict() for queue in self.chip_queues]
        return payload


def summarize_quafu_tasks_response(payload: Mapping[str, Any] | list[Mapping[str, Any]]) -> QuafuTaskEvidence:
    """Summarize an authorized Quafu `/api/tasks` response.

    Expected response shape from the Quafu frontend is `{"items": [...],
    "total": N}`. A plain row list is also accepted for exports.
    """

    rows, total_reported = _extract_rows(payload)
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    warnings: list[str] = []
    for row in rows:
        chip_name = str(row.get("chipName") or "UNKNOWN")
        grouped[chip_name].append(row)

    queues = [_summarize_chip(chip_name, chip_rows) for chip_name, chip_rows in sorted(grouped.items())]
    chips_seen = [queue.chip_name for queue in queues]
    active_values = [queue.active_tasks for queue in queues]
    max_active = max(active_values, default=0)
    min_active = min(active_values, default=0)
    queue_imbalance_ratio = round((max_active + 1) / (min_active + 1), 4) if active_values else 1.0
    status_counts = Counter(str(row.get("status") or "UNKNOWN") for row in rows)
    evidence_codes = _evidence_codes(rows, queues, queue_imbalance_ratio)

    missing_known = [chip for chip in KNOWN_QUAFU_CHIPS if chip not in chips_seen]
    if missing_known:
        warnings.append("not_all_public_frontend_chip_filters_appear_in_this_page")
    if not rows:
        warnings.append("no_task_rows_seen; login_or_exported_api_response_required")
    if "UNKNOWN" in chips_seen:
        warnings.append("some_rows_missing_chipName")
    if any("UNKNOWN" in queue.status_counts for queue in queues):
        warnings.append("some_rows_missing_status")

    return QuafuTaskEvidence(
        total_reported=total_reported,
        rows_seen=len(rows),
        chips_seen=chips_seen,
        active_statuses=[status for status in sorted(status_counts) if status in ACTIVE_STATUSES],
        terminal_statuses=[status for status in sorted(status_counts) if status in TERMINAL_STATUSES],
        chip_queues=queues,
        queue_imbalance_ratio=queue_imbalance_ratio,
        router_need_score=_router_need_score(rows, queues, queue_imbalance_ratio),
        evidence_codes=evidence_codes,
        warnings=warnings,
        provenance={
            "source": "authorized_quafu_api_tasks_response_or_export",
            "api_path_observed_in_public_frontend": "/api/tasks",
            "read_only": True,
            "token_access": False,
            "hardware_submission": False,
            "official_endorsement": False,
            "public_frontend_chip_filters": KNOWN_QUAFU_CHIPS,
        },
    )


def quafu_tasks_to_backend_states(
    payload: Mapping[str, Any] | list[Mapping[str, Any]],
    *,
    avg_job_seconds: float = 35.0,
    num_qubits_by_chip: Mapping[str, int] | None = None,
) -> list[BackendState]:
    """Build Router backend states from authorized Quafu task rows."""

    evidence = summarize_quafu_tasks_response(payload)
    qubit_counts = dict(num_qubits_by_chip or {})
    backends: list[BackendState] = []
    for queue in evidence.chip_queues:
        backends.append(
            BackendState(
                backend_id=f"quafu_{queue.chip_name.lower()}",
                provider="quafu_task_list",
                num_qubits=qubit_counts.get(queue.chip_name, 5),
                queue_depth=queue.active_tasks,
                avg_job_seconds=avg_job_seconds,
                utilization=min(0.95, queue.active_tasks / max(queue.total_tasks, 1)),
                calibration_age_seconds=3600,
                single_qubit_error=0.0015,
                two_qubit_error=0.018,
                readout_error=0.025,
                multiprogramming_slots=1,
                access_mode="quafu_authorized_task_list_readonly",
                real_adapter_ready=True,
                dry_run_only=True,
            )
        )
    return backends


def _extract_rows(payload: Mapping[str, Any] | list[Mapping[str, Any]]) -> tuple[list[Mapping[str, Any]], int]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, Mapping)], len(payload)
    items = payload.get("items", [])
    if not isinstance(items, list):
        items = []
    total = payload.get("total", len(items))
    try:
        total_reported = int(total)
    except (TypeError, ValueError):
        total_reported = len(items)
    return [row for row in items if isinstance(row, Mapping)], total_reported


def _summarize_chip(chip_name: str, rows: Iterable[Mapping[str, Any]]) -> QuafuChipQueue:
    materialized = list(rows)
    status_counts = Counter(str(row.get("status") or "UNKNOWN") for row in materialized)
    active_rows = [row for row in materialized if str(row.get("status") or "") in ACTIVE_STATUSES]
    submit_times = [_parse_submit_time(row.get("submitTime")) for row in materialized]
    active_times = [_parse_submit_time(row.get("submitTime")) for row in active_rows]
    task_names = {str(row.get("taskName")) for row in materialized if row.get("taskName")}
    return QuafuChipQueue(
        chip_name=chip_name,
        total_tasks=len(materialized),
        active_tasks=sum(status_counts.get(status, 0) for status in ACTIVE_STATUSES),
        terminal_tasks=sum(status_counts.get(status, 0) for status in TERMINAL_STATUSES),
        status_counts=dict(sorted(status_counts.items())),
        latest_submit_time=_format_time(max((value for value in submit_times if value), default=None)),
        oldest_active_submit_time=_format_time(min((value for value in active_times if value), default=None)),
        task_names_seen=len(task_names),
    )


def _parse_submit_time(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    candidates = [
        text,
        text.replace("T", " "),
        _quafu_dash_time_to_iso(text),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d-%H-%M-%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(candidate[:19], fmt)
            except ValueError:
                continue
    return None


def _quafu_dash_time_to_iso(text: str) -> str | None:
    parts = text.split("-")
    if len(parts) >= 6:
        return "-".join(parts[:3]) + " " + ":".join(parts[3:6])
    return None


def _format_time(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.strftime("%Y-%m-%d %H:%M:%S")


def _evidence_codes(rows: list[Mapping[str, Any]], queues: list[QuafuChipQueue], imbalance: float) -> list[str]:
    codes = ["quafu_tasks_api_shape_observed"]
    if rows:
        codes.append("authorized_task_rows_parseable")
    if len(queues) >= 2:
        codes.append("multi_chip_task_history")
    if any(queue.active_tasks > 0 for queue in queues):
        codes.append("active_queue_visible")
    if imbalance >= 2.0:
        codes.append("queue_imbalance_supports_routing")
    if any(queue.terminal_tasks > 0 for queue in queues):
        codes.append("historical_outcome_status_visible")
    return codes


def _router_need_score(rows: list[Mapping[str, Any]], queues: list[QuafuChipQueue], imbalance: float) -> float:
    score = 0.0
    if rows:
        score += 0.25
    if len(queues) >= 2:
        score += 0.25
    if any(queue.active_tasks > 0 for queue in queues):
        score += 0.20
    if any(queue.terminal_tasks > 0 for queue in queues):
        score += 0.10
    if imbalance >= 1.5:
        score += min(0.20, (imbalance - 1.0) / 10)
    return round(min(score, 1.0), 4)
