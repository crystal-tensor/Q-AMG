# This file is independently implemented for QuantumBridge SDK.
"""Command line entry point for the QuantumBridge Studio local backend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .api_models import to_json_safe
from .benchmark_service import run_benchmark_suite_by_id
from .execution_service import execute_workflow
from .export_service import export_workflow_sample
from .frontend_contract import DEFAULT_FRONTEND_SEED_DIR, write_frontend_seed_bundle
from .workflow_inputs import get_default_inputs
from .workflow_registry import list_workflows


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        result = args.func(args)
        print(json.dumps(to_json_safe(result), indent=2, sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI returns structured local errors.
        print(json.dumps({"success": False, "error": {"code": type(exc).__name__, "message": str(exc)}}, indent=2, sort_keys=True))
        return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m quantumbridge.studio")
    subcommands = parser.add_subparsers(dest="command", required=True)

    generate = subcommands.add_parser("generate-seed", help="Generate local frontend seed data.")
    generate.add_argument("--output-dir", default=str(DEFAULT_FRONTEND_SEED_DIR))
    generate.set_defaults(func=_cmd_generate_seed)

    list_cmd = subcommands.add_parser("list-workflows", help="List registered local Studio workflows.")
    list_cmd.set_defaults(func=_cmd_list_workflows)

    execute = subcommands.add_parser("execute", help="Execute a local Studio workflow with default inputs.")
    execute.add_argument("--workflow", required=True)
    execute.set_defaults(func=_cmd_execute)

    benchmark = subcommands.add_parser("benchmark", help="Run a local benchmark suite.")
    benchmark.add_argument("--suite", required=True)
    benchmark.set_defaults(func=_cmd_benchmark)

    export = subcommands.add_parser("export", help="Export a local workflow sample.")
    export.add_argument("--workflow", required=True)
    export.add_argument("--format", default="json", choices=["json", "markdown", "python", "notebook"])
    export.set_defaults(func=_cmd_export)

    return parser


def _cmd_generate_seed(args: argparse.Namespace) -> dict[str, Any]:
    written = write_frontend_seed_bundle(Path(args.output_dir))
    return {"success": True, "written_files": [str(path) for path in written]}


def _cmd_list_workflows(args: argparse.Namespace) -> dict[str, Any]:
    workflows = [item.to_dict() for item in list_workflows()]
    return {"success": True, "workflow_count": len(workflows), "workflows": workflows}


def _cmd_execute(args: argparse.Namespace) -> dict[str, Any]:
    workflow_id = str(args.workflow)
    result = execute_workflow(workflow_id, get_default_inputs(workflow_id))
    return {"success": result.status == "succeeded", "execution": result.to_dict()}


def _cmd_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    report = run_benchmark_suite_by_id(_normalize_suite_id(args.suite))
    return {"success": True, "benchmark": report.to_dict()}


def _cmd_export(args: argparse.Namespace) -> dict[str, Any]:
    exported = export_workflow_sample(str(args.workflow), str(args.format))
    return {"success": True, "export": exported.to_dict()}


def _normalize_suite_id(value: str) -> str:
    suite_id = str(value).strip().lower().replace("-", "_")
    if suite_id == "basic":
        return "circuit_basic"
    return suite_id


if __name__ == "__main__":
    raise SystemExit(main())
