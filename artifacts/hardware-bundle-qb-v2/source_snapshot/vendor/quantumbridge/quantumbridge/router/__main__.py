# This file is independently implemented for QuantumBridge SDK.
"""Command line entry point for QuantumBridge Router."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmark import run_router_benchmark
from .core import RouterSLA, route_workload
from .demo import run_router_demo
from .quafu_tasks import quafu_tasks_to_backend_states, summarize_quafu_tasks_response
from .reports import write_router_demo_artifacts
from .workloads import workload_by_name


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        payload = args.func(args)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI returns structured local errors.
        print(json.dumps({"success": False, "error": {"code": type(exc).__name__, "message": str(exc)}}, indent=2, sort_keys=True))
        return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m quantumbridge.router")
    subcommands = parser.add_subparsers(dest="command", required=True)

    route = subcommands.add_parser("route", help="Route one named demo workload.")
    route.add_argument("--workload", default="qaoa-small", choices=["bell", "ghz", "ghz-4", "qaoa-small", "vqe-small"])
    route.add_argument("--min-fidelity", type=float, default=0.75)
    route.add_argument("--max-wait", type=float, default=900.0)
    route.add_argument("--max-cost", type=float, default=5.0)
    route.add_argument("--shots", type=int, default=1024)
    route.set_defaults(func=_cmd_route)

    benchmark = subcommands.add_parser("benchmark", help="Run Router-vs-baseline benchmark.")
    benchmark.add_argument("--jobs", type=int, default=100)
    benchmark.add_argument("--seed", type=int, default=7)
    benchmark.set_defaults(func=_cmd_benchmark)

    demo = subcommands.add_parser("demo", help="Generate complete Router demo artifacts.")
    demo.add_argument("--output-dir", default="output/router_demo")
    demo.add_argument("--jobs", type=int, default=100)
    demo.add_argument("--seed", type=int, default=7)
    demo.set_defaults(func=_cmd_demo)

    quafu = subcommands.add_parser("quafu-tasks", help="Summarize an authorized Quafu /api/tasks JSON export.")
    quafu.add_argument("--input", required=True, help="Path to JSON shaped like {'items': [...], 'total': N}.")
    quafu.add_argument("--route-workload", default=None, choices=["bell", "ghz", "ghz-4", "qaoa-small", "vqe-small"])
    quafu.add_argument("--min-fidelity", type=float, default=0.50)
    quafu.add_argument("--max-wait", type=float, default=1200.0)
    quafu.set_defaults(func=_cmd_quafu_tasks)

    return parser


def _cmd_route(args: argparse.Namespace) -> dict:
    sla = RouterSLA(min_fidelity=args.min_fidelity, max_wait=args.max_wait, max_cost=args.max_cost, shots=args.shots)
    return {"success": True, "decision": route_workload(workload_by_name(args.workload), sla=sla, workload_id=f"cli-{args.workload}").to_dict()}


def _cmd_benchmark(args: argparse.Namespace) -> dict:
    return {"success": True, "benchmark": run_router_benchmark(job_count=args.jobs, seed=args.seed).to_dict()}


def _cmd_demo(args: argparse.Namespace) -> dict:
    payload = run_router_demo(output_dir=Path(args.output_dir), job_count=args.jobs, seed=args.seed)
    return {"success": True, **payload}


def _cmd_quafu_tasks(args: argparse.Namespace) -> dict:
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    evidence = summarize_quafu_tasks_response(payload)
    result = {"success": True, "evidence": evidence.to_dict()}
    if args.route_workload:
        backends = quafu_tasks_to_backend_states(payload)
        sla = RouterSLA(min_fidelity=args.min_fidelity, max_wait=args.max_wait, max_cost=5.0, allow_mock=False)
        decision = route_workload(workload_by_name(args.route_workload), sla=sla, backends=backends, workload_id=f"quafu-{args.route_workload}")
        result["router_decision"] = decision.to_dict()
    return result


if __name__ == "__main__":
    raise SystemExit(main())
