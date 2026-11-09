"""Fresh Python processes versus repeated full generation in the same process.

No OS cache flushing, no device reset and no claim of machine-cold startup.
Python 3.12 monotonic clocks are shared across processes on supported platforms:
https://docs.python.org/3.12/library/time.html#time.monotonic
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

WORKER_ENTRY_NS = time.monotonic_ns()
ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def worker(request_path, parent_start_ns):
    request = json.loads(request_path.read_text())
    before_import = time.monotonic_ns()
    from qamg.pipeline import generate
    from qamg.problem import build_problem
    from qamg.schema import DesignSpec, PlannerSettings
    imported = time.monotonic_ns()
    settings = PlannerSettings(backend=request["backend"], encoding="compact_binary", iterations=12)
    spec = DesignSpec.model_validate(request["spec"])
    rows = []
    for repetition in range(request["warm_repeats"]+1):
        before = time.monotonic_ns()
        result = generate(spec, settings, request_path.parent/f"generation-{repetition}")
        finished = time.monotonic_ns()
        # Evaluate output quality after the timed generation, never as a quantum warm start.
        p = build_problem(spec)
        optimum = min(p.cost(s) for s in p.assignments() if not p.violations(s))
        cost = p.cost(p.selection(result["selected_bits"]))
        rows.append({"repetition": repetition, "population": "first_in_fresh_process" if not repetition else "same_process_warm",
                     "generation_wall_s": (finished-before)/1e9,
                     "parent_spawn_to_first_artifacts_s": (finished-parent_start_ns)/1e9 if not repetition else None,
                     "valid": result["validation"]["valid"], "relative_gap": (cost-optimum)/max(1, abs(optimum)),
                     "target_met": cost <= optimum+.05*max(1, abs(optimum))+1e-9,
                     "selected_bits": result["selected_bits"], "source": result["provenance"]["execution_source"],
                     "device": result["resources"].get("capabilities", {}).get("name"),
                     "pipeline_timings": result["timings"]})
    evidence = {"worker_entry_after_parent_spawn_s": (WORKER_ENTRY_NS-parent_start_ns)/1e9,
                "pipeline_import_s": (imported-before_import)/1e9,
                "rows": rows}
    save(request_path.parent/"worker.json", evidence)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--parent-start-ns", type=int)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--processes-per-family-backend", type=int, default=5)
    parser.add_argument("--warm-repeats", type=int, default=2)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.parent_start_ns)
        return
    if not args.corpus or not args.output or args.processes_per_family_backend < 1 or args.warm_repeats < 1:
        parser.error("Provide corpus/output and positive repetition counts")
    if args.output.exists():
        raise SystemExit("Choose a new output directory")
    output = args.output.resolve()
    output.mkdir(parents=True)
    corpus = json.loads(args.corpus.read_text())
    heldout = [c for c in corpus["instances"] if c["split"] == "heldout" and c["spec"]["groups"] == 5]
    cases = [next(c for c in heldout if c["spec"]["family"] == family) for family in ["drive", "clamp"]]
    requests = [{"case": case["id"], "family": case["spec"]["family"], "backend": backend,
                 "repeat": repeat, "warm_repeats": args.warm_repeats,
                 "spec": {**case["spec"], "seed": 1000+repeat}}
                for repeat in range(args.processes_per_family_backend) for case in cases
                for backend in ["auto", "classical"]]
    random.Random(20261109).shuffle(requests)
    sources = [*sorted((ROOT/"src").rglob("*.py")), Path(__file__), ROOT/"uv.lock"]
    save(output/"manifest.json", {
        "requests": requests, "platform": platform.platform(), "python": sys.version,
        "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "settings": {"encoding": "compact_binary", "iterations": 12, "shots": 512, "depth": 2},
        "timing_scope": "Parent process spawn through first STEP/GLB/JSON artifacts; later calls repeat full pipeline with fresh encoding/compile",
        "cache_scope": "Fresh Python interpreter; OS page cache and GPU driver cache warm/uncontrolled; no OS/machine cold claim",
        "coverage": "Two representative five-group inputs only, five fresh processes per family/backend by default; not the 30-instance acceleration benchmark",
        "quality_rule": "Same original feasible objective gap <=5%; quality checked after timed generation; failed quality kept",
        "hardware_execution": False,
    })
    records = []
    with (output/"raw.jsonl").open("w") as raw:
        for index, request in enumerate(requests):
            folder = output/f"process-{index:03d}"
            folder.mkdir()
            save(folder/"request.json", request)
            started_ns = time.monotonic_ns()
            row = {**request, "worker_folder": str(folder)}
            try:
                completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", str(folder/"request.json"),
                                            "--parent-start-ns", str(started_ns)], cwd=ROOT, capture_output=True,
                                           text=True, timeout=180, check=False)
                (folder/"stdout.txt").write_text(completed.stdout)
                (folder/"stderr.txt").write_text(completed.stderr)
                row["return_code"] = completed.returncode
                row["result"] = json.loads((folder/"worker.json").read_text()) if completed.returncode == 0 else None
            except subprocess.TimeoutExpired:
                row.update(return_code=None, result=None, error="worker exceeded 180 seconds")
            row["parent_process_total_s_including_warm_calls"] = (time.monotonic_ns()-started_ns)/1e9
            records.append(row)
            raw.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
            raw.flush()
            print(f"cold-start process {index+1}/{len(requests)} {request['family']} {request['backend']} exit={row['return_code']}", flush=True)
    summary = {"registered_processes": len(requests), "completed_processes": sum(r["return_code"] == 0 for r in records),
               "scope": "Python-process cold only; caches warm/uncontrolled; representative inputs; descriptive timing, no speedup claim", "methods": {}}
    for family in ["drive", "clamp"]:
        for backend in ["auto", "classical"]:
            selected = [r for r in records if r["family"] == family and r["backend"] == backend]
            results = [r["result"] for r in selected if r["result"] is not None]
            first = [r["rows"][0]["parent_spawn_to_first_artifacts_s"] for r in results]
            warm = [x["generation_wall_s"] for r in results for x in r["rows"][1:]]
            summary["methods"][f"{family}:{backend}"] = {
                "registered_processes": len(selected), "completed_processes": len(results),
                "fresh_process_to_first_artifacts_mean_s": sum(first)/len(first) if first else None,
                "fresh_process_range_s": [min(first), max(first)] if first else None,
                "pipeline_import_mean_s": sum(r["pipeline_import_s"] for r in results)/len(results) if results else None,
                "same_process_warm_generation_mean_s": sum(warm)/len(warm) if warm else None,
                "registered_generations": len(selected)*(args.warm_repeats+1),
                "valid_target_generations": sum(x["valid"] and x["target_met"] for r in results for x in r["rows"]),
                "sources": sorted({x["source"] for r in results for x in r["rows"]}),
                "devices": sorted({x["device"] for r in results for x in r["rows"] if x["device"]}),
            }
    save(output/"summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
