"""Reproducible held-out experiments. Measurements never imply hardware advantage.

The target is an independent predicate of the original discrete design problem.
It excludes the deliberately lazy envelope constraint (not present in corpus).
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import shutil
import time
from pathlib import Path

import numpy as np

from .backends import make_backend
from .classical import solve_classical
from .problem import build_problem
from .quantum import QuantumStructurePrior
from .schema import DesignSpec, PlannerSettings

CLASSICAL_METHODS = ["exact", "cpsat", "cpsat_qubo", "anneal", "factorized"]


def dump(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def snapshot_sources(directory: Path) -> dict:
    root = Path(__file__).resolve().parents[2]
    paths = sorted((root/"src").rglob("*.py")) + [root/"pyproject.toml", root/"uv.lock"]
    hashes = {}
    for path in paths:
        relative = path.relative_to(root)
        dest = directory/"source_snapshot"/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def create_corpus(path: Path, heldout: int = 30) -> dict:
    rng = np.random.default_rng(20261109)
    instances = []
    for split, size in [("train", 12), ("heldout", heldout)]:
        for i in range(size):
            groups = 3+i % 3
            family = "drive" if (i//3) % 2 == 0 else "clamp"
            spec = DesignSpec(family=family, groups=groups,
                              seed=int(rng.integers(0, 2**24)),
                              load="heavy" if i % 2 else "light", compact=bool(i % 4 == 0),
                              shaft_length_mm=float(rng.uniform(75, 140)),
                              opening_mm=float(rng.uniform(20, 45)),
                              scale=float(rng.uniform(.8, 1.2)))
            p = build_problem(spec)
            spec.cost_overrides = {g.key: list(map(int, rng.integers(1, 21, 2))) for g in p.groups}
            instances.append({"id": f"{split}-{i:03d}", "split": split, "spec": spec.model_dump()})
    corpus = {"schema_version": 1, "generator_seed": 20261109,
              "description": "Synthetic assembly catalogue preferences; scores are not measured monetary or load data.",
              "instances": instances}
    dump(path, corpus)
    return corpus


def summarize(records: list[dict], seed: int = 20261109) -> dict:
    methods = sorted({r["method"] for r in records})
    summary = {}
    for method in methods:
        rows = [r for r in records if r["method"] == method]
        hits = [r["time_to_target_s"] for r in rows if r.get("time_to_target_s") is not None]
        gaps = [r["relative_gap"] for r in rows if r.get("relative_gap") is not None]
        summary[method] = {
            "runs": len(rows), "success_rate": len(hits)/len(rows),
            "mean_time_to_target_successes_s": float(np.mean(hits)) if hits else None,
            "median_wall_s": float(np.median([r["wall_s"] for r in rows])),
            "mean_relative_gap_returned": float(np.mean(gaps)) if gaps else None,
            "failure_count": len(rows)-len(hits),
            "timing_population": "process_warm_problem_cold",
        }
    # Paired, instance-clustered bootstrap; incomplete pairs are disclosed, not imputed as wins.
    speedups = {}
    classical = [r for r in records if r["method"] in CLASSICAL_METHODS]
    rng = np.random.default_rng(seed)
    for method in methods:
        if not method.startswith("qaoa"):
            continue
        logs, excluded = [], 0
        for instance in sorted({r["instance"] for r in records}):
            qrows = [r for r in records if r["instance"] == instance and r["method"] == method]
            cgroups = [[r for r in classical if r["instance"] == instance and r["method"] == cm]
                       for cm in CLASSICAL_METHODS]
            valid_classical = [g for g in cgroups if g and all(r.get("time_to_target_s") is not None for r in g)]
            if not qrows or any(r.get("time_to_target_s") is None for r in qrows) or not valid_classical:
                excluded += 1
                continue
            best_classical = min(float(np.mean([r["time_to_target_s"] for r in g])) for g in valid_classical)
            quantum = float(np.mean([r["time_to_target_s"] for r in qrows]))
            logs.append(math.log(max(1e-12, best_classical)/max(1e-12, quantum)))
        if logs:
            arr = np.array(logs)
            boot = np.exp(np.mean(rng.choice(arr, size=(2000, len(arr)), replace=True), axis=1))
            speedups[method] = {"geomean_speedup_vs_fastest_observed_classical": float(np.exp(arr.mean())),
                                "ci95": list(map(float, np.quantile(boot, [.025, .975]))),
                                "paired_instances": len(arr), "excluded_instances": excluded,
                                "claim_scope": "simulator wall-clock only; not physical quantum acceleration"}
    return {"methods": summary, "speedups": speedups,
            "hardware_validated": False, "quantum_acceleration_target_met": False}


def run_benchmark(output: str | Path, settings: PlannerSettings | None = None,
                  instances: int = 30, seeds: int = 5, geometry: bool = True,
                  corpus_path: str | Path | None = None,
                  methods: list[str] | None = None, split: str = "heldout",
                  warm_start_file: str | Path | None = None,
                  classical_evaluations: int = 1792) -> dict:
    start = time.perf_counter()
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory/"raw.jsonl").exists():
        raise FileExistsError("Refusing to overwrite an experiment; choose a new output directory")
    settings = settings or PlannerSettings(iterations=12, shots=256)
    if geometry:
        from .geometry import decode_geometry, validate_geometry
    cp = Path(corpus_path) if corpus_path else directory/"corpus.json"
    corpus = json.loads(cp.read_text()) if cp.exists() else create_corpus(cp, max(30, instances))
    cases = [i for i in corpus["instances"] if i["split"] == split][:instances]
    methods = methods or ["qaoa_xy", *CLASSICAL_METHODS]
    if set(methods) - set(CLASSICAL_METHODS + ["qaoa_xy", "qaoa_x", "qaoa_binary"]):
        raise ValueError("Unknown benchmark method")
    transfer = json.loads(Path(warm_start_file).read_text()) if warm_start_file else None
    if transfer and (transfer["depth"] != settings.depth or
                     transfer["corpus_sha256"] != hashlib.sha256(cp.read_bytes()).hexdigest()):
        raise ValueError("Transfer parameters must match depth and the exact corpus split")
    amortization_count = len(cases)*seeds*sum(m.startswith("qaoa") for m in methods)
    training_per_request = transfer["training_wall_s"]/max(1, amortization_count) if transfer else 0
    manifest = {"settings": settings.model_dump(), "instances": [i["id"] for i in cases],
                "seeds": seeds, "split": split, "methods": methods, "geometry": geometry,
                "corpus_sha256": hashlib.sha256(cp.read_bytes()).hexdigest(),
                "target": {"feasible": True, "relative_gap": .05,
                           "gap_denominator": "max(1,abs(optimum))", "reference": "exact categorical optimum"},
                "budget": {"timeout_s_per_method": settings.timeout_s,
                           "quantum_calls_max": max(settings.iterations, 2*settings.depth+2)+1,
                           "classical_stochastic_evaluations": classical_evaluations},
                "timing_scope": "Imported libraries warm, each problem encoding/compile cold; angle transfer recorded separately.",
                "platform": platform.platform(), "machine": platform.machine(),
                "packages": {n: importlib.metadata.version(n) for n in ["qiskit", "qiskit-aer", "numpy", "scipy", "cadquery", "ortools"]},
                "hardware_execution_requested": settings.backend == "company",
                "advantage_claim": "unverified"}
    manifest["source_sha256"] = snapshot_sources(directory)
    manifest["warm_start"] = ({"file": str(warm_start_file),
                                "sha256": hashlib.sha256(Path(warm_start_file).read_bytes()).hexdigest(),
                                "training_wall_s": transfer["training_wall_s"],
                                "amortization_requests": amortization_count,
                                "charged_training_s_per_request": training_per_request}
                               if transfer else None)
    dump(directory/"manifest.json", manifest)  # Written BEFORE timing or looking at results.
    quantum_backend = make_backend(settings.backend, settings.max_qubits) if any(
        m.startswith("qaoa") for m in methods) else None
    records = []
    raw_path = directory/"raw.jsonl"
    if raw_path.exists():
        raise FileExistsError("Refusing to mix benchmark reruns; choose a new output directory")
    with raw_path.open("w") as stream:
        for instance in cases:
            problem = build_problem(DesignSpec.model_validate(instance["spec"]))
            reference = solve_classical(problem, "exact")
            if reference.best is None:
                raise RuntimeError("Corpus contains an infeasible reference instance")
            optimum = problem.cost(problem.selection(reference.best.bits))
            threshold = optimum+.05*max(1, abs(optimum))

            def target(bits, problem=problem, threshold=threshold):
                s = problem.selection(bits)
                return s is not None and not problem.violations(s) and problem.cost(s) <= threshold+1e-9

            for repeat in range(seeds):
                # Randomized method order reduces thermal/cache-order bias; recorded in raw rows.
                order = np.random.default_rng(problem.spec.seed+repeat).permutation(methods)
                for method in order:
                    seed = 1000+repeat
                    before = time.perf_counter()
                    row = {"instance": instance["id"], "family": problem.spec.family,
                           "variables": problem.encode().n, "method": str(method), "seed": seed,
                           "optimum": optimum, "threshold": threshold,
                           "reference_certified": True, "repair": "none"}
                    try:
                        if method.startswith("qaoa"):
                            updates = ({"encoding": "compact_binary", "mixer": "x"}
                                       if method == "qaoa_binary" else
                                       {"encoding": "onehot", "mixer": str(method).split("_")[1]})
                            local = settings.model_copy(update=updates)
                            parameters = (transfer["contexts"][f"{problem.spec.family}:{len(problem.groups)}"]["parameters"]
                                          if transfer else None)
                            result = QuantumStructurePrior(quantum_backend).plan(
                                problem, local, seed, target, initial_parameters=parameters)
                        else:
                            result = solve_classical(problem, str(method), seed,
                                                     evaluations=classical_evaluations,
                                                     target=target, timeout_s=settings.timeout_s)
                        solver_return_s = time.perf_counter()-before
                        best = result.best
                        cost = problem.cost(problem.selection(best.bits)) if best else None
                        row.update({"status": result.status, "execution_source": result.execution_source,
                                    "bits": best.bits if best else None,
                                    "relative_gap": (cost-optimum)/max(1, abs(optimum)) if cost is not None else None,
                                    "time_to_target_s": result.target_hit_s,
                                    "wall_s": solver_return_s, "solver_timings": result.timings,
                                    "resources": result.resources, "trace": result.trace})
                        if transfer and method.startswith("qaoa"):
                            row["inference_time_to_target_s"] = row["time_to_target_s"]
                            row["amortized_training_s"] = training_per_request
                            if row["time_to_target_s"] is not None:
                                row["time_to_target_s"] += training_per_request
                        if geometry and best:
                            geom_start = time.perf_counter()
                            model = decode_geometry(problem, best.bits)
                            decoded = time.perf_counter()
                            valid = validate_geometry(model)
                            row["geometry"] = {"valid": valid.valid, "validation": valid.model_dump(),
                                               "geometry_s": decoded-geom_start,
                                               "validation_s": time.perf_counter()-decoded}
                            row["generation_to_validated_geometry_s"] = time.perf_counter()-before
                            row["end_to_end_note"] = "Actual decision, decoding and B-rep validation. Artifact export and process startup measured separately by generate."
                    except Exception as error:  # noqa: BLE001 -- failed runs must remain in the corpus.
                        row.update({"status": "error", "error": f"{type(error).__name__}: {error}",
                                    "time_to_target_s": None, "wall_s": time.perf_counter()-before})
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
                    stream.flush()
                    records.append(row)
            print(f"bench {instance['id']} {problem.spec.family} {problem.encode().n} original QUBO variables complete", flush=True)
    summary = summarize(records)
    qrows = [r for r in records if r["method"].startswith("qaoa")]
    summary["quantum_geometry_valid_rate_all_runs"] = (sum(r.get("geometry", {}).get("valid", False)
        for r in qrows)/len(qrows)) if geometry and qrows else None
    summary["elapsed_s"] = time.perf_counter()-start
    summary["record_count"] = len(records)
    summary["geometry_validation_is_partial"] = not geometry
    dump(directory/"summary.json", summary)
    return summary


def train_transfer(output: str | Path, corpus_path: str | Path, depth: int = 1) -> dict:
    """Train context-specific angles on train ONLY; this is not a learned 3D foundation model."""
    start = time.perf_counter()
    output, cp = Path(output), Path(corpus_path)
    corpus = json.loads(cp.read_text()) if cp.exists() else create_corpus(cp)
    contexts, evidence = {}, []
    settings = PlannerSettings(encoding="compact_binary", depth=depth, iterations=16,
                               training_shots=128, shots=256)
    backend = make_backend("aer")
    for instance in corpus["instances"]:
        if instance["split"] != "train":
            continue
        p = build_problem(DesignSpec.model_validate(instance["spec"]))
        context = f"{p.spec.family}:{len(p.groups)}"
        if context in contexts:
            continue
        result = QuantumStructurePrior(backend).plan(p, settings, p.spec.seed)
        contexts[context] = {"parameters": result.parameters, "trained_on": instance["id"]}
        evidence.append({"instance": instance["id"], "result": result.as_dict()})
    transfer = {"depth": depth, "encoding": "compact_binary", "contexts": contexts,
                "corpus_sha256": hashlib.sha256(cp.read_bytes()).hexdigest(),
                "training_wall_s": time.perf_counter()-start,
                "description": "First training instance per family/size context; heldout data never tunes angles."}
    dump(output, transfer)
    dump(output.with_suffix(".training.json"), evidence)
    return transfer
