"""Full artifact-level acceptance, including failed jobs and STEP/GLB round trips."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

START = time.perf_counter()

import cadquery as cq
import numpy as np
import trimesh

from qamg.benchmark import dump, snapshot_sources
from qamg.pipeline import generate
from qamg.schema import DesignSpec, PlannerSettings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--instances", type=int, default=30)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--quantum-backend", choices=["auto", "qb", "torch-gpu"], default="auto")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Choose a new artifact validation directory")
    args.output.mkdir(parents=True)
    cases = [i for i in json.loads(args.corpus.read_text())["instances"] if i["split"] == "heldout"][:args.instances]
    dump(args.output/"manifest.json", {"corpus": str(args.corpus), "instances": len(cases),
                                       "seeds": args.seeds, "source_sha256": snapshot_sources(args.output),
                                       "settings": PlannerSettings(encoding="compact_binary", iterations=12).model_dump(),
                                       "import_and_driver_setup_s": time.perf_counter()-START,
                                       "mode_order": "seeded random per case/repetition"})
    records = []
    with (args.output/"raw.jsonl").open("w") as stream:
        for case in cases:
            for seed in range(args.seeds):
                for mode in np.random.default_rng(seed).permutation([args.quantum_backend, "classical"]):
                    label = f"{case['id']}-seed{seed}-{mode}"
                    output = args.output/label
                    spec = DesignSpec.model_validate({**case["spec"], "seed": 1000+seed})
                    row = {"case": case["id"], "seed": seed, "mode": str(mode), "output": str(output)}
                    before = time.perf_counter()
                    try:
                        r = generate(spec, PlannerSettings(backend=str(mode), encoding="compact_binary",
                                                           iterations=12), output)
                        solids = len(cq.importers.importStep(str(output/"assembly.step")).solids().vals())
                        meshes = len(trimesh.load(output/"assembly.glb", force="scene").geometry)
                        row.update({"valid": r["validation"]["valid"] and solids == meshes == r["part_count"],
                                    "part_count": r["part_count"], "step_solids": solids, "glb_meshes": meshes,
                                    "pipeline_timings": r["timings"], "disassembly": r["disassembly"],
                                    "selected_bits": r["selected_bits"], "provenance": r["provenance"]})
                        row["resources"] = r["resources"]
                    except Exception as error:  # noqa: BLE001 -- acceptance failures remain in the denominator.
                        row.update({"valid": False, "error": f"{type(error).__name__}: {error}"})
                    row["total_with_roundtrip_s"] = time.perf_counter()-before
                    records.append(row)
                    stream.write(json.dumps(row, ensure_ascii=False)+"\n")
                    stream.flush()
            print(f"artifact acceptance {case['id']} complete", flush=True)
    summary = {}
    for mode in [args.quantum_backend, "classical"]:
        rows = [r for r in records if r["mode"] == mode]
        successful = [r for r in rows if r["valid"]]
        summary[mode] = {"runs": len(rows), "valid": len(successful),
                         "valid_rate_all_runs": len(successful)/max(1, len(rows)),
                         "mean_end_to_end_pipeline_s": float(np.mean([r["pipeline_timings"]["end_to_end_s"] for r in successful])) if successful else None,
                         "mean_roundtrip_inclusive_s": float(np.mean([r["total_with_roundtrip_s"] for r in successful])) if successful else None}
    summary["real_quantum_speedup_verified"] = False
    summary["elapsed_with_imports_s"] = time.perf_counter()-START
    dump(args.output/"summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
