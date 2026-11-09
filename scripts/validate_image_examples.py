"""Run real pinned TripoSR weights on two categories, then QB partition and merge.

This is representative evidence, not proof of accurate reconstruction for every image.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from qamg.benchmark import dump, snapshot_sources
from qamg.image_pipeline import generate_image, merge_run
from qamg.schema import Image3DSettings

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=128)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Choose a new evidence directory; failures must be retained")
    args.output.mkdir(parents=True)
    settings = Image3DSettings(resolution=args.resolution)
    inputs = {name: ROOT/"vendor/triposr/examples"/f"{name}.png" for name in ["chair", "teapot"]}
    dump(args.output/"manifest.json", {"settings": settings.model_dump(),
        "inputs": {name: {"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for name, path in inputs.items()},
        "source_sha256": snapshot_sources(args.output),
        "model_manifest": json.loads((ROOT/"models/manifest.json").read_text()),
        "vendor": json.loads((ROOT/"vendor/SOURCES.json").read_text()),
        "scope": "Two real neural image reconstructions, no model stub; neither universal accuracy nor hardware acceleration evidence"})
    rows = []
    for name, path in inputs.items():
        before = time.perf_counter()
        row = {"input": name, "success": False}
        try:
            result = generate_image(path.read_bytes(), settings, args.output/name, path.name,
                lambda message, name=name: print(f"{name}: {message}", flush=True))
            structure = json.loads((args.output/name/"structure.json").read_text())
            chosen = [p["id"] for p in structure["parts"][:2]]
            merged = merge_run(args.output/name, chosen, args.output/f"{name}-merged", name)
            row.update(success=True, part_count=result["part_count"], merged_part_count=merged["part_count"],
                       validation=result["validation"], merge_validation=merged["validation"],
                       model=result["model"], resources=result["resources"], disassembly=result["disassembly"],
                       selected_measurement=result["provenance"]["selected_measurement"],
                       original_restoration_available=merged["original_run_id"] == name)
        except Exception as error:  # noqa: BLE001 -- preserve actual failures, never substitute a test fixture.
            row["error"] = f"{type(error).__name__}: {error}"
            print(f"{name}: failed {row['error']}", flush=True)
        row["wall_s"] = time.perf_counter()-before
        rows.append(row)
        dump(args.output/"results.json", rows)
    print(json.dumps({"real_inputs": len(rows), "passed": sum(r["success"] for r in rows),
                      "quantum_sdk": "QuantumBridge", "hardware_verified": False}, indent=2))


if __name__ == "__main__":
    main()
