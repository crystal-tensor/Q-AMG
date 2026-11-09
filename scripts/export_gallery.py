"""Package only curated, generated examples for the public static viewer."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    target = ROOT/"web/public/gallery"
    target.mkdir(parents=True, exist_ok=True)
    examples = []
    for name in ["chair", "teapot", "drive", "clamp"]:
        source = ROOT/"artifacts/examples"/name
        if not (source/"run.json").is_file():
            continue
        summary = json.loads((source/"run.json").read_text())
        if summary.get("status") != "generated":
            continue
        id = f"example-{name}"
        output = target/id
        output.mkdir(exist_ok=True)
        for file in ["run.json", "structure.json", "trace.json", "assembly.glb", "assembly.step", "assembly.obj",
                     "parts.zip", "input.png", "disassembly.json"]:
            if (source/file).is_file():
                shutil.copy2(source/file, output/file)
        examples.append({"id": id, "kind": summary.get("kind", "parametric_cad"), "part_count": summary["part_count"]})
    (target/"index.json").write_text(json.dumps({"project": "Q-AMG", "quantum_sdk": "QuantumBridge",
        "examples": examples, "generation_requires_backend": True,
        "hardware_verified": False, "quantum_speedup_verified": False}, indent=2))
    print(f"Exported {len(examples)} curated interactive examples; user uploads are excluded")


if __name__ == "__main__":
    main()
