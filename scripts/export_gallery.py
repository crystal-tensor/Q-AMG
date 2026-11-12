"""Package only curated, generated examples for the public static viewer."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from qamg.gallery import public_catalogue

ROOT = Path(__file__).resolve().parents[1]


def main():
    target = ROOT/"web/public/gallery"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    examples, excluded = [], []
    for entry in public_catalogue():
        name = entry["slug"]
        source = ROOT/"artifacts/examples"/name
        if not (source/"run.json").is_file():
            continue
        summary = json.loads((source/"run.json").read_text())
        if summary.get("status") != "generated":
            continue
        if summary["spec"]["family"] == "image_parts":
            review_path = source/"quality-review.json"
            review = json.loads(review_path.read_text()) if review_path.exists() else {}
            image_hash = hashlib.sha256((source/"input.png").read_bytes()).hexdigest()
            mesh_hash = hashlib.sha256((source/"assembly.glb").read_bytes()).hexdigest()
            if not review.get("accepted"):
                excluded.append({"id": f"example-{name}", "reason": review.get("reason", "尚未完成逐件核对")})
                continue
            if not (review.get("input_png_sha256") == image_hash and review.get("assembly_glb_sha256") == mesh_hash):
                raise ValueError(f"Public image example needs a review bound to these exact files: {name}")
        id = f"example-{name}"
        output = target/id
        output.mkdir(exist_ok=True)
        for file in ["run.json", "structure.json", "trace.json", "assembly.glb", "assembly.step", "assembly.obj",
                     "parts.zip", "input.png", "disassembly.json", "quality-review.json", "demo.json", "thumbnail.png",
                     "bom.json", "bom.csv", "references.json", "reference-photo.png", "photo-attribution.json", "hardware-evidence.json"]:
            if (source/file).is_file():
                shutil.copy2(source/file, output/file)
        examples.append({"id": id, "title": entry["title"], "category": entry["category"],
                         "description": entry["description"], "kind": summary.get("kind", "parametric_cad"),
                         "part_count": summary["part_count"], "decomposition": summary.get("decomposition"),
                         "hardware_claim": summary.get("hardware_claim"),
                         "photo": summary.get("demo", {}).get("photo"),
                         "thumbnail": "thumbnail.png" if (output/"thumbnail.png").is_file() else None})
    (target/"index.json").write_text(json.dumps({"project": "Q-AMG", "quantum_sdk": "QuantumBridge",
        "examples": examples, "excluded_examples": excluded, "generation_requires_backend": True,
        "hardware_verified": False, "quantum_speedup_verified": False}, indent=2))
    print(f"Exported {len(examples)} curated interactive examples; user uploads are excluded")
    for entry in excluded:
        print(f"Excluded {entry['id']}: {entry['reason']}")


if __name__ == "__main__":
    main()
