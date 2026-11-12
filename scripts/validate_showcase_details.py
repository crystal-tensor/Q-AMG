"""Audit exported detailed examples and their public copies, rather than builders."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import trimesh

from qamg.benchmark import dump
from qamg.gallery import CATALOGUE
from qamg.mesh_parts import as_manifold


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root, public):
    records = []
    for entry in CATALOGUE:
        slug = entry["slug"]
        directory = root/slug
        run, structure = read(directory/"run.json"), read(directory/"structure.json")
        inventory, references = read(directory/"bom.json"), read(directory/"references.json")
        count = len(structure["parts"])
        assert count == run["part_count"] == inventory["component_count"]
        assert run["validation"]["valid"] and not references["oem_cad"]
        assert not references["complete_factory_bom"] and not references["dimensions_measured"]
        source_ids = {source["id"] for source in references["sources"]}
        source_ids.update(source["id"] for source in references.get("reference_images", []))
        groups = {g["id"]: g for g in structure["groups"]}
        geometry = trimesh.load(directory/"assembly.glb", force="scene")
        assert len(geometry.graph.nodes_geometry) == count
        witnesses = read(directory/"trace.json")[0]["plan"]["trace"]
        selected = run["provenance"]["selected_measurement"]
        observed = sum(batch["counts"].get(selected["bits"], 0) for batch in witnesses)
        assert observed == selected["observed_count"] > 0
        assert any(batch["job_id"] == selected["job_id"] and batch["counts"].get(selected["bits"], 0)
                   for batch in witnesses)
        assert run["provenance"]["quantum_sdk"] == "QuantumBridge"
        assert not run["provenance"]["quantum_speedup_verified"]
        hardware = structure.get("hardware_claim")
        if hardware:
            assert hardware == run["hardware_claim"] == inventory["hardware_claim"]
            assert hardware == read(directory/"hardware-evidence.json")
            assert count >= 500 and hardware["modeled_part_instances"] == count
            assert hardware["source_linked_reference_instances"] == count
            assert hardware["verified_real_part_instances"] == 0
            assert hardware["fully_real_500_requirement_met"] is False
            assert sha(directory/"hardware-evidence.json") == sha(public/f"example-{slug}"/"hardware-evidence.json")
        with ZipFile(directory/"parts.zip") as archive:
            assert len(archive.namelist()) == count+4+bool(hardware) == len(set(archive.namelist()))
            assert json.loads(archive.read("bom.json")) == inventory
            assert json.loads(archive.read("structure.json")) == structure
            for part, component in zip(structure["parts"], inventory["components"], strict=True):
                assert part["label"] == component["name"] and part["id"] == component["id"]
                assert set(part["source_refs"]) <= source_ids
                visited, parent = set(), part["parent"]
                while parent != "root":
                    assert parent in groups and parent not in visited
                    visited.add(parent)
                    parent = groups[parent]["parent"]
                assert len(visited) == 2
                assert np.shape(part["explode_levels_mm"]) == (3, 3)
                assert np.isfinite(part["explode_levels_mm"]).all()
                assert part["explode_levels_mm"][-1] == part["explode_mm"]
                mesh_path = directory/part["mesh_file"]
                assert archive.read(mesh_path.name) == mesh_path.read_bytes()
                mesh = trimesh.load(mesh_path, force="mesh")
                assert np.isfinite(mesh.vertices).all() and mesh.is_watertight, (slug, part["id"], part["label"])
                solid = as_manifold(mesh)
                assert abs(solid.volume()-part["volume_mm3"])/part["volume_mm3"] < .002, (
                    slug, part["label"], solid.volume(), part["volume_mm3"])
                substantial = [s for s in solid.decompose() if s.volume() > max(1e-8, solid.volume()*1e-6)]
                assert len(substantial) == 1, (slug, part["label"])
        for name in ["assembly.glb", "parts.zip", "structure.json", "bom.csv", "references.json", "thumbnail.png"]:
            assert sha(directory/name) == sha(public/f"example-{slug}"/name)
        if references.get("photo"):
            photo = references["photo"]
            assert photo["license"] == "CC BY-SA 4.0" and photo["author"]
            assert sha(directory/photo["filename"]) == photo["sha256"]
            assert sha(directory/photo["filename"]) == sha(public/f"example-{slug}"/photo["filename"])
            assert photo["photo_is_3d_reconstruction_input"] is False
        records.append({"id": slug, "parts": count, "assemblies": run["decomposition"]["counts"],
                        "closed_connected_component_instances": count,
                        "glb_sha256": sha(directory/"assembly.glb"), "public_files_match": True,
                        "observed_qb_count": observed, "references": sorted(source_ids),
                        "complete_factory_bom": False, "physical_disassembly_verified": False})
    return {"examples": records, "total_components": sum(r["parts"] for r in records),
            "scope": "Export and public-file integrity, not OEM equivalence or physical quantum speedup"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts/examples"))
    parser.add_argument("--public", type=Path, default=Path("web/public/gallery"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root, args.public)
    dump(args.output, result)
    print(f"Audited {len(result['examples'])} detailed examples / {result['total_components']} components")


if __name__ == "__main__":
    main()
