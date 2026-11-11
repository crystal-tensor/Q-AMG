"""Export the current named component inventory, including edited assemblies."""
from __future__ import annotations

import csv
from zipfile import ZIP_DEFLATED, ZipFile

from .benchmark import dump


def write_component_manifest(directory, structure, references):
    rows = [{"id": p["id"], "name": p["label"], "assembly": p.get("assembly_path", "编辑组合"),
             "role": p.get("role", "component"), "material": p.get("material", "未标注"),
             "basis": p.get("component_basis", "geometry_edit"), "quantity": 1,
             "volume_mm3": p["volume_mm3"], "source_refs": ";".join(p.get("source_refs", []))}
            for p in structure["parts"]]
    dump(directory/"bom.json", {"scope": "当前展示模型的逐件清单，非完整工厂 BOM", "complete_factory_bom": False,
                               "component_count": len(rows), "components": rows})
    with (directory/"bom.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    dump(directory/"references.json", references)
    # Rebuild the current inventory, without stale BOM or duplicate ZIP members.
    with ZipFile(directory/"parts.zip", "w", compression=ZIP_DEFLATED) as archive:
        for part in structure["parts"]:
            source = directory/part["mesh_file"]
            archive.write(source, source.name)
        for name in ["structure.json", "bom.json", "bom.csv", "references.json"]:
            archive.write(directory/name, name)
