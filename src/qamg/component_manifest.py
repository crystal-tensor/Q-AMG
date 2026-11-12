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
    claim = None
    if references.get("hardware_truth_required"):
        from .hardware_evidence import assess_hardware_evidence

        claim = assess_hardware_evidence(structure)
        structure["hardware_claim"] = claim
        dump(directory/"structure.json", structure)
        dump(directory/"hardware-evidence.json", claim)
        for row, part in zip(rows, structure["parts"], strict=True):
            evidence = part.get("hardware_evidence", {})
            row.update({"instance_id": evidence.get("instance_id", part["id"]),
                        "component_type": evidence.get("component_type", "edited_composite"),
                        "reference_part_number": evidence.get("reference_part_number") or "未公开/待核",
                        "source_locator": evidence.get("source_locator", "用户几何编辑"),
                        "reference_image_id": evidence.get("reference_image_id", "无"),
                        "quantity_basis": evidence.get("quantity_basis", "geometry_edit"),
                        "dimensions_basis": evidence.get("dimensions_basis", "geometry_edit"),
                        "position_basis": evidence.get("position_basis", "geometry_edit"),
                        "as_built_verified": False,
                        "missing_evidence": ";".join(evidence.get("missing_evidence", claim["required_to_verify"]))})
    inventory = {"scope": "当前展示模型的逐件清单，非完整工厂 BOM", "complete_factory_bom": False,
                 "component_count": len(rows), "components": rows}
    if claim:
        inventory["hardware_claim"] = claim
    dump(directory/"bom.json", inventory)
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
        metadata = ["structure.json", "bom.json", "bom.csv", "references.json"]
        if claim:
            metadata.append("hardware-evidence.json")
        for name in metadata:
            archive.write(directory/name, name)
