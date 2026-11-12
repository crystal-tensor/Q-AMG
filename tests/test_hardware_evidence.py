import json
from zipfile import ZipFile

import pytest

from qamg.component_manifest import write_component_manifest
from qamg.engineering_references import reference_bundle
from qamg.hardware_evidence import assess_hardware_evidence
from qamg.superconducting import superconducting_computer


def test_reference_links_and_self_declared_flags_cannot_certify_real_hardware():
    parts = [{"id": f"p-{i}", "hardware_evidence": {
        "instance_id": f"instance-{i}", "component_type": "amplifier",
        "source_refs": ["ibm-cryostat"], "source_locator": "cryostat architecture",
        "as_built_verified": True, "missing_evidence": []}} for i in range(500)]
    result = assess_hardware_evidence({"parts": parts})
    assert result["minimum_model_count_met"]
    assert result["source_linked_reference_instances"] == 500
    assert result["verified_real_part_instances"] == 0
    assert result["missing_evidence_counts"]["physical_inspection"] == 500
    assert result["fully_real_500_requirement_met"] is False
    parts[-1]["hardware_evidence"]["instance_id"] = "instance-0"
    with pytest.raises(ValueError, match="unique"):
        assess_hardware_evidence({"parts": parts})


def test_superconducting_reference_preserves_continuous_cables_and_physical_part_classes():
    model = superconducting_computer()
    assert len(model.parts) >= 500
    assert len({p.evidence["instance_id"] for p in model.parts}) == len(model.parts)
    source_ids = {s["id"] for s in reference_bundle("superconducting-computer")["sources"]}
    for part in model.parts:
        assert part.solid.volume() > 0
        positive = [p for p in part.solid.decompose() if p.volume() > max(1e-8, part.solid.volume()*1e-6)]
        assert len(positive) == 1, part.label
        assert part.group.count(" / ") == 1
        assert set(part.evidence["source_refs"]) <= source_ids
        assert part.evidence["as_built_verified"] is False
        assert len(part.evidence["missing_evidence"]) == 4
    kinds = [p.evidence["component_type"] for p in model.parts]
    # Five cold stages are not five artificial cuts of every continuous cable.
    assert kinds.count("rf_cable")+kinds.count("superconducting_coax") == 166
    assert kinds.count("hemt_amplifier") == 4
    # Two purchased four-channel arrays are two modules, not eight faux parts.
    assert kinds.count("isolator_array") == 2
    assert kinds.count("quantum_processor") == 1


def test_exported_bom_and_zip_keep_unverified_hardware_scope_after_edits(tmp_path):
    mesh = tmp_path/"part.ply"
    mesh.write_bytes(b"test mesh placeholder")
    structure = {"parts": [{"id": "part-000", "label": "用户合并组合", "volume_mm3": 1,
                             "mesh_file": mesh.name, "component_basis": "geometry_edit"}]}
    references = reference_bundle("superconducting-computer")
    write_component_manifest(tmp_path, structure, references)
    report = json.loads((tmp_path/"hardware-evidence.json").read_text())
    assert report["modeled_part_instances"] == 1
    assert report["source_linked_reference_instances"] == 0
    assert report["verified_real_part_instances"] == 0
    assert report["minimum_model_count_met"] is False
    with ZipFile(tmp_path/"parts.zip") as archive:
        bom = json.loads(archive.read("bom.json"))
        assert bom["hardware_claim"] == report
        assert bom["components"][0]["as_built_verified"] is False
        assert json.loads(archive.read("hardware-evidence.json")) == report
        assert json.loads(archive.read("structure.json"))["hardware_claim"] == report
