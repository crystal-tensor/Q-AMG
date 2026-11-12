"""Keep rendered-part counts separate from physical/as-built verification.

A source link, a valid mesh, a catalogue number or a self-declared boolean cannot
certify that a component is present, dimensionally correct and correctly placed
in an actual machine. This viewer currently has no such inspection record.
"""
from __future__ import annotations

from collections import Counter


def assess_hardware_evidence(structure: dict, minimum: int = 500) -> dict:
    parts = structure["parts"]
    types = Counter()
    gaps = Counter()
    ids = []
    documented = 0
    for part in parts:
        evidence = part.get("hardware_evidence", {})
        ids.append(evidence.get("instance_id", part["id"]))
        types[evidence.get("component_type", "unclassified")] += 1
        documented += bool(evidence.get("source_refs") and evidence.get("source_locator"))
        required_gaps = {"original_as_built_bom", "dimensioned_cad", "installed_pose", "physical_inspection"}
        for gap in sorted(required_gaps | set(evidence.get("missing_evidence", []))):
            gaps[gap] += 1
    if len(ids) != len(set(ids)):
        raise ValueError("Hardware instance IDs must be unique")
    return {
        "status": "reference_design_not_as_built_verified",
        "target_verified_real_parts": minimum,
        "modeled_part_instances": len(parts),
        "source_linked_reference_instances": documented,
        "verified_real_part_instances": 0,
        "minimum_model_count_met": len(parts) >= minimum,
        "fully_real_500_requirement_met": False,
        "component_types": dict(types), "missing_evidence_counts": dict(gaps),
        "brand_scope": "相干一号：费曼为用户指定展示名称；非 Google / IBM 产品名称",
        "scope": "照片与器件类别参考；网格数量、引用或软件布尔检查均不证明500件实物完全真实" if structure.get("references", {}).get("photo_guided") else "公开架构与器件类别参考；网格数量、引用或软件布尔检查均不证明500件实物完全真实",
        "required_to_verify": ["同一台设备的原厂或经授权 BOM 及版本", "逐件料号与数量对应", "有尺寸的 CAD 或实测记录",
                               "零件在整机中的装配坐标与连接关系", "与这些文件绑定的实机逐件核对记录"],
    }
