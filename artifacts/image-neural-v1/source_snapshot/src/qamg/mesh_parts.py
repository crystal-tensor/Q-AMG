"""Quantum-selected geometric part hypotheses and reversible exact mesh booleans.

Image-derived partitions are visual hypotheses, not inferred manufacturing joints.
Manifold closes cut surfaces; merging must preserve the source solid's volume.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations

import manifold3d as mf
import numpy as np
import trimesh

from .problem import StructureProblem
from .schema import ChoiceGroup, ImagePlanningContext, Option, PairRule


@dataclass(frozen=True)
class CutPlane:
    axis: int
    offset_mm: float
    score: int
    label: str


def as_manifold(mesh: trimesh.Trimesh) -> mf.Manifold:
    colors = np.asarray(mesh.visual.vertex_colors[:, :3], dtype=np.float32)/255
    properties = np.ascontiguousarray(np.column_stack([mesh.vertices, colors]), dtype=np.float32)
    solid = mf.Manifold(mf.Mesh(vert_properties=properties,
                                tri_verts=np.ascontiguousarray(mesh.faces, dtype=np.uint32)))
    if solid.status() != mf.Error.NoError or solid.is_empty() or solid.volume() <= 0:
        raise ValueError(f"生成网格未形成有效封闭实体：{solid.status()}")
    return solid


def as_trimesh(solid: mf.Manifold) -> trimesh.Trimesh:
    value = solid.to_mesh()
    properties = np.asarray(value.vert_properties)
    colors = np.clip(properties[:, 3:6], 0, 1) if properties.shape[1] >= 6 else None
    return trimesh.Trimesh(vertices=properties[:, :3], faces=value.tri_verts,
                           vertex_colors=colors, process=False)


def propose_planes(mesh: trimesh.Trimesh, maximum: int = 9) -> list[CutPlane]:
    """Deterministic local neck/color evidence; never solve the subset objective."""
    proposals = []
    bounds, extents = mesh.bounds, mesh.extents
    centers, area = mesh.triangles_center, mesh.area_faces
    rgb = np.asarray(mesh.visual.vertex_colors[:, :3], dtype=float)/255
    for axis in np.argsort(extents)[::-1]:
        axis = int(axis)
        if extents[axis] < .12*max(extents):
            continue
        edges = np.linspace(bounds[0, axis], bounds[1, axis], 49)
        histogram, _ = np.histogram(centers[:, axis], bins=edges, weights=area)
        smoothed = np.convolve(histogram, [1/3, 1/3, 1/3], mode="same")
        options = []
        for index in range(9, 39):
            offset = float((edges[index]+edges[index+1])/2)
            fraction = (offset-bounds[0, axis])/extents[axis]
            left = mesh.vertices[:, axis] < offset
            contrast = float(np.linalg.norm(rgb[left].mean(axis=0)-rgb[~left].mean(axis=0)))
            neck = smoothed[index]/max(float(smoothed.max()), 1e-8)
            score = round(20*neck+6*abs(fraction-.5)-12*contrast)
            options.append(CutPlane(axis, offset, score, f"{'XYZ'[axis]} = {offset:.1f} mm"))
        selected = []
        for candidate in sorted(options, key=lambda p: (p.score, p.offset_mm)):
            if all(abs(candidate.offset_mm-p.offset_mm) >= .16*extents[axis] for p in selected):
                selected.append(candidate)
            if len(selected) == 3:
                break
        proposals.extend(selected)
    if not proposals:
        raise ValueError("模型没有足够空间生成可用的分件候选")
    return proposals[:maximum]


def partition_problem(planes: list[CutPlane], image_sha: str, desired_parts: int,
                      seed: int) -> StructureProblem:
    context = ImagePlanningContext(seed=seed, image_sha256=image_sha, desired_parts=desired_parts)
    target_cuts = min(len(planes), max(1, round(np.log2(desired_parts))))
    weight = 40
    groups = [ChoiceGroup(key=f"cut_{i}", label=plane.label, options=[
        Option(key="keep", label="保持连接", cost=0, attributes={"cut": False}),
        Option(key="separate", label=f"沿 {plane.label} 分件", cost=plane.score+weight*(1-2*target_cuts),
               attributes={"cut": True, **asdict(plane)})]) for i, plane in enumerate(planes)]
    pairs = [PairRule(left=(i, 1), right=(j, 1), cost=2*weight,
                       reason="soft_cut_count_target") for i, j in combinations(range(len(planes)), 2)]
    return StructureProblem(context, groups, pairs)


def cut_parts(source: mf.Manifold, planes: list[CutPlane], selection: tuple[int, ...],
              max_parts: int = 24) -> list[mf.Manifold]:
    if not any(selection):
        raise ValueError("该测量没有选择分件平面")
    pieces = [source]
    tolerance = max(1e-8, source.volume()*1e-10)
    for plane, active in zip(planes, selection, strict=True):
        if not active:
            continue
        normal = [0., 0., 0.]
        normal[plane.axis] = 1.
        updated = []
        for piece in pieces:
            positive, negative = piece.split_by_plane(normal, plane.offset_mm)
            for half in [positive, negative]:
                if not half.is_empty() and half.volume() > tolerance:
                    updated.extend(part for part in half.decompose() if part.volume() > tolerance)
        pieces = updated
        if len(pieces) > max_parts:
            raise ValueError("该测量会产生过多碎片，请使用另一组测量候选")
    if len(pieces) < 2:
        raise ValueError("该测量未形成至少两个独立分件")
    if abs(sum(p.volume() for p in pieces)-source.volume())/source.volume() > 1e-5:
        raise ArithmeticError("分件体积未守恒，拒绝导出该候选")
    return sorted(pieces, key=lambda p: tuple(np.round(as_trimesh(p).centroid, 5)))


def merge_solids(parts: list[mf.Manifold]) -> mf.Manifold:
    if not parts:
        raise ValueError("没有可合并的分件")
    merged = mf.Manifold.batch_boolean(parts, mf.OpType.Add)
    if merged.status() != mf.Error.NoError or merged.is_empty():
        raise ValueError("网格布尔合并失败")
    return merged


def validate_partition(source: mf.Manifold, parts: list[mf.Manifold]) -> dict:
    union = merge_solids(parts)
    relative_volume_error = abs(union.volume()-source.volume())/source.volume()
    # Volume alone cannot prove equality: independently measure symmetric difference.
    missing = (source-union).volume()
    extra = (union-source).volume()
    symmetric_difference = max(0., missing)+max(0., extra)
    valid = (all(p.status() == mf.Error.NoError and p.volume() > 0 for p in parts)
             and relative_volume_error < 1e-5 and symmetric_difference/source.volume() < 1e-5)
    return {"valid": bool(valid), "issues": [], "checks": {
        "manifold_solid_status": True, "split_volume_conservation": relative_volume_error < 1e-5,
        "union_symmetric_difference": symmetric_difference/source.volume() < 1e-5,
        "hidden_geometry_recovered": False, "manufacturing_joints_verified": False,
        "load_or_manufacturing_validated": False},
        "source_volume_mm3": source.volume(), "merged_volume_mm3": union.volume(),
        "relative_volume_error": relative_volume_error, "symmetric_difference_mm3": symmetric_difference}


def sampled_disassembly(parts: list[mf.Manifold], ids: list[str], samples: int = 16) -> dict:
    """Fixed orientation, sampled translations of generated parts. No path proof."""
    meshes = [as_trimesh(p) for p in parts]
    bounds = np.array([m.bounds for m in meshes])
    diagonal = float(np.linalg.norm(bounds[:, 1].max(axis=0)-bounds[:, 0].min(axis=0)))
    remaining = set(range(len(parts)))
    moves, checks = [], 0
    tolerance = max(1e-7, sum(p.volume() for p in parts)*1e-9)
    while remaining:
        found = False
        for index in sorted(remaining, key=lambda i: -float(np.linalg.norm(meshes[i].centroid))):
            for axis in range(3):
                for sign in [1., -1.]:
                    delta = np.zeros(3)
                    delta[axis] = sign*diagonal*1.4
                    clear = True
                    for fraction in np.linspace(1/samples, 1., samples):
                        shifted_bounds = bounds[index]+delta*fraction
                        for other in remaining-{index}:
                            overlap = np.minimum(shifted_bounds[1], bounds[other, 1])-np.maximum(shifted_bounds[0], bounds[other, 0])
                            if np.any(overlap <= 1e-7):
                                continue
                            checks += 1
                            if (parts[index].translate(delta*fraction) ^ parts[other]).volume() > tolerance:
                                clear = False
                                break
                        if not clear:
                            break
                    if clear:
                        moves.append({"part": ids[index], "translation_mm": delta.tolist(), "samples": samples})
                        remaining.remove(index)
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if not found:
            break
    return {"moves": moves, "complete_for_generated_mesh": not remaining,
            "complete_for_envelope_model": not remaining, "remaining_parts": [ids[i] for i in sorted(remaining)],
            "sampled_intersection_checks": checks, "samples_per_translation": samples,
            "continuous_collision_proof": False, "physical_assembly_verified": False,
            "scope": "Fixed-orientation sampled translations of generated mesh partitions; no tool, joint or continuous-path certification"}
