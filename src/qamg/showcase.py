"""Source-backed reference assemblies, separate from photo inference and OEM CAD.

Shapes are authored approximations. References support the component taxonomy,
not measured mating surfaces, a complete factory BOM or certified service paths.
QB measurements choose display layout parameters, not the component geometry.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

from .assembly_plan import selected_disassembly
from .backends import make_backend
from .benchmark import dump
from .component_manifest import write_component_manifest
from .gallery import CATALOGUE
from .image_pipeline import export_parts
from .mesh_parts import as_manifold, as_trimesh, merge_solids, validate_partition
from .problem import StructureProblem
from .quantum import QuantumStructurePrior
from .schema import ChoiceGroup, Option, PairRule, PlannerSettings, ShowcaseContext

WHITE = [226, 230, 229]
SILVER = [151, 167, 180]
DARK = [37, 47, 58]
BLUE = [60, 108, 143]
TEAL = [46, 135, 124]
GOLD = [201, 164, 84]
ORANGE = [214, 112, 53]
RED = [179, 65, 66]


def box(size, center=(0., 0., 0.)):
    return as_manifold(trimesh.creation.box(size)).translate(center)


def orient(solid, axis):
    matrix = trimesh.geometry.align_vectors([0, 0, 1], axis)
    return solid.transform(matrix[:3].tolist())


def turn(solid, degrees, axis=(0, 0, 1)):
    matrix = trimesh.transformations.rotation_matrix(math.radians(degrees), axis)
    return solid.transform(matrix[:3].tolist())


def cyl(radius, height, center=(0., 0., 0.), axis=(0, 0, 1), sections=64):
    solid = as_manifold(trimesh.creation.cylinder(radius=radius, height=height, sections=sections))
    return orient(solid, axis).translate(center)


def tube(radius, inner, height, center=(0., 0., 0.), axis=(0, 0, 1)):
    return cyl(radius, height, center, axis)-cyl(inner, height+.1, center, axis)


def ellipsoid(size, center=(0., 0., 0.)):
    mesh = trimesh.creation.icosphere(subdivisions=3)
    mesh.apply_scale(np.array(size)/2)
    return as_manifold(mesh).translate(center)


def beam(start, end, radius):
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    return cyl(radius, float(np.linalg.norm(end-start)), (start+end)/2, end-start)


def union(*solids):
    return merge_solids(list(solids))


def prism(points, height, z=0):
    points = np.asarray(points, dtype=float)
    n = len(points)
    vertices = np.concatenate([np.column_stack([points, np.full(n, z-height/2)]),
                               np.column_stack([points, np.full(n, z+height/2)]),
                               [[*points.mean(axis=0), z-height/2], [*points.mean(axis=0), z+height/2]]])
    faces = []
    for i in range(n):
        j = (i+1) % n
        faces += [[i, j, n+j], [i, n+j, n+i], [2*n, j, i], [2*n+1, n+i, n+j]]
    mesh = trimesh.Trimesh(vertices, faces, process=True)
    mesh.fix_normals()
    return as_manifold(mesh)


def rounded_rect(width, length, radius, height, z=0):
    radius = min(radius, width/2, length/2)
    points = []
    for x, y, angle in [(width/2-radius, length/2-radius, 0),
                         (-width/2+radius, length/2-radius, 90),
                         (-width/2+radius, -length/2+radius, 180),
                         (width/2-radius, -length/2+radius, 270)]:
        for a in np.linspace(angle, angle+90, 9):
            points.append([x+radius*np.cos(np.deg2rad(a)), y+radius*np.sin(np.deg2rad(a))])
    return prism(points, height, z)


def loft_x(stations):
    """Closed elliptical stations (x, lateral radius, vertical radius, z)."""
    count = 64
    angles = np.arange(count)*2*np.pi/count
    rings = [np.column_stack([np.full(count, x), ry*np.cos(angles), z+rz*np.sin(angles)])
             for x, ry, rz, z in stations]
    vertices = np.concatenate([*rings, [[stations[0][0], 0, stations[0][3]],
                                        [stations[-1][0], 0, stations[-1][3]]]])
    faces = []
    for row in range(len(stations)-1):
        for i in range(count):
            a, b = row*count+i, row*count+(i+1) % count
            c, d = b+count, a+count
            faces += [[a, b, c], [a, c, d]]
    for i in range(count):
        j = (i+1) % count
        faces += [[len(vertices)-2, j, i], [len(vertices)-1, (len(stations)-1)*count+i,
                                            (len(stations)-1)*count+j]]
    mesh = trimesh.Trimesh(vertices, faces, process=True)
    mesh.fix_normals()
    return as_manifold(mesh)


def fan(radius, blades=18):
    pieces = [cyl(radius*.2, 4)]
    blade = prism([[radius*.1, -1], [radius*.75, -radius*.13], [radius, -1],
                   [radius*.8, radius*.10], [radius*.1, 1]], 1.4)
    blade = turn(blade, 18, [1, 0, 0])
    for angle in np.linspace(0, 360, blades, endpoint=False):
        pieces.append(turn(blade, angle))
    return union(*pieces)


def gear(radius, teeth=20, height=2):
    angles = np.arange(teeth*4)*2*np.pi/(teeth*4)
    radii = np.tile([radius*.86, radius, radius, radius*.86], teeth)
    return prism(np.column_stack([radii*np.cos(angles), radii*np.sin(angles)]), height)


@dataclass
class DemoPart:
    label: str
    solid: object
    color: list[int]
    group: str
    explode: list[float]
    appearance: dict | None = None
    role: str = "component"
    basis: str = "reference_component_class"
    material: str = "工程示意材料"


class Model:
    def __init__(self):
        self.parts = []
        self.groups = {}

    def group(self, path, explode):
        self.groups[path] = list(explode)
        return path

    def add(self, label, solid, color, group, explode, appearance=None, **metadata):
        self.parts.append(DemoPart(label, solid, color, group, list(explode), appearance,
                                   **metadata))

    def part(self, label, solid, color, group, offset=(0, 0, 0), **metadata):
        delta = np.asarray(self.groups[group])+np.asarray(offset)
        self.add(label, solid, color, group, delta, **metadata)


def generate_showcase(slug: str, output: Path) -> dict:
    started = time.perf_counter()
    entry = next(e for e in CATALOGUE if e["slug"] == slug)
    from .engineering_models import BUILDERS
    from .engineering_references import PROFILES, reference_bundle

    profile = PROFILES[slug]
    model = BUILDERS[slug]()
    parts = []
    for item in model.parts:
        rgb = np.asarray(item.color)/255
        # Remove sub-micron boolean slivers before float32 PLY/GLB export. This
        # is bounded simplification of authored geometry, never neural re-cutting.
        solid = item.solid.simplify(1e-4)
        for _ in range(3):
            probe = as_trimesh(solid)
            probe.process()
            if probe.is_watertight:
                break
            solid = as_manifold(probe).simplify(1e-4)
        else:
            raise ValueError(f"Float32 export is not watertight: {slug} / {item.label}")
        parts.append(solid.set_properties(3, lambda position, properties, color=rgb: color))
    context = ShowcaseContext(example_id=slug, title=entry["title"], seed=42)
    groups = []
    for key, label, low, high in [("lateral", "横向分离", .8, 1.15), ("vertical", "竖向分离", .85, 1.2),
                                   ("longitudinal", "纵向间距", .9, 1.25)]:
        groups.append(ChoiceGroup(key=key, label=label, options=[
            Option(key="compact", label="紧凑展开", cost=1, attributes={"factor": low}),
            Option(key="spread", label="宽松展开", cost=0, attributes={"factor": high})]))
    problem = StructureProblem(context, groups, [PairRule(left=(0, 1), right=(1, 1), cost=3,
                                  reason="illustrative_layout_spread_penalty_not_measured_physics")])
    settings = PlannerSettings(backend="qb", encoding="compact_binary", iterations=4, depth=1, shots=512)
    plan = QuantumStructurePrior(make_backend("qb", settings.max_qubits)).plan(problem, settings, 42)
    candidate = plan.best
    if candidate is None:
        raise ValueError("No observed layout candidate; no classical substitution")
    factors = [g.options[o].attributes["factor"] for g, o in zip(groups, problem.selection(candidate.bits), strict=True)]
    scaling = [factors[2], factors[0], factors[1]]
    vectors = [(np.asarray(p.explode)*scaling).tolist() for p in model.parts]
    ids = [f"part-{i:03d}" for i in range(len(parts))]
    anchor = int(np.argmax([p.volume() for p in parts]))
    sequence = selected_disassembly(parts, ids, vectors, anchor, samples=12)
    validation = validate_partition(merge_solids(parts), parts)
    validation["checks"].update({"authored_conceptual_example": True, "photo_reconstruction_performed": False,
                                  "official_product_cad": False, "manufacturing_joints_verified": False})
    validation["assembly_overlap_fraction"] = max(0., sum(p.volume() for p in parts)/validation["merged_volume_mm3"]-1)
    provenance = {"mode": "qb", "execution_source": plan.execution_source, "quantum_sdk": "QuantumBridge",
                  "quantum_role": "illustrative_layout_parameter_selection", "geometry_source": "authored_procedural_concept",
                  "geometry_is_inferred": False, "photo_reconstruction_performed": False,
                  "official_product_cad": False, "generation_rounds": 1, "fallback_used": False,
                  "component_taxonomy_source": "public_technical_references_with_declared_inferences",
                  "export_surface_simplification_tolerance_mm": 1e-4,
                  "hardware_verified": False, "quantum_speedup_verified": False,
                  "selected_measurement": {"bits": candidate.bits, "job_id": candidate.first_job_id,
                                           "observed_count": candidate.count, "count_scope": "all_batches_in_this_plan",
                                           "kind": "direct_or_exactly_decoded"}}
    output.mkdir(parents=True, exist_ok=True)
    structure = export_parts(parts, output, provenance, sequence, validation, [p.label for p in model.parts],
                              explosion_vectors=vectors, appearance=[p.appearance or {} for p in model.parts])
    names = list(model.groups)
    group_ids = {name: f"demo-group-{i}" for i, name in enumerate(names)}
    structure["groups"] = []
    for name in names:
        parent = name.rpartition(" / ")[0]
        structure["groups"].append({"id": group_ids[name], "parent": group_ids.get(parent, "root"),
                                     "label": name.split(" / ")[-1], "assembly_path": name,
                                     "depth": name.count(" / ")+1,
                                     "explode_mm": (np.asarray(model.groups[name])*scaling).tolist()})
    for record, item in zip(structure["parts"], model.parts, strict=True):
        top = item.group.split(" / ")[0]
        record.update({"parent": group_ids[item.group], "assembly_path": item.group,
                       "role": item.role, "material": item.material,
                       "component_basis": profile.get("internal_basis", item.basis),
                       "source_refs": profile["sources"], "geometry_accuracy": "parametric_approximation",
                       "is_cover": any(word in item.label for word in ["外壳", "蒙皮", "筒体", "整流罩", "护壳", "前罩", "后壳", "后盖", "底盖", "表盘", "显示屏", "玻璃车顶", "风挡", "船壳", "甲板板", "端盖", "机匣", "表镜"]),
                       "explode_levels_mm": [(np.asarray(model.groups[top])*scaling).tolist(),
                                             (np.asarray(model.groups[item.group])*scaling).tolist(),
                                             record["explode_mm"]]})
    references = reference_bundle(slug)
    if slug == "model-y":
        source = Path(__file__).resolve().parents[2]/"assets/references/model-y"
        photo = json.loads((source/"attribution.json").read_text())
        photo_file = source/photo["filename"]
        if hashlib.sha256(photo_file.read_bytes()).hexdigest() != photo["sha256"]:
            raise ValueError("Model Y photo does not match its attribution manifest")
        shutil.copy2(photo_file, output/photo["filename"])
        dump(output/"photo-attribution.json", photo)
        references["photo"] = photo
    counts = {"assemblies": sum(" / " not in name for name in names),
              "subassemblies": sum(" / " in name for name in names), "components": len(parts),
              "fasteners": sum(item.role == "fastener" for item in model.parts)}
    structure["decomposition"] = {"revision": 2, "levels": ["总成", "子总成", "零件"],
                                   "counts": counts, "complete_factory_bom": False}
    structure["references"] = references
    structure["engineering_scope"] = profile["coverage"]+profile["inferred"]
    dump(output/"structure.json", structure)
    write_component_manifest(output, structure, references)
    dump(output/"trace.json", [{"round": 0, "problem": problem.as_dict(), "plan": plan.as_dict()}])
    dump(output/"problem.json", problem.as_dict())
    if plan.circuit_qasm3:
        (output/"layout.qasm").write_text(plan.circuit_qasm3)
    description = {**entry, "kind": "procedural_demo", "detail_revision": 2,
                   "geometry_source": "Q-AMG independently authored reference component geometry",
                   "variant": profile["variant"], "coverage": profile["coverage"],
                   "reference_scope": "公开技术资料支持部件类别；形状、数量和装配位置仍以各例边界为准",
                   "references": references, "decomposition": structure["decomposition"],
                   "photo": references.get("photo"),
                   "limitations": [profile["inferred"], "未提供完整官方 CAD / 工厂 BOM", "碰撞采样不认证连续拆解路径"]}
    dump(output/"demo.json", description)
    summary = {"status": "generated", "kind": "procedural_demo", "spec": context.model_dump(),
               "settings": {"planner": settings.model_dump()}, "title": entry["title"],
               "description": entry["description"], "part_count": len(parts), "selected_bits": candidate.bits,
               "decisions": problem.describe(candidate.bits), "resources": plan.resources,
               "provenance": provenance, "validation": validation, "demo": description,
               "decomposition": structure["decomposition"],
               "timings": {"end_to_end_s": time.perf_counter()-started},
               "disassembly": {"complete_for_generated_mesh": sequence["complete_for_generated_mesh"],
                               "moves": len(sequence["moves"]), "continuous_collision_proof": False}}
    dump(output/"run.json", summary)
    return summary
