"""Executable B-rep decoder for two bounded assembly grammars; units are mm.

Gears are polygonal illustrative profiles (not validated involute gears), and
actuators are envelope models. Neither load ratings nor manufacturability are claimed.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import cadquery as cq
import numpy as np
import trimesh

from .problem import StructureProblem
from .schema import ValidationIssue, ValidationResult

STEEL = (0.57, 0.67, 0.76)
TEAL = (0.12, 0.67, 0.61)
ORANGE = (0.96, 0.53, 0.22)
BLUE = (0.22, 0.39, 0.65)
DARK = (0.25, 0.30, 0.37)


@dataclass
class Part:
    id: str
    label: str
    parent: str
    shape: cq.Shape
    color: tuple[float, float, float]
    explode: tuple[float, float, float]
    features: list[dict] = field(default_factory=list)

    def metadata(self) -> dict:
        b = self.shape.BoundingBox()
        return {"id": self.id, "label": self.label, "parent": self.parent, "color": self.color,
                "explode_mm": self.explode, "volume_mm3": self.shape.Volume(),
                "bounds_mm": [[b.xmin, b.ymin, b.zmin], [b.xmax, b.ymax, b.zmax]],
                "features": self.features}


@dataclass
class AssemblyModel:
    problem: StructureProblem
    bits: str
    parts: list[Part]
    groups: list[dict]
    dimensions: dict
    mates: list[dict]

    def structure(self) -> dict:
        return {"family": self.problem.spec.family, "units": "mm", "glb_units": "m",
                "selected_bits": self.bits, "bit_order": "qubit0_left",
                "decisions": self.problem.describe(self.bits),
                "spec": self.problem.spec.model_dump(), "groups": self.groups,
                "parts": [p.metadata() for p in self.parts], "dimensions": self.dimensions,
                "mates": self.mates,
                "modeling_order": [{"part": p.id, "features": p.features} for p in self.parts],
                "explosion": {"kind": "illustrative", "collision_checked": False},
                "limitations": ["Polygonal gear profiles are illustrative, not involute tooth design.",
                                "No force, fatigue, thread, bearing life or manufacturing validation.",
                                "STEP stores B-rep geometry; editable parameters live in design/structure JSON."]}


def box(x: float, y: float, z: float, center=(0., 0., 0.)) -> cq.Shape:
    return cq.Workplane("XY").box(x, y, z).val().translate(center)


def cylinder(radius: float, length: float, start=(0., 0., 0.), axis=(0., 0., 1.)) -> cq.Shape:
    return cq.Solid.makeCylinder(radius, length, cq.Vector(*start), cq.Vector(*axis))


def ring(outer: float, inner: float, thickness: float, z: float) -> cq.Shape:
    return cylinder(outer, thickness, (0, 0, z)).cut(cylinder(inner, thickness+2, (0, 0, z-1)))


def _drive(problem: StructureProblem, bits: str) -> AssemblyModel:
    decisions = problem.describe(bits)
    r = decisions["shaft"]["attributes"]["radius"]
    bore = decisions["bearing"]["attributes"]["bore"]
    gear_r = decisions["gear"]["attributes"]["radius"]
    teeth = decisions["gear"]["attributes"]["teeth"]
    split = decisions.get("housing", {}).get("attributes", {}).get("split", False)
    guard = decisions.get("guard", {}).get("attributes", {}).get("present", False)
    length = problem.spec.shaft_length_mm
    parts: list[Part] = []
    groups = [{"id": "rotor", "label": "转子组件", "parent": "root"},
              {"id": "front_support", "label": "前支承组件", "parent": "root"},
              {"id": "rear_support", "label": "后支承组件", "parent": "root"}]

    def add(id, label, parent, shape, color, explode, features):
        parts.append(Part(id, label, parent, shape, color, explode, features))

    add("base", "底座", "root", box(2*r+38, 4, length+12, (0, -r-22, length/2)), DARK,
        (0, -30, 0), [{"op": "box", "dimensions_mm": [2*r+38, 4, length+12]}])
    add("shaft", "主轴", "rotor", cylinder(r, length), STEEL, (0, 0, 70),
        [{"op": "circle_extrude", "radius_mm": r, "length_mm": length}])
    points = [(math.cos(i*math.pi/(2*teeth)) * (gear_r if i % 4 in [1, 2] else gear_r-2),
               math.sin(i*math.pi/(2*teeth)) * (gear_r if i % 4 in [1, 2] else gear_r-2))
              for i in range(4*teeth)]
    gear = cq.Workplane("XY").polyline(points).close().extrude(8).val()
    gear = gear.cut(cylinder(r+.3, 10, (0, 0, -1))).translate((0, 0, length/2-4))
    add("gear", "示意齿轮", "rotor", gear, ORANGE, (0, 0, 35),
        [{"op": "polygon_extrude", "teeth": teeth, "radius_mm": gear_r,
          "thickness_mm": 8}, {"op": "bore", "radius_mm": r+.3}])
    for prefix, z, sign in [("front", .2*length-4, -1), ("rear", .8*length-4, 1)]:
        outer = bore+5
        add(f"{prefix}_bearing", "前轴承" if prefix == "front" else "后轴承",
            f"{prefix}_support", ring(outer, bore, 8, z), TEAL, (0, 0, sign*42),
            [{"op": "annulus_extrude", "inner_radius_mm": bore,
              "outer_radius_mm": outer, "thickness_mm": 8}])
        w, h = 2*(r+10), 2*r+30
        housing = box(w, h, 8, (0, -5, z+4)).cut(cylinder(outer+.15, 10, (0, 0, z-1)))
        if split:
            housing_group = f"{prefix}_housing"
            groups.append({"id": housing_group, "label": "分体轴承座", "parent": f"{prefix}_support"})
            for side, sign_x in [("left", -1), ("right", 1)]:
                half = housing.intersect(box(w/2-.15, h+2, 10, (sign_x*w/4, -5, z+4)))
                add(f"{prefix}_housing_{side}", f"轴承座{'左' if sign_x < 0 else '右'}半体",
                    housing_group, half, BLUE, (sign_x*40, 0, sign*10),
                    [{"op": "box_bore_split", "side": side, "width_mm": w,
                      "height_mm": h, "bore_mm": outer+.15}])
        else:
            add(f"{prefix}_housing", "整体轴承座", f"{prefix}_support", housing, BLUE,
                (0, 30, sign*15), [{"op": "box_bore", "width_mm": w,
                                  "height_mm": h, "bore_mm": outer+.15}])
    for name, z, sign in [("front_retainer", 5, -1), ("rear_retainer", length-8, 1)]:
        add(name, "定位环", "rotor", ring(r+2, r+.15, 3, z), STEEL, (0, 0, sign*85),
            [{"op": "annulus_extrude", "outer_radius_mm": r+2,
              "inner_radius_mm": r+.15, "thickness_mm": 3}])
    if guard:
        groups.append({"id": "protection", "label": "防护组件", "parent": "root"})
        guard_shape = ring(gear_r+5, gear_r+2, 12, length/2-6)
        foot_height = (r+20)-(gear_r+2)
        foot = box(8, foot_height, 12,
                   (0, -((r+20)+(gear_r+2))/2, length/2))
        guard_shape = guard_shape.fuse(foot)
        add("guard", "防护环与支脚", "protection", guard_shape, BLUE, (40, 20, 0),
            [{"op": "annulus_extrude", "outer_radius_mm": gear_r+5,
              "inner_radius_mm": gear_r+2}, {"op": "fuse_foot"}])
    return AssemblyModel(problem, bits, parts, groups,
                         {"shaft_radius_mm": r, "bearing_bore_mm": bore,
                          "gear_radius_mm": gear_r, "shaft_length_mm": length},
                         [{"kind": "shaft_bearing_fit", "parts": ["shaft", f"{p}_bearing"],
                           "radial_clearance_mm": bore-r, "allowed_mm": [.15, .5]}
                          for p in ["front", "rear"]])


def _clamp(problem: StructureProblem, bits: str) -> AssemblyModel:
    d = problem.describe(bits)
    jaw = d["jaw"]["attributes"]["width"]
    inner = d["frame"]["attributes"]["inner"]
    actuator = d["actuator"]["attributes"]["type"]
    guide = d.get("guide", {}).get("attributes", {}).get("type", "dual")
    grooved = d.get("pad", {}).get("attributes", {}).get("type") == "grooved"
    opening = problem.spec.opening_mm
    length = opening+100
    support_x = length/2-12
    jaw_x = opening/2+7.5
    parts: list[Part] = []
    groups = [{"id": "guidance", "label": "导向组件", "parent": "root"},
              {"id": "fixed_jaw_group", "label": "固定夹爪组件", "parent": "root"},
              {"id": "slider", "label": "移动夹爪组件", "parent": "root"},
              {"id": "actuation", "label": "丝杆驱动" if actuator == "screw" else "活塞式驱动",
               "parent": "root"}]

    def add(id, label, parent, shape, color, explode, features):
        parts.append(Part(id, label, parent, shape, color, explode, features))

    add("base", "底板", "root", box(length, inner+18, 6, (0, 0, 3)), DARK, (0, 0, -25),
        [{"op": "box", "dimensions_mm": [length, inner+18, 6]}])
    if guide == "dual":
        for label, sign in [("left", -1), ("right", 1)]:
            add(f"rail_{label}", "侧导轨", "guidance",
                box(length-36, 6, 4, (0, sign*(inner/2+3), 8)), TEAL,
                (0, sign*30, 0), [{"op": "box", "dimensions_mm": [length-36, 6, 4]}])
    else:
        a, b = opening/2, support_x-6
        add("rail_central", "中央导轨", "guidance", box(b-a, 8, 4, ((a+b)/2, 0, 8)), TEAL,
            (0, 0, -15), [{"op": "box", "dimensions_mm": [b-a, 8, 4]}])
    for id, sign, parent in [("fixed", -1, "fixed_jaw_group"), ("moving", 1, "slider")]:
        z_min = 6 if id == "fixed" else 10
        add(f"{id}_jaw", "固定夹爪" if id == "fixed" else "移动夹爪", parent,
            box(12, jaw, 34-z_min, (sign*jaw_x, 0, (34+z_min)/2)), BLUE,
            (sign*25, 0, 25), [{"op": "box", "dimensions_mm": [12, jaw, 34-z_min]}])
        pad_x = sign*(opening/2+.75)
        pad = box(1.5, jaw-2, 16, (pad_x, 0, 22))
        if grooved:
            # A shallow groove leaves a continuous backing web in the 1.5-mm pad.
            pad = pad.cut(cylinder(.75, 18, (sign*opening/2, 0, 13)))
        add(f"{id}_pad", "带槽夹垫" if grooved else "平面夹垫", parent, pad, ORANGE,
            (sign*10, 0, 40), [{"op": "box", "dimensions_mm": [1.5, jaw-2, 16]},
                             *([{"op": "cut_groove", "radius_mm": .75}] if grooved else [])])
    support = box(8, inner+12, 24, (support_x, 0, 18)).cut(
        cylinder(3.4, 10, (support_x-5, 0, 22), (1, 0, 0)))
    add("actuator_support", "驱动支座", "actuation", support, BLUE, (30, 0, 15),
        [{"op": "box_bore", "bore_radius_mm": 3.4}])
    rod_start, rod_end = jaw_x+6, support_x+16
    add("rod", "丝杆示意轴" if actuator == "screw" else "活塞杆", "actuation",
        cylinder(3, rod_end-rod_start, (rod_start, 0, 22), (1, 0, 0)), STEEL, (55, 0, 0),
        [{"op": "circle_extrude", "radius_mm": 3, "axis": "X", "length_mm": rod_end-rod_start}])
    if actuator == "screw":
        add("knob", "调节旋钮", "actuation", cylinder(8, 6, (rod_end, 0, 22), (1, 0, 0)),
            ORANGE, (70, 0, 0), [{"op": "circle_extrude", "radius_mm": 8, "axis": "X"}])
    else:
        # Rod runs inside the bore; an envelope cylinder is not a pneumatic simulation.
        body = cylinder(8, 18, (support_x+4, 0, 22), (1, 0, 0)).cut(
            cylinder(3.3, 20, (support_x+3, 0, 22), (1, 0, 0)))
        add("actuator_body", "活塞式驱动包络", "actuation", body, TEAL, (65, 0, 0),
            [{"op": "annulus_extrude", "outer_radius_mm": 8, "inner_radius_mm": 3.3,
              "axis": "X", "length_mm": 18}])
    return AssemblyModel(problem, bits, parts, groups,
                         {"jaw_width_mm": jaw, "guide_inner_mm": inner, "opening_mm": opening},
                         [{"kind": "jaw_guide_clearance", "parts": ["moving_jaw"],
                           "clearance_mm": inner-jaw, "minimum_mm": 4}])


def decode_geometry(problem: StructureProblem, bits: str) -> AssemblyModel:
    model = _drive(problem, bits) if problem.spec.family == "drive" else _clamp(problem, bits)
    scale = problem.spec.scale
    if scale != 1:
        for part in model.parts:
            part.shape = part.shape.scale(scale)
            part.explode = tuple(v*scale for v in part.explode)
    model.dimensions["global_scale"] = scale
    return model


def bbox_overlap(a: cq.Shape, b: cq.Shape, tolerance: float = 1e-7) -> bool:
    a, b = a.BoundingBox(), b.BoundingBox()
    return all(min(getattr(a, f"{axis}max"), getattr(b, f"{axis}max")) -
               max(getattr(a, f"{axis}min"), getattr(b, f"{axis}min")) > tolerance
               for axis in "xyz")


def validate_geometry(model: AssemblyModel, intersections: bool = True) -> ValidationResult:
    issues: list[ValidationIssue] = []
    selection = model.problem.selection(model.bits)
    if selection is None:
        raise ValueError("Geometry needs a categorical assignment")
    for violation in model.problem.violations(selection):
        issues.append(ValidationIssue(code="structural_constraint", message=violation))
    for part in model.parts:
        if not part.shape.isValid() or part.shape.Volume() <= 1e-7:
            issues.append(ValidationIssue(code="invalid_solid", message=f"Invalid solid: {part.id}",
                                          parts=[part.id]))
        elif len(part.shape.Solids()) != 1:
            issues.append(ValidationIssue(code="disconnected_part", message=f"零件不是单一连通实体: {part.id}",
                                          parts=[part.id]))
    for mate in model.mates:
        if mate["kind"] == "shaft_bearing_fit":
            if not mate["allowed_mm"][0] <= mate["radial_clearance_mm"] <= mate["allowed_mm"][1]:
                issues.append(ValidationIssue(code="bearing_fit", message="轴与轴承的径向间隙不合格",
                                              parts=mate["parts"], culprit_choices=[(0, selection[0]),
                                                                                   (1, selection[1])]))
        elif mate["kind"] == "jaw_guide_clearance" and mate["clearance_mm"] < mate["minimum_mm"]:
            issues.append(ValidationIssue(code="jaw_clearance", message="夹爪导向间距不足",
                                          parts=mate["parts"], culprit_choices=[(0, selection[0]),
                                                                               (1, selection[1])]))
    max_radius = model.problem.spec.max_gear_radius_mm
    if model.problem.spec.family == "drive" and max_radius is not None:
        gear = next(p for p in model.parts if p.id == "gear")
        b = gear.shape.BoundingBox()
        actual_radius = max(abs(b.xmin), abs(b.xmax), abs(b.ymin), abs(b.ymax))
        if actual_radius > max_radius+1e-6:
            issues.append(ValidationIssue(code="gear_envelope", message="齿轮几何超出指定包络",
                                          parts=[gear.id], culprit_choices=[(2, selection[2])]))
    pairs, maximum = 0, 0.
    if intersections:
        for a, b in combinations(model.parts, 2):
            if not bbox_overlap(a.shape, b.shape):
                continue
            pairs += 1
            volume = max(0., a.shape.intersect(b.shape).Volume())
            maximum = max(maximum, volume)
            if volume > 1e-5*model.problem.spec.scale**3:
                issues.append(ValidationIssue(code="interference", message=f"零件干涉: {a.id}/{b.id}",
                                              parts=[a.id, b.id]))
    return ValidationResult(valid=not issues, issues=issues, checked_pairs=pairs,
                            max_intersection_mm3=maximum,
                            checks={"solid_validity": True, "one_connected_solid_per_part": True,
                                    "mate_dimensions": True,
                                    "brep_pairwise_intersections": intersections,
                                    "load_or_manufacturing_validated": False})


def export_model(model: AssemblyModel, directory: str | Path, metadata: dict | None = None) -> dict:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    assembly = cq.Assembly(name="root")
    groups = {"root": assembly}
    for group in model.groups:
        groups[group["id"]] = cq.Assembly(name=group["id"])
    for part in model.parts:
        groups[part.parent].add(part.shape, name=part.id, color=cq.Color(*part.color))
    # Attach children deepest-first: CadQuery copies assembly contents on add.
    for group in reversed(model.groups):
        groups[group["parent"]].add(groups[group["id"]])
    assembly.export(str(directory/"assembly.step"))
    scene = trimesh.Scene(base_frame="root")
    for group in model.groups:
        scene.graph.update(frame_to=group["id"], frame_from=group["parent"], matrix=np.eye(4))
    for part in model.parts:
        vertices, triangles = part.shape.tessellate(.12*model.problem.spec.scale, .15)
        mesh = trimesh.Trimesh(vertices=np.array([v.toTuple() for v in vertices])/1000,
                               faces=np.array(triangles), process=False)
        mesh.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
            baseColorFactor=[*part.color, 1.0], metallicFactor=.55, roughnessFactor=.32))
        scene.add_geometry(mesh, node_name=part.id, geom_name=part.id, parent_node_name=part.parent)
    (directory/"assembly.glb").write_bytes(scene.export(file_type="glb"))
    structure = {**model.structure(), **(metadata or {})}
    (directory/"structure.json").write_text(json.dumps(structure, indent=2, ensure_ascii=False))
    (directory/"design.json").write_text(model.problem.spec.model_dump_json(indent=2))
    (directory/"problem.json").write_text(json.dumps(model.problem.as_dict(), indent=2, ensure_ascii=False))
    return {"step": "assembly.step", "glb": "assembly.glb", "structure": "structure.json",
            "parameters": "design.json", "problem": "problem.json"}
