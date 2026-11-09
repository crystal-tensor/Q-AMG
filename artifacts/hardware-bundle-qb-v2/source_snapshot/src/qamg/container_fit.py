"""Explicit image-guided rotational-container reconstruction, with stated priors.

This is an assisted edit of a neural reference, not a claim that the neural model
recovered the true hidden construction. The cup remains continuous inside its
sleeve; hollow walls and assembly clearances are explicit assumptions.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import trimesh

from .assembly_plan import direction_problem, selected_disassembly, selected_vectors
from .backends import make_backend
from .benchmark import dump
from .classical import solve_classical
from .image_pipeline import export_parts, load_run_parts
from .mesh_parts import as_manifold, as_trimesh, merge_solids, validate_partition
from .problem import StructureProblem
from .quantum import QuantumStructurePrior
from .schema import ChoiceGroup, ImagePlanningContext, Option, PairRule


def hollow_cup(radius_bottom: float, radius_top: float, height: float, wall: float):
    if not (0 < wall < min(radius_bottom, radius_top)/3 and height > wall*3):
        raise ValueError("杯体壁厚或高度不合理")
    def radii(z):
        return radius_bottom+(radius_top-radius_bottom)*z/height
    z = np.linspace(0, height, 24)
    profile = [[0, 0], *[[radii(v), v] for v in z],
               *[[radii(v)-wall, v] for v in np.linspace(height, wall, 24)], [0, wall], [0, 0]]
    mesh = trimesh.creation.revolve(np.array(profile), sections=128)
    mesh.fix_normals()
    return mesh


def sleeve_mesh(radius_bottom: float, radius_top: float, height: float, clearance: float,
                 thickness: float, ribs: int = 48):
    count = ribs*4
    angle = np.arange(count)*2*np.pi/count
    ripples = thickness*.15*(1+np.cos(ribs*angle))
    rings = []
    for radius, z, outer in [(radius_bottom+clearance, 0, True), (radius_top+clearance, height, True),
                              (radius_top+clearance, height, False), (radius_bottom+clearance, 0, False)]:
        r = radius+thickness+ripples if outer else np.full(count, radius)
        rings.append(np.column_stack([r*np.cos(angle), r*np.sin(angle), np.full(count, z)]))
    faces = []
    for ring in range(4):
        for i in range(count):
            a, b = ring*count+i, ring*count+(i+1) % count
            c, d = ((ring+1) % 4)*count+(i+1) % count, ((ring+1) % 4)*count+i
            faces += [[a, b, c], [a, c, d]]
    mesh = trimesh.Trimesh(np.concatenate(rings), faces, process=True)
    mesh.fix_normals()
    return mesh


def assisted_lid(radius: float, base_height: float, scale_height: float):
    """A connected cap/rim/tab proxy, explicitly assumed rather than recovered."""
    thickness, rim_height = scale_height*.025, scale_height*.045
    disk = as_manifold(trimesh.creation.cylinder(radius=radius, height=thickness, sections=128))
    disk = disk.translate([0, 0, base_height+thickness/2])
    rim = as_manifold(trimesh.creation.annulus(r_min=radius*.9, r_max=radius,
                                               height=rim_height, sections=128))
    rim = rim.translate([0, 0, base_height+rim_height/2])
    tab_height, tab_width, tab_depth = scale_height*.12, radius*.55, radius*.16
    tab = as_manifold(trimesh.creation.box([tab_width, tab_depth, tab_height]))
    hole = as_manifold(trimesh.creation.box([tab_width*.65, tab_depth*2, tab_height*.55]))
    tab = (tab-hole).translate([0, radius*.5, base_height+tab_height/2])
    return merge_solids([disk, rim, tab])


def _colorize(solid, color):
    rgb = np.asarray(color, dtype=float)/255
    return solid.set_properties(3, lambda position, properties: rgb)


def _region_observations(image, mask, progress):
    """SAM prompts are derived from the explicitly selected container prior."""
    import torch
    from scipy import ndimage

    from .vision import predictor

    yy, xx = np.where(mask)
    x0, x1, y0, y1 = xx.min(), xx.max(), yy.min(), yy.max()
    height, width = y1-y0, x1-x0
    model = predictor("cpu")
    progress("按杯类结构约束核对盖子和杯套区域")
    observations = {}
    with torch.inference_mode():
        model.set_image(np.array(image.convert("RGB"), copy=True))
        for name, band, position in [("lid", (0., .32), .13), ("sleeve", (.58, 1.), .86)]:
            row = int(y0+height*position)
            columns = np.flatnonzero(mask[row])
            column = float(np.median(columns)) if len(columns) else float((x0+x1)/2)
            box = np.array([x0, y0+height*band[0], x1, y0+height*band[1]], dtype=np.float32)
            points = np.array([[column, row], [(x0+x1)/2, y0+height*.48]], dtype=np.float32)
            candidates, scores, _ = model.predict(point_coords=points, point_labels=np.array([1, 0]),
                                                   box=box, multimask_output=True)
            candidates = np.asarray(candidates) > 0
            zone = mask.copy()
            zone[:round(box[1])] = False
            zone[round(box[3])+1:] = False
            values = [np.logical_and(c, zone).sum()/max(1, np.logical_or(c, zone).sum()) for c in candidates]
            selected = candidates[int(np.argmax(values))] & mask
            components, count = ndimage.label(selected)
            if count:
                areas = np.bincount(components.ravel())
                areas[0] = 0
                selected = components == int(areas.argmax())
            sy, sx = np.where(selected)
            if len(sx) < mask.sum()*.03 or len(sx) > mask.sum()*.6:
                raise ValueError(f"无法可靠定位 {name} 区域，请核对输入与结构类型")
            pixels = np.asarray(image.convert("RGB"))[selected]
            brightness = pixels.mean(axis=1)
            illuminated = pixels[(brightness >= np.quantile(brightness, .55)) &
                                 (brightness <= np.quantile(brightness, .85))]
            color = np.median(illuminated if len(illuminated) else pixels, axis=0).astype(int).tolist()
            observations[name] = {"mask": selected, "color": color,
                                   "bbox": [int(sx.min()), int(sy.min()), int(sx.max()), int(sy.max())],
                                   "scores": np.asarray(scores).tolist()}
    observations["subject_bbox"] = [int(x0), int(y0), int(x1), int(y1)]
    observations["sleeve_height_fraction"] = float(np.clip(
        (observations["sleeve"]["bbox"][3]-observations["sleeve"]["bbox"][1])/height, .16, .45))
    observations["width_px"] = int(width)
    samples = []
    for fraction in np.linspace(.35, .6, 20):
        row = round(y0+height*fraction)
        columns = np.flatnonzero(mask[row])
        if len(columns) > 8:
            samples.append([float(fraction), float(columns.max()-columns.min())])
    coefficients = np.polyfit(np.array(samples)[:, 0], np.array(samples)[:, 1], 1)
    top_width = float(np.polyval(coefficients, .29))
    observations["wall_silhouette_samples"] = samples
    observations["taper_radius_ratio"] = float(np.clip(np.polyval(coefficients, .83)/top_width, .65, 1.))
    observations["lid_radius_ratio"] = float(np.clip(
        (observations["lid"]["bbox"][2]-observations["lid"]["bbox"][0])/top_width, 1.01, 1.2))
    observations["projection_assumption"] = "Taper fitted from projected side silhouette; perspective and hidden geometry remain uncertain"
    return observations


def fit_container(source_directory: Path, output: Path, source_run_id: str, progress=lambda _: None):
    from PIL import Image

    from .image_model import prepare_image, read_image
    from .schema import Image3DSettings

    started = time.perf_counter()
    original = json.loads((source_directory/"run.json").read_text())
    structure = json.loads((source_directory/"structure.json").read_text())
    image = read_image((source_directory/"input.png").read_bytes())
    output.mkdir(parents=True, exist_ok=True)
    image.save(output/"input.png")
    settings = Image3DSettings.model_validate(original["settings"])
    prepared, preprocessing = prepare_image(image, settings.model_copy(update={"device": "cpu"}),
                                             clean_foreground=True, background=1., mask_path=output/"foreground-mask.png")
    prepared.save(output/"processed.png")
    mask = np.array(Image.open(output/"foreground-mask.png")) > 127
    observations = _region_observations(image, mask, progress)
    for key in ["lid", "sleeve"]:
        Image.fromarray(np.uint8(observations[key].pop("mask"))*255).save(output/f"{key}-mask.png")
    dump(output/"image-observations.json", observations)
    reference = merge_solids(list(load_run_parts(source_directory, structure).values()))
    mesh = as_trimesh(reference)
    values, axes = np.linalg.eigh(np.cov(mesh.vertices.T))
    scores = [abs(values[(i+1) % 3]-values[(i+2) % 3])/max(values[(i+1) % 3]+values[(i+2) % 3], 1e-8) for i in range(3)]
    axis = axes[:, int(np.argmin(scores))]
    if axis[2] < 0:
        axis = -axis
    rotation = trimesh.geometry.align_vectors(axis, [0., 0., 1.])
    mesh.apply_transform(rotation)
    center = np.median(mesh.vertices[:, :2], axis=0)
    low, high = np.quantile(mesh.vertices[:, 2], [.003, .997])
    mesh.apply_translation([-center[0], -center[1], -low])
    height = float(high-low)
    radius = np.linalg.norm(mesh.vertices[:, :2], axis=1)
    samples = []
    for fraction in np.linspace(.4, .78, 16):
        selected = abs(mesh.vertices[:, 2]-height*fraction) < height*.025
        if selected.sum() >= 8:
            samples.append([height*fraction, float(np.quantile(radius[selected], .94))])
    if len(samples) < 6:
        raise ValueError("外形参考不足以拟合连续杯体")
    top_radius = float(np.median(np.array(samples)[:, 1]))
    bottom_radius = top_radius*observations["taper_radius_ratio"]
    coefficients = [(top_radius-bottom_radius)/(height*.84), bottom_radius]
    def body_radius(z):
        return float(np.polyval(coefficients, z))
    if not (.08*height < body_radius(0) < height and .08*height < body_radius(height) < height):
        raise ValueError("此对象不符合所选的杯类旋转体结构")
    progress("QB 选择图像拟合与配合参数")
    image_sha = hashlib.sha256((source_directory/"input.png").read_bytes()).hexdigest()
    groups = [ChoiceGroup(key="lid_interface", label="盖子与杯口界面", options=[
        Option(key="lower", label="较低界面", cost=1, attributes={"fraction": .82}),
        Option(key="upper", label="较高界面", cost=0, attributes={"fraction": .86})]),
        ChoiceGroup(key="wall", label="不可见杯壁假设", options=[
        Option(key="thin", label="较薄壁", cost=0, attributes={"wall_mm": max(.8, body_radius(0)*.03)}),
        Option(key="thick", label="较厚壁", cost=1, attributes={"wall_mm": max(1.2, body_radius(0)*.045)})]),
        ChoiceGroup(key="clearance", label="杯套配合间隙假设", options=[
        Option(key="close", label="较小间隙", cost=0, attributes={"clearance_mm": .2}),
        Option(key="loose", label="较大间隙", cost=1, attributes={"clearance_mm": .4})])]
    problem = StructureProblem(ImagePlanningContext(seed=settings.seed, image_sha256=image_sha, desired_parts=3,
                                source="User-selected lid/cup/sleeve prior; silhouette fit and hidden wall assumptions"), groups,
                                [PairRule(left=(1, 0), right=(2, 0), cost=2, reason="fit_regularization_not_measured_load")])
    planner = settings.planner.model_copy(update={"encoding": "compact_binary"})
    backend = make_backend(planner.backend, planner.max_qubits) if planner.backend != "classical" else None
    result = (QuantumStructurePrior(backend).plan(problem, planner, settings.seed) if backend else
              solve_classical(problem, "cpsat", settings.seed, timeout_s=planner.timeout_s))
    chosen, parts, parameters = None, None, None
    attempts = []
    for candidate in [c for c in result.candidates if c.feasible]:
        selection = problem.selection(candidate.bits)
        lid_z = height*groups[0].options[selection[0]].attributes["fraction"]
        wall = groups[1].options[selection[1]].attributes["wall_mm"]
        clearance = groups[2].options[selection[2]].attributes["clearance_mm"]
        sleeve_height = height*observations["sleeve_height_fraction"]
        cap = assisted_lid(body_radius(lid_z)*observations["lid_radius_ratio"], lid_z, height)
        body = hollow_cup(body_radius(0), body_radius(lid_z), lid_z, wall)
        thickness = max(.9, body_radius(0)*.065)
        sleeve = sleeve_mesh(body_radius(0), body_radius(sleeve_height), sleeve_height, clearance, thickness)
        solids = [cap, as_manifold(body), as_manifold(sleeve)]
        colors = [observations["lid"]["color"], [184, 208, 214], observations["sleeve"]["color"]]
        solids = [_colorize(solid, color) for solid, color in zip(solids, colors, strict=True)]
        overlap = sum(p.volume() for p in solids)-merge_solids(solids).volume()
        attempts.append({"bits": candidate.bits, "overlap_mm3": overlap})
        if overlap <= max(1e-5, sum(p.volume() for p in solids)*1e-7):
            chosen, parts = candidate, solids
            parameters = {"axis_from_reference": axis.tolist(), "reference_to_upright": rotation.tolist(),
                          "profile_samples_mm": samples, "radius_fit_coefficients": list(coefficients),
                          "body_height_mm": lid_z, "wall_mm_assumed": wall,
                          "sleeve_height_mm": sleeve_height, "clearance_mm_assumed": clearance,
                          "sleeve_wall_mm_assumed": thickness, "ribs_assumed": 48}
            parameters["lid_details_assumed"] = "Connected disk/rim/fold-tab display proxy; not recovered latch or seal geometry"
            break
    if parts is None:
        dump(output/"fit-attempts.json", attempts)
        raise ValueError("测量得到的配合参数未形成无交叠三部件，保留原模型")
    direction, choices, anchor = direction_problem(parts, image_sha, settings.seed, anchor_hint=1)
    direction_result = (QuantumStructurePrior(backend).plan(direction, planner, settings.seed+1) if backend else
                        solve_classical(direction, "cpsat", settings.seed+1, timeout_s=planner.timeout_s))
    ids = [f"part-{i:03d}" for i in range(3)]
    selected = None
    for candidate in [c for c in direction_result.candidates if c.feasible]:
        vectors = selected_vectors(direction, candidate.bits, choices, 3)
        sequence = selected_disassembly(parts, ids, vectors, anchor)
        selected = candidate
        if sequence["complete_for_generated_mesh"]:
            break
    if selected is None:
        raise ValueError("没有观测到有效的展开方向")
    union = merge_solids(parts)
    validation = validate_partition(union, parts)
    validation["checks"].update({"three_named_component_constraints": True, "continuous_hollow_cup": True,
                                   "sleeve_does_not_replace_cup_bottom": True, "manufacturing_joints_verified": False})
    provenance = {"mode": planner.backend, "execution_source": direction_result.execution_source,
                  "quantum_sdk": "QuantumBridge" if backend else None,
                  "quantum_role": "assisted_fit_parameters_and_assembly_directions",
                  "source_run_id": source_run_id, "geometry_is_inferred": True,
                  "generation_rounds": 2, "fallback_used": False, "hardware_verified": False,
                  "quantum_speedup_verified": False, "structure_prior": "explicit_lid_cup_sleeve",
                  "selected_measurement": {"bits": selected.bits, "job_id": selected.first_job_id, "observed_count": selected.count,
                                            "kind": "direct_or_exactly_decoded"},
                  "selected_fit_measurement": {"bits": chosen.bits, "job_id": chosen.first_job_id,
                                                "observed_count": chosen.count}}
    appearance = [{}, {"type": "transparent_display_proxy", "rgba": [184, 208, 214, 105]}, {}]
    export_parts(parts, output, provenance, sequence, validation, ["盖子", "杯子", "杯套"],
                  explosion_vectors=vectors, appearance=appearance)
    dump(output/"trace.json", [{"round": 0, "problem": problem.as_dict(), "plan": result.as_dict(), "geometry_attempts": attempts},
                               {"round": 1, "problem": direction.as_dict(), "plan": direction_result.as_dict()}])
    dump(output/"problem.json", direction.as_dict())
    for filename, plan in [("fit.qasm", result), ("partition.qasm", direction_result)]:
        if plan.circuit_qasm3:
            (output/filename).write_text(plan.circuit_qasm3)
    dump(output/"container-fit.json", {"reference_run_id": source_run_id, "observations": observations,
             "parameters": parameters, "selected_fit_bits": chosen.bits,
             "selected_fit_job_id": chosen.first_job_id, "preprocessing": preprocessing,
             "scope": "Explicit rotational prior and completed hidden cup wall; not recovered manufacturing truth"})
    shutil.copy2(source_directory/"assembly.glb", output/"source.glb")
    elapsed = time.perf_counter()-started
    summary = {"status": "generated", "kind": "assisted_image3d", "source_name": original["source_name"],
               "spec": direction.spec.model_dump(), "settings": settings.model_dump(), "part_count": 3,
               "selected_bits": selected.bits, "decisions": direction.describe(selected.bits),
               "provenance": provenance, "validation": validation, "reference_run_id": source_run_id,
               "resources": {**direction_result.resources, "fit_sampling_resources": result.resources,
                             "peak_logical_qubits": max(result.resources.get("logical_qubits", 0), direction_result.resources.get("logical_qubits", 0)),
                             "total_shots": result.resources.get("shots", 0)+direction_result.resources.get("shots", 0),
                             "total_backend_calls": result.resources.get("backend_calls", 0)+direction_result.resources.get("backend_calls", 0)},
               "model": {"model": "图像引导结构拟合", "reference_model": original["model"]["model"],
                         "appearance": "source_region_color_and_transparency_proxy", "timings": {"total_s": elapsed}},
               "timings": {"reference_generation_s": original["timings"]["end_to_end_s"], "assisted_fit_s": elapsed,
                           "end_to_end_s": original["timings"]["end_to_end_s"]+elapsed},
               "disassembly": {"complete_for_generated_mesh": sequence["complete_for_generated_mesh"], "moves": len(sequence["moves"])},
               "files": {"glb": "assembly.glb", "obj": "assembly.obj", "parts": "parts.zip", "structure": "structure.json", "fit": "container-fit.json", "trace": "trace.json"},
               "limitations": ["显式旋转体结构先验", "盖子圆盘、外圈与提片为形状近似", "内壁厚度与配合间隙为假设", "透明为显示近似", "连续路径和制造连接未认证"]}
    dump(output/"run.json", summary)
    return summary
