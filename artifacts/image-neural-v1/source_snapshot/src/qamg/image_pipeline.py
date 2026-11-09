"""Image -> neural mesh -> QB-selected part hypotheses -> split/assemble/export."""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import trimesh

from .backends import make_backend
from .benchmark import dump
from .classical import solve_classical
from .image_model import generate_mesh, read_image
from .mesh_parts import (
    as_manifold,
    as_trimesh,
    cut_parts,
    merge_solids,
    partition_problem,
    propose_planes,
    sampled_disassembly,
    validate_partition,
)
from .quantum import QuantumStructurePrior
from .schema import Image3DSettings


def export_parts(parts, output: Path, provenance: dict, disassembly: dict, validation: dict,
                 labels: list[str] | None = None, members: list[list[str]] | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    directory = output/"parts"
    directory.mkdir(exist_ok=True)
    scene, records = trimesh.Scene(), []
    centers = np.array([as_trimesh(part).centroid for part in parts])
    origin = centers.mean(axis=0)
    span = max(float(np.ptp(centers, axis=0).max()), 10)
    for i, solid in enumerate(parts):
        id = f"part-{i:03d}"
        mesh = as_trimesh(solid)
        mesh.export(directory/f"{id}.ply")
        delta = centers[i]-origin
        norm = float(np.linalg.norm(delta))
        direction = delta/norm if norm > 1e-8 else np.array([0., 0., 1.])
        label = labels[i] if labels else f"候选分件 {i+1:02d}"
        records.append({"id": id, "label": label, "parent": "generated-object",
                        "mesh_file": f"parts/{id}.ply", "volume_mm3": solid.volume(),
                        "bounds_mm": mesh.bounds.tolist(), "center_mm": centers[i].tolist(),
                        "explode_mm": (direction*max(span*.8, 25)).tolist(),
                        "members": members[i] if members else [id],
                        "connected_components": len(solid.decompose()),
                        "inferred_partition": True})
        meters = mesh.copy()
        meters.apply_scale(.001)
        scene.add_geometry(meters, geom_name=id, node_name=id)
    scene.export(output/"assembly.glb")
    combined = as_trimesh(merge_solids(parts))
    combined.export(output/"assembly.obj")
    combined.export(output/"assembly.stl")
    with ZipFile(output/"parts.zip", "w", compression=ZIP_DEFLATED) as archive:
        for file in sorted(directory.glob("*.ply")):
            archive.write(file, file.name)
    structure = {"schema_version": 2, "representation": "generated_mesh_parts",
                 "units": "mm", "glb_units": "m", "parts": records,
                 "groups": [{"id": "generated-object", "parent": "root", "label": "图像生成 / 候选分件"}],
                 "provenance": provenance, "validation": validation, "disassembly": disassembly,
                 "engineering_scope": "可见外观的生成式三维与几何分件；背面、内部、真实连接和尺寸存在不确定性"}
    dump(output/"structure.json", structure)
    dump(output/"disassembly.json", disassembly)
    return structure


def generate_image(data: bytes, settings: Image3DSettings, output: Path,
                   source_name: str = "image", progress=lambda _: None) -> dict:
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    image = read_image(data)
    image.save(output/"input.png")
    image_sha = hashlib.sha256(data).hexdigest()
    mesh, processed, model_metadata = generate_mesh(image, settings, progress)
    processed.save(output/"processed.png")
    source = as_manifold(mesh)
    source_scene = trimesh.Scene()
    meters = mesh.copy()
    meters.apply_scale(.001)
    source_scene.add_geometry(meters, node_name="generated-source")
    source_scene.export(output/"source.glb")
    progress("构建分件候选与量子目标")
    planes = propose_planes(mesh)
    problem = partition_problem(planes, image_sha, settings.desired_parts, settings.seed)
    dump(output/"problem.json", problem.as_dict())
    dump(output/"design.json", {"image_sha256": image_sha, "source_name": Path(source_name).name,
                                "settings": settings.model_dump(), "model": model_metadata,
                                "planes": [asdict(plane) for plane in planes]})
    planner = settings.planner.model_copy(update={"encoding": "compact_binary"})
    progress("QB 采样分件结构" if planner.backend != "classical" else "经典对照生成分件结构")
    before = time.perf_counter()
    backend = make_backend(planner.backend, planner.max_qubits) if planner.backend != "classical" else None
    result = (QuantumStructurePrior(backend).plan(problem, planner, settings.seed)
              if backend is not None else solve_classical(problem, "cpsat", settings.seed, timeout_s=planner.timeout_s))
    planning_s = time.perf_counter()-before
    round_trace = {"round": 0, "problem": problem.as_dict(), "plan": result.as_dict(),
                   "geometry_attempts": [], "feedback_rules": []}
    if result.circuit_qasm3:
        (output/"partition.qasm").write_text(result.circuit_qasm3)
    selected, parts, validation = None, None, None
    progress("切分封闭实体并校验可合并性")
    before = time.perf_counter()
    for candidate in [c for c in result.candidates if c.feasible][:32]:
        try:
            pieces = cut_parts(source, planes, problem.selection(candidate.bits), max_parts=24)
            checked = validate_partition(source, pieces)
            round_trace["geometry_attempts"].append({"bits": candidate.bits, "part_count": len(pieces), "validation": checked})
            if checked["valid"]:
                selected, parts, validation = candidate, pieces, checked
                break
        except (ValueError, ArithmeticError) as error:
            round_trace["geometry_attempts"].append({"bits": candidate.bits, "rejected": str(error)})
    dump(output/"trace.json", [round_trace])
    if selected is None:
        raise ValueError("当前测量候选没有形成有效分件，请调整分件数、图片或随机种子；没有使用固定模型替代")
    geometry_s = time.perf_counter()-before
    ids = [f"part-{i:03d}" for i in range(len(parts))]
    progress("采样检查拆解路径")
    before = time.perf_counter()
    disassembly = sampled_disassembly(parts, ids)
    disassembly_s = time.perf_counter()-before
    provenance = {"mode": planner.backend, "execution_source": result.execution_source,
                  "quantum_sdk": "QuantumBridge" if backend else None,
                  "selected_measurement": {"bits": selected.bits, "job_id": selected.first_job_id,
                                           "observed_count": selected.count, "kind": "direct_or_exactly_decoded"},
                  "generation_rounds": 1, "fallback_used": False, "hardware_verified": False,
                  "quantum_speedup_verified": False, "image_sha256": image_sha,
                  "geometry_is_inferred": True}
    progress("导出三维模型与分件")
    export_start = time.perf_counter()
    structure = export_parts(parts, output, provenance, disassembly, validation)
    summary = {"status": "generated", "kind": "image3d", "spec": problem.spec.model_dump(),
               "source_name": Path(source_name).name, "settings": settings.model_dump(),
               "validation": validation, "provenance": provenance, "model": model_metadata,
               "selected_bits": selected.bits, "decisions": problem.describe(selected.bits),
               "part_count": len(structure["parts"]), "resources": result.resources,
               "files": {"glb": "assembly.glb", "obj": "assembly.obj", "stl": "assembly.stl", "parts": "parts.zip",
                         "structure": "structure.json", "trace": "trace.json", "input": "input.png"},
               "disassembly": {"complete_for_generated_mesh": disassembly["complete_for_generated_mesh"],
                                "moves": len(disassembly["moves"]), "continuous_collision_proof": False},
               "timings": {"neural_generation_s": model_metadata["timings"]["total_s"], "planning_s": planning_s,
                           "partition_and_validation_s": geometry_s, "disassembly_s": disassembly_s,
                           "export_s": time.perf_counter()-export_start, "end_to_end_s": time.perf_counter()-started},
               "limitations": ["背面和内部由单图模型推测", "分件基于形状与颜色，真实装配连接未经验证",
                               "尺寸由输入标尺确定，非测量尺寸", "离散碰撞采样不构成连续路径证明"]}
    dump(output/"run.json", summary)
    progress("完成")
    return summary


def load_run_parts(directory: Path, structure: dict):
    scene = None
    results = {}
    for part in structure["parts"]:
        relative = part.get("mesh_file")
        if relative:
            path = (directory/relative).resolve()
            if not path.is_relative_to(directory.resolve()):
                raise ValueError("分件路径无效")
            mesh = trimesh.load(path, force="mesh", process=True)
        else:
            if scene is None:
                scene = trimesh.load(directory/"assembly.glb", force="scene")
            transform, geometry = scene.graph[part["id"]]
            mesh = scene.geometry[geometry].copy()
            mesh.apply_transform(transform)
            mesh.apply_scale(1000)
            mesh.merge_vertices()
            mesh.fix_normals(multibody=True)
        results[part["id"]] = as_manifold(mesh)
    return results


def merge_run(source_directory: Path, selected_ids: list[str], output: Path,
              source_run_id: str, progress=lambda _: None) -> dict:
    started = time.perf_counter()
    summary = json.loads((source_directory/"run.json").read_text())
    structure = json.loads((source_directory/"structure.json").read_text())
    if len(set(selected_ids)) != len(selected_ids) or len(selected_ids) < 2:
        raise ValueError("请选择至少两个不同分件")
    all_ids = {p["id"] for p in structure["parts"]}
    if not set(selected_ids) <= all_ids:
        raise ValueError("选中的分件不属于当前模型")
    progress("合并选中的封闭实体")
    loaded = load_run_parts(source_directory, structure)
    combined = merge_solids([loaded[id] for id in selected_ids])
    kept = [p for p in structure["parts"] if p["id"] not in selected_ids]
    parts = [loaded[p["id"]] for p in kept]+[combined]
    labels = [p["label"] for p in kept]+[f"合并组合 · {len(selected_ids)} 件"]
    members = [p.get("members", [p["id"]]) for p in kept]+[
        [member for p in structure["parts"] if p["id"] in selected_ids for member in p.get("members", [p["id"]])]]
    validation = validate_partition(merge_solids(list(loaded.values())), parts)
    if not validation["valid"]:
        raise ValueError("合并前后的几何不一致，已保留原模型")
    output.mkdir(parents=True, exist_ok=True)
    ids = [f"part-{i:03d}" for i in range(len(parts))]
    progress("检查组合结果并导出")
    disassembly = sampled_disassembly(parts, ids) if len(parts) > 1 else {"moves": [], "complete_for_generated_mesh": True,
                    "complete_for_envelope_model": True, "continuous_collision_proof": False}
    provenance = {**summary["provenance"], "source_run_id": source_run_id,
                  "original_run_id": summary.get("original_run_id", source_run_id),
                  "edit": "classical_mesh_boolean_union", "merged_ids": selected_ids}
    export_parts(parts, output, provenance, disassembly, validation, labels, members)
    for filename in ["input.png", "processed.png", "source.glb", "design.json", "problem.json", "trace.json"]:
        if (source_directory/filename).is_file():
            shutil.copy2(source_directory/filename, output/filename)
    edit_s = time.perf_counter()-started
    result = {**summary, "kind": "mesh_edit", "part_count": len(parts), "validation": validation,
              "provenance": provenance, "source_run_id": source_run_id,
              "original_run_id": provenance["original_run_id"], "files": {
                  "glb": "assembly.glb", "obj": "assembly.obj", "stl": "assembly.stl", "parts": "parts.zip", "structure": "structure.json"},
              "timings": {**summary["timings"], "latest_merge_s": edit_s,
                          "end_to_end_s": summary["timings"]["end_to_end_s"]+edit_s}}
    dump(output/"merge.json", {"source_run_id": source_run_id, "selected_ids": selected_ids,
                               "validation": validation, "edit_s": edit_s,
                               "quantum_resampling_performed": False})
    dump(output/"run.json", result)
    progress("完成")
    return result
