"""Image -> jointly generated neural parts -> QB assembly planning -> edit/export."""
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

from .assembly_plan import direction_problem, selected_disassembly, selected_vectors
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
from .part_model import generate_parts
from .quantum import QuantumStructurePrior
from .schema import Image3DSettings


def export_parts(parts, output: Path, provenance: dict, disassembly: dict, validation: dict,
                 labels: list[str] | None = None, members: list[list[str]] | None = None,
                 explosion_vectors: list[list[float]] | None = None,
                 appearance: list[dict] | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    directory = output/"parts"
    directory.mkdir(exist_ok=True)
    scene, records = trimesh.Scene(), []
    inferred = bool(provenance.get("geometry_is_inferred", False))
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
                        "explode_mm": explosion_vectors[i] if explosion_vectors is not None else (direction*max(span*.8, 25)).tolist(),
                        "members": members[i] if members else [id],
                        "connected_components": len(solid.decompose()),
                        "inferred_partition": inferred,
                        "appearance": appearance[i] if appearance else {}})
        meters = mesh.copy()
        meters.apply_scale(.001)
        if appearance and appearance[i].get("type") == "transparent_display_proxy":
            material = trimesh.visual.material.PBRMaterial(
                baseColorFactor=appearance[i]["rgba"], alphaMode="BLEND", doubleSided=True,
                metallicFactor=0., roughnessFactor=.15)
            meters.visual = trimesh.visual.TextureVisuals(material=material)
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
                 "groups": [{"id": "generated-object", "parent": "root", "label": "图像生成 / 候选分件" if inferred else "网格组合 / 原始零件"}],
                 "provenance": provenance, "validation": validation, "disassembly": disassembly,
                 "engineering_scope": "可见外观的生成式三维与几何分件；背面、内部、真实连接和尺寸存在不确定性"}
    dump(output/"structure.json", structure)
    dump(output/"disassembly.json", disassembly)
    return structure


def generate_image(data: bytes, settings: Image3DSettings, output: Path,
                   source_name: str = "image", progress=lambda _: None) -> dict:
    if settings.image_model == "triposr-geometric":
        return _generate_geometric_image(data, settings, output, source_name, progress)
    return _generate_part_image(data, settings, output, source_name, progress)


def _generate_part_image(data: bytes, settings: Image3DSettings, output: Path,
                         source_name: str, progress) -> dict:
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    image = read_image(data)
    image.save(output/"input.png")
    image_sha = hashlib.sha256(data).hexdigest()
    meshes, processed, metadata = generate_parts(image, settings, progress,
                                                 steps=settings.denoising_steps, tokens=settings.part_tokens,
                                                 artifact_directory=output)
    processed.save(output/"processed.png")
    if len(meshes) != settings.desired_parts:
        raise ValueError("神经模型输出的部件数与请求不符，拒绝用切块补足")
    progress("校验独立部件实体")
    parts = [as_manifold(mesh) for mesh in meshes]
    source = merge_solids(parts)
    validation = validate_partition(source, parts)
    overlap = max(0., sum(p.volume() for p in parts)-source.volume())
    validation["generated_part_overlap_mm3"] = overlap
    validation["generated_part_overlap_fraction"] = overlap/max(source.volume(), 1e-8)
    validation["checks"].update({"requested_neural_part_count": len(parts) == settings.desired_parts,
                                  "part_boundaries_from_neural_model": True,
                                  "semantic_parts_verified": False,
                                  "generated_parts_nonoverlapping": overlap/max(source.volume(), 1e-8) < 1e-5})
    if not validation["valid"]:
        raise ValueError("生成部件实体校验未通过，未替换成其他模型")
    scene = trimesh.Scene()
    for i, mesh in enumerate(meshes):
        scaled = mesh.copy()
        scaled.apply_scale(.001)
        scene.add_geometry(scaled, node_name=f"neural-part-{i:03d}")
    scene.export(output/"source.glb")
    progress("构建装配方向与相互遮挡目标")
    problem, directions, anchor = direction_problem(parts, image_sha, settings.seed)
    dump(output/"problem.json", problem.as_dict())
    dump(output/"design.json", {"image_sha256": image_sha, "source_name": Path(source_name).name,
                                "settings": settings.model_dump(), "model": metadata,
                                "anchor_part": anchor, "direction_candidates": directions,
                                "geometric_slicing_performed": False})
    planner = settings.planner.model_copy(update={"encoding": "compact_binary"})
    progress("QB 采样装配展开方向" if planner.backend != "classical" else "经典对照选择装配方向")
    before = time.perf_counter()
    backend = make_backend(planner.backend, planner.max_qubits) if planner.backend != "classical" else None
    result = (QuantumStructurePrior(backend).plan(problem, planner, settings.seed) if backend is not None
              else solve_classical(problem, "cpsat", settings.seed, timeout_s=planner.timeout_s))
    planning_s = time.perf_counter()-before
    selected = next((c for c in result.candidates if c.feasible), None)
    if selected is None:
        dump(output/"trace.json", [{"round": 0, "problem": problem.as_dict(), "plan": result.as_dict()}])
        raise ValueError("本次测量没有产生有效装配方向，已保留原始神经部件")
    vectors = selected_vectors(problem, selected.bits, directions, len(parts))
    if result.circuit_qasm3:
        (output/"partition.qasm").write_text(result.circuit_qasm3)
    progress("采样检查所选方向的拆解次序")
    before = time.perf_counter()
    ids = [f"part-{i:03d}" for i in range(len(parts))]
    disassembly = selected_disassembly(parts, ids, vectors, anchor)
    disassembly_s = time.perf_counter()-before
    provenance = {"mode": planner.backend, "execution_source": result.execution_source,
                  "quantum_sdk": "QuantumBridge" if backend else None,
                  "quantum_role": "assembly_explosion_direction_selection",
                  "selected_measurement": {"bits": selected.bits, "job_id": selected.first_job_id,
                                           "observed_count": selected.count, "kind": "direct_or_exactly_decoded",
                                           "count_scope": "all_batches_in_this_round", "job_id_scope": "first batch observing the candidate"},
                  "generation_rounds": 1, "fallback_used": False, "hardware_verified": False,
                  "quantum_speedup_verified": False, "image_sha256": image_sha,
                  "geometry_is_inferred": True, "neural_parts_preserved": True,
                  "geometric_slicing_performed": False}
    dump(output/"trace.json", [{"round": 0, "problem": problem.as_dict(), "plan": result.as_dict(),
                                "selected_explosion_vectors_mm": vectors, "disassembly": disassembly}])
    labels = [f"生成部件 {i+1:02d}" for i in range(len(parts))]
    structure = export_parts(parts, output, provenance, disassembly, validation, labels,
                              explosion_vectors=vectors)
    summary = {"status": "generated", "kind": "image3d", "spec": problem.spec.model_dump(),
               "source_name": Path(source_name).name, "settings": settings.model_dump(),
               "validation": validation, "provenance": provenance, "model": metadata,
               "selected_bits": selected.bits, "decisions": problem.describe(selected.bits),
               "part_count": len(structure["parts"]), "resources": result.resources,
               "files": {"glb": "assembly.glb", "obj": "assembly.obj", "stl": "assembly.stl", "parts": "parts.zip",
                         "structure": "structure.json", "trace": "trace.json", "input": "input.png"},
               "disassembly": {"complete_for_generated_mesh": disassembly["complete_for_generated_mesh"],
                                "moves": len(disassembly["moves"]), "continuous_collision_proof": False},
               "timings": {"neural_generation_s": metadata["timings"]["total_s"], "planning_s": planning_s,
                           "disassembly_s": disassembly_s, "end_to_end_s": time.perf_counter()-started},
               "limitations": ["部件来自联合生成模型，未经真实拆件验证", "背面、内部与连接由单图推测",
                               "按部件配色，未恢复照片纹理", "采样路径不构成连续碰撞证明"]}
    dump(output/"run.json", summary)
    progress("完成")
    return summary


def _generate_geometric_image(data: bytes, settings: Image3DSettings, output: Path,
                               source_name: str, progress) -> dict:
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
                                           "observed_count": selected.count,
                                           "kind": "conditioned_composite" if result.resources.get("decomposed") else "direct_or_exactly_decoded",
                                           "count_scope": "selected_block_candidate" if result.resources.get("decomposed") else "all_batches_in_this_round",
                                           "job_id_scope": "first batch observing the candidate"},
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


def rename_run(source_directory: Path, labels: dict[str, str], output: Path,
               source_run_id: str, progress=lambda _: None) -> dict:
    structure = json.loads((source_directory/"structure.json").read_text())
    summary = json.loads((source_directory/"run.json").read_text())
    ids = {p["id"] for p in structure["parts"]}
    labels = {key: value.strip() for key, value in labels.items()}
    if not labels or not set(labels) <= ids or any(not value or len(value) > 64 for value in labels.values()):
        raise ValueError("部件名称需为 1–64 个字符，且部件必须属于当前模型")
    progress("保存部件名称，保留原始几何")
    shutil.copytree(source_directory, output, dirs_exist_ok=True)
    changes = []
    for part in structure["parts"]:
        if part["id"] in labels:
            changes.append({"part": part["id"], "old_label": part["label"], "new_label": labels[part["id"]]})
            part["label"] = labels[part["id"]]
    provenance = {**summary["provenance"], "source_run_id": source_run_id,
                  "original_run_id": summary.get("original_run_id", source_run_id),
                  "edit": "part_label_annotation", "annotation_source": "user_supplied_labels",
                  "geometry_modified": False, "quantum_resampling_performed": False}
    structure["provenance"] = provenance
    result = {**summary, "kind": "mesh_annotation", "source_run_id": source_run_id,
              "original_run_id": provenance["original_run_id"], "provenance": provenance}
    dump(output/"structure.json", structure)
    dump(output/"run.json", result)
    dump(output/"annotation.json", {"source_run_id": source_run_id, "changes": changes,
                                   "geometry_modified": False, "quantum_resampling_performed": False})
    return result


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
                  "edit": "classical_mesh_boolean_union", "merged_ids": selected_ids,
                  "neural_parts_preserved": False, "edit_directions_source": "classical_centroid_layout",
                  "quantum_resampling_performed": False}
    if provenance.get("quantum_role") == "assembly_explosion_direction_selection":
        provenance["quantum_role"] = "source_assembly_direction_selection"
    elif provenance.get("quantum_role") == "illustrative_layout_parameter_selection":
        provenance["quantum_role"] = "source_illustrative_layout_parameter_selection"
    appearance = [p.get("appearance", {}) for p in kept]+[{}]
    export_parts(parts, output, provenance, disassembly, validation, labels, members, appearance=appearance)
    for filename in ["input.png", "processed.png", "source.glb", "design.json", "problem.json", "trace.json", "container-fit.json", "demo.json", "thumbnail.png"]:
        if (source_directory/filename).is_file():
            shutil.copy2(source_directory/filename, output/filename)
    edit_s = time.perf_counter()-started
    result = {**summary, "kind": "mesh_edit", "part_count": len(parts), "validation": validation,
              "provenance": provenance, "source_run_id": source_run_id,
              "disassembly": {"complete_for_generated_mesh": disassembly["complete_for_generated_mesh"],
                              "moves": len(disassembly["moves"]), "continuous_collision_proof": False},
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
