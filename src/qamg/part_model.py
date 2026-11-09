"""Part-aware image generation: jointly denoise independent parts, never slice a monolith."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from functools import lru_cache

import numpy as np

from .image_model import ROOT, prepare_image
from .torch_backend import available_gpu


def model_status() -> dict:
    manifest = json.loads((ROOT/"configs/partcrafter-files.json").read_text())
    total = sum(f["bytes"] for f in manifest["files"].values())
    downloaded, ready = 0, True
    for name, record in manifest["files"].items():
        path = ROOT/"models/partcrafter"/name
        partial = path.with_suffix(path.suffix+".part")
        complete = path.exists() and path.stat().st_size == record["bytes"]
        ready = ready and complete
        available = path.stat().st_size if path.exists() else partial.stat().st_size if partial.exists() else 0
        segments = path.with_suffix(path.suffix+".segments")
        if segments.exists():
            shard_sizes = {}
            for shard in segments.iterdir():
                name = shard.name.removesuffix(".part")
                if name.isdecimal():
                    shard_sizes[name] = max(shard_sizes.get(name, 0), shard.stat().st_size)
            available = max(available, sum(shard_sizes.values()))
        downloaded += min(record["bytes"], available)
    return {"model": "PartCrafter", "ready": ready, "downloaded_bytes": downloaded,
            "checkpoint_bytes": total, "local_inference": True,
            "geometry_scope": "直接联合生成多个部件；隐藏结构和制造连接仍需核验"}


@lru_cache(maxsize=1)
def _load_pipeline():
    import torch
    from transformers import BitImageProcessor, Dinov2Model

    if not model_status()["ready"]:
        raise FileNotFoundError("部件生成模型尚未准备完成，请运行 uv run python scripts/prepare_part_model.py")
    manifest = json.loads((ROOT/"configs/partcrafter-files.json").read_text())
    folder = ROOT/"models/partcrafter"
    for relative, record in manifest["files"].items():
        with (folder/relative).open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != record["sha256"]:
            raise ValueError(f"部件模型校验失败：{relative}")
    source = str(ROOT/"vendor/partcrafter")
    if source not in sys.path:
        sys.path.insert(0, source)
    from partcrafter.models.autoencoders import TripoSGVAEModel
    from partcrafter.models.transformers import PartCrafterDiTModel
    from partcrafter.pipelines.pipeline_partcrafter import PartCrafterPipeline

    scheduler = load_native_scheduler(folder)
    kwargs = {"torch_dtype": torch.float16, "local_files_only": True}
    pipeline = PartCrafterPipeline(
        vae=TripoSGVAEModel.from_pretrained(folder/"vae", **kwargs),
        transformer=PartCrafterDiTModel.from_pretrained(folder/"transformer", **kwargs),
        scheduler=scheduler,
        image_encoder_dinov2=Dinov2Model.from_pretrained(folder/"image_encoder_dinov2", **kwargs),
        feature_extractor_dinov2=BitImageProcessor.from_pretrained(folder/"feature_extractor_dinov2", local_files_only=True),
    )
    for model in [pipeline.vae, pipeline.transformer, pipeline.image_encoder_dinov2]:
        model.eval()
    return pipeline


def load_native_scheduler(folder):
    """The trained rectified flow uses the opposite update sign to generic SD3 flow."""
    source = str(ROOT/"vendor/partcrafter")
    if source not in sys.path:
        sys.path.insert(0, source)
    from partcrafter.schedulers.scheduling_rectified_flow import RectifiedFlowScheduler

    index = json.loads((folder/"model_index.json").read_text())
    config = json.loads((folder/"scheduler/scheduler_config.json").read_text())
    if index["scheduler"][-1] != "RectifiedFlowScheduler" or config["_class_name"] != "RectifiedFlowScheduler":
        raise ValueError("采样调度器与权重配置不匹配；拒绝替换为通用调度器")
    return RectifiedFlowScheduler.from_pretrained(folder/"scheduler", local_files_only=True)


def clean_neural_components(mesh):
    """Keep legitimate multi-body groups; remove only a small measured dust mass.

    A pair of robot arms can contain two equally large connected components.
    A largest-component-only rule wrongly rejects it. Preserve the raw output
    separately and reject distributed fragment clouds by dominant volume mass.
    """
    import trimesh

    components = mesh.split()
    volumes = np.array([abs(c.volume) for c in components])
    total = max(float(volumes.sum()), 1e-12)
    dominant = volumes >= total*.01
    mass = float(volumes[dominant].sum()/total)
    record = {"components_before": len(components), "component_abs_volumes": volumes.tolist(),
              "dominant_threshold_fraction": .01, "dominant_volume_fraction": mass,
              "removed_volume_fraction": 0., "components_after": len(components)}
    if len(components) > 8 and mass < .98:
        raise ValueError(f"出现大量分散碎片（{len(components)} 个连通体），主体体积不足 98%；原始输出已保留")
    if len(components) > 1 and mass >= .98 and not dominant.all():
        mesh = trimesh.util.concatenate([c for c, keep in zip(components, dominant, strict=True) if keep])
        record.update({"removed_volume_fraction": 1-mass, "components_after": int(dominant.sum())})
    return mesh, record


def generate_parts(image, settings, progress=lambda _: None, *, steps: int = 50, tokens: int = 1024,
                   artifact_directory=None):
    import torch
    import trimesh
    from skimage.measure import marching_cubes

    started = time.perf_counter()
    progress("清理主体前景")
    prepared, preprocessing = prepare_image(image, settings, clean_foreground=True, background=1.)
    if artifact_directory is not None:
        prepared.save(artifact_directory/"processed.png")
    preprocessed = time.perf_counter()
    progress("校验并载入部件感知模型")
    pipeline = _load_pipeline()
    device = available_gpu() if settings.device != "cpu" else None
    if settings.device == "gpu" and not device:
        raise ValueError("当前没有可用 GPU，请选择自动或 CPU")
    device = device or "cpu"
    dtype = torch.float32 if device == "cpu" else torch.float16
    pipeline.to(device=device, dtype=dtype)
    # MPS fp16 DINO features failed the independent CPU comparison on this host.
    # Keep the visual encoder in fp32; denoising and decoding retain fp16 on GPU.
    pipeline.image_encoder_dinov2.to(dtype=torch.float32)
    loaded = time.perf_counter()

    def on_step(_pipe, index, timestep, tensors):
        progress(f"联合生成 {settings.desired_parts} 个部件 · {index+1}/{steps} · {device.upper()}")
        return {}

    with torch.inference_mode():
        latents = pipeline(
            image=[prepared]*settings.desired_parts,
            attention_kwargs={"num_parts": settings.desired_parts},
            num_tokens=tokens, generator=torch.Generator(device="cpu").manual_seed(settings.seed),
            num_inference_steps=steps, guidance_scale=7., use_flash_decoder=False,
            output_type="latent", callback_on_step_end=on_step,
        ).samples
        if artifact_directory is not None:
            np.save(artifact_directory/"neural-latents.npy", latents.float().cpu().numpy())
        denoised = time.perf_counter()
        meshes, component_reviews = [], []
        resolution = settings.resolution
        axis = torch.linspace(-1.005, 1.005, resolution, device=device, dtype=dtype)
        points = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1).reshape(1, -1, 3)
        for i, latent in enumerate(latents):
            progress(f"提取独立部件表面 · {i+1}/{len(latents)}")
            values = pipeline.vae.decode(latent.unsqueeze(0), sampled_points=points, num_chunks=4096,
                                         to_cpu=True).sample.float().numpy().reshape((resolution,)*3)
            if not np.isfinite(values).all() or values.min() >= 0 or values.max() <= 0:
                raise ValueError(f"部件 {i+1} 没有形成可用表面；未用占位部件代替")
            # Outside of the learned SDF volume is positive; retain every generated part.
            padded = np.pad(values, 1, constant_values=max(1., float(values.max())))
            vertices, faces, _, _ = marching_cubes(padded, 0, allow_degenerate=False)
            vertices = ((vertices-1)/(resolution-1)*2-1)*1.005
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
            # Preserve cavity orientation; a global inversion does not fill inner shells.
            if mesh.volume < 0:
                mesh.invert()
            mesh.remove_unreferenced_vertices()
            if artifact_directory is not None:
                raw = artifact_directory/"raw-neural-parts"
                raw.mkdir(exist_ok=True)
                mesh.export(raw/f"part-{i:03d}.ply")
            if not mesh.is_watertight or mesh.volume <= 0:
                raise ValueError(f"部件 {i+1} 不是有效封闭实体，拒绝伪造成功结果")
            mesh, review = clean_neural_components(mesh)
            component_reviews.append(review)
            meshes.append(mesh)
    vertices = np.concatenate([m.vertices for m in meshes])
    bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
    scale = settings.size_mm/float(np.ptp(vertices, axis=0).max())
    palette = ["#d6bc75", "#86a7aa", "#bfb297", "#899f79", "#c99887", "#989ac0"]
    for i, mesh in enumerate(meshes):
        mesh.apply_translation(-bounds.mean(axis=0))
        mesh.apply_scale(scale)
        # Upstream uses Y up; the Q-AMG viewer uses Z up. This is a proper rotation.
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0]))
        mesh.visual.vertex_colors = trimesh.visual.color.hex_to_rgba(palette[i % len(palette)])
    finished = time.perf_counter()
    manifest = json.loads((ROOT/"configs/partcrafter-files.json").read_text())
    metadata = {"model": "PartCrafter", "model_revision": manifest["revision"], "device": device,
                "scheduler": type(pipeline.scheduler).__name__, "native_scheduler_manifest_verified": True,
                "requested_device": settings.device, "dtype": str(dtype), "preprocessing": preprocessing,
                "image_encoder_dtype": str(pipeline.image_encoder_dinov2.dtype),
                "denoiser_dtype": str(pipeline.transformer.dtype),
                "neural_parts_generated_jointly": True, "geometric_slicing_performed": False,
                "num_inference_steps": steps, "tokens_per_part": tokens,
                "sampling_resolution": resolution, "size_mm": settings.size_mm,
                "scale_is_user_assumption": True, "appearance": "part_identity_palette_not_photo_texture",
                "part_count": len(meshes), "latent_sha256": hashlib.sha256(latents.float().cpu().numpy().tobytes()).hexdigest(),
                "component_cleanup": component_reviews, "raw_neural_output_preserved": True,
                "timings": {"preprocessing_s": preprocessed-started, "model_load_and_upload_s": loaded-preprocessed,
                            "joint_denoising_s": denoised-loaded, "surface_extraction_s": finished-denoised,
                            "total_s": finished-started}}
    return meshes, prepared, metadata
