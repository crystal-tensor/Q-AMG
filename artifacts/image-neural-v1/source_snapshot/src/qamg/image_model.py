"""Local single-image neural 3D generation, with explicit inferred geometry.

TripoSR weights/source are pinned in vendor/SOURCES.json and models/manifest.json.
No image is sent to an external API. Units and invisible surfaces are inferred.
"""
from __future__ import annotations

import hashlib
import io
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .schema import Image3DSettings

ROOT = Path(__file__).resolve().parents[2]
MODEL_SHA256 = "429e2c6b22a0923967459de24d67f05962b235f79cde6b032aa7ed2ffcd970ee"
MAX_IMAGE_BYTES = 20*1024*1024


def read_image(data: bytes) -> Image.Image:
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise ValueError("请选择不超过 20 MB 的图片")
    try:
        image = Image.open(io.BytesIO(data))
        if image.width*image.height > 40_000_000 or min(image.size) < 16:
            raise ValueError("图片需至少 16×16 像素，总像素不超过 4000 万")
        image.seek(0)
        image = ImageOps.exif_transpose(image).convert("RGBA")
        image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
        return image
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError("无法读取图片，请使用 PNG、JPEG、WEBP、BMP 或 TIFF") from error


def model_status() -> dict:
    folder = ROOT/"models"
    checkpoint = folder/"triposr/model.ckpt"
    partial = folder/"triposr/model.ckpt.part"
    ready = all((folder/name).is_file() for name in
                ["triposr/model.ckpt", "triposr/config.yaml", "dino/config.json", "background/u2netp.onnx"])
    return {"model": "TripoSR", "ready": ready,
            "downloaded_bytes": checkpoint.stat().st_size if checkpoint.exists() else partial.stat().st_size if partial.exists() else 0,
            "checkpoint_bytes": 1677246742, "local_inference": True,
            "geometry_scope": "单图生成的外观与候选分件；背面、内部和真实尺寸存在不确定性"}


@lru_cache(maxsize=1)
def _background_session():
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    return ort.InferenceSession(str(ROOT/"models/background/u2netp.onnx"), sess_options=options,
                               providers=["CPUExecutionProvider"])


def prepare_image(image: Image.Image, settings: Image3DSettings) -> tuple[Image.Image, dict]:
    rgba = np.asarray(image)
    alpha = rgba[..., 3].astype(np.float32)/255
    method = "input_alpha"
    if settings.remove_background and np.min(alpha) > .99:
        session = _background_session()
        rgb = np.asarray(image.convert("RGB").resize((320, 320), Image.Resampling.LANCZOS), dtype=np.float32)
        rgb /= max(float(rgb.max()), 1)
        rgb = (rgb-np.array([.485, .456, .406], dtype=np.float32))/np.array([.229, .224, .225], dtype=np.float32)
        result = session.run(None, {session.get_inputs()[0].name: rgb.transpose(2, 0, 1)[None]})[0][0, 0]
        result = (result-result.min())/max(float(result.max()-result.min()), 1e-6)
        mask = Image.fromarray(np.uint8(result*255)).resize(image.size, Image.Resampling.LANCZOS)
        alpha = np.asarray(mask, dtype=np.float32)/255
        method = "u2netp_foreground_estimate"
    elif not settings.remove_background:
        method = "full_image_with_input_alpha"
    occupied = alpha > .1
    if occupied.sum() < 16:
        raise ValueError("未找到足够的前景，请关闭背景移除或使用主体更清晰的图片")
    yy, xx = np.where(occupied)
    bbox = (int(xx.min()), int(yy.min()), int(xx.max()+1), int(yy.max()+1))
    rgb = rgba[..., :3].astype(np.float32)/255
    composited = rgb*alpha[..., None]+.5*(1-alpha[..., None])
    cropped = Image.fromarray(np.uint8(np.clip(composited, 0, 1)*255)).crop(bbox)
    side = max(cropped.size)
    canvas_side = max(side, round(side/settings.foreground_ratio))
    canvas = Image.new("RGB", (canvas_side, canvas_side), (128, 128, 128))
    canvas.paste(cropped, ((canvas_side-cropped.width)//2, (canvas_side-cropped.height)//2))
    return canvas, {"method": method, "foreground_bbox_px": list(bbox),
                    "foreground_fraction": float(occupied.mean()), "input_size_px": list(image.size)}


@lru_cache(maxsize=1)
def _load_model():
    import torch
    from omegaconf import OmegaConf
    folder = ROOT/"models"
    if not model_status()["ready"]:
        raise FileNotFoundError("图片模型尚未准备完成，请运行 uv run python scripts/prepare_models.py")
    checkpoint = folder/"triposr/model.ckpt"
    with checkpoint.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    if sha != MODEL_SHA256:
        raise ValueError("TripoSR 权重校验失败，请重新下载模型")
    source = str(ROOT/"vendor/triposr")
    if source not in sys.path:
        sys.path.insert(0, source)
    from tsr.system import TSR
    cfg = OmegaConf.load(folder/"triposr/config.yaml")
    cfg.image_tokenizer.pretrained_model_name_or_path = str(folder/"dino")
    model = TSR(cfg)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    model.renderer.set_chunk_size(8192)
    return model


def generate_mesh(image: Image.Image, settings: Image3DSettings, progress=lambda _: None):
    import torch
    import trimesh
    from skimage.measure import marching_cubes

    from .torch_backend import available_gpu
    started = time.perf_counter()
    progress("提取前景")
    prepared, preprocessing = prepare_image(image, settings)
    preprocessed = time.perf_counter()
    progress("载入单图 3D 模型")
    model = _load_model()
    requested = available_gpu() if settings.device in {"auto", "gpu"} else None
    if settings.device == "gpu" and not requested:
        raise ValueError("当前没有可用 GPU，请选择自动或 CPU")
    device = requested or "cpu"
    model.to(device)
    loaded = time.perf_counter()
    progress(f"生成三维特征 · {device.upper()}")
    fallback = None
    try:
        with torch.inference_mode():
            scene_code = model([prepared], device=device)[0]
    except RuntimeError as error:
        if device == "cpu" or settings.device == "gpu":
            raise
        fallback = f"{type(error).__name__}: {str(error)[:200]}"
        device = "cpu"
        model.to(device)
        progress("GPU 运算未完成，改用 CPU 继续同一模型")
        with torch.inference_mode():
            scene_code = model([prepared], device=device)[0]
    encoded = time.perf_counter()
    progress("提取三维表面")
    resolution = settings.resolution
    radius = float(model.renderer.cfg.radius)
    axis = torch.linspace(-radius, radius, resolution, device=device)
    points = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1).reshape(-1, 3)
    with torch.inference_mode():
        density = model.renderer.query_triplane(model.decoder, points, scene_code)["density_act"]
    grid = density.float().cpu().numpy().reshape((resolution,)*3)
    if not np.isfinite(grid).all() or float(grid.max()) <= settings.density_threshold:
        raise ValueError("此输入没有产生可提取的三维表面，请调整图片、前景或密度阈值")
    # Padding closes surfaces clipped by the sampled bounding box, retained in metadata.
    padded = np.pad(grid, 1, constant_values=0)
    vertices, faces, _, _ = marching_cubes(padded, settings.density_threshold, allow_degenerate=False)
    vertices = ((vertices-1)/(resolution-1)*2-1)*radius
    progress("生成表面颜色")
    with torch.inference_mode():
        colors = model.renderer.query_triplane(model.decoder, torch.as_tensor(vertices.copy(), device=device), scene_code)["color"]
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors.float().cpu().numpy(), process=True)
    mesh.fix_normals(multibody=True)
    mesh.remove_unreferenced_vertices()
    extent = float(np.max(mesh.extents))
    if not np.isfinite(extent) or extent <= 1e-8:
        raise ValueError("生成模型的尺寸异常")
    mesh.apply_translation(-mesh.bounds.mean(axis=0))
    mesh.apply_scale(settings.size_mm/extent)
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()
    complete = time.perf_counter()
    metadata = {"model": "TripoSR", "checkpoint_sha256": MODEL_SHA256,
                "device": device, "requested_device": settings.device, "cpu_fallback_reason": fallback,
                "preprocessing": preprocessing, "sampling_resolution": resolution,
                "density_threshold": settings.density_threshold, "bounding_box_padding_voxels": 1,
                "size_mm": settings.size_mm, "scale_is_user_assumption": True,
                "vertices": len(mesh.vertices), "faces": len(mesh.faces), "watertight": bool(mesh.is_watertight),
                "timings": {"preprocessing_s": preprocessed-started, "model_load_and_upload_s": loaded-preprocessed,
                            "image_encoding_s": encoded-loaded, "surface_color_and_scale_s": complete-encoded,
                            "total_s": complete-started}}
    return mesh, prepared, metadata
