"""Local SAM2.1 large foreground refinement, guided by an explicit coarse mask."""
from __future__ import annotations

import hashlib
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

from .torch_backend import available_gpu

ROOT = Path(__file__).resolve().parents[2]


def model_ready() -> bool:
    record = json.loads((ROOT/"configs/vision-model.json").read_text())
    path = ROOT/"models/vision"/record["filename"]
    return path.exists() and path.stat().st_size == record["bytes"]


@lru_cache(maxsize=1)
def predictor(device: str):
    record = json.loads((ROOT/"configs/vision-model.json").read_text())
    checkpoint = ROOT/"models/vision"/record["filename"]
    with checkpoint.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != record["sha256"]:
            raise ValueError("SAM2 分割模型校验失败")
    source = str(ROOT/"vendor/sam2")
    if source not in sys.path:
        sys.path.insert(0, source)
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model = build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", str(checkpoint),
                      device=device, apply_postprocessing=False)
    return SAM2ImagePredictor(model)


def refine_foreground(image, coarse_mask: np.ndarray, requested_device: str = "auto"):
    import torch

    yy, xx = np.where(coarse_mask)
    if len(xx) < 16:
        raise ValueError("粗前景不足，无法构建视觉分割提示")
    box = np.array([xx.min(), yy.min(), xx.max(), yy.max()], dtype=np.float32)
    positive = []
    for fraction in [.2, .5, .8]:
        row = int(np.quantile(yy, fraction))
        columns = np.flatnonzero(coarse_mask[row])
        if len(columns):
            column = int(columns[np.argmin(np.abs(columns-np.median(columns)))])
            positive.append([column, row])
    height, width = coarse_mask.shape
    negative = [[x, y] for x, y in [[0, 0], [width-1, 0], [0, height-1], [width-1, height-1]]
                if not coarse_mask[y, x]]
    points = np.array(positive+negative, dtype=np.float32)
    labels = np.array([1]*len(positive)+[0]*len(negative))
    device = (available_gpu() if requested_device != "cpu" else None) or "cpu"
    fallback = None

    def infer(selected_device):
        model = predictor(selected_device)
        with torch.inference_mode():
            model.set_image(np.array(image.convert("RGB"), copy=True))
            masks, scores, _ = model.predict(point_coords=points, point_labels=labels,
                                             box=box, multimask_output=True)
        return masks, scores

    try:
        masks, scores = infer(device)
    except RuntimeError as error:
        if device == "cpu" or requested_device == "gpu":
            raise
        fallback = f"{type(error).__name__}: {str(error)[:200]}"
        device = "cpu"
        masks, scores = infer(device)
    overlap = [float(np.logical_and(mask, coarse_mask).sum()/max(1, np.logical_or(mask, coarse_mask).sum()))
               for mask in masks]
    chosen = int(np.argmax(overlap))
    if masks[chosen].sum() < 16:
        raise ValueError("分割模型没有生成可用主体轮廓")
    return masks[chosen].astype(np.float32), {
        "model": "SAM2.1-hiera-large", "device": device, "cpu_fallback_reason": fallback,
        "prompt_source": "coarse_foreground_box_and_interior_points",
        "prompt_box_xyxy": box.tolist(), "prompt_points_xy": points.tolist(),
        "prompt_labels": labels.tolist(), "candidate_model_scores": np.asarray(scores).tolist(),
        "candidate_coarse_mask_iou": overlap, "selected_candidate": chosen,
        "semantic_part_identity_verified": False,
    }
