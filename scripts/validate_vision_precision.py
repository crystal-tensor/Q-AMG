"""Compare visual encoder features with an independent CPU fp32 reference."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from PIL import Image
from transformers import BitImageProcessor, Dinov2Model

from qamg.benchmark import dump
from qamg.part_model import ROOT
from qamg.torch_backend import available_gpu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    device = available_gpu()
    if not device:
        raise RuntimeError("This validation needs a GPU; CPU reference alone cannot establish GPU correctness")
    torch.set_num_threads(8)
    model = Dinov2Model.from_pretrained(ROOT/"models/partcrafter/image_encoder_dinov2",
                                       local_files_only=True, torch_dtype=torch.float32).eval()
    processor = BitImageProcessor.from_pretrained(ROOT/"models/partcrafter/feature_extractor_dinov2",
                                                  local_files_only=True)
    public_image = ROOT/"vendor/partcrafter/examples/robot.png"
    yy, xx = np.indices((224, 224))
    checker = np.uint8(np.stack([(xx*7) % 256, (yy*5) % 256, ((xx//16+yy//16) % 2)*255], axis=-1))
    sources = [("public_robot", Image.open(public_image).convert("RGB")),
               ("synthetic_pattern", Image.fromarray(checker))]
    records = []
    with torch.inference_mode():
        for name, image in sources:
            model.to(device="cpu", dtype=torch.float32)
            pixels = processor(image, return_tensors="pt").pixel_values
            reference = model(pixels).last_hidden_state.cpu()
            for dtype in [torch.float16, torch.float32]:
                model.to(device=device, dtype=dtype)
                before = time.perf_counter()
                value = model(pixels.to(device=device, dtype=dtype)).last_hidden_state.float().cpu()
                error = float((reference-value).norm()/reference.norm())
                records.append({"input": name, "gpu": device, "dtype": str(dtype),
                                "relative_l2_error": error, "cpu_reference_norm": float(reference.norm()),
                                "gpu_norm": float(value.norm()), "elapsed_s": time.perf_counter()-before,
                                "accuracy_threshold": 1e-3, "passed": error < 1e-3})
    result = {"torch": torch.__version__, "transformers": transformers.__version__,
              "public_image_sha256": hashlib.sha256(public_image.read_bytes()).hexdigest(),
              "records": records,
              "selected_fp32_encoder_passed": all(r["passed"] for r in records if r["dtype"] == "torch.float32"),
              "scope": "Feature correctness on two public/synthetic inputs; timings are diagnostic, not a speed benchmark"}
    dump(args.output/"result.json", result)
    if not result["selected_fp32_encoder_passed"]:
        raise RuntimeError("Selected visual encoder precision failed CPU reference comparison")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
