"""Prepare the pinned, ungated SAM2.1 large model for local image segmentation."""
from __future__ import annotations

import json

from prepare_part_model import ROOT, digest, ranged_download


def main():
    record = json.loads((ROOT/"configs/vision-model.json").read_text())
    target = ROOT/"models/vision"/record["filename"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        ranged_download(target, record)
    if target.stat().st_size != record["bytes"] or digest(target) != record["sha256"]:
        raise ValueError("SAM2 model checksum mismatch")
    path = ROOT/"models/manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {}
    manifest["vision/"+record["filename"]] = record
    path.write_text(json.dumps(manifest, indent=2))
    print("Verified SAM2.1 large; local inference only.", flush=True)


if __name__ == "__main__":
    main()
