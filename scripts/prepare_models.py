"""Download pinned public model files with resumable transfers and SHA-256 checks.

Uses the system TLS trust store through curl, retaining partial files on failure.
Uploaded images never participate in these network requests.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "triposr/model.ckpt": (
        "https://huggingface.co/stabilityai/TripoSR/resolve/5b521936b01fbe1890f6f9baed0254ab6351c04a/model.ckpt",
        "429e2c6b22a0923967459de24d67f05962b235f79cde6b032aa7ed2ffcd970ee"),
    "triposr/config.yaml": (
        "https://huggingface.co/stabilityai/TripoSR/resolve/5b521936b01fbe1890f6f9baed0254ab6351c04a/config.yaml",
        "74ca708ce086bf68e97709ea6b3d91f14717921c04691e84043f0eb8fcc68e62"),
    "dino/config.json": (
        "https://huggingface.co/facebook/dino-vitb16/resolve/f205d5d8e640a89a2b8ef0369670dfc37cc07fc2/config.json",
        "b87c0270b97db085fd82cf114a761fd0f62ae7914fbd407c752a2260646b689c"),
    "background/u2netp.onnx": (
        "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx",
        "309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8"),
}


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    manifest = {}
    for relative, (url, expected) in FILES.items():
        target = ROOT/"models"/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            partial = target.with_suffix(target.suffix+".part")
            for attempt in range(6):
                previous = partial.stat().st_size if partial.exists() else 0
                print(f"Downloading {relative}: resuming at {previous:,} bytes (attempt {attempt+1}/6)", flush=True)
                result = subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error", "--continue-at", "-",
                                         "--connect-timeout", "20", "--max-time", "1800", "--output", str(partial), url], check=False)
                if result.returncode == 0:
                    break
                current = partial.stat().st_size if partial.exists() else 0
                print(f"Transfer ended with curl={result.returncode}; retained {current:,} bytes", flush=True)
                if current <= previous and attempt >= 2:
                    raise RuntimeError(f"Download made no progress: {relative}")
            else:
                raise RuntimeError(f"Download attempt budget exhausted: {relative}")
            if sha256(partial) != expected:
                raise ValueError(f"Checksum mismatch: {relative}; partial file retained")
            partial.rename(target)
        actual = sha256(target)
        if actual != expected:
            raise ValueError(f"Existing model checksum mismatch: {relative}")
        manifest[relative] = {"url": url, "sha256": actual, "bytes": target.stat().st_size}
        print(f"Verified {relative}", flush=True)
    (ROOT/"models/manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Local image models ready; no uploaded image was sent to any service.")


if __name__ == "__main__":
    main()
