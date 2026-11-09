"""Download the pinned part-aware image model; no user image leaves this machine."""
from __future__ import annotations

import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def ranged_download(target: Path, record: dict):
    """Bounded independent HTTP ranges avoid one stalled multi-GB connection."""
    size, block = record["bytes"], 32*1024*1024
    directory = target.with_suffix(target.suffix+".segments")
    directory.mkdir(exist_ok=True)
    partial = target.with_suffix(target.suffix+".part")
    if partial.exists():
        with partial.open("rb") as source:
            for i, start in enumerate(range(0, size, block)):
                expected = min(block, size-start)
                chunk = source.read(expected)
                if len(chunk) != expected:
                    break
                file = directory/f"{i:04d}"
                if not file.exists():
                    file.write_bytes(chunk)

    def transfer(i):
        start, end = i*block, min((i+1)*block, size)-1
        expected = end-start+1
        destination = directory/f"{i:04d}"
        if destination.exists() and destination.stat().st_size == expected:
            return
        temporary, headers = directory/f"{i:04d}.part", directory/f"{i:04d}.headers"
        for attempt in range(6):
            result = subprocess.run([
                "curl", "--fail", "--location", "--silent", "--show-error", "--range", f"{start}-{end}",
                "--connect-timeout", "20", "--max-time", "300", "--speed-limit", "65536", "--speed-time", "30",
                "--max-filesize", str(expected), "--dump-header", str(headers),
                "--output", str(temporary), record["url"],
            ], check=False)
            response = headers.read_text().lower() if headers.exists() else ""
            correct_range = f"content-range: bytes {start}-{end}/{size}" in response
            if result.returncode == 0 and temporary.stat().st_size == expected and correct_range:
                temporary.rename(destination)
                return
            print(f"Retry range {target.name} {i}: curl={result.returncode}, attempt={attempt+1}", flush=True)
        raise RuntimeError(f"Range download failed: {target.name}/{i}; valid ranges retained")

    count = (size+block-1)//block
    with ThreadPoolExecutor(max_workers=8) as pool:
        for completed, future in enumerate(as_completed([pool.submit(transfer, i) for i in range(count)]), 1):
            future.result()
            if completed % 4 == 0 or completed == count:
                print(f"Model ranges {target.parent.name}: {completed}/{count}", flush=True)
    with partial.open("wb") as output:
        for i in range(count):
            with (directory/f"{i:04d}").open("rb") as source:
                while chunk := source.read(1024*1024):
                    output.write(chunk)
    if digest(partial) != record["sha256"]:
        raise ValueError(f"Assembled model checksum mismatch: {target.name}; ranges retained")
    partial.rename(target)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(item):
    relative, record = item
    target = ROOT/"models/partcrafter"/relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() and record["bytes"] > 256*1024*1024:
        ranged_download(target, record)
    if not target.exists():
        partial = target.with_suffix(target.suffix+".part")
        for attempt in range(6):
            before = partial.stat().st_size if partial.exists() else 0
            print(f"Downloading {relative}: {before:,} bytes, attempt {attempt+1}/6", flush=True)
            result = subprocess.run([
                "curl", "--fail", "--location", "--silent", "--show-error", "--continue-at", "-",
                "--connect-timeout", "20", "--max-time", "1800", "--speed-limit", "65536", "--speed-time", "30",
                "--output", str(partial), record["url"],
            ], check=False)
            if result.returncode == 0:
                break
            after = partial.stat().st_size if partial.exists() else 0
            if after <= before and attempt >= 2:
                raise RuntimeError(f"Download made no progress: {relative}; partial file retained")
        else:
            raise RuntimeError(f"Download budget exhausted: {relative}; partial file retained")
        if digest(partial) != record["sha256"]:
            raise ValueError(f"Checksum mismatch: {relative}; partial file retained")
        partial.rename(target)
    if target.stat().st_size != record["bytes"] or digest(target) != record["sha256"]:
        raise ValueError(f"Existing model checksum mismatch: {relative}")
    print(f"Verified {relative}", flush=True)
    return "partcrafter/"+relative, record


def main():
    source = json.loads((ROOT/"configs/partcrafter-files.json").read_text())
    records = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(prepare, item) for item in source["files"].items()]):
            name, record = future.result()
            records[name] = record
    path = ROOT/"models/manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {}
    manifest.update(records)
    path.write_text(json.dumps(manifest, indent=2))
    print("Part-aware model ready; images remain local.", flush=True)


if __name__ == "__main__":
    main()
