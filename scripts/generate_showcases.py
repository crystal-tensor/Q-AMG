"""Build curated authored examples with native QB layout measurement records."""
from __future__ import annotations

import argparse
from pathlib import Path

from qamg.benchmark import dump
from qamg.gallery import CATALOGUE
from qamg.showcase import generate_showcase


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/examples"))
    parser.add_argument("--only", choices=[e["slug"] for e in CATALOGUE])
    args = parser.parse_args()
    records = []
    for entry in CATALOGUE:
        if args.only and args.only != entry["slug"]:
            continue
        print(f"Building {entry['slug']} · {entry['title']}", flush=True)
        summary = generate_showcase(entry["slug"], args.output/entry["slug"])
        records.append({"id": entry["slug"], "part_count": summary["part_count"],
                        "valid": summary["validation"]["valid"], "quantum_sdk": "QuantumBridge",
                        "measured_bits": summary["selected_bits"], "seconds": summary["timings"]["end_to_end_s"],
                        "photo_reconstruction_performed": summary["provenance"]["photo_reconstruction_performed"], "official_product_cad": False})
        print(f"  {summary['part_count']} parts, valid={summary['validation']['valid']}, "
              f"{summary['timings']['end_to_end_s']:.2f}s", flush=True)
    receipt = f"showcases-{args.only}-manifest.json" if args.only else "showcases-manifest.json"
    dump(args.output/receipt, {"examples": records, "hardware_verified": False,
            "quantum_speedup_verified": False, "scope": "Authored reference assemblies; photo-guided cases explicitly marked, not OEM CAD or verified physical parts"})


if __name__ == "__main__":
    main()
