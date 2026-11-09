"""Offline preview by default. Execution requires explicit endpoint and budget."""
import argparse
import json
from pathlib import Path

from qamg.hardware import execute_bundle, verify_bundle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--max-shots", type=int)
    args = parser.parse_args()
    manifest = verify_bundle(args.bundle)
    if not args.execute:
        print(json.dumps({"status": "verified_offline_no_submission", "budget": manifest["budget"],
                          "registered_instances": len(manifest["instances"]), "hardware_validated": False}, indent=2))
        return
    if args.output is None or args.max_jobs is None or args.max_shots is None:
        parser.error("Execution requires --output, --max-jobs and --max-shots reflecting actual authorization")
    result = execute_bundle(args.bundle, args.output, args.max_jobs, args.max_shots)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
