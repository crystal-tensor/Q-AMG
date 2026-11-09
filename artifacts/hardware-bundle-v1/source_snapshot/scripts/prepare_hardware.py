"""Create a fully offline logical circuit/budget bundle; no endpoint is contacted."""
import argparse
import json
from pathlib import Path

from qmech.hardware import prepare_bundle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--transfer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare_bundle(args.output, args.corpus, args.transfer)
    print(json.dumps({"status": result["status"], "budget": result["budget"],
                      "hardware_requirements": result["hardware_requirements"]}, indent=2))


if __name__ == "__main__":
    main()
