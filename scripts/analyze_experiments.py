"""Recompute statistics without changing or rerunning the original experiments."""
import argparse
import json
import shutil
from pathlib import Path

from qamg.analysis import analyze_experiment
from qamg.benchmark import dump, snapshot_sources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Choose a new output directory")
    snapshot_sources(args.output)
    shutil.copy2(__file__, args.output/"source_snapshot"/"analyze_experiments.py")
    dump(args.output/"analysis-policy.json", {
        "kind": "posthoc_independent_reanalysis_of_immutable_records_not_an_optimization_round",
        "registered_population": "Original manifest instances x methods x seeds, no exclusions",
        "bootstrap_repeats": 4000, "bootstrap_seed": 20261109,
        "no_solver_or_hardware_calls": True,
    })
    for name in ["round-1-xy", "round-2-binary", "round-3-transfer"]:
        result = analyze_experiment(args.experiments/name, args.experiments/"corpus.json", args.output/f"{name}.json")
        print(json.dumps({"round": name, "status": result["review_status"],
                          "registered_slots": result["registered_slots"], "missing": result["missing_rows"]}))


if __name__ == "__main__":
    main()
