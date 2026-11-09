"""Three registered engineering rounds; CPU/GPU simulator results are not hardware advantage."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT/"artifacts"/"benchmarks"/datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S")
    output = output.resolve()
    if output.exists():
        raise SystemExit("Choose a new output directory to preserve the experiment record")
    output.mkdir(parents=True)
    corpus = output/"corpus.json"
    registration = {
        "instances": 30, "seeds": 5, "split": "heldout", "hardware_claim": False,
        "rounds": [
            {"name": "round-1-xy", "change": "Reference one-hot, feasible-subspace QAOA", "depth": 2, "iterations": 12},
            {"name": "round-2-binary", "change": "Exact binary substitution removes redundant qubits and XY gates", "depth": 2, "iterations": 12},
            {"name": "round-3-transfer", "change": "Depth-1 context angle transfer trained only on train; training amortized over 150 requests", "depth": 1, "iterations": 0},
        ],
        "note": "Registered before this reproduction. The development pilot established encoding and driver overhead as bottlenecks."
    }
    (output/"registration.json").write_text(json.dumps(registration, indent=2))
    prefix = [sys.executable, "-m", "qamg.cli"]

    def run(*arguments):
        subprocess.run(prefix+list(map(str, arguments)), cwd=ROOT, check=True)

    common = ["--instances", "30", "--seeds", "5", "--corpus", corpus]
    classical = ["exact", "cpsat", "cpsat_qubo", "anneal", "factorized"]
    run("benchmark", "--output", output/"round-1-xy", *common,
        "--methods", "qaoa_xy", *classical)
    run("benchmark", "--output", output/"round-2-binary", *common,
        "--methods", "qaoa_binary", *classical)
    run("train-prior", "--output", output/"transfer.json", "--corpus", corpus)
    run("benchmark", "--output", output/"round-3-transfer", *common, "--depth", "1",
        "--iterations", "0", "--warm-start-file", output/"transfer.json",
        "--methods", "qaoa_binary", *classical)
    print(f"Reproduction saved to: {output}")


if __name__ == "__main__":
    main()
