"""Re-read full acceptance outputs, quantum witnesses and archived benchmark sources."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from qamg.benchmark import dump


def main():
    root = Path("artifacts")
    acceptance = root/"acceptance/v3-gpu"
    rows = [json.loads(line) for line in (acceptance/"raw.jsonl").read_text().splitlines()]
    quantum = [r for r in rows if r["mode"] == "auto"]
    assert len(rows) == 300 and len(quantum) == 150
    assert all(r["valid"] for r in rows)
    witnesses = 0
    for row in quantum:
        directory = Path(row["output"])
        summary = json.loads((directory/"run.json").read_text())
        structure = json.loads((directory/"structure.json").read_text())
        trace = json.loads((directory/"trace.json").read_text())[-1]["plan"]["trace"]
        selected = summary["provenance"]["selected_measurement"]
        bits = selected["bits"]
        assert structure["selected_bits"] == bits
        matching = [t for t in trace if t["job_id"] == selected["job_id"]]
        assert len(matching) == 1 and matching[0]["counts"].get(bits, 0) > 0
        physical = "".join(pair[1] for pair in [bits[i:i+2] for i in range(0, len(bits), 2)])
        assert matching[0]["raw_measurement_counts"].get(physical, 0) > 0
        assert sum(t["counts"].get(bits, 0) for t in trace) == selected["observed_count"]
        assert all(t["metadata"]["state_evolution_device"] == "mps" for t in trace)
        assert summary["provenance"]["fallback_used"] is False
        assert summary["provenance"]["quantum_speedup_verified"] is False
        assert row["step_solids"] == row["glb_meshes"] == row["part_count"]
        witnesses += 1
    archived = 0
    for manifest in sorted((root/"benchmarks/registered-v2").glob("round-*/manifest.json")):
        d = json.loads(manifest.read_text())
        for path, digest in d["source_sha256"].items():
            assert hashlib.sha256((manifest.parent/"source_snapshot"/path).read_bytes()).hexdigest() == digest
            archived += 1
        records = [json.loads(line) for line in (manifest.parent/"raw.jsonl").read_text().splitlines()]
        assert len(records) == 900
        assert len({r["instance"] for r in records}) == 30
        assert len({r["seed"] for r in records}) == 5
        assert all(r.get("geometry", {}).get("valid") for r in records)
    report = {"full_artifact_runs": len(rows), "valid_runs": sum(r["valid"] for r in rows),
              "gpu_measurement_witnesses_checked": witnesses,
              "archived_source_hashes_checked": archived,
              "hardware_verified": False, "quantum_speedup_verified": False,
              "whole_goal_complete": False,
              "remaining": ["Company superconducting QPU connection and calibration", "Physical quantum acceleration >=1.2 with CI lower bound >1"]}
    dump(root/"delivery-audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
