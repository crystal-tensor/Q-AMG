import json

import pytest

from qamg.benchmark import create_corpus, run_benchmark, summarize
from qamg.schema import PlannerSettings


def test_corpus_split_has_30_heldout_and_three_scales(tmp_path):
    corpus = create_corpus(tmp_path/"corpus.json")
    heldout = [i for i in corpus["instances"] if i["split"] == "heldout"]
    assert len(heldout) == 30
    assert {i["spec"]["groups"] for i in heldout} == {3, 4, 5}
    assert len({json.dumps(i["spec"]["cost_overrides"], sort_keys=True) for i in heldout}) == 30
    assert not {i["id"] for i in heldout} & {i["id"] for i in corpus["instances"] if i["split"] == "train"}


def test_smoke_benchmark_keeps_failure_semantics_and_snapshots(tmp_path):
    summary = run_benchmark(tmp_path/"bench", PlannerSettings(iterations=0, shots=64),
                            instances=2, seeds=1, geometry=False,
                            methods=["qaoa_binary", "exact"])
    assert summary["record_count"] == 4
    assert summary["hardware_validated"] is False
    assert summary["quantum_acceleration_target_met"] is False
    assert summary["geometry_validation_is_partial"] is True
    manifest = json.loads((tmp_path/"bench/manifest.json").read_text())
    assert "src/qamg/quantum.py" in manifest["source_sha256"]
    with pytest.raises(FileExistsError):
        run_benchmark(tmp_path/"bench", instances=1, seeds=1, geometry=False)


def test_failed_runs_stay_in_success_rate():
    records = [{"method": "exact", "instance": "one", "time_to_target_s": .1, "wall_s": .11},
               {"method": "exact", "instance": "one", "time_to_target_s": None, "wall_s": 1.0}]
    result = summarize(records)
    assert result["methods"]["exact"]["success_rate"] == .5
    assert result["methods"]["exact"]["failure_count"] == 1
