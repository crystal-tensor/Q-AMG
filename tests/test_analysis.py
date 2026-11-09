import copy

import pytest

from qamg.analysis import reanalyze, wilson
from qamg.benchmark import create_corpus
from qamg.problem import build_problem
from qamg.schema import DesignSpec


def fixture(tmp_path):
    corpus = create_corpus(tmp_path/"corpus.json")
    cases = [c for c in corpus["instances"] if c["split"] == "heldout"][:2]
    manifest = {"instances": [c["id"] for c in cases], "methods": ["qaoa_binary", "exact"],
                "seeds": 2, "geometry": False, "budget": {"timeout_s_per_method": 10},
                "target": {"relative_gap": .05}}
    records = []
    for case in cases:
        p = build_problem(DesignSpec.model_validate(case["spec"]))
        selection = min((s for s in p.assignments() if not p.violations(s)), key=p.cost)
        optimum = p.cost(selection)
        for method in manifest["methods"]:
            for seed in [1000, 1001]:
                records.append({"instance": case["id"], "method": method, "seed": seed,
                                "bits": p.bits(selection), "relative_gap": 0, "optimum": optimum,
                                "threshold": optimum+.05*max(1, abs(optimum)), "status": "target_reached",
                                "time_to_target_s": 1 if method.startswith("qaoa") else .1, "wall_s": 1,
                                "trace": [{"counts": {p.bits(selection): 32}}]})
    return records, manifest, corpus


def test_failures_and_missing_slots_cannot_disappear_from_speedups(tmp_path):
    records, manifest, corpus = fixture(tmp_path)
    records[0].update(status="error", time_to_target_s=None)
    records.pop(1)  # Another registered quantum run is missing, not silently dropped.
    result = reanalyze(records, manifest, corpus, bootstrap_repeats=100)
    assert result["registered_slots"] == 8 and result["missing_rows"] == 1
    q = result["methods"]["qaoa_binary"]
    assert q["success_rate"] == .5 and q["failures_including_missing"] == 2
    assert q["budget_charged_time_s"]["mean"] == 5.5
    speed = result["speedups"]["qaoa_binary"]["budget_charged_time_s"]["exact"]
    assert speed["paired_instances"] == 2 and speed["excluded_instances"] == 0
    assert speed["geomean_ratio"] == pytest.approx((.01*.1)**.5)
    assert result["hardware_validated"] is False


def test_diversity_is_within_same_input_not_across_inputs(tmp_path):
    records, manifest, corpus = fixture(tmp_path)
    result = reanalyze(records, manifest, corpus, bootstrap_repeats=100)
    q = result["methods"]["qaoa_binary"]
    assert q["distinct_final_targets_per_instance"]["mean"] == 1
    assert all(r["pairwise_normalized_hamming_successes"] == 0 for r in q["diversity_by_instance"])
    assert wilson(30, 30)[0] < .95  # All runs succeeding does not imply population probability 1.


def test_duplicate_or_false_reference_is_rejected(tmp_path):
    records, manifest, corpus = fixture(tmp_path)
    with pytest.raises(ValueError, match="duplicate"):
        reanalyze(records+[copy.deepcopy(records[0])], manifest, corpus, bootstrap_repeats=100)
    records[0]["optimum"] += 1
    with pytest.raises(ValueError, match="reference disagrees"):
        reanalyze(records, manifest, corpus, bootstrap_repeats=100)
