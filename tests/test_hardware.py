import json

import httpx
import pytest
from quantumbridge import Circuit
from quantumbridge.serialization import qpy_load

from qamg.backends import CompanyBackend, QBBackend
from qamg.benchmark import create_corpus
from qamg.hardware import (
    SubmissionJournal,
    canary_circuits,
    digest,
    execute_bundle,
    prepare_bundle,
    verify_bundle,
)
from qamg.problem import build_problem
from qamg.schema import DesignSpec


def make_bundle(tmp_path, instances=1):
    cp = tmp_path/"corpus.json"
    corpus = create_corpus(cp)
    contexts = {f"{c['spec']['family']}:{c['spec']['groups']}":
                {"parameters": [.4, .2], "trained_on": c["id"]}
                for c in corpus["instances"] if c["split"] == "train"}
    transfer = tmp_path/"transfer.json"
    transfer.write_text(json.dumps({"encoding": "compact_binary", "depth": 1, "training_wall_s": .2,
                                    "corpus_sha256": digest(cp), "contexts": contexts}))
    bundle = tmp_path/"bundle"
    manifest = prepare_bundle(bundle, cp, transfer, instances=instances, seeds=1, canary_repeats=1)
    return bundle, manifest


def test_offline_bundle_is_bound_hash_checked_and_not_hardware_evidence(tmp_path):
    bundle, manifest = make_bundle(tmp_path, instances=2)
    assert manifest["budget"]["jobs_max"] == 7
    assert manifest["budget"]["shots_max"] == 7*256
    assert manifest["hardware_validated"] is False
    assert verify_bundle(bundle)["status"] == "offline_prepared_not_submitted"
    structure = next(j for j in manifest["jobs"] if j["kind"] == "structure")
    with (bundle/structure["circuit"]).open("rb") as stream:
        circuit = qpy_load(stream)[0]
    assert circuit.num_qubits == 3 and not circuit.parameters
    (bundle/structure["circuit"]).with_suffix(".qasm").write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_bundle(bundle)


def test_canary_expectations_independently_match_simulated_circuits():
    backend = QBBackend()
    for circuit, expected in canary_circuits().values():
        batch = backend.sample(backend.compile(circuit, 42), 1024, 42)
        assert all(abs(batch.counts.get(bits, 0)/1024-prob) < .1 for bits, prob in expected.items())
        assert set(batch.counts) <= set(expected)


def test_declared_budget_is_checked_before_remote_contact(tmp_path):
    bundle, manifest = make_bundle(tmp_path)
    contacts = []

    def handle(request):
        contacts.append(request)
        return httpx.Response(500)

    client = httpx.Client(base_url="https://contract.invalid/", transport=httpx.MockTransport(handle))
    backend = CompanyBackend("https://contract.invalid", client=client)
    with pytest.raises(ValueError, match="authorization"):
        execute_bundle(bundle, tmp_path/"execution", 1, manifest["budget"]["shots_max"], backend)
    assert not contacts


def test_submission_journal_stops_second_post_before_budget_overrun(tmp_path):
    posts = []

    def handle(request):
        if request.url.path == "/capabilities":
            return httpx.Response(200, json={"name": "contract-mock-not-a-qpu", "max_qubits": 3,
                                            "max_shots": 1024, "is_simulator": False,
                                            "basis_gates": ["rz", "sx", "x", "cx"], "bit_order": "qubit0_left"})
        if request.url.path == "/jobs":
            posts.append(json.loads(request.content))
            return httpx.Response(200, json={"job_id": "contract-only"})
        if request.url.path.endswith("/result"):
            return httpx.Response(200, json={"counts": {"0": 32}, "bit_order": "qubit0_left"})
        return httpx.Response(200, json={"status": "completed"})

    client = httpx.Client(base_url="https://contract.invalid/", transport=httpx.MockTransport(handle))
    backend = CompanyBackend("https://contract.invalid", client=client, poll_interval_s=0)
    journal = SubmissionJournal(tmp_path, max_jobs=1, max_shots=32)
    backend.event_sink = journal
    circuit = Circuit(1)
    circuit.measure_all()
    backend.sample(circuit, 32, 42)
    with pytest.raises(RuntimeError, match="budget exhausted"):
        backend.sample(circuit, 32, 42)
    assert len(posts) == 1
    events = [json.loads(x) for x in journal.path.read_text().splitlines()]
    assert events[0]["event"] == "submit_intent"
    assert events[1]["event"] == "submitted" and events[1]["job_id"] == "contract-only"
    assert events[0]["client_request_id"] == posts[0]["client_request_id"]
    assert journal.jobs == 1 and journal.shots == 32


def test_ambiguous_submission_is_durable_and_freezes_further_posts(tmp_path):
    posts = []

    def handle(request):
        if request.url.path == "/capabilities":
            return httpx.Response(200, json={"name": "contract-mock-not-a-qpu", "max_qubits": 3,
                                            "max_shots": 1024, "is_simulator": False,
                                            "basis_gates": ["rz", "sx", "x", "cx"], "bit_order": "qubit0_left"})
        posts.append(request)
        raise httpx.ReadError("contract-only lost submission reply", request=request)

    client = httpx.Client(base_url="https://contract.invalid/", transport=httpx.MockTransport(handle))
    backend = CompanyBackend("https://contract.invalid", client=client)
    journal = SubmissionJournal(tmp_path, max_jobs=2, max_shots=64)
    backend.event_sink = journal
    circuit = Circuit(1)
    circuit.measure_all()
    with pytest.raises(httpx.ReadError):
        backend.sample(circuit, 32, 42)
    with pytest.raises(RuntimeError, match="stopped after an error"):
        backend.sample(circuit, 32, 42)
    events = [json.loads(x) for x in journal.path.read_text().splitlines()]
    assert [e["event"] for e in events] == ["submit_intent", "submit_failed"]
    assert len(posts) == 1 and journal.jobs == 1


def test_hardware_runner_contract_preserves_unverified_status_and_budget(tmp_path):
    bundle, manifest = make_bundle(tmp_path)
    corpus = json.loads((bundle/"corpus.json").read_text())
    case = next(c for c in corpus["instances"] if c["id"] == manifest["instances"][0])
    problem = build_problem(DesignSpec.model_validate(case["spec"]))
    best = min((s for s in problem.assignments() if not problem.violations(s)), key=problem.cost)
    # A declared-hardware response is only a contract mock, never experimental evidence.
    count_sets = [
        {bits: round(prob*256) for bits, prob in j["expected_probabilities"].items()}
        for j in manifest["jobs"] if j["kind"] == "canary"] + [{"".join(map(str, best)): 256}]
    posts = []

    def handle(request):
        if request.url.path == "/capabilities":
            return httpx.Response(200, json={"name": "contract-mock-not-a-qpu", "max_qubits": 5,
                                            "max_shots": 1024, "is_simulator": False,
                                            "basis_gates": ["rz", "sx", "x", "cx"], "bit_order": "qubit0_left"})
        if request.url.path == "/jobs":
            posts.append(json.loads(request.content))
            return httpx.Response(200, json={"job_id": str(len(posts)-1)})
        if request.url.path.endswith("/result"):
            index = int(request.url.path.split("/")[-2])
            return httpx.Response(200, json={"counts": count_sets[index], "bit_order": "qubit0_left"})
        return httpx.Response(200, json={"status": "completed"})

    client = httpx.Client(base_url="https://contract.invalid/", transport=httpx.MockTransport(handle))
    backend = CompanyBackend("https://contract.invalid", client=client, poll_interval_s=0)
    result = execute_bundle(bundle, tmp_path/"execution", 6, 1536, backend)
    assert result["job_budget_usage"] == {"attempts": 6, "shots": 1536}
    assert result["hardware_validated"] is False and result["quantum_acceleration_target_met"] is False
    assert len(posts) == 6 and result["record_count"] == 6
    assert "provider-declared" in result["speedups"]["qaoa_binary"]["claim_scope"]
    assert all(c["passed"] for c in json.loads((tmp_path/"execution/canaries.json").read_text()))
