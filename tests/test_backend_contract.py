import httpx
import pytest
from quantumbridge import Circuit

from qamg.backends import CompanyBackend, SampleBatch


def test_company_reference_contract_is_not_hardware_verification():
    requests = []

    def handle(request):
        requests.append((request.method, request.url.path))
        if request.url.path == "/capabilities":
            return httpx.Response(200, json={"name": "contract-mock-not-a-qpu", "max_qubits": 8,
                                            "max_shots": 1024, "is_simulator": True,
                                            "basis_gates": ["rz", "sx", "x", "cx"],
                                            "coupling_map": [[a,b] for a,b in [(0,1),(1,0),(1,2),(2,1),(2,3),(3,2),(3,4),(4,3),(4,5),(5,4),(5,6),(6,5),(6,7),(7,6)]],
                                            "bit_order": "qubit0_left"})
        if request.url.path == "/jobs" and request.method == "POST":
            assert b"OPENQASM" in request.content
            return httpx.Response(200, json={"job_id": "contract-job"})
        if request.url.path == "/jobs/contract-job":
            return httpx.Response(200, json={"status": "completed"})
        if request.url.path == "/jobs/contract-job/result":
            return httpx.Response(200, json={"counts": {"10": 32, "01": 32},
                                            "bit_order": "qubit0_left"})
        return httpx.Response(404)

    client = httpx.Client(base_url="https://contract.invalid/", transport=httpx.MockTransport(handle))
    b = CompanyBackend("https://contract.invalid", client=client, poll_interval_s=0)
    c = Circuit(2, 2)
    c.h(0)
    c.cx(0, 1)
    for wire, bit in [(0, 0), (1, 1)]:
        c.measure(wire, bit)
    compiled = b.compile(c, 4)
    assert compiled.num_qubits >= 2 and compiled.num_clbits == 2
    r = b.sample(compiled, 64, 4)
    assert r.execution_source == "remote_simulator"
    assert b.capabilities().verified is False
    assert r.timings["quantum_execution_s"] is None
    assert ("GET", "/jobs/contract-job/result") in requests


def test_invalid_sample_counts_rejected():
    with pytest.raises(ValueError):
        SampleBatch({"01": 2}, 3, "bad", "ideal_simulator", {})
    with pytest.raises(ValueError):
        SampleBatch({"XX": 3}, 3, "bad", "ideal_simulator", {})


def test_company_not_configured_is_explicit():
    with pytest.raises(ValueError, match="not been configured"):
        CompanyBackend("")


@pytest.mark.parametrize("status", ["failed", "cancelled", "unknown-state"])
def test_terminal_or_unknown_job_states_are_errors(status):
    def handle(request):
        if request.url.path == "/capabilities":
            return httpx.Response(200, json={"name": "contract-mock-not-a-qpu", "max_qubits": 2,
                                            "max_shots": 64, "is_simulator": True,
                                            "basis_gates": ["rz", "sx", "x", "cx"],
                                            "bit_order": "qubit0_left"})
        if request.url.path == "/jobs":
            return httpx.Response(200, json={"job_id": "terminal"})
        return httpx.Response(200, json={"status": status})
    client = httpx.Client(base_url="https://contract.invalid/", transport=httpx.MockTransport(handle))
    backend = CompanyBackend("https://contract.invalid", client=client, poll_interval_s=0)
    c = Circuit(1, 1)
    c.measure(0, 0)
    with pytest.raises((ValueError, RuntimeError)):
        backend.sample(c, 32, 1)


def test_timeout_attempts_cancel_and_never_returns_fake_counts():
    cancelled = []

    def handle(request):
        if request.url.path == "/capabilities":
            return httpx.Response(200, json={"name": "contract-mock-not-a-qpu", "max_qubits": 2,
                                            "max_shots": 64, "is_simulator": True,
                                            "basis_gates": ["rz", "sx", "x", "cx"],
                                            "bit_order": "qubit0_left"})
        if request.url.path.endswith("/cancel"):
            cancelled.append(True)
            return httpx.Response(200, json={"status": "cancelled"})
        if request.url.path == "/jobs":
            return httpx.Response(200, json={"job_id": "slow"})
        return httpx.Response(200, json={"status": "queued"})
    client = httpx.Client(base_url="https://contract.invalid/", transport=httpx.MockTransport(handle))
    backend = CompanyBackend("https://contract.invalid", client=client, timeout_s=.005,
                              poll_interval_s=.001)
    c = Circuit(1, 1)
    c.measure(0, 0)
    with pytest.raises(TimeoutError):
        backend.sample(c, 32, 1)
    assert cancelled

