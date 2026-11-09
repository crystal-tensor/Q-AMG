# Copyright 2026 QuantumBridge Contributors. Apache-2.0.
"""Offline security studies: native toy circuits and established crypto libraries."""

import copy
import hashlib
import hmac
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np

from quantumbridge.core import Circuit
from quantumbridge.devices import StatevectorDevice
from quantumbridge.finance.kernels import labels
from .common import ApplicationCase


def pqc_case():
    executable = shutil.which("openssl")
    if not executable:
        raise RuntimeError(
            "OpenSSL with ML-KEM-768 and ML-DSA-65 is required; no simulated fallback."
        )
    version = subprocess.run(
        [executable, "version"], capture_output=True, text=True, check=True, timeout=20
    ).stdout.strip()
    with tempfile.TemporaryDirectory(prefix="qb-pqc-") as directory:
        root = Path(directory)
        root.chmod(0o700)
        commands = []

        def run(*arguments, valid=True):
            result = subprocess.run(
                [executable, *arguments], cwd=root, capture_output=True, timeout=20
            )
            for path in root.iterdir():
                path.chmod(0o600)
            commands.append(
                {"arguments": " ".join(arguments), "returncode": result.returncode}
            )
            if valid and result.returncode:
                raise RuntimeError(
                    "OpenSSL operation failed; verify local provider and algorithm support."
                )
            return result.returncode

        run("genpkey", "-algorithm", "ML-KEM-768", "-out", "kem-private.pem")
        run("pkey", "-in", "kem-private.pem", "-pubout", "-out", "kem-public.pem")
        run(
            "pkey",
            "-in",
            "kem-private.pem",
            "-pubout",
            "-outform",
            "DER",
            "-out",
            "kem-public.der",
        )
        run(
            "pkeyutl",
            "-encap",
            "-pubin",
            "-inkey",
            "kem-public.pem",
            "-out",
            "ciphertext.bin",
            "-secret",
            "sender.bin",
        )
        run(
            "pkeyutl",
            "-decap",
            "-inkey",
            "kem-private.pem",
            "-in",
            "ciphertext.bin",
            "-secret",
            "receiver.bin",
        )
        sender, receiver = (
            (root / "sender.bin").read_bytes(),
            (root / "receiver.bin").read_bytes(),
        )
        ciphertext = (root / "ciphertext.bin").read_bytes()
        changed = bytes([ciphertext[0] ^ 1]) + ciphertext[1:]
        (root / "changed.bin").write_bytes(changed)
        run(
            "pkeyutl",
            "-decap",
            "-inkey",
            "kem-private.pem",
            "-in",
            "changed.bin",
            "-secret",
            "changed-secret.bin",
        )
        implicit_rejection = not hmac.compare_digest(
            sender, (root / "changed-secret.bin").read_bytes()
        )
        (root / "message.bin").write_bytes(
            b"QuantumBridge synthetic migration manifest v1"
        )
        run("genpkey", "-algorithm", "ML-DSA-65", "-out", "signature-private.pem")
        run(
            "pkey",
            "-in",
            "signature-private.pem",
            "-pubout",
            "-out",
            "signature-public.pem",
        )
        run(
            "pkeyutl",
            "-sign",
            "-inkey",
            "signature-private.pem",
            "-in",
            "message.bin",
            "-out",
            "signature.bin",
        )
        valid_signature = (
            run(
                "pkeyutl",
                "-verify",
                "-pubin",
                "-inkey",
                "signature-public.pem",
                "-in",
                "message.bin",
                "-sigfile",
                "signature.bin",
            )
            == 0
        )
        (root / "message.bin").write_bytes(
            b"QuantumBridge altered migration manifest v1"
        )
        rejected_signature = (
            run(
                "pkeyutl",
                "-verify",
                "-pubin",
                "-inkey",
                "signature-public.pem",
                "-in",
                "message.bin",
                "-sigfile",
                "signature.bin",
                valid=False,
            )
            != 0
        )
        sizes = {
            "kem_ciphertext_bytes": len(ciphertext),
            "shared_secret_bytes": len(sender),
            "kem_public_spki_der_bytes": (root / "kem-public.der").stat().st_size,
            "mldsa65_signature_bytes": (root / "signature.bin").stat().st_size,
        }
        matched = hmac.compare_digest(sender, receiver)
    return ApplicationCase(
        "pqc",
        "security",
        "Actual ML-KEM-768 and ML-DSA-65 library round trips",
        {
            "implementation": version,
            "kem": "ML-KEM-768",
            "signature": "ML-DSA-65",
            "execution": "classical OpenSSL provider; not a quantum computer",
        },
        {
            "operation_log": commands,
            "public_sizes": sizes,
            "chart_rows": [
                {"label": key, "value": value} for key, value in sizes.items()
            ],
            "chart_units": "Bytes",
            "chart_title": "Actual local algorithm / container sizes",
        },
        {
            "kem_ciphertext_bytes": 1088,
            "shared_secret_bytes": 32,
            "mldsa65_signature_bytes": 3309,
        },
        {},
        outputs={
            **sizes,
            "shared_secrets_equal": matched,
            "tampered_kem_implicit_rejection": implicit_rejection,
            "valid_signature_accepted": valid_signature,
            "altered_message_rejected": rejected_signature,
            "private_keys_exported": False,
            "temporary_directory_removed": not root.exists(),
            "certification": "not a FIPS module certification or completed migration",
        },
    )


def bb84_circuit(bit, basis, measurement_basis):
    circuit = Circuit(1, name="bb84-prepare-and-measure")
    if bit:
        circuit.x(0)
    if basis:
        circuit.h(0)
    if measurement_basis:
        circuit.h(0)
    return circuit


def bb84_transcript(*, eve=False, photons=1024, seed=20261023):
    if type(photons) is not int or not 8 <= photons <= 4096:
        raise ValueError("Use eight to 4096 simulated photons.")
    rng, device = np.random.default_rng(seed), StatevectorDevice()
    rows = []
    for index in range(photons):
        bit, basis, bob_basis, eve_basis = map(int, rng.integers(0, 2, size=4))
        send_bit, send_basis = bit, basis
        if eve:
            sample = device.sample(
                bb84_circuit(bit, basis, eve_basis), shots=1, seed=seed + 2 * index
            )[0]
            send_bit, send_basis = int(sample), eve_basis
        received = int(
            device.sample(
                bb84_circuit(send_bit, send_basis, bob_basis),
                shots=1,
                seed=seed + 2 * index + 1,
            )[0]
        )
        rows.append(
            {
                "index": index,
                "alice_bit": bit,
                "alice_basis": basis,
                "bob_basis": bob_basis,
                "bob_bit": received,
                "sifted": basis == bob_basis,
            }
        )
    sifted = [row for row in rows if row["sifted"]]
    errors = sum(row["alice_bit"] != row["bob_bit"] for row in sifted)
    qber = errors / len(sifted)
    return {
        "photons": photons,
        "sifted_count": len(sifted),
        "sifted_errors": errors,
        "qber": qber,
        "qber_standard_error": float(
            np.sqrt(qber * (1 - qber) / max(1, len(sifted) - 1))
        ),
        "eve": eve,
        "public_teaching_transcript_first24": rows[:24],
    }


def hamming_encode(data):
    if len(data) != 4 or any(bit not in (0, 1) for bit in data):
        raise ValueError("Four binary data symbols are required.")
    word = [0] * 8
    for index, bit in zip((3, 5, 6, 7), data):
        word[index] = int(bit)
    for parity in (1, 2, 4):
        word[parity] = sum(word[i] for i in range(1, 8) if i & parity) % 2
    return word[1:]


def hamming_correct(word):
    if len(word) != 7 or any(bit not in (0, 1) for bit in word):
        raise ValueError("Seven binary code symbols are required.")
    corrected = list(word)
    syndrome = 0
    for index, bit in enumerate(word, 1):
        if bit:
            syndrome ^= index
    if syndrome:
        corrected[syndrome - 1] ^= 1
    return [corrected[i - 1] for i in (3, 5, 6, 7)], syndrome


def qkd_case():
    clean, intercepted = bb84_transcript(), bb84_transcript(eve=True)
    encoded = hamming_encode([1, 0, 1, 1])
    changed = list(encoded)
    changed[4] ^= 1
    recovered, syndrome = hamming_correct(changed)
    return ApplicationCase(
        "qkd",
        "security",
        "BB84 sifting and intercept-resend experiment",
        {
            "photons_per_scenario": 1024,
            "seed": 20261023,
            "channel": "ideal local single-qubit simulation",
            "eve_model": "random-basis intercept-resend",
            "illustrative_abort_qber": 0.11,
        },
        {
            "scenarios": [
                {
                    k: v
                    for k, v in row.items()
                    if k != "public_teaching_transcript_first24"
                }
                for row in (clean, intercepted)
            ],
            "clean_transcript_first24": clean["public_teaching_transcript_first24"],
            "eve_transcript_first24": intercepted["public_teaching_transcript_first24"],
            "single_error_correction_example": {
                "data": [1, 0, 1, 1],
                "encoded": encoded,
                "corrupted": changed,
                "syndrome": syndrome,
                "decoded": recovered,
                "limit": "Hamming(7,4) single-error teaching block; not a QKD reconciliation stack; no double-error guarantee",
            },
            "chart_rows": [
                {"label": "No Eve", "value": clean["qber"]},
                {"label": "Intercept-resend", "value": intercepted["qber"]},
            ],
            "chart_units": "Sifted QBER",
            "chart_title": "Seeded local BB84 protocol experiment",
        },
        {"ideal_clean_qber": 0.0, "ideal_intercept_resend_qber": 0.25},
        {
            "z_basis_bit1": bb84_circuit(1, 0, 0),
            "x_basis_bit1": bb84_circuit(1, 1, 1),
            "mismatched_bases": bb84_circuit(1, 0, 1),
        },
        {
            name: [0.0, 1.0]
            for name in ("z_basis_bit1", "x_basis_bit1", "mismatched_bases")
        },
        {
            "clean_qber": clean["qber"],
            "eve_qber": intercepted["qber"],
            "eve_scenario_aborted": intercepted["qber"] > 0.11,
            "production_key_emitted": False,
            "authentication_and_finite_key_proof": "not implemented; all bits are public seeded teaching data",
        },
    )


def four_bit_phase(circuit):
    circuit.ccx(0, 1, 4)
    circuit.h(3).ccx(2, 4, 3).h(3)
    circuit.ccx(0, 1, 4)


def grover_circuit(iterations, marked="1011"):
    if (
        type(iterations) is not int
        or not 0 <= iterations <= 8
        or len(marked) != 4
        or set(marked) - {"0", "1"}
    ):
        raise ValueError("Use a four-bit marked state and zero to eight iterations.")
    circuit = Circuit(5, name="four-bit-grover-one-work-wire")
    for q in range(4):
        circuit.h(q)
    for _ in range(iterations):
        for q, bit in enumerate(marked):
            if bit == "0":
                circuit.x(q)
        four_bit_phase(circuit)
        for q, bit in enumerate(marked):
            if bit == "0":
                circuit.x(q)
        for q in range(4):
            circuit.h(q).x(q)
        four_bit_phase(circuit)
        for q in range(4):
            circuit.x(q).h(q)
    return circuit


def threat_case():
    theta, marked = np.arcsin(1 / 4), "1011"
    device, rows = StatevectorDevice(), []
    for iterations in range(6):
        circuit = grover_circuit(iterations, marked)
        probability = device.probabilities(circuit).get(marked + "0", 0.0)
        rows.append(
            {
                "iterations": iterations,
                "oracle_queries": iterations,
                "primitive_gates": len(circuit.operations),
                "native_success_probability": probability,
                "analytic_success_probability": float(
                    np.sin((2 * iterations + 1) * theta) ** 2
                ),
            }
        )
    return ApplicationCase(
        "threat",
        "security",
        "Grover toy search and explicit logical-query assumptions",
        {
            "search_bits": 4,
            "candidates": 16,
            "marked": marked,
            "work_qubits": 1,
            "resource_model": "ideal oracle queries only; not seconds or physical qubits",
        },
        {
            "amplification_trace": rows,
            "chart_rows": [
                {
                    "label": str(row["iterations"]),
                    "value": row["native_success_probability"],
                }
                for row in rows
            ],
            "chart_units": "Success probability",
            "chart_title": "Grover iterations on a known toy oracle",
        },
        {
            "three_iteration_success": float(np.sin(7 * theta) ** 2),
            "classical_sequential_expected_queries": 8.5,
        },
        {"grover3": grover_circuit(3)},
        {
            "grover3": [
                float(bits[:4] == marked and bits[4] == "0") for bits in labels(5)
            ]
        },
        {
            "search_space": 16,
            "oracle_calls_per_run": 3,
            "rsa_or_aes_broken": False,
            "excluded_costs": [
                "real reversible cipher oracle",
                "fault tolerance",
                "routing",
                "physical error correction",
                "hardware runtime",
            ],
            "toy_oracle_contains_known_marked_value": True,
        },
    )


def von_neumann_extract(bits):
    values = list(bits)
    if any(bit not in (0, 1) for bit in values):
        raise ValueError("Binary symbols are required.")
    return [a for a, b in zip(values[::2], values[1::2]) if a != b]


def stream_health(bits):
    values = list(bits)
    if not values or any(bit not in (0, 1) for bit in values):
        raise ValueError("A nonempty binary stream is required.")
    longest, run = 1, 1
    for previous, value in zip(values, values[1:]):
        run = run + 1 if value == previous else 1
        longest = max(longest, run)
    windows = [
        {
            "window": i // 128,
            "samples": len(values[i : i + 128]),
            "ones": sum(values[i : i + 128]),
        }
        for i in range(0, len(values), 128)
    ]
    ok = longest < 32 and all(
        32 <= row["ones"] <= 112 for row in windows if row["samples"] == 128
    )
    return {
        "max_run": longest,
        "illustrative_repeat_limit": 32,
        "illustrative_window_ones_bounds": [32, 112],
        "diagnostics_pass": ok,
        "windows": windows,
        "nist_90b_certified": False,
    }


def qrng_case():
    probability = 0.72
    circuit = Circuit(1, name="biased-qrng-simulation").ry(
        2 * np.arcsin(np.sqrt(probability)), 0
    )
    raw = [
        int(bit)
        for bit in StatevectorDevice().sample(circuit, shots=4096, seed=20261023)
    ]
    extracted, health = von_neumann_extract(raw), stream_health(raw)
    return ApplicationCase(
        "qrng",
        "security",
        "Simulated entropy pipeline, health diagnostics and extraction",
        {
            "source": "seeded local statevector simulation",
            "raw_samples": 4096,
            "seed": 20261023,
            "one_probability": probability,
            "extractor": "01 -> 0; 10 -> 1; 00/11 discarded; unpaired tail discarded",
        },
        {
            "ordered_raw_stream": "".join(map(str, raw)),
            "extracted_stream": "".join(map(str, extracted)),
            "health": health,
            "window_statistics": health["windows"],
            "chart_rows": [
                {"label": "Raw zeros", "value": raw.count(0)},
                {"label": "Raw ones", "value": sum(raw)},
                {"label": "Extracted zeros", "value": extracted.count(0)},
                {"label": "Extracted ones", "value": sum(extracted)},
            ],
            "chart_units": "Bit count",
            "chart_title": "Biased simulated stream and Von Neumann output",
        },
        {
            "ideal_input_one_probability": probability,
            "ideal_output_one_probability_under_iid_assumption": 0.5,
            "expected_output_fraction_of_input": probability * (1 - probability),
        },
        {"biased_sampling": circuit},
        {"biased_sampling": [0.0, 1.0]},
        {
            "raw_ones_fraction": sum(raw) / len(raw),
            "extracted_length": len(extracted),
            "extracted_ones_fraction": sum(extracted) / len(extracted)
            if extracted
            else None,
            "diagnostics_pass": health["diagnostics_pass"],
            "physical_entropy_claim": False,
            "production_random_bytes_emitted": False,
            "hardware_receipt": None,
        },
    )


def audit_record(payload, index, previous, key):
    record = {"index": index, "previous": previous, "payload": payload}
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    return {**record, "tag": hmac.new(key, canonical, hashlib.sha256).hexdigest()}


def verify_audit(records, key, expected_length, expected_tail):
    if len(records) != expected_length:
        return False
    previous = "0" * 64
    for index, record in enumerate(records):
        expected = audit_record(record["payload"], index, previous, key)
        if (
            record["index"] != index
            or record["previous"] != previous
            or not hmac.compare_digest(expected["tag"], record["tag"])
        ):
            return False
        previous = record["tag"]
    return hmac.compare_digest(previous, expected_tail)


def privacy_case():
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    cipher = AESGCM(AESGCM.generate_key(bit_length=256))
    nonce, aad = os.urandom(12), b"QuantumBridge/synthetic-job/local-only"
    message = json.dumps(
        {"model": "synthetic-H2", "counts": {"00": 520, "11": 504}}, sort_keys=True
    ).encode()
    encrypted = cipher.encrypt(nonce, message, aad)
    roundtrip = cipher.decrypt(nonce, encrypted, aad) == message
    attempts = [
        ("ciphertext", bytes([encrypted[0] ^ 1]) + encrypted[1:], aad),
        ("metadata", encrypted, aad + b"-altered"),
    ]
    checks = []
    for name, altered, metadata in attempts:
        try:
            cipher.decrypt(nonce, altered, metadata)
        except InvalidTag:
            checks.append({"attack": name, "rejected": True})
        else:
            checks.append({"attack": name, "rejected": False})
    permissions = {"viewer": {"read"}, "operator": {"read", "prepare_local"}}
    access = [
        {
            "role": role,
            "action": action,
            "allowed": action in permissions.get(role, set()),
        }
        for role, action in (
            ("viewer", "read"),
            ("viewer", "prepare_local"),
            ("operator", "prepare_local"),
            ("unknown", "read"),
        )
    ]
    key, records, previous = os.urandom(32), [], "0" * 64
    for index, payload in enumerate(access):
        record = audit_record(payload, index, previous, key)
        records.append(record)
        previous = record["tag"]
    changed = copy.deepcopy(records)
    changed[1]["payload"]["allowed"] = True
    good = verify_audit(records, key, len(records), previous)
    tampered = not verify_audit(changed, key, len(records), previous)
    truncated = not verify_audit(records[:-1], key, len(records), previous)
    return ApplicationCase(
        "privacy",
        "security",
        "Authenticated result encryption, access decisions and anchored audit",
        {
            "encryption": "cryptography AES-256-GCM",
            "nonce_bytes": 12,
            "nonce_policy": "fresh OS randomness with fresh key per example",
            "data": "synthetic quantum-job counts only",
            "audit": "HMAC-SHA256 chain; expected length and tail retained by verifier",
        },
        {
            "access_decisions": access,
            "integrity_checks": checks,
            "public_audit_records": records,
            "chart_rows": [
                {"label": "Round trip", "value": int(roundtrip)},
                {"label": "Ciphertext rejected", "value": int(checks[0]["rejected"])},
                {"label": "Metadata rejected", "value": int(checks[1]["rejected"])},
                {"label": "Audit tamper rejected", "value": int(tampered)},
            ],
            "chart_units": "Check passed (1/0)",
            "chart_title": "Actual classical cryptography checks",
        },
        {"tag_overhead_bytes": 16},
        {},
        outputs={
            "roundtrip_ok": roundtrip,
            "ciphertext_bytes": len(encrypted),
            "plaintext_bytes": len(message),
            "altered_ciphertext_rejected": checks[0]["rejected"],
            "altered_aad_rejected": checks[1]["rejected"],
            "valid_audit_accepted": good,
            "audit_tamper_rejected": tampered,
            "truncation_rejected_with_anchor": truncated,
            "secret_material_exported": False,
            "identity_authentication": "not implemented; roles are explicit teaching inputs",
            "homomorphic_or_mpc_implemented": False,
        },
    )


BUILDERS = {
    "pqc": pqc_case,
    "qkd": qkd_case,
    "threat": threat_case,
    "qrng": qrng_case,
    "privacy": privacy_case,
}
NOTEBOOK_HELPERS = {
    "qkd": ["bb84_circuit", "bb84_transcript", "hamming_encode", "hamming_correct"],
    "threat": ["four_bit_phase", "grover_circuit"],
    "qrng": ["von_neumann_extract", "stream_health"],
    "privacy": ["audit_record", "verify_audit"],
}
