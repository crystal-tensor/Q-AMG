# Copyright 2026 QuantumBridge Contributors. Licensed under Apache-2.0.
"""Explicit origins and reversible label traces, without inventing provider facts."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json

from quantumbridge.compat.quafu.cloud_backend import _extract_counts

from .execution import probabilities_from_counts


CAPABILITY_STATES = frozenset({"VERIFIED_READ_ONLY", "DOCUMENTED_UNVERIFIED", "ABSENT", "UNSUPPORTED", "ERROR", "REQUIRES_PAID_EXECUTION"})
RESPONSE_FIELDS = ("bit_order", "measurements", "compiler", "calibration_id", "executed_qasm", "transpiled", "mapping", "layout", "options", "corrected")


def response_field_matrix(observed, *, documentation=None):
    """Documentation and synthetic probes can never fill a historical response."""
    if not isinstance(observed, Mapping):
        raise ValueError("observed client response must be a mapping")
    return {field: {"status": "VERIFIED_READ_ONLY" if field in observed else "ABSENT",
                    "origin": "saved_Task.result_response", "observed_value": observed.get(field),
                    "documentation_hint": (documentation or {}).get(field),
                    "hint_is_historical_evidence": False} for field in RESPONSE_FIELDS}


def permute_counts(counts, width, positions):
    """positions[c] gives the raw-label position hypothesized to carry c."""
    if type(width) is not int or not 1 <= width <= 6:
        raise ValueError("label trace permits one to six bits")
    probabilities_from_counts(counts, width)
    if (not isinstance(positions, (list, tuple)) or any(type(i) is not int for i in positions)
            or sorted(positions) != list(range(width))):
        raise ValueError("bijective label position permutation required")
    result = {"".join(label[i] for i in positions): int(value) for label, value in counts.items()}
    if len(result) != len(counts) or sum(result.values()) != sum(counts.values()):
        raise ValueError("label permutation changed counts")
    inverse = [positions.index(i) for i in range(width)]
    restored = {"".join(label[i] for i in inverse): value for label, value in result.items()}
    if restored != counts:
        raise ValueError("label permutation is not exactly reversible")
    return result


def trace_count_labels(payload, width, *, ordered_physical_mapping, financial_counts=None, decoder_order=None):
    """Prove the project-side pipeline; don't claim access to raw HTTP decoding."""
    if type(width) is not int or not 1 <= width <= 6:
        raise ValueError("label trace permits one to six bits")
    extracted = _extract_counts(payload)
    probabilities_from_counts(extracted, width)
    if payload.get("count") != extracted:
        raise ValueError("D3 historical contract requires the observed top-level count field")
    if (len(ordered_physical_mapping) != width or len(set(ordered_physical_mapping)) != width
            or any(type(i) is not int or i < 0 for i in ordered_physical_mapping)):
        raise ValueError("unique actual classical-to-physical mapping required")
    encoded = json.dumps(extracted, sort_keys=True, separators=(",", ":"))
    sorted_classical = sorted(range(width), key=lambda i: ordered_physical_mapping[i])
    alternatives = {}
    for name, positions, origin in (
            ("identity", list(range(width)), "diagnostic identity interpretation"),
            ("reverse", list(reversed(range(width))), "finance q0-last assumption, not service metadata"),
            ("ascending_physical_register", [sorted_classical.index(i) for i in range(width)],
             "HYPOTHESIS: raw labels sorted by physical index; mapping alone does not prove it")):
        alternatives[name] = {"counts": permute_counts(extracted, width, positions),
                              "positions": positions, "origin": origin, "applied_to_original": False}
    if financial_counts is not None:
        probabilities_from_counts(financial_counts, width)
        if decoder_order not in {"q0-first", "q0-last"}:
            raise ValueError("explicit saved financial decoder order required")
        positions = list(range(width)) if decoder_order == "q0-first" else list(reversed(range(width)))
        if financial_counts != permute_counts(extracted, width, positions):
            raise ValueError("saved normalized finance counts do not match the recorded permutation")
    provider_order = payload.get("bit_order")
    return {"raw_count_field": "count", "client_response_counts_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
            "counts": extracted, "label_width": width, "label_lengths": sorted({len(key) for key in extracted}),
            "padding": "none", "register_delimiter": "none; nonbinary labels refused", "labels_merged": False,
            "shots": sum(extracted.values()), "project_extraction_changes_labels": False,
            "provider_order_observed_in_this_response": provider_order,
            "count_order_status": "UNKNOWN" if provider_order is None else "OBSERVED_FIELD_NOT_INDEPENDENTLY_CERTIFIED",
            "finance_decoder_status": "GATEWAY_ASSUMPTION_UNVERIFIED" if financial_counts is not None else "NOT_USED_FOR_FINANCE_DECODING",
            "saved_finance_decoder_order": decoder_order, "saved_normalized_counts": financial_counts,
            "project_transforms_exactly_invertible": True, "hypotheses": alternatives,
            "client_decoding_stage": {"status": "DOCUMENTED_UNVERIFIED",
                                      "reason": "saved Task.result payload is already decoded; raw HTTP labels and upstream internals were not inspected"},
            "stages": ["server raw HTTP labels UNOBSERVED", "installed Task.result decoding UNVERIFIED",
                       "saved client response count", "gateway _extract_counts: labels unchanged",
                       "finance probabilities_from_counts: explicit identity/reverse if used", "FinanceCase.analyze: q0-first indexing"]}


def layout_pinning_verdict(requested, observed):
    if (not isinstance(requested, (list, tuple)) or not requested or len(set(requested)) != len(requested)
            or any(type(i) is not int or i < 0 for i in requested)
            or not isinstance(observed, (list, tuple)) or not observed or len(set(observed)) != len(observed)
            or any(type(i) is not int or i < 0 for i in observed)):
        raise ValueError("unique explicit physical layouts required")
    same_set = set(requested) == set(observed)
    return {"requested": list(requested), "observed": list(observed), "physical_set_matches": same_set,
            "ordered_mapping_matches": list(requested) == list(observed),
            "status": "ERROR" if not same_set else "DOCUMENTED_UNVERIFIED",
            "layout_pinning": "PHYSICAL_LAYOUT_PINNING_NOT_VERIFIED",
            "future_task_guarantee": False,
            "reason": "accepting target_qubits or matching one receipt does not guarantee another task's placement"}
