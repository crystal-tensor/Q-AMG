# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is part of QuantumBridge SDK.
# This implementation is developed for the QuantumBridge native architecture.
# Design source: docs/mvp/api_contract_v0.1.md and docs/mvp/simulator_design_v0.1.md.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Union

import numpy as np

from quantumbridge.schema.result_schema import RESULT_SCHEMA_VERSION, make_result_payload
from quantumbridge.schema.validation import validate_result_schema


def _complex_list(values: Optional[np.ndarray]) -> Optional[list[dict[str, float]]]:
    if values is None:
        return None
    return [{"real": float(np.real(value)), "imag": float(np.imag(value))} for value in values]


def _complex_array(values, *, matrix=False) -> Optional[np.ndarray]:
    if values is None:
        return None
    from quantumbridge.execution.resource_context import current_resources
    scope = current_resources()
    if scope is not None:
        from quantumbridge.execution.array_reservations import numeric_array, numeric_shape
        from quantumbridge.execution.array_ownership import retained_array
        from quantumbridge.core.classical import UnsupportedFeature
        if issubclass(type(values), np.ndarray):
            if len(numeric_shape(values)) != (2 if matrix else 1):raise ValueError("Result numeric payload has an invalid rank.")
            return numeric_array(values, category="result_complex_import", copy=True)
        if type(values) is not list and type(values) is not tuple:raise UnsupportedFeature("result_numeric_import")
        rows = values if matrix else (values,)
        if matrix and rows and type(rows[0]) is not list and type(rows[0]) is not tuple:raise UnsupportedFeature("result_density_rows")
        width = len(rows[0]) if matrix and rows else len(values) if not matrix else 0
        if any(type(row) is not list and type(row) is not tuple or len(row) != width for row in rows):
            raise ValueError("Result density rows must have matching lengths.")
        def scalar(item):
            if type(item) is dict:
                real, imag = item.get("real", 0.), item.get("imag", 0.)
                if numeric_shape(real, real_only=True) or numeric_shape(imag, real_only=True):
                    raise ValueError("Result complex components must be real scalars.")
                return complex(real, imag)
            if numeric_shape(item):raise ValueError("Result values must be numeric scalars.")
            return complex(item)
        for row in rows:
            for item in row:scalar(item)
        shape = (len(rows), width) if matrix else (width,)
        result = retained_array(scope.ledger, "result_complex_import", shape, complex)
        scope.ledger.unavailable("result_import_python_heap",
            "Result import owns its numerical destination; decoder containers/scalars and Python allocator memory are not measured RSS.")
        for index, row in enumerate(rows):
            target = result[index] if matrix else result
            for column, item in enumerate(row):target[column] = scalar(item)
        return result
    if matrix:return np.array([_complex_array(row) for row in values])
    return np.asarray([complex(item.get("real", 0.0), item.get("imag", 0.0)) if isinstance(item, dict) else complex(item) for item in values], dtype=complex)


@dataclass
class Result:
    state: Optional[np.ndarray] = None
    probabilities_data: Optional[dict[str, float]] = None
    counts_data: Optional[dict[str, int]] = None
    expectation_data: Optional[Union[float, dict[str, float]]] = None
    metadata_data: dict[str, Any] = field(default_factory=dict)
    trace: Optional[list[dict[str, Any]]] = None
    density: Optional[np.ndarray] = None
    mps: Any = None
    mps_snapshots: dict[str, Any] = field(default_factory=dict, repr=False)

    def statevector(self) -> Optional[np.ndarray]:
        from quantumbridge.execution.array_ownership import copied_array
        return None if self.state is None else copied_array(self.state, category="result_statevector_copy")

    def density_matrix(self) -> Optional[np.ndarray]:
        from quantumbridge.execution.array_ownership import copied_array
        return None if self.density is None else copied_array(self.density, category="result_density_copy")

    def matrix_product_state(self, label=None):
        """Copy the final MPS or a native labeled save, preserving group nesting.

        data()/JSON retain their serialized dict/list contract. Labeled native
        saves are available on live execution Results, not inferred from JSON.
        """
        if label is None:
            return None if self.mps is None else self.mps.copy()
        if label not in self.mps_snapshots:
            raise KeyError("Unknown native MPS snapshot label: "+str(label))
        def copied(value):
            if isinstance(value, dict): return {key: copied(child) for key, child in value.items()}
            if isinstance(value, list): return [copied(child) for child in value]
            return value.copy()
        return copied(self.mps_snapshots[label])

    def probabilities(self) -> Optional[dict[str, float]]:
        return None if self.probabilities_data is None else dict(self.probabilities_data)

    def counts(self) -> Optional[dict[str, int]]:
        return None if self.counts_data is None else dict(self.counts_data)

    def memory(self) -> Optional[list[str]]:
        samples = self.metadata_data.get("memory")
        return None if samples is None else list(samples)

    def data(self,label=None):
        from quantumbridge.simulator.snapshots import decode
        values=decode(self.metadata_data.get("saved_snapshots",{}))
        if label is None:return values
        if label not in values:raise KeyError("Unknown saved snapshot label: "+str(label))
        return values[label]

    def get_counts(self): return self.counts()
    def get_memory(self): return self.memory()

    def expectation_value(self):
        return self.expectation_data

    def metadata(self) -> dict[str, Any]:
        return dict(self.metadata_data)

    def to_dict(self, *, serialization_limits=None) -> dict[str, Any]:
        from quantumbridge.schema.serialization import check_numeric_export
        check_numeric_export([array for array in (self.state, self.density) if array is not None], limits=serialization_limits)
        metadata = self.metadata()
        expectation_values = self.expectation_data if isinstance(self.expectation_data, dict) else None
        if self.expectation_data is not None and not isinstance(self.expectation_data, dict):
            expectation_values = {"default": float(self.expectation_data)}
        payload = make_result_payload(
            job_id=metadata.get("job_id"),
            backend_name=metadata.get("backend_name", metadata.get("device")),
            backend_version=metadata.get("backend_version"),
            experiment_id=metadata.get("experiment_id"),
            shots=metadata.get("shots"),
            counts=self.counts(),
            probabilities=self.probabilities(),
            statevector=_complex_list(self.state),
            density_matrix=metadata.get("density_matrix") if self.density is None else [_complex_list(row) for row in self.density],
            expectation_values=expectation_values,
            observable_metadata=metadata.get("observable_metadata", {}),
            circuit_metadata=metadata.get("circuit_metadata", {}),
            noise_model_metadata=metadata.get("noise_model_metadata", metadata.get("noise_model", {})),
            seed=metadata.get("seed"),
            timing=metadata.get("timing", {}),
            errors=metadata.get("errors", []),
            warnings=metadata.get("warnings", []),
            provenance=metadata.get("provenance", {}),
            metadata=metadata,
        )
        payload["expectation"] = self.expectation_data
        payload["trace"] = self.trace
        if self.mps is not None:
            payload["matrix_product_state"] = self.mps.to_dict(serialization_limits=serialization_limits)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, serialization_limits=None) -> "Result":
        from quantumbridge.schema.serialization import validate_payload_structure
        validate_payload_structure(payload, limits=serialization_limits)
        if "schema_version" not in payload:
            payload = {
                "schema_version": "0.1",
                "counts": payload.get("counts"),
                "probabilities": payload.get("probabilities"),
                "statevector": payload.get("statevector"),
                "density_matrix": None,
                "expectation_values": {"default": payload.get("expectation")} if payload.get("expectation") is not None else None,
                "metadata": payload.get("metadata", {}),
                "provenance": {},
                "errors": [],
                "warnings": [],
            }
        validate_result_schema(payload)
        expectation = payload.get("expectation")
        if expectation is None:
            values = payload.get("expectation_values")
            if isinstance(values, dict):
                expectation = values if set(values) != {"default"} else values.get("default")
        metadata = dict(payload.get("metadata") or {})
        for key in (
            "job_id",
            "backend_name",
            "backend_version",
            "experiment_id",
            "shots",
            "seed",
            "observable_metadata",
            "circuit_metadata",
            "noise_model_metadata",
            "timing",
            "errors",
            "warnings",
            "provenance",
        ):
            if payload.get(key) not in (None, {}, []):
                metadata.setdefault(key, payload.get(key))
        metadata.setdefault("schema_version", payload.get("schema_version", RESULT_SCHEMA_VERSION))
        from quantumbridge.information.matrix_product_state import MatrixProductState

        return cls(
            state=_complex_array(payload.get("statevector")),
            probabilities_data=None if payload.get("probabilities") is None else dict(payload.get("probabilities")),
            counts_data=None if payload.get("counts") is None else {str(k): int(v) for k, v in payload.get("counts").items()},
            expectation_data=expectation,
            metadata_data=metadata,
            trace=payload.get("trace"),
            density=_complex_array(payload.get("density_matrix"), matrix=True),
            mps=None if payload.get("matrix_product_state") is None else MatrixProductState.from_dict(payload["matrix_product_state"], serialization_limits=serialization_limits),
        )

    def to_json(self, *, indent: int | None = None, serialization_limits=None) -> str:
        from quantumbridge.schema.serialization import to_json
        return to_json(self.to_dict(serialization_limits=serialization_limits), indent=indent, limits=serialization_limits)

    @classmethod
    def from_json(cls, text: str, *, serialization_limits=None) -> "Result":
        from quantumbridge.schema.serialization import from_json
        return cls.from_dict(from_json(text, limits=serialization_limits), serialization_limits=serialization_limits)

    def validate_schema(self) -> bool:
        return validate_result_schema(self.to_dict())
