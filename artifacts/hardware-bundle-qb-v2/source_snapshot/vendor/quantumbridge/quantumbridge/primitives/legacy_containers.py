# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is part of QuantumBridge SDK.
# This implementation is developed for the QuantumBridge native architecture.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Sequence

__all__ = ["BitArray", "DataBin", "PrimitiveResult", "PubResult"]


@dataclass(frozen=True)
class BitArray:
    """Compact bitstring sample container compatible with common Sampler V2 access."""

    bitstrings: tuple[str, ...] = ()
    num_bits: int | None = None

    @classmethod
    def from_counts(cls, counts: Mapping[str, int], num_bits: int | None = None) -> "BitArray":
        bitstrings: list[str] = []
        for label, count in sorted(counts.items()):
            bitstrings.extend([str(label)] * int(count))
        if num_bits is None and bitstrings:
            num_bits = len(bitstrings[0])
        return cls(tuple(bitstrings), num_bits)

    @property
    def num_shots(self) -> int:
        return len(self.bitstrings)

    def __len__(self) -> int:
        return len(self.bitstrings)

    def __iter__(self) -> Iterator[str]:
        return iter(self.bitstrings)

    def __getitem__(self, index: int) -> str:
        return self.bitstrings[index]

    def get_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for bitstring in self.bitstrings:
            counts[bitstring] = counts.get(bitstring, 0) + 1
        return counts

    def to_bool_array(self):
        import numpy as np

        return np.asarray([[char == "1" for char in bitstring] for bitstring in self.bitstrings], dtype=bool)

    def to_dict(self) -> dict[str, Any]:
        return {"bitstrings": list(self.bitstrings), "num_bits": self.num_bits, "num_shots": self.num_shots}


@dataclass
class DataBin:
    """Attribute and mapping-style primitive data container."""

    _fields: dict[str, Any] = field(default_factory=dict)

    def __init__(self, **fields: Any) -> None:
        object.__setattr__(self, "_fields", dict(fields))

    def __getattr__(self, name: str) -> Any:
        try:
            return self._fields[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __getitem__(self, name: str) -> Any:
        return self._fields[name]

    def __contains__(self, name: object) -> bool:
        return name in self._fields

    def __iter__(self) -> Iterator[str]:
        return iter(self._fields)

    def __len__(self) -> int:
        return len(self._fields)

    def keys(self):
        return self._fields.keys()

    def items(self):
        return self._fields.items()

    def values(self):
        return self._fields.values()

    def get(self, name: str, default: Any = None) -> Any:
        return self._fields.get(name, default)

    def to_dict(self) -> dict[str, Any]:
        return _to_plain_dict(self._fields)


@dataclass(frozen=True)
class PubResult:
    """Result for one primitive unified bloc."""

    data: DataBin
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"data": self.data.to_dict(), "metadata": _to_plain_dict(dict(self.metadata))}


@dataclass(frozen=True)
class PrimitiveResult:
    """Sequence-like container for primitive pub results."""

    pub_results: Sequence[PubResult]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    _job_id: str | None = None

    def __iter__(self) -> Iterator[PubResult]:
        return iter(self.pub_results)

    def __len__(self) -> int:
        return len(self.pub_results)

    def __getitem__(self, index: int) -> PubResult:
        return self.pub_results[index]

    def tolist(self) -> list[PubResult]:
        return list(self.pub_results)

    def result(self, timeout: float | None = None) -> "PrimitiveResult":
        if timeout is not None and timeout < 0:
            raise TimeoutError("QuantumBridge primitive result timeout must be non-negative.")
        return self

    def status(self) -> str:
        return "DONE"

    def done(self) -> bool:
        return True

    def cancelled(self) -> bool:
        return False

    def running(self) -> bool:
        return False

    def job_id(self) -> str:
        return self._job_id or str(self.metadata.get("job_id", "quantumbridge-primitive-result"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "results": [result.to_dict() for result in self.pub_results],
            "metadata": _to_plain_dict(dict(self.metadata)),
        }


def _to_plain_dict(value: Any) -> Any:
    if isinstance(value, DataBin):
        return value.to_dict()
    if isinstance(value, BitArray):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): _to_plain_dict(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_to_plain_dict(item) for item in value]
    if isinstance(value, list):
        return [_to_plain_dict(item) for item in value]
    return value
