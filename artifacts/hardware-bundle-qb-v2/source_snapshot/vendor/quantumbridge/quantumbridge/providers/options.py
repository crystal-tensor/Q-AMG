# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is part of QuantumBridge SDK.
# This implementation is developed for the QuantumBridge native architecture.

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any


class Options:
    """Small Qiskit-style options container with attribute access."""

    def __init__(self, **fields: Any) -> None:
        object.__setattr__(self, "_fields", dict(fields))

    def __getattr__(self, name: str) -> Any:
        try:
            return self._fields[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_fields":
            object.__setattr__(self, name, value)
            return
        self._fields[name] = value

    def __contains__(self, name: object) -> bool:
        return name in self._fields

    def __iter__(self) -> Iterator[str]:
        return iter(self._fields)

    def get(self, name: str, default: Any = None) -> Any:
        return self._fields.get(name, default)

    def update_options(self, **fields: Any) -> "Options":
        self._fields.update(fields)
        return self

    def copy(self) -> "Options":
        return Options(**self._fields)

    def to_dict(self) -> dict[str, Any]:
        return dict(self._fields)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "Options":
        return cls(**dict(mapping))
