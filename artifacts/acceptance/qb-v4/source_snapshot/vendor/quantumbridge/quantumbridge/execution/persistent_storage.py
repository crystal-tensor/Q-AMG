# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
"""Explicit native construction pool, separate from callback admission budgets."""

from dataclasses import dataclass
from weakref import WeakValueDictionary

from .array_ownership import native_configuration_copy
from .resource_ledger import UnifiedResourceLedger


@dataclass(frozen=True)
class PersistentStorageOwner:
    pool: object
    payload: object
    roots: tuple

    @property
    def nbytes(self):
        return sum(root.nbytes for root in self.roots)

    def copy_memo(self):
        return {id(root): root for root in self.roots}

    def admit(self, scope):
        return scope.stack.enter_context(scope.ledger.borrow_resident(self.roots,
            request_token=scope.request_token))


class PersistentStoragePool:
    """A caller-shareable pool for known copied ndarray data, not process RSS.

    Default pools belong to one kernel. An explicitly shared pool can constrain
    multiple kernels. Neither policy reuses a single execution request's cap.
    Weak records do not keep dead kernel/view storage alive.
    """

    def __init__(self, limits=None):
        self.ledger = UnifiedResourceLedger(limits)
        self._roots = WeakValueDictionary()
        self._allocations = 0

    def freeze(self, value):
        roots = []
        def copied(array):
            array.flags.writeable = False
            roots.append(array)
            with self.ledger._lock:
                self._roots[id(array)] = array
                self._allocations += 1
        payload = native_configuration_copy(value, ledger=self.ledger,
            strict=True, on_array=copied, category="persistent_native_storage")
        return PersistentStorageOwner(self, payload, tuple(roots))

    def receipt(self):
        with self.ledger._lock:
            receipt = self.ledger.receipt()
            return {"persistent_owned_bytes": receipt["live_memory_bytes"],
                "new_physical_allocation_count": self._allocations,
                "live_physical_storage_count": len(self._roots),
                "scope": "known native ndarray construction pool, not process RSS",
                "ledger": receipt}
