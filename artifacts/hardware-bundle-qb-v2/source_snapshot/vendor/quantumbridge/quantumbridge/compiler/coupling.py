# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is part of QuantumBridge SDK.
# This implementation is developed for the QuantumBridge native architecture.

from collections import deque
from numbers import Integral


class CouplingMap:
    def __init__(self, edges):
        normalized = []
        for a, b in edges:
            if any(isinstance(wire, bool) or not isinstance(wire, Integral) or wire < 0 for wire in (a, b)) or a == b:
                raise ValueError("Coupling edges need two distinct non-negative integer wires.")
            normalized.append((int(a), int(b)))
        self.edges = tuple(sorted(set(normalized)))

    def is_connected(self, a: int, b: int) -> bool:
        edge = (int(a), int(b))
        return edge in self.edges or (edge[1], edge[0]) in self.edges

    def neighbors(self, wire: int) -> tuple[int, ...]:
        out = []
        for a, b in self.edges:
            if a == wire:
                out.append(b)
            elif b == wire:
                out.append(a)
        return tuple(sorted(set(out)))

    def has_edge(self, a, b): return (a, b) in self.edges

    def shortest_path(self, start, end):
        queue, seen = deque([(start, (start,))]), {start}
        while queue:
            vertex, path = queue.popleft()
            if vertex == end: return path
            for neighbor in self.neighbors(vertex):
                if neighbor not in seen:
                    seen.add(neighbor); queue.append((neighbor, path + (neighbor,)))
        raise ValueError("Physical target topology is disconnected for this interaction.")

    def distance(self, start, end): return len(self.shortest_path(start, end)) - 1

    @classmethod
    def from_line(cls, num_qubits, bidirectional=True):
        edges = [(wire, wire + 1) for wire in range(num_qubits - 1)]
        return cls(edges + [(b, a) for a, b in edges] if bidirectional else edges)

    @classmethod
    def from_ring(cls, num_qubits, bidirectional=True):
        edges = [(wire, (wire + 1) % num_qubits) for wire in range(num_qubits)]
        return cls(edges + [(b, a) for a, b in edges] if bidirectional else edges)

    @classmethod
    def from_full(cls, num_qubits): return cls((a, b) for a in range(num_qubits) for b in range(num_qubits) if a != b)
