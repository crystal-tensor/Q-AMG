# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# Native wire-dependency DAG; no upstream graph/circuit implementation copied.

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from graphlib import TopologicalSorter, CycleError

from quantumbridge.core import Circuit, Operation, Measurement
from quantumbridge.core.classical import ClassicalExpression, UnsupportedFeature
from quantumbridge.core.control_flow import validate_control_flow
from .limits import CompilationLimits


def instruction_resources(item):
    resources = set()
    def expression(value):
        if isinstance(value, ClassicalExpression):
            if value.op == "bit": resources.add(("c", value.value))
            if value.op == "register": resources.update(("c", bit) for bit in value.value)
            if value.op == "var": resources.add(("v", value.value[0]))
            for child in value.args: expression(child)
        elif isinstance(value, Circuit):
            for child in value.iter_instructions(): resources.update(instruction_resources(child))
        elif isinstance(value, dict):
            for child in value.values(): expression(child)
        elif isinstance(value, (tuple, list)):
            for child in value: expression(child)
    if isinstance(item, Measurement): resources.update((("q", item.wire), ("c", item.bit)))
    else: resources.update(("q", wire) for wire in item.wires)
    for condition in [item.metadata.get("condition"), item.metadata.get("gate_condition")] + [frame.get("condition") for frame in item.metadata.get("control_flow_stack", ())]:
        if condition: resources.update(("c", bit) for bit in condition["bits"])
    node = item.metadata.get("control_node")
    if node is not None: expression(node.data)
    return resources


@dataclass(frozen=True)
class DAGNode:
    node_id: int
    kind: str
    instruction: object = None
    wire: object = None

    @property
    def name(self):
        return "measure" if isinstance(self.instruction, Measurement) else getattr(self.instruction, "name", self.kind)


class DAGCircuit:
    def __init__(self, num_qubits, num_bits=0, *, name=None, metadata=None, limits=None):
        template = Circuit(num_qubits, num_bits, name, metadata)
        self.num_qubits, self.num_bits = template.num_qubits, template.num_bits
        self.name, self.metadata = name, deepcopy(template.metadata)
        self.limits = limits or CompilationLimits()
        self._operations, self._next_id = [], 0
        self._extra_edges = set()
        self.nodes, self.edges = {}, {}
        self._rebuild()

    def _rebuild(self):
        resources = {("q", index) for index in range(self.num_qubits)} | {("c", index) for index in range(self.num_bits)}
        for node in self._operations: resources.update(instruction_resources(node.instruction))
        if len(self._operations) + 2 * len(resources) > self.limits.max_dag_nodes:
            raise MemoryError("Native DAG exceeds max_dag_nodes.")
        self.nodes = {node.node_id: node for node in self._operations}
        self.edges = {node.node_id: set() for node in self._operations}
        last, identifier = {}, -1
        for resource in sorted(resources, key=str):
            self.nodes[identifier] = DAGNode(identifier, "input", wire=resource)
            self.edges[identifier], last[resource] = set(), identifier
            identifier -= 1
        for node in self._operations:
            used = instruction_resources(node.instruction)
            control = node.instruction.metadata.get("control_node")
            if control is not None and control.kind not in {"declare", "store"}:
                used = resources
            for resource in used:
                self.edges[node.node_id].add(last[resource])
                last[resource] = node.node_id
        for resource in sorted(resources, key=str):
            self.nodes[identifier] = DAGNode(identifier, "output", wire=resource)
            self.edges[identifier] = {last[resource]}
            identifier -= 1
        self._extra_edges = {(a,b) for a,b in self._extra_edges if a in self.nodes and b in self.nodes}
        for a,b in self._extra_edges: self.edges[b].add(a)

    def _check_item(self, item):
        if not isinstance(item, (Operation, Measurement)): raise TypeError("DAG operation requires a native Operation/Measurement.")
        for kind, index in instruction_resources(item):
            if kind in {"q", "c"}:
                width = self.num_qubits if kind == "q" else self.num_bits
                if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < width: raise ValueError("DAG references an invalid wire.")

    def apply_operation_back(self, item): return self.insert_operation(len(self._operations), item)
    def apply_operation_front(self, item): return self.insert_operation(0, item)

    def insert_operation(self, index, item):
        self._check_item(item)
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index <= len(self._operations): raise ValueError("Invalid DAG insertion index.")
        node = DAGNode(self._next_id, "operation", deepcopy(item))
        self._next_id += 1
        self._operations.insert(index, node)
        try: self._rebuild()
        except Exception:
            self._operations.remove(node)
            self._rebuild()
            raise
        return node

    def _index(self, node):
        identifier = node.node_id if isinstance(node, DAGNode) else node
        return next(index for index, current in enumerate(self._operations) if current.node_id == identifier)

    def remove_op_node(self, node):
        self._operations.pop(self._index(node)); self._rebuild()

    def substitute_node(self, node, operation):
        self._check_item(operation)
        index = self._index(node)
        old = self._operations[index]
        if instruction_resources(old.instruction) != instruction_resources(operation): raise ValueError("DAG substitution must preserve the resource interface.")
        replacement = DAGNode(old.node_id, "operation", deepcopy(operation))
        self._operations[index] = replacement
        self._rebuild()
        return replacement

    def substitute_node_with_dag(self, node, replacement):
        replacement.validate()
        index = self._index(node)
        old = self._operations[index]
        items = [entry.instruction for entry in replacement.topological_op_nodes()]
        used = set().union(*(instruction_resources(item) for item in items)) if items else set()
        if used != instruction_resources(old.instruction): raise ValueError("Replacement DAG resource interface differs.")
        self._operations.pop(index)
        for item in reversed(items): self.insert_operation(index, item)
        self._rebuild()

    def replace_block(self, nodes, operation):
        indices = sorted(self._index(node) for node in nodes)
        if not indices or indices != list(range(indices[0], indices[-1] + 1)):
            raise UnsupportedFeature("noncontiguous_dag_block", "Block replacement currently requires consecutive nodes.")
        used = set().union(*(instruction_resources(self._operations[index].instruction) for index in indices))
        if used != instruction_resources(operation): raise ValueError("Block replacement resource interface differs.")
        del self._operations[indices[0]:indices[-1] + 1]
        self.insert_operation(indices[0], operation)

    def add_dependency(self, predecessor, successor):
        a = predecessor.node_id if isinstance(predecessor, DAGNode) else predecessor
        b = successor.node_id if isinstance(successor, DAGNode) else successor
        if a not in self.nodes or b not in self.nodes: raise ValueError("Unknown DAG dependency node.")
        self.edges[b].add(a)
        self._extra_edges.add((a,b))

    def _order(self):
        if set(self.edges) != set(self.nodes) or any(parent not in self.nodes for values in self.edges.values() for parent in values): raise ValueError("Malformed DAG dependency references.")
        try: return tuple(TopologicalSorter({key: sorted(values) for key, values in self.edges.items()}).static_order())
        except CycleError as exc: raise ValueError("Native circuit DAG contains a cycle.") from exc

    def topological_op_nodes(self): return tuple(self.nodes[key] for key in self._order() if self.nodes[key].kind == "operation")
    def predecessors(self, node): return tuple(self.nodes[key] for key in sorted(self.edges[node.node_id]))
    def successors(self, node): return tuple(self.nodes[key] for key in sorted(self.nodes) if node.node_id in self.edges[key])
    def size(self): return len(self._operations)
    def width(self): return sum(node.kind == "input" for node in self.nodes.values())
    def count_ops(self): return dict(Counter(node.name for node in self._operations))

    def longest_path(self):
        distances, parents = {}, {}
        for key in self._order():
            parent = max(self.edges[key], key=lambda value: distances[value], default=None)
            parents[key] = parent
            distances[key] = (0 if parent is None else distances[parent]) + int(self.nodes[key].kind == "operation")
        key = max(distances, key=distances.get, default=None)
        path = []
        while key is not None:
            if self.nodes[key].kind == "operation": path.append(self.nodes[key])
            key = parents[key]
        return tuple(reversed(path))

    def depth(self, *, structural=False):
        if not structural and any(node.instruction.metadata.get("control_node") is not None for node in self._operations):
            raise UnsupportedFeature("dynamic_depth", "Use structural=True for block structure; runtime loop depth is path-dependent.")
        return len(self.longest_path())

    def validate(self):
        positions = {key: index for index, key in enumerate(self._order())}
        if {key for key, node in self.nodes.items() if node.kind == "operation"} != {node.node_id for node in self._operations}: raise ValueError("DAG operation inventory is inconsistent.")
        def precedes(parent, child):
            pending, seen = [child], set()
            while pending:
                current = pending.pop()
                if parent in self.edges[current]: return True
                seen.add(current)
                pending.extend(value for value in self.edges[current] if value not in seen)
            return False
        last = {}
        for node in self._operations:
            self._check_item(node.instruction)
            for resource in instruction_resources(node.instruction):
                if resource in last and (positions[last[resource]] >= positions[node.node_id] or not precedes(last[resource], node.node_id)): raise ValueError("DAG violates circuit wire ordering.")
                last[resource] = node.node_id
        from quantumbridge.core.control_flow import ExecutionLimits
        validate_control_flow(dag_to_circuit(self, validate=False), limits=ExecutionLimits(max_control_depth=self.limits.max_control_nesting))
        if self.depth(structural=True) > self.limits.max_compilation_depth: raise MemoryError("DAG exceeds max_compilation_depth.")
        return True


def circuit_to_dag(circuit, limits=None):
    from quantumbridge.core.circuit_metadata import is_separated, inherit_execution
    dag = DAGCircuit(circuit.num_qubits, circuit.num_bits, name=circuit.name,
        metadata=None if is_separated(circuit) else circuit.metadata, limits=limits)
    if is_separated(circuit):dag.metadata=circuit.metadata;inherit_execution(circuit,dag)
    from quantumbridge.core.control_flow import ExecutionLimits
    validate_control_flow(circuit, limits=ExecutionLimits(max_control_depth=dag.limits.max_control_nesting))
    total = 0
    def budget(body):
        nonlocal total
        levels = {}
        for item in body.iter_instructions():
            total += 1
            if total > dag.limits.max_dag_nodes: raise MemoryError("Nested native DAG exceeds max_dag_nodes.")
            node = item.metadata.get("control_node")
            used = instruction_resources(item)
            if node is not None and node.kind not in {"declare", "store"}:
                used |= {(kind,index) for kind,width in (("q",body.num_qubits),("c",body.num_bits)) for index in range(width)} | set(levels)
            level = 1 + max((levels.get(resource,0) for resource in used),default=0)
            if level > dag.limits.max_compilation_depth: raise MemoryError("Block DAG exceeds max_compilation_depth.")
            for resource in used: levels[resource] = level
            if node is not None:
                for child in node.bodies: budget(child)
    budget(circuit)
    for index, item in enumerate(circuit.iter_instructions()):
        if index >= dag.limits.max_dag_nodes: raise MemoryError("Native DAG exceeds max_dag_nodes.")
        dag._check_item(item)
        dag._operations.append(DAGNode(index, "operation", deepcopy(item)))
    dag._next_id = len(dag._operations)
    dag._rebuild()
    dag.validate()
    return dag


def dag_to_circuit(dag, validate=True):
    if validate: dag._order()
    from quantumbridge.core.circuit_metadata import empty_execution_circuit
    circuit = empty_execution_circuit(dag)
    for node in dag.topological_op_nodes():
        item = deepcopy(node.instruction)
        if isinstance(item, Measurement):
            item.metadata["operation_index"] = len(circuit.operations)
            circuit.measurements.append(item)
        else: circuit.operations.append(item)
    return circuit
