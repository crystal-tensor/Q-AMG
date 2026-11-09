# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# Bounded native analysis/transformation execution with versioned properties.

from copy import deepcopy
import json
from time import perf_counter

from .limits import CompilationLimits
from .pass_base import CompilerPass, AnalysisPass, PassResult
from .report import CompilerReport


def _comparison_snapshot(circuit):
    # Internal pass mutation detection is not a wire-format import/export.
    from quantumbridge.execution.resource_context import current_resources
    scope = current_resources()
    if scope is not None:
        scope.ledger.unavailable("compiler_comparison_python_heap",
            "Canonical pass-comparison strings use the existing JSON encoder; Python container/string allocator memory is not an owned-array or strict RSS bound.")
    from quantumbridge.core.symbolic_limits import current_symbolic_limits
    from quantumbridge.serialization.limits import SerializationLimits
    limits = SerializationLimits(max_symbolic_expression_depth=current_symbolic_limits().max_depth)
    from quantumbridge.ir.qb_ir import _DefinitionCodec
    context=_DefinitionCodec(limits,include_user_metadata=False)
    return json.dumps(circuit.to_ir().to_dict(_context=context), sort_keys=True)


class PropertySet(dict):
    def __init__(self):
        super().__init__()
        self.version = 0

    def invalidate(self, preserved=()):
        keep = set(preserved)
        for key in tuple(self):
            if key not in keep: del self[key]
        self.version += 1


class PassManager:
    def __init__(self, passes=None, *, limits=None, callback=None, seed=0, max_iterations=1):
        self.passes = list(passes or [])
        self.limits = limits or CompilationLimits()
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or not 1 <= max_iterations <= self.limits.max_pass_iterations:
            raise ValueError("Invalid bounded pass iteration count.")
        self.callback, self.seed, self.max_iterations = callback, seed, max_iterations
        self.property_set = PropertySet()

    def run(self, circuit, callback=None):
        circuit = circuit.lowered()
        from .dag import circuit_to_dag
        from .traversal import transform_segments
        circuit_to_dag(circuit, self.limits)
        from quantumbridge.core.circuit_metadata import execution_copy
        current = execution_copy(circuit)
        analyses, trace, changed = {}, [], False
        self.property_set = PropertySet()
        count = 0
        self.metrics = []
        for iteration in range(self.max_iterations):
            iteration_changed = False
            for compiler_pass in self.passes:
                if count >= self.limits.max_pass_iterations: raise MemoryError("Compiler exceeds max_pass_iterations.")
                if not isinstance(compiler_pass, CompilerPass): raise TypeError("PassManager entries must be CompilerPass instances.")
                missing = set(getattr(compiler_pass, "requires", ())) - self.property_set.keys()
                if missing: raise ValueError(f"Pass requirements are absent or invalidated: {sorted(missing)}.")
                compiler_pass.property_set, compiler_pass.seed = self.property_set, self.seed
                before = _comparison_snapshot(current)
                started = perf_counter()
                if current.has_dynamic_operations and not getattr(compiler_pass, "handles_dynamic", False) and not isinstance(compiler_pass, AnalysisPass):
                    results = []
                    def transform(segment):
                        result = compiler_pass.run(segment)
                        if not isinstance(result, PassResult): raise TypeError("Compiler passes must return PassResult.")
                        results.append(result)
                        return result.circuit
                    compiled = transform_segments(current, transform)
                    result = PassResult(compiled, {"segment_analyses": [item.analyses for item in results]}, any(item.changed for item in results))
                else:
                    result = compiler_pass.run(current)
                if not isinstance(result, PassResult): raise TypeError("Compiler passes must return PassResult.")
                actual_change = before != _comparison_snapshot(result.circuit)
                if isinstance(compiler_pass, AnalysisPass) and (actual_change or before != _comparison_snapshot(current) or result.changed):
                    raise ValueError("AnalysisPass must not mutate the native circuit.")
                current = result.circuit
                circuit_to_dag(current, self.limits)
                if actual_change:
                    self.property_set.invalidate(getattr(compiler_pass, "preserves", ()))
                self.property_set.update(result.analyses)
                analyses.update(result.analyses)
                changed |= actual_change
                iteration_changed |= actual_change
                duration = perf_counter() - started
                entry = {"pass": type(compiler_pass).__name__, "iteration": iteration, "count": count,
                         "changed": actual_change, "property_version": self.property_set.version}
                trace.append(entry)
                self.metrics.append({**entry, "duration_seconds": duration})
                selected_callback = callback or self.callback
                if selected_callback is not None:
                    selected_callback(pass_=compiler_pass, circuit=current.copy(), property_set=deepcopy(self.property_set),
                                      count=count, duration=duration)
                count += 1
            if not iteration_changed: break
        analyses["pass_trace"] = trace
        if circuit.has_dynamic_operations: analyses["dynamic_execution_order"] = True
        analyses["current_properties"] = dict(self.property_set)
        analyses["compiler_report"] = CompilerReport(len(circuit.operations), len(current.operations), analyses.copy(), changed).to_dict()
        return PassResult(current, analyses, changed)


class StagedPassManager:
    def __init__(self, stages):
        self.stages = dict(stages)

    def run(self, circuit):
        current, analyses, changed = circuit, {}, False
        for name, manager in self.stages.items():
            if not isinstance(manager, PassManager): raise TypeError("Stages must contain native PassManager objects.")
            result = manager.run(current)
            current, changed = result.circuit, changed or result.changed
            analyses[name] = result.analyses
        return PassResult(current, {"stages": analyses}, changed)
