# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is part of QuantumBridge SDK.
# This implementation is developed for the QuantumBridge native architecture.


from dataclasses import dataclass, field
from copy import deepcopy

from quantumbridge.core.measurements import Measurement


def copy_terminal_measurements(source, target):
    if source.has_dynamic_operations:
        raise ValueError("Use PassManager for circuits with dynamic execution boundaries.")
    target.measurements = [
        Measurement(item.wire, item.bit, item.kind, {**deepcopy(item.metadata), "operation_index": len(target.operations)})
        for item in source.measurements
    ]


@dataclass
class PassResult:
    circuit: object
    analyses: dict = field(default_factory=dict)
    changed: bool = False


class CompilerPass:
    def run(self, circuit):
        raise NotImplementedError("QuantumBridge compiler passes must implement run().")


class AnalysisPass(CompilerPass):
    pass


class TransformationPass(CompilerPass):
    pass
