# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
from dataclasses import dataclass
from numbers import Integral
from .errors import ResourceLimitError


@dataclass(frozen=True)
class SerializationLimits:
    max_input_bytes: int = 16777216
    max_tokens: int = 250000
    max_ast_nodes: int = 100000
    max_expression_depth: int = 48
    max_include_depth: int = 8
    max_gate_expansion: int = 100000
    max_circuits: int = 256
    max_instructions: int = 100000
    max_registers: int = 1024
    max_qubits: int = 4096
    max_clbits: int = 65536
    max_parameter_nodes: int = 100000
    max_metadata_bytes: int = 1048576
    max_custom_definitions: int = 1024
    max_nesting_depth: int = 32
    max_symbolic_expression_depth: int = 48
    max_numeric_elements: int = 1048576
    max_result_labels: int = 1024
    max_json_depth: int = 256

    def __post_init__(self):
        if any(isinstance(value,bool) or not isinstance(value,Integral) or value < 1 for value in self.__dict__.values()):
            raise ValueError("Serialization limits must be positive integers.")

    def check(self, field, value, **location):
        if value > getattr(self,field): raise ResourceLimitError(f"Input exceeds {field}.",**location)
