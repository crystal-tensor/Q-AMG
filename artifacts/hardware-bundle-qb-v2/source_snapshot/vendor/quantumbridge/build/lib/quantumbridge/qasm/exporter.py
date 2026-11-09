# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# Native legacy entry point, sharing the strict supported-subset serializer.


def dumps(circuit) -> str:
    from quantumbridge.serialization import qasm2_dumps
    return qasm2_dumps(circuit)
