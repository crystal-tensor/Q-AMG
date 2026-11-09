# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# Standard payloads and explicitly external native metadata, never fake QPY.
from copy import deepcopy
from dataclasses import dataclass, field
import json
from quantumbridge.core import Parameter
from quantumbridge.compat.qiskit_adapter import circuit_from_ir
from quantumbridge.ir import IRProgram
from .limits import SerializationLimits
from .errors import FormatValidationError, UnsupportedFeature
from .text_api import qasm2_dumps,qasm2_loads,qasm3_dumps,qasm3_loads
from .qpy import qpy_dumps,qpy_loads


@dataclass(frozen=True)
class SerializedCircuit:
    format: str
    version: str
    data: object
    sidecar: dict = field(default_factory=dict)


def serialize_circuit(circuit,format="qasm3",*,metadata_policy="reject",**options):
    if metadata_policy not in {"reject","external"}: raise ValueError("metadata_policy must be reject or external.")
    sidecar = {}
    source = circuit
    if metadata_policy == "external":
        from quantumbridge.core.circuit_metadata import execution_metadata, is_separated, serialized_user_metadata
        if format in {"qasm2","qasm3","qpy"} and "classical_scope" in execution_metadata(circuit):
            raise UnsupportedFeature("An external metadata artifact cannot replace typed classical scope transport.",format=format)
        sidecar = {"name":circuit.name,"metadata":deepcopy(serialized_user_metadata(circuit)),
                   "parameter_uuids":{parameter.name:parameter.uuid for parameter in circuit.parameters}}
        source = circuit.copy()
        source.name = None
        source.metadata = {} if is_separated(source) else {key:value for key,value in source.metadata.items() if key in {"global_phase","registers"}}
        if format in {"qasm2","qasm3"}:
            mapping = {parameter:Parameter(parameter.name) for parameter in source.parameters if parameter.uuid is not None}
            from quantumbridge.core.circuit import Circuit
            source = Circuit.assign_parameters(source,mapping,strict=False,_copy_user_metadata=False)
            source.name = None
    if format == "qasm2": data,version = qasm2_dumps(source,**options),"2.0"
    elif format == "qasm3": data,version = qasm3_dumps(source,**options),options.get("version","3.0")
    elif format == "qpy": data,version = qpy_dumps(source,**options),str(options.get("version",13))
    elif format == "ir-json": data,version = source.to_ir().to_json(),"qb-ir-v0.1"
    else: raise UnsupportedFeature("Unknown circuit serialization format.",format=format)
    return SerializedCircuit(format,version,data,sidecar)


def deserialize_circuit(data,format=None,**options):
    sidecar = None
    if isinstance(data,SerializedCircuit):
        if format is not None and format != data.format: raise FormatValidationError("Artifact/source format mismatch.",format=format)
        format,sidecar,data = data.format,data.sidecar,data.data
    if format == "qasm2": result = qasm2_loads(data,**options)
    elif format == "qasm3": result = qasm3_loads(data,**options)
    elif format == "qpy":
        circuits = qpy_loads(data,**options)
        if len(circuits) != 1: raise FormatValidationError("Use qpy_loads for multiple circuits.",format=format)
        result = circuits[0]
    elif format == "ir-json":
        limits = options.get("limits") or SerializationLimits()
        limits.check("max_input_bytes",len(data.encode()),format=format)
        try: result = circuit_from_ir(IRProgram.from_json(data))
        except (ValueError,KeyError,TypeError,RecursionError) as exc: raise FormatValidationError("Invalid executable IR JSON.",format=format) from exc
    else: raise UnsupportedFeature("Unknown circuit serialization format.",format=format)
    if sidecar:
        if set(sidecar) != {"name","metadata","parameter_uuids"}: raise FormatValidationError("Invalid external sidecar schema.",format=format)
        result.name = sidecar["name"]
        from quantumbridge.core.circuit_metadata import is_separated
        if not is_separated(result):result._execution_metadata=deepcopy(result.metadata)
        result.metadata=deepcopy(sidecar["metadata"])
        mapping = {parameter:Parameter(parameter.name,sidecar["parameter_uuids"][parameter.name]) for parameter in result.parameters if parameter.name in sidecar["parameter_uuids"]}
        result = result.assign_parameters(mapping,strict=False)
    return result


def convert_circuit_format(data,source_format,target_format,*,source_options=None,target_options=None):
    return serialize_circuit(deserialize_circuit(data,source_format,**(source_options or {})),target_format,**(target_options or {}))


def validate_serialization_roundtrip(circuit,format="qasm3",*,verify_execution=False,**options):
    restored = deserialize_circuit(serialize_circuit(circuit,format,**options))
    if (restored.num_qubits,restored.num_bits) != (circuit.num_qubits,circuit.num_bits) or {p.name for p in restored.parameters} != {p.name for p in circuit.parameters}:
        raise FormatValidationError("Roundtrip changed resources or free parameter names.",format=format)
    if verify_execution:
        if circuit.parameters or circuit.num_qubits > 8: raise UnsupportedFeature("Explicit binding and at most eight qubits are required for exhaustive execution validation.",format=format)
        import numpy as np
        if circuit.has_dynamic_operations or restored.has_dynamic_operations:
            from quantumbridge import DensityMatrixDevice
            device = DensityMatrixDevice()
            np.testing.assert_allclose(device.density_matrix(circuit),device.density_matrix(restored),atol=1e-12)
            a,b = device.run(circuit).probabilities(),device.run(restored).probabilities()
            labels = set(a)|set(b)
            np.testing.assert_allclose([a.get(key,0) for key in labels],[b.get(key,0) for key in labels],atol=1e-12)
        else:
            from quantumbridge.information import Operator
            np.testing.assert_allclose(Operator.from_circuit(circuit).data,Operator.from_circuit(restored).data,atol=1e-12)
    return {"circuit":restored,"format":format,"structural_fields_validated":True,"execution_checked":verify_execution}
