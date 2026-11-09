# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# Method selection reuses native devices, the structured interpreter and Job.
from copy import deepcopy
from dataclasses import replace
from itertools import count
from numbers import Integral
import numpy as np
from quantumbridge.core import Circuit
from quantumbridge.core.classical import UnsupportedFeature
from quantumbridge.core.control_flow import ExecutionLimits
from quantumbridge.providers import Backend,Job
from quantumbridge.providers.options import Options
from quantumbridge.providers.backend import _as_circuit_sequence,_parameter_sets_for_circuits
from quantumbridge.results import Result
from .limits import AerExecutionLimits
from .snapshots import SnapshotContext


class LinearMapDevice:
    def __init__(self,method,limits):
        self.simulation=method;self.limits=limits;self.noise_model=None;self.max_qubits=None

    def _initial(self,width):
        size=1<<width
        if self.simulation=="superop":size*=size
        self.limits.check("max_matrix_dimension",size);self.limits.check("max_state_bytes",16*size*size*4)
        from quantumbridge.execution.resource_context import current_resources
        if current_resources() is None:return np.eye(size,dtype=complex)
        from quantumbridge.execution import mps_arrays as arrays
        result=arrays.filled("linear_map_initial",(size,size),complex,0)
        result.flat[::size+1]=1
        return result

    def map_gate(self,state,matrix,wires,width):
        from quantumbridge.information.operator import _expand_subsystem_operator
        full=_expand_subsystem_operator(matrix,tuple(wires),width)
        from quantumbridge.execution.resource_context import current_resources
        if current_resources() is not None:
            from quantumbridge.execution import mps_arrays as arrays
            from quantumbridge.information import SuperOp
            return arrays.contracted(full if self.simulation=="unitary" else SuperOp.from_operator(full).data,state)
        return full@state if self.simulation=="unitary" else np.kron(full.conj(),full)@state

    def map_channel(self,state,operators,wires,width):
        if self.simulation!="superop":raise UnsupportedFeature("unitary_channel")
        from quantumbridge.information.operator import _expand_subsystem_operator
        from quantumbridge.execution.resource_context import current_resources
        if current_resources() is not None:
            from quantumbridge.execution import mps_arrays as arrays
            from quantumbridge.information import SuperOp
            result=arrays.filled("linear_map_channel",state.shape,complex,0)
            for operator in operators:
                full=_expand_subsystem_operator(operator,tuple(wires),width)
                channel=SuperOp.from_operator(full).data
                np.add(result,channel,out=result)
            return arrays.contracted(result,state)
        full=[_expand_subsystem_operator(op,tuple(wires),width) for op in operators]
        return sum(np.kron(op.conj(),op) for op in full)@state


class AerResult:
    def __init__(self,results):
        self.results=tuple(results);self.success=True;self.status="COMPLETED"

    def data(self,index=0):return self.results[index].data()
    def get_counts(self,index=None):
        if index is None:return self.results[0].counts() if len(self.results)==1 else [r.counts() for r in self.results]
        return self.results[index].counts()
    def get_memory(self,index=0):return self.results[index].memory()
    def to_dict(self,*,serialization_limits=None):return {"format":"qb-aer-result-v1","success":self.success,"status":self.status,"results":[r.to_dict(serialization_limits=serialization_limits) for r in self.results]}
    def to_json(self,*,serialization_limits=None):
        from quantumbridge.schema.serialization import to_json
        return to_json(self.to_dict(serialization_limits=serialization_limits),limits=serialization_limits,allow_nan=False)
    @classmethod
    def from_dict(cls,value,*,serialization_limits=None):
        from quantumbridge.schema.serialization import validate_payload_structure
        validate_payload_structure(value,limits=serialization_limits)
        if set(value)!={"format","success","status","results"} or value["format"]!="qb-aer-result-v1" or value["success"] is not True or value["status"]!="COMPLETED":raise ValueError("Invalid native Aer result envelope.")
        if not isinstance(value["results"],list) or not value["results"]:raise ValueError("Native Aer result needs experiments.")
        result=cls([Result.from_dict(v,serialization_limits=serialization_limits) for v in value["results"]])
        for item in result.results:item.data()
        return result
    @classmethod
    def from_json(cls,value,*,serialization_limits=None):
        from quantumbridge.schema.serialization import from_json
        return cls.from_dict(from_json(value,limits=serialization_limits),serialization_limits=serialization_limits)


_COUNTER=count(1)


class NativeAerSimulator(Backend):
    METHODS=("automatic","statevector","density_matrix","matrix_product_state","stabilizer","unitary","superop")
    DEFAULTS={"method":"automatic","shots":1024,"seed_simulator":None,"memory":False,"noise_model":None,
              "precision":"double","device":"CPU","max_memory_mb":256,"max_bond_dim":None,"cutoff":1e-16,
              "aer_limits":None,"execution_limits":None}

    def __init__(self,method="automatic",**options):
        super().__init__(name="quantumbridge_native_aer")
        self.options=Options(**self._validate({**self.DEFAULTS,"method":method,**options}))

    @classmethod
    def _validate(cls,options):
        if set(options)-set(cls.DEFAULTS):raise UnsupportedFeature("aer_options","Unsupported Aer options: "+str(sorted(set(options)-set(cls.DEFAULTS))))
        if options["method"] not in cls.METHODS:raise UnsupportedFeature("simulator_method",str(options["method"]))
        if options["device"]!="CPU" or options["precision"]!="double":raise UnsupportedFeature("execution_device_precision","Only actual CPU double-precision execution is implemented.")
        limits=options.get("aer_limits")
        if limits is None:limits=AerExecutionLimits()
        if not isinstance(limits,AerExecutionLimits):raise TypeError("aer_limits requires AerExecutionLimits.")
        shots=options["shots"]
        if shots is not None:
            if isinstance(shots,bool) or not isinstance(shots,Integral) or shots<1:raise ValueError("shots must be positive integer or None.")
            limits.check("max_shots",shots);limits.check("max_trajectories",shots)
        if not isinstance(options["memory"],bool):raise TypeError("memory must be boolean.")
        seed=options["seed_simulator"]
        if seed is not None and (isinstance(seed,bool) or not isinstance(seed,Integral) or seed<0):raise ValueError("seed_simulator must be a nonnegative integer.")
        if not isinstance(options["max_memory_mb"],(int,float)) or isinstance(options["max_memory_mb"],bool) or not np.isfinite(options["max_memory_mb"]) or options["max_memory_mb"]<=0:raise ValueError("max_memory_mb must be finite positive.")
        if options["max_bond_dim"] is not None and (isinstance(options["max_bond_dim"],bool) or not isinstance(options["max_bond_dim"],Integral) or options["max_bond_dim"]<1):raise ValueError("Invalid MPS bond limit.")
        if isinstance(options["cutoff"],bool) or not np.isfinite(options["cutoff"]) or not 0<=options["cutoff"]<1:raise ValueError("Invalid MPS cutoff.")
        if options["noise_model"] is not None:
            from quantumbridge.noise import NoiseModel
            if not isinstance(options["noise_model"],NoiseModel):raise TypeError("Only native NoiseModel is accepted.")
        if options["execution_limits"] is not None and not isinstance(options["execution_limits"],ExecutionLimits):raise TypeError("Invalid execution limits.")
        return dict(options)

    def set_options(self,**fields):self.options=Options(**self._validate({**self.options.to_dict(),**fields}))
    def available_methods(self):return self.METHODS
    def available_devices(self):return ("CPU",)

    def configuration(self):
        return {**super().configuration(),"conditional":True,"simulation_methods":list(self.METHODS),"native_implementation":True,"full_aer_parity":False}

    def run(self,circuits,**options):
        parameter_values=options.pop("parameter_values",None)
        actual=Options(**self._validate({**deepcopy(self.options.to_dict()),**options}))
        sequence=_as_circuit_sequence(circuits)
        return Job(f"native-aer-{next(_COUNTER)}",backend=self,
                   executor=lambda:self._execute(sequence,actual,parameter_values),metadata={"backend_name":self.name,"num_circuits":len(sequence)})

    @staticmethod
    def _walk(circuit):
        for item in circuit.iter_instructions():
            yield item
            node=item.metadata.get("control_node")
            if node is not None:
                for body in node.bodies:yield from NativeAerSimulator._walk(body)

    def _select(self,circuit,options):
        method=options["method"]
        if method!="automatic":return method
        kinds={op.metadata["simulator_instruction"]["kind"] for op in self._walk(circuit) if "simulator_instruction" in op.metadata}
        for name in ("unitary","superop","matrix_product_state","stabilizer"):
            if "set_"+name in kinds or "save_"+name in kinds:return name
        if "save_clifford" in kinds:return "stabilizer"
        if kinds & {"save_statevector","save_statevector_dict","save_amplitudes","set_statevector"}:return "statevector"
        if circuit.num_qubits>20:return "matrix_product_state"
        return "density_matrix" if options["noise_model"] is not None or "set_density_matrix" in kinds else "statevector"

    def _execute(self,circuits,options,parameter_values=None,observable=None):
        from quantumbridge.devices import StatevectorDevice,DensityMatrixDevice,MatrixProductStateDevice,StabilizerDevice
        from quantumbridge.execution.control_flow_engine import DynamicExecutionEngine
        options=options.to_dict()
        results=[]
        for circuit,parameters in zip(circuits,_parameter_sets_for_circuits(parameter_values,len(circuits))):
            model = options["noise_model"]
            if model is not None and (model.quantum_errors or model.local_quantum_errors) and any(
                    "gate" in op.metadata or "annotated_operation" in op.metadata for op in self._walk(circuit)):
                raise UnsupportedFeature("composite_gate_noise_boundary", "Named instruction noise needs preserved composite execution boundaries.")
            circuit=circuit.bind(parameters) if parameters else circuit.copy()
            method=self._select(circuit,options);limits=options["aer_limits"] or AerExecutionLimits()
            if method!="matrix_product_state" and (options["max_bond_dim"] is not None or options["cutoff"]!=1e-16):raise UnsupportedFeature("non_mps_options")
            if options["shots"] is None and options["memory"]:raise UnsupportedFeature("analytic_memory","Shot memory requires finite shots.")
            dense=method in {"statevector","density_matrix","unitary","superop"}
            if dense:
                limits.check("max_qubits_dense",circuit.num_qubits)
                exponent=circuit.num_qubits*(1 if method=="statevector" else 4 if method=="superop" else 2)
                limits.check("max_state_bytes",16*(1<<exponent)*8)
                if 16*(1<<exponent)*8>options["max_memory_mb"]*1024**2:raise MemoryError("Dense method exceeds max_memory_mb.")
            if method=="stabilizer":
                estimated=32*circuit.num_qubits**2
                limits.check("max_state_bytes",estimated)
                if estimated>options["max_memory_mb"]*1024**2:raise MemoryError("Tableau method exceeds max_memory_mb.")
            model=options["noise_model"]
            if model is not None:
                model.validate(circuit.num_qubits)
                errors=list(model.quantum_errors.values())+list(model.local_quantum_errors.values())
                limits.check("max_noise_channels",len(errors)+len(model.channels)+len(model.readout_errors)+len(model.correlated_readout_errors)+int(model.default_readout_error is not None))
                for error in errors:limits.check("max_noise_kraus_ops",len(error.kraus()))
            execution=options["execution_limits"] or ExecutionLimits(max_shots=limits.max_shots,max_branches=limits.max_dynamic_branches)
            execution=replace(execution,max_shots=min(execution.max_shots,limits.max_shots),max_branches=min(execution.max_branches,limits.max_dynamic_branches))
            memory=min(options["max_memory_mb"],limits.max_state_bytes/1024**2)
            if method=="statevector":device=StatevectorDevice(max_memory_mb=memory,execution_limits=execution)
            elif method=="density_matrix":device=DensityMatrixDevice(max_memory_mb=memory,execution_limits=execution)
            elif method=="matrix_product_state":device=MatrixProductStateDevice(max_memory_mb=memory,max_bond_dim=options["max_bond_dim"],cutoff=options["cutoff"],execution_limits=execution)
            elif method=="stabilizer":device=StabilizerDevice(execution_limits=execution)
            else:
                if circuit.has_measurements or method=="unitary" and (model is not None or any(op.name in {"reset","kraus"} for op in self._walk(circuit))):
                    raise UnsupportedFeature("linear_map_nonunitary","Unitary rejects reset/noise/measurement; superop rejects measurement.")
                device=LinearMapDevice(method,limits);device.execution_limits=execution
            device.noise_model=model
            context=SnapshotContext(circuit,method,limits,finite_shots=options["shots"] is not None)
            engine=DynamicExecutionEngine(device,circuit,limits=execution,snapshot_context=context)
            shots=options["shots"]
            if shots is not None and method not in {"unitary","superop"}:
                result=engine.shots(shots,options["seed_simulator"],options["memory"])
            else:
                records=engine.branches();probs={}
                if circuit.has_measurements:
                    for _,bits,_,weight in records:
                        label="".join(map(str,bits));probs[label]=probs.get(label,0.)+weight
                state=density=mps=None
                from quantumbridge.execution.resource_context import current_resources
                scope=current_resources()
                if method=="statevector":
                    if len(records)==1:state=records[0][0]
                    else:
                        limits.check("max_state_bytes",16*(1<<(2*circuit.num_qubits)))
                        if scope is None:density=sum(w*np.outer(s,s.conj()) for s,_,_,w in records)
                elif method=="density_matrix" and scope is None:density=sum(w*s for s,_,_,w in records)
                elif method=="matrix_product_state" and len(records)==1:mps=records[0][0]
                if scope is not None and (method=="density_matrix" or method=="statevector" and len(records)>1):
                    from quantumbridge.execution import mps_arrays as arrays
                    dimension=1<<circuit.num_qubits
                    density=arrays.filled("aer_ensemble_density",(dimension,dimension),complex,0)
                    for value,_,_,weight in records:
                        contribution=(arrays.einsum("i,j->ij",(dimension,dimension),value,arrays.conjugated(value))
                            if method=="statevector" else arrays.copied(value,category="aer_ensemble_contribution"))
                        np.multiply(contribution,weight,out=contribution)
                        np.add(density,contribution,out=density)
                result=Result(state=state,density=density,mps=mps,probabilities_data=probs or None,metadata_data=records.metadata,
                              mps_snapshots=getattr(records,"mps_snapshots",{}))
            result.metadata_data.update(method=method,requested_method=options["method"],backend_name=self.name,
                shots=None if method in {"unitary","superop"} else shots,requested_shots=shots,
                seed=options["seed_simulator"],success=True,noise_model_metadata={} if model is None else model.summary(),
                native_implementation=True,full_aer_parity=False,qubit_order="q0-first",classical_order="c0-first")
            results.append(result)
        return AerResult(results)
