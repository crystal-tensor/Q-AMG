# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# Bounded in-program observation; no final-state substitution for snapshots.
from copy import deepcopy
from itertools import product
import json
import numpy as np
from quantumbridge.core.classical import UnsupportedFeature
from quantumbridge.ir.qb_ir import matrix_from_ir
from .limits import AerExecutionLimits


def _snapshot_array(shape,dtype,category="snapshot_array"):
    from quantumbridge.execution.resource_context import current_resources
    from quantumbridge.execution.array_ownership import retained_array
    scope=current_resources()
    if scope is not None:
        scope.ledger.unavailable("snapshot_python_heap",
            "Snapshot numerical buffers are owned; JSON transport sizes and Python/container/iterator allocator memory are not observed RSS.")
    return retained_array(None if scope is None else scope.ledger,category,shape,dtype)


def _snapshot_copy(value):
    result=_snapshot_array(value.shape,value.dtype,"snapshot_defensive_copy")
    np.copyto(result,value,casting="no")
    return result


def encode(value):
    from quantumbridge.information.matrix_product_state import MatrixProductState
    if isinstance(value,MatrixProductState): return value.to_dict()
    if isinstance(value,np.ndarray):
        return {"qb_simulator_array":True,"real":value.real.tolist(),"imag":value.imag.tolist(),"complex":bool(np.iscomplexobj(value))}
    if isinstance(value,np.generic): return encode(value.item())
    if isinstance(value,complex): return {"qb_simulator_complex":[value.real,value.imag]}
    if isinstance(value,dict): return {key:encode(child) for key,child in value.items()}
    if isinstance(value,(tuple,list)): return [encode(child) for child in value]
    return value


def decode(value):
    if isinstance(value,dict):
        if value.get("qb_simulator_array") is True:
            if set(value)!={"qb_simulator_array","real","imag","complex"}: raise ValueError("Malformed simulator array.")
            from quantumbridge.execution.resource_context import current_resources
            scope=current_resources()
            if scope is not None:
                from quantumbridge.execution.array_reservations import numeric_shape,numeric_array,_numeric_leaves
                from quantumbridge.execution.array_ownership import retained_array
                if not isinstance(value["complex"],bool):raise ValueError("Invalid simulator array type.")
                shape=numeric_shape(value["real"],real_only=True)
                if shape!=numeric_shape(value["imag"],real_only=True):raise ValueError("Invalid simulator array values.")
                if value["complex"]:
                    result=matrix_from_ir({"real":value["real"],"imag":value["imag"]})
                else:
                    if any(scalar!=0 for scalar in _numeric_leaves(value["imag"])):raise ValueError("Invalid simulator array type.")
                    result=numeric_array(value["real"],float,category="snapshot_real_import",copy=True)
                valid=retained_array(scope.ledger,"snapshot_import_validation",shape,bool)
                np.isfinite(result,out=valid)
                if not np.all(valid):raise ValueError("Invalid simulator array values.")
                return result
            real,imag=np.asarray(value["real"],float),np.asarray(value["imag"],float)
            if real.shape!=imag.shape or not np.all(np.isfinite(real)) or not np.all(np.isfinite(imag)): raise ValueError("Invalid simulator array values.")
            if not isinstance(value["complex"],bool) or not value["complex"] and np.any(imag):raise ValueError("Invalid simulator array type.")
            return real+1j*imag if value["complex"] else real
        if set(value)=={"qb_simulator_complex"}:
            result=complex(*value["qb_simulator_complex"])
            if not np.isfinite(result):raise ValueError("Invalid simulator complex value.")
            return result
        return {key:decode(child) for key,child in value.items()}
    if isinstance(value,list): return [decode(child) for child in value]
    return value


class SnapshotContext:
    def __init__(self,circuit,method,limits=None,*,finite_shots=False):
        self.circuit,self.method,self.limits=circuit,method,limits or AerExecutionLimits()
        self.finite_shots=finite_shots
        self.cells,self.labels,self.bytes,self.peak_bytes={},{},0,0
        self.count,self.groups=0,set()
        def visit(body):
            for item in body.operations:
                node=item.metadata.get("control_node")
                if node is not None:
                    for child in node.bodies: visit(child)
                spec=item.metadata.get("simulator_instruction")
                if spec is None: continue
                if item.controls or item.name!=spec.get("kind"): raise ValueError("Malformed simulator instruction.")
                from quantumbridge.devices.execution import validate_wires
                if not item.wires or tuple(item.wires)!=validate_wires(item.wires,circuit.num_qubits):raise ValueError("Invalid simulator wires.")
                if item.name.startswith("save_"):
                    if any(not isinstance(spec.get(flag),bool) for flag in ("pershot","conditional","unnormalized")):raise ValueError("Malformed snapshot modes.")
                    label=spec.get("label")
                    if not isinstance(label,str) or not label or label in self.labels: raise ValueError("Missing or duplicate snapshot label.")
                    self.labels[label]=spec
                    if "paulis" in spec and any(not isinstance(local,str) or len(local)!=len(item.wires) or set(local)-set("IXYZ") or not np.isfinite(coefficient) or not np.isreal(coefficient) for local,coefficient in spec["paulis"]):raise ValueError("Invalid saved Pauli observable.")
                    if "indices" in spec:
                        from numbers import Integral
                        if not spec["indices"] or any(isinstance(i,bool) or not isinstance(i,Integral) or not 0<=i<1<<circuit.num_qubits for i in spec["indices"]):raise ValueError("Invalid saved amplitude indices.")
                    self.limits.check("max_snapshot_count",len(self.labels))
                    if method in {"unitary","superop"} and spec["pershot"]:raise UnsupportedFeature("map_pershot","Linear-map snapshots are deterministic, not per-shot state trajectories.")
                    if spec["unnormalized"] and spec["kind"] not in {"save_density_matrix","save_probabilities","save_probabilities_dict","save_expectation_value","save_expectation_value_variance","save_amplitudes_squared"}:
                        raise UnsupportedFeature("unnormalized_save_kind")
        visit(circuit)

    def _density(self,state,wires):
        size=1<<len(wires)
        self.limits.check("max_saved_density_dimension",size)
        self.limits.check("max_snapshot_bytes",16*size*size)
        if self.method=="matrix_product_state": return state.reduced_density_matrix(wires)
        if self.method=="density_matrix":
            from quantumbridge.execution.resource_context import current_resources
            if current_resources() is not None:
                # Preserve the existing partial_trace ordering (sorted kept
                # axes) without intermediate trace matrices.
                keep=tuple(sorted(wires))
                rest=tuple(q for q in range(self.circuit.num_qubits) if q not in keep)
                result=_snapshot_array((size,size),complex,"snapshot_reduced_density")
                result.fill(0)
                def embedded(value,axes):
                    return sum(((value>>(len(axes)-1-index))&1)<<(self.circuit.num_qubits-1-wire) for index,wire in enumerate(axes))
                for row in range(size):
                    left=embedded(row,keep)
                    for column in range(size):
                        right=embedded(column,keep)
                        result[row,column]=sum(state[left|embedded(index,rest),right|embedded(index,rest)] for index in range(1<<len(rest)))
                return result
            from quantumbridge.information.partial_trace import partial_trace
            return partial_trace(state,wires,self.circuit.num_qubits)
        if self.method=="statevector":
            rest=tuple(q for q in range(self.circuit.num_qubits) if q not in wires)
            from quantumbridge.execution.resource_context import current_resources
            if current_resources() is not None:
                from quantumbridge.execution import mps_arrays as arrays
                tensor=arrays.reshaped(state,(2,)*self.circuit.num_qubits).transpose(wires+rest)
                partition=arrays.reshaped(tensor,(size,1<<len(rest)))
                return arrays.contracted(partition,arrays.conjugated(partition).T)
            partition=state.reshape((2,)*self.circuit.num_qubits).transpose(wires+rest).reshape(size,-1)
            return partition@partition.conj().T
        if self.method=="stabilizer":
            if len(wires)>5: raise UnsupportedFeature("tableau_reduced_density","Tableau Pauli expansion is bounded to five saved wires.")
            from quantumbridge.utils.math import pauli_matrix
            from quantumbridge.operators import Pauli
            from quantumbridge.execution.resource_context import current_resources
            if current_resources() is not None:
                from quantumbridge.utils.math import _pauli_dense
                result=_snapshot_array((size,size),complex,"snapshot_tableau_density")
                result.fill(0)
                for local in product("IXYZ",repeat=len(wires)):
                    label=["I"]*self.circuit.num_qubits
                    for wire,char in zip(wires,local):label[wire]=char
                    matrix=_pauli_dense("".join(local))
                    np.multiply(matrix,state.expectation_value(Pauli("".join(label)))/size,out=matrix)
                    np.add(result,matrix,out=result)
                return result
            result=np.zeros((size,size),complex)
            for local in product("IXYZ",repeat=len(wires)):
                label=["I"]*self.circuit.num_qubits; matrix=np.ones((1,1),complex)
                for wire,char in zip(wires,local): label[wire]=char; matrix=np.kron(matrix,pauli_matrix(char))
                result+=state.expectation_value(Pauli("".join(label)))*matrix/size
            return result
        raise UnsupportedFeature("density_snapshot_method")

    def _vector(self,state):
        self.limits.check("max_saved_state_dimension",1<<self.circuit.num_qubits)
        self.limits.check("max_snapshot_bytes",16*(1<<self.circuit.num_qubits))
        if self.method=="statevector": return _snapshot_copy(state)
        if self.method=="matrix_product_state": return state.statevector()
        if self.method=="stabilizer": return state.to_statevector().data
        raise UnsupportedFeature("statevector_snapshot_method","A mixed-state method cannot save a single pure statevector.")

    def _amplitudes(self,state,indices):
        if self.method=="matrix_product_state":
            from quantumbridge.execution import mps_arrays as arrays
            values=_snapshot_array((len(indices),),complex,"snapshot_mps_amplitudes")
            for offset,index in enumerate(indices):
                vector=arrays.filled("snapshot_amplitude_boundary",(1,),complex,1.)
                for wire,tensor in enumerate(state._tensors):
                    vector=arrays.contracted(vector,tensor[:,(index>>(self.circuit.num_qubits-1-wire))&1,:])
                values[offset]=vector[0]
            return values
        vector=self._vector(state)
        result=_snapshot_array((len(indices),),complex,"snapshot_selected_amplitudes")
        for offset,index in enumerate(indices):result[offset]=vector[index]
        return result

    def _moments(self,state,wires,spec):
        if "paulis" in spec:
            from quantumbridge.operators import SparsePauliOperator,Pauli
            terms=[]
            for local,coefficient in spec["paulis"]:
                label=["I"]*self.circuit.num_qubits
                for wire,char in zip(wires,local): label[wire]=char
                terms.append(("".join(label),coefficient))
            operator=SparsePauliOperator(terms)
            if self.method in {"matrix_product_state","stabilizer"}:
                def expectation(op):
                    return float(sum(c*(state.expectation_pauli(label) if self.method=="matrix_product_state" else state.expectation_value(Pauli(label))) for label,c in op.terms).real)
                return expectation(operator),expectation(operator.dot(operator))
            from quantumbridge.utils.math import pauli_matrix,apply_unitary
            if self.method not in {"statevector","density_matrix"}:raise UnsupportedFeature("expectation_snapshot_method")
            # Act on vector/row tensor axes, never allocate an exponential Pauli matrix.
            def expectation(op):
                total=0j
                for label,coefficient in op.terms:
                    from quantumbridge.execution.resource_context import current_resources
                    if current_resources() is not None:
                        from quantumbridge.utils.math import _pauli_permutation
                        flip,parity,phase=_pauli_permutation(label)
                        value=0j
                        for column in range(1<<self.circuit.num_qubits):
                            row=column^flip
                            signed=-phase if (column&parity).bit_count()&1 else phase
                            value+=signed*(state[column,row] if self.method=="density_matrix" else complex(state[row]).conjugate()*state[column])
                        total+=coefficient*value
                        continue
                    evolved=state.reshape(-1)
                    width=self.circuit.num_qubits*(2 if self.method=="density_matrix" else 1)
                    for wire,char in enumerate(label):
                        if char!="I":evolved=apply_unitary(evolved,pauli_matrix(char),(wire,),width)
                    value=np.trace(evolved.reshape(state.shape)) if self.method=="density_matrix" else np.vdot(state,evolved)
                    total+=coefficient*value
                return float(total.real)
            return expectation(operator),expectation(operator.dot(operator))
        self.limits.check("max_saved_density_dimension",1<<len(wires))
        self.limits.check("max_snapshot_bytes",16*(1<<(2*len(wires))))
        matrix=matrix_from_ir(spec["observable"])
        from quantumbridge.utils.math import validate_observable_matrix
        if not validate_observable_matrix(matrix,len(wires),atol=1e-8,rtol=1e-5):raise ValueError("Invalid saved observable.")
        rho=self._density(state,wires)
        from quantumbridge.execution.resource_context import current_resources
        if self.method=="matrix_product_state" or current_resources() is not None:
            from quantumbridge.execution import mps_arrays as arrays
            product=arrays.contracted(rho,matrix)
            return float(np.trace(product).real),float(np.trace(arrays.contracted(product,matrix)).real)
        return float(np.trace(rho@matrix).real),float(np.trace(rho@matrix@matrix).real)

    def observe(self,item,branch,rng):
        spec=item.metadata["simulator_instruction"]; kind=spec["kind"][5:]
        if spec["pershot"] and not self.finite_shots: raise UnsupportedFeature("analytic_pershot","pershot saves require finite-shot trajectories.")
        wires=tuple(item.wires);state=branch.state
        if kind=="state": kind={"statevector":"statevector","density_matrix":"density_matrix","matrix_product_state":"matrix_product_state","stabilizer":"stabilizer","unitary":"unitary","superop":"superop"}[self.method]
        full={"statevector","statevector_dict","matrix_product_state","stabilizer","clifford","unitary","superop","amplitudes","amplitudes_squared"}
        if kind in full and wires!=tuple(range(self.circuit.num_qubits)): raise ValueError("This save instruction requires the full ordered register.")
        if kind=="matrix_product_state" and self.method==kind:
            return self._observe_mps(spec,branch)
        group=hex(sum(bit<<index for index,bit in enumerate(branch.store.bits))) if spec["conditional"] else ""
        key=(spec["label"],group)
        groups=self.groups|{key} if spec["conditional"] else self.groups
        self.limits.check("max_conditional_groups",len(groups))
        raw=kind in {"statevector","statevector_dict","matrix_product_state","stabilizer","clifford","unitary","superop","amplitudes"}
        if kind=="statevector": value=self._vector(state)
        elif kind=="statevector_dict": value={hex(i):complex(v) for i,v in enumerate(self._vector(state)) if abs(v)>1e-15}
        elif kind=="density_matrix": value=self._density(state,wires)
        elif kind in {"probabilities","probabilities_dict"}:
            self.limits.check("max_saved_state_dimension",1<<len(wires))
            if self.method=="matrix_product_state": probabilities=state.probabilities(wires)
            elif self.method=="statevector":
                rest=tuple(q for q in range(self.circuit.num_qubits) if q not in wires)
                from quantumbridge.execution.resource_context import current_resources
                if current_resources() is not None:
                    from quantumbridge.execution import mps_arrays as arrays
                    partition=arrays.reshaped(state,(2,)*self.circuit.num_qubits).transpose(wires+rest)
                    partition=arrays.reshaped(partition,(1<<len(wires),1<<len(rest)))
                    values=_snapshot_array((partition.shape[0],),float,"snapshot_probability_reduction")
                    for index,row in enumerate(partition):values[index]=np.vdot(row,row).real
                else:
                    partition=state.reshape((2,)*self.circuit.num_qubits).transpose(wires+rest).reshape(1<<len(wires),-1)
                    values=np.sum(abs(partition)**2,axis=1)
                probabilities={format(i,f"0{len(wires)}b"):float(v) for i,v in enumerate(values)}
            else:
                rho=self._density(state,wires); probabilities={format(i,f"0{len(wires)}b"):float(v.real) for i,v in enumerate(np.diag(rho))}
            value=_snapshot_array((1<<len(wires),),float,"snapshot_probabilities")
            for i in range(len(value)):value[i]=probabilities.get(format(i,f"0{len(wires)}b"),0.)
            if kind.endswith("dict"): value={hex(i):float(v) for i,v in enumerate(value) if v>0}
        elif kind in {"amplitudes","amplitudes_squared"}:
            if kind=="amplitudes_squared" and self.method=="density_matrix":
                value=_snapshot_array((len(spec["indices"]),),float,"snapshot_selected_probabilities")
                for offset,index in enumerate(spec["indices"]):value[offset]=state[index,index].real
            else:
                value=self._amplitudes(state,spec["indices"])
                if kind.endswith("squared"):
                    squared=_snapshot_array(value.shape,float,"snapshot_amplitude_probabilities")
                    np.absolute(value,out=squared);np.square(squared,out=squared)
                    value=squared
        elif kind in {"expectation_value","expectation_value_variance"}:
            first,second=self._moments(state,wires,spec)
            if kind.endswith("variance"):
                value=_snapshot_array((2,),float,"snapshot_expectation_variance")
                value[:]=first,max(0.,second-first*first)
            else:value=first
        elif kind in {"stabilizer","clifford"} and self.method=="stabilizer": value=state.clifford.to_dict()
        elif kind==self.method and kind in {"unitary","superop"}: value=_snapshot_copy(state)
        else: raise UnsupportedFeature("save_method",f"save_{kind} is unsupported by {self.method}.")
        size=len(json.dumps(encode(value),allow_nan=False).encode())
        from quantumbridge.execution.resource_context import current_resources
        if current_resources() is not None:
            return self._retain_managed(spec,branch,key,groups,value,size,raw)
        cell=self.cells.setdefault(key,{"weight":0.,"value":None,"bytes":0})
        self.groups=groups
        if spec["pershot"]:
            self.bytes+=size;self.limits.check("max_pershot_snapshot_bytes",self.bytes)
            if cell["value"] is None:cell["value"]=[]
            cell["value"].append(deepcopy(value))
        elif raw:
            if not self.finite_shots and cell["value"] is not None and json.dumps(encode(cell["value"]))!=json.dumps(encode(value)):
                raise UnsupportedFeature("mixed_pure_snapshot","An analytic mixed ensemble cannot be saved as one pure state; use density or pershot.")
            self.bytes+=size-cell["bytes"];cell["bytes"]=size;cell["value"]=deepcopy(value)
        else:
            if cell["value"] is None:cell["value"]={k:0. for k in value} if isinstance(value,dict) else np.zeros_like(value)
            if isinstance(value,dict):
                for k,v in value.items(): cell["value"][k]=cell["value"].get(k,0.)+branch.weight*v
            else: cell["value"]+=branch.weight*value
            actual=len(json.dumps(encode(cell["value"]),allow_nan=False).encode())
            self.bytes+=actual-cell["bytes"];cell["bytes"]=actual
        cell["weight"]+=branch.weight
        self.limits.check("max_snapshot_bytes",self.bytes);self.peak_bytes=max(self.peak_bytes,self.bytes)
        cell["raw"]=raw
        return [branch]

    def _retain_managed(self,spec,branch,key,groups,value,size,raw):
        old=self.cells.get(key)
        weight=branch.weight+(0 if old is None else old["weight"])
        if spec["pershot"]:
            saved=[*([] if old is None else old["value"]),value]
            prospective=self.bytes+size
            self.limits.check("max_pershot_snapshot_bytes",prospective)
        elif raw:
            if not self.finite_shots and old is not None and encode(old["value"])!=encode(value):
                raise UnsupportedFeature("mixed_pure_snapshot","An analytic mixed ensemble cannot be saved as one pure state; use density or pershot.")
            # Native producers already return an independent owned value.
            saved=value
            prospective=self.bytes+size-(0 if old is None else old["bytes"])
        else:
            if isinstance(value,dict):
                saved={} if old is None else dict(old["value"])
                for name,scalar in value.items():saved[name]=saved.get(name,0.)+branch.weight*scalar
            elif isinstance(value,np.ndarray):
                saved=_snapshot_array(value.shape,value.dtype,"snapshot_retained_aggregation")
                np.multiply(value,branch.weight,out=saved)
                if old is not None:np.add(saved,old["value"],out=saved)
            else:
                saved=_snapshot_array((),float,"snapshot_retained_aggregation")
                np.multiply(value,branch.weight,out=saved)
                if old is not None:np.add(saved,old["value"],out=saved)
            size=len(json.dumps(encode(saved),allow_nan=False).encode())
            prospective=self.bytes+size-(0 if old is None else old["bytes"])
        self.limits.check("max_snapshot_bytes",prospective)
        self.cells[key]={"weight":weight,"value":saved,"bytes":size,"raw":raw}
        self.groups=groups
        self.bytes=prospective
        self.peak_bytes=max(self.peak_bytes,self.bytes)
        return [branch]

    def _observe_mps(self,spec,branch):
        from quantumbridge.execution.resource_context import current_resources
        state=branch.state
        self.limits.check("max_snapshot_bytes",4*state.tensor_bytes)
        group=hex(sum(bit<<index for index,bit in enumerate(branch.store.bits))) if spec["conditional"] else ""
        key=(spec["label"],group)
        groups=self.groups|{key} if spec["conditional"] else self.groups
        self.limits.check("max_conditional_groups",len(groups))
        encoded=state.to_dict()
        size=len(json.dumps(encoded,allow_nan=False).encode())
        old=self.cells.get(key)
        prospective=self.bytes+size-(0 if spec["pershot"] or old is None else old["bytes"])
        self.limits.check("max_snapshot_bytes",prospective)
        if spec["pershot"]:self.limits.check("max_pershot_snapshot_bytes",prospective)
        elif not self.finite_shots and old is not None and encode(old["value"])!=encoded:
            raise UnsupportedFeature("mixed_pure_snapshot","An analytic mixed ensemble cannot be saved as one pure state; use density or pershot.")
        resources=current_resources()
        if resources is not None:
            resources.ledger.unavailable("snapshot_python_heap",
                "Snapshot JSON sizes are transport bytes, not observed Python heap, allocator peak or total process RSS.")
        # Copy only after transport/group checks and whole-MPS array admission.
        saved=state.copy()
        cell={"weight":branch.weight+(0 if old is None else old["weight"]),
              "value":([*([] if old is None else old["value"]),saved] if spec["pershot"] else saved),
              "bytes":size,"raw":True,"native_mps":True}
        self.cells[key]=cell
        self.groups=groups
        self.bytes=prospective
        self.peak_bytes=max(self.peak_bytes,self.bytes)
        return [branch]

    def native_mps_snapshots(self):
        """Transfer references to already-owned roots, without another copy."""
        result={}
        for (label,group),cell in self.cells.items():
            if not cell.get("native_mps"):continue
            value=list(cell["value"]) if self.labels[label]["pershot"] else cell["value"]
            if self.labels[label]["conditional"]:result.setdefault(label,{})[group]=value
            else:result[label]=value
        return result

    def replace(self,item,branch):
        spec=item.metadata["simulator_instruction"]; kind=spec["kind"][4:]
        if tuple(item.wires)!=tuple(range(self.circuit.num_qubits)):raise ValueError("Set requires the full ordered register.")
        if kind=="matrix_product_state" and self.method==kind:
            from quantumbridge.information import MatrixProductState
            state=MatrixProductState.from_dict(spec["value"],max_memory_mb=self.limits.max_state_bytes/1024**2)
        elif kind=="stabilizer" and self.method==kind:
            from quantumbridge.information import StabilizerState,Clifford
            state=StabilizerState(Clifford.from_dict(spec["value"]))
        elif kind in {"statevector","density_matrix","unitary","superop"}:
            state=matrix_from_ir(spec["value"])
            self.limits.check("max_state_bytes",state.nbytes)
            # Revalidate IR-deserialized payloads rather than trusting builder validation.
            from quantumbridge.core import Circuit
            from .instructions import set_state
            set_state(Circuit(self.circuit.num_qubits),kind,state)
            if kind=="statevector" and self.method=="density_matrix":state=np.outer(state,state.conj())
            elif kind=="statevector" and self.method=="matrix_product_state":
                from quantumbridge.information import MatrixProductState
                state=MatrixProductState.from_statevector(state)
            elif kind!=self.method:raise UnsupportedFeature("set_method",f"set_{kind} is unsupported by {self.method}.")
        else: raise UnsupportedFeature("set_method",f"set_{kind} is unsupported by {self.method}.")
        if kind in {"matrix_product_state","stabilizer"} and state.num_qubits!=self.circuit.num_qubits:raise ValueError("Replacement state width mismatch.")
        branch.state=state
        return [branch]

    def metadata(self):
        from quantumbridge.execution.array_ownership import native_storage_bytes
        result={}
        native_tensors=[]
        for (label,group),cell in self.cells.items():
            spec=self.labels[label];value=cell["value"]
            if cell.get("native_mps"):
                for state in value if spec["pershot"] else (value,):native_tensors.extend(state._tensors)
            if not spec["pershot"] and not cell["raw"] and not spec["unnormalized"]:
                if isinstance(value,dict):value={k:v/cell["weight"] for k,v in value.items()}
                elif isinstance(value,np.ndarray):
                    normalized=_snapshot_array(value.shape,value.dtype,"snapshot_normalized_export")
                    np.divide(value,cell["weight"],out=normalized)
                    value=normalized
                else:value=value/cell["weight"]
            if spec["conditional"]: result.setdefault(label,{})[group]=encode(value)
            else:result[label]=encode(value)
        return {"saved_snapshots":result,"snapshot_storage_bytes":self.peak_bytes,
                "snapshot_native_tensor_bytes":native_storage_bytes(native_tensors),"snapshot_labels":list(self.labels),
                "snapshot_aggregation":"trajectory/branch-weight average; raw states use last finite trajectory; conditional groups renormalized"}
