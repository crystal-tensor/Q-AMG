# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is part of QuantumBridge SDK.
# This implementation is developed for the QuantumBridge native architecture.

from .channels import _validate_probability
from numbers import Integral
import numpy as np
from .quantum_error import QuantumError


class ReadoutError:
    def __init__(self, p0_to_1, p1_to_0=None):
        if p1_to_0 is None:
            matrix=np.asarray(p0_to_1,float)
        else:
            a,b=_validate_probability(p0_to_1),_validate_probability(p1_to_0)
            matrix=np.array([[1-a,a],[b,1-b]])
        if matrix.ndim!=2 or matrix.shape[0]!=matrix.shape[1] or len(matrix)<2 or len(matrix)&(len(matrix)-1) or len(matrix)>64 or not np.all(np.isfinite(matrix)) or np.any(matrix<0) or not np.allclose(matrix.sum(axis=1),1,atol=1e-12,rtol=0):
            raise ValueError("Readout errors require a bounded row-stochastic matrix.")
        self.probabilities=matrix.copy()
        self.num_qubits=len(matrix).bit_length()-1
        if self.num_qubits==1: self.p0_to_1,self.p1_to_0=float(matrix[0,1]),float(matrix[1,0])


class NoiseModel:
    def __init__(self):
        self.channels = []
        self.readout_errors = {}
        self.quantum_errors = {}
        self.local_quantum_errors = {}
        self.default_readout_error = None
        self.correlated_readout_errors = {}

    def add_channel(self, channel, wires=None):
        self.channels.append({"channel": channel, "wires": None if wires is None else tuple(wires)})
        return self

    def add_readout_error(self, wire, error):
        if isinstance(wire,ReadoutError):
            wire,error=tuple(error),wire
        else: wire=(wire,)
        self._qargs(wire,error.num_qubits)
        if len(wire)==1: self.readout_errors[int(wire[0])]=error
        else: self.correlated_readout_errors[tuple(wire)]=error
        return self

    @staticmethod
    def _qargs(wires,width):
        if len(wires)!=width or len(set(wires))!=len(wires) or any(isinstance(w,bool) or not isinstance(w,Integral) or w<0 for w in wires):
            raise ValueError("Noise qargs must match error width and be distinct nonnegative integers.")

    @staticmethod
    def _names(instructions):
        names=(instructions,) if isinstance(instructions,str) else tuple(instructions)
        if not names or any(not isinstance(n,str) or not n or n.startswith(("save_","set_")) or n in {"barrier","delay","control_flow"} for n in names):
            raise ValueError("Noise bindings need executable gate/reset/measure names.")
        return names

    def add_all_qubit_quantum_error(self,error,instructions):
        if not isinstance(error,QuantumError): raise TypeError("Expected native QuantumError.")
        for name in self._names(instructions):
            previous=self.quantum_errors.get(name)
            self.quantum_errors[name]=error if previous is None else previous.compose(error)
        return self

    def add_quantum_error(self,error,instructions,qubits):
        if not isinstance(error,QuantumError): raise TypeError("Expected native QuantumError.")
        qubits=tuple(qubits); self._qargs(qubits,error.num_qubits)
        for name in self._names(instructions):
            key=(name,qubits); previous=self.local_quantum_errors.get(key)
            self.local_quantum_errors[key]=error if previous is None else previous.compose(error)
        return self

    def add_all_qubit_readout_error(self,error):
        if not isinstance(error,ReadoutError) or error.num_qubits!=1: raise ValueError("Global readout error must be one-qubit.")
        self.default_readout_error=error
        return self

    def error_for(self,name,wires):
        error=self.local_quantum_errors.get((name,tuple(wires)),self.quantum_errors.get(name))
        if error is not None and error.num_qubits!=len(wires): raise ValueError("Bound gate and quantum error widths differ.")
        return error

    def readout_for(self,wires):
        wires=tuple(wires)
        return self.correlated_readout_errors.get(wires) if len(wires)>1 else self.readout_errors.get(wires[0],self.default_readout_error)

    @property
    def requires_dynamic_engine(self):
        return bool(self.quantum_errors or self.local_quantum_errors or self.correlated_readout_errors or self.default_readout_error)

    def validate(self,width):
        for _,wires in self.local_quantum_errors:
            if max(wires)>=width: raise ValueError("Physical gate noise qargs exceed circuit width.")
        for wires in self.correlated_readout_errors:
            if max(wires)>=width: raise ValueError("Correlated readout qargs exceed circuit width.")
        if any(w>=width for w in self.readout_errors): raise ValueError("Readout wire exceeds circuit width.")
        if len(self.quantum_errors)+len(self.local_quantum_errors)>1024: raise MemoryError("Too many noise bindings.")
        return True

    def summary(self):
        basic={
            "channels": [entry["channel"].name for entry in self.channels],
            "readout_errors": sorted(self.readout_errors),
        }
        if not self.requires_dynamic_engine:return basic
        return {**basic,
            "gate_errors":sorted(self.quantum_errors),
            "physical_gate_errors":[{"instruction":n,"qubits":list(q)} for n,q in self.local_quantum_errors],
            "correlated_readout":[list(q) for q in self.correlated_readout_errors],
            "global_readout":self.default_readout_error is not None,
            "noise_binding":"compiled physical instructions; legacy add_channel remains terminal",
        }
