# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
"""Independent public snapshots of the native circuit parameter ordering."""
from collections.abc import Iterator, MappingView
from copy import deepcopy
from itertools import islice

from .gate_limits import current_gate_limits
from .symbolic_limits import current_symbolic_limits


def _admit(*values):
    from .composition import _admit_graph
    _admit_graph(values)


def _check_size(size):
    current_gate_limits().check("max_gate_parameters",size)
    current_symbolic_limits().check("max_parameters",size)


def _stream(values):
    _admit(values)
    if type(values) is ParameterView:values=values.data
    if type(values) in (tuple,list,set,frozenset,dict,range):_check_size(len(values))
    limit=min(current_gate_limits().max_gate_parameters,current_symbolic_limits().max_parameters)
    for index,value in enumerate(islice(iter(values),limit+1),1):
        _check_size(index)
        yield value


def _items(values):return tuple(_stream(values))


def _membership_operand(values):
    # Membership in a one-shot public operand consumes it, unlike a set copy.
    return _stream(values) if isinstance(values,Iterator) else values


class ParameterView(MappingView):
    """Ordered parameter snapshot with public set queries and local mutation.

    The data list belongs to the snapshot, never to a circuit or binding plan.
    """

    def __init__(self,parameters=None):
        self.data=list(_items(()) if parameters is None else _items(parameters))

    def _validate(self,*others):
        _admit(self,*others)
        _check_size(len(self.data))
        for other in others:
            if type(other) is ParameterView:other=other.data
            if type(other) in (tuple,list,set,frozenset,dict,range):_check_size(len(other))

    def __len__(self):
        self._validate()
        return len(self.data)

    def __iter__(self):
        self._validate()
        return iter(self.data)

    def __getitem__(self,index):
        self._validate(index)
        return self.data[index]

    def __contains__(self,value):
        self._validate(value)
        return value in self.data

    def __repr__(self):
        self._validate()
        return "ParameterView("+repr(self.data)+")"

    def copy(self):
        self._validate()
        return ParameterView(self.data.copy())

    def __copy__(self):
        self._validate()
        result=object.__new__(type(self))
        result.data=self.data
        return result

    def __deepcopy__(self,memo):
        self._validate()
        result=object.__new__(type(self))
        memo[id(self)]=result
        result.data=deepcopy(self.data,memo)
        return result

    def remove(self,value):
        self._validate(value)
        self.data.remove(value)

    def __eq__(self,other):
        self._validate(other)
        return set(self.data)==set(_items(other))

    def __ne__(self,other):return not self==other

    def __le__(self,other):
        self._validate(other)
        other=_membership_operand(other)
        return all(value in other for value in self.data)

    def __lt__(self,other):return self!=other and self<=other

    def __ge__(self,other):
        self._validate(other)
        return all(value in self.data for value in _stream(other))

    def __gt__(self,other):return self!=other and self>=other

    def __and__(self,other):
        self._validate(other)
        other=_membership_operand(other)
        return ParameterView([value for value in self.data if value in other])

    def __rand__(self,other):return self&other

    def __iand__(self,other):
        result=self&other
        self.data[:]=result.data
        return self

    def __or__(self,other):
        self._validate(other)
        left,right=set(self.data),set(_items(other))
        _check_size(len(left)+sum(value not in left for value in right))
        return left|right

    def __ror__(self,other):return self|other

    def __xor__(self,other):
        self._validate(other)
        left,right=set(self.data),set(_items(other))
        _check_size(sum(value not in right for value in left)+sum(value not in left for value in right))
        return left^right

    def __sub__(self,other):
        self._validate(other)
        return set(self.data)-set(_items(other))

    def isdisjoint(self,other):
        self._validate(other)
        return not any(value in self.data for value in _stream(other))
