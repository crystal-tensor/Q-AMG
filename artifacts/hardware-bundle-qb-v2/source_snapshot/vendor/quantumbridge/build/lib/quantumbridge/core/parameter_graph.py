# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
"""Intern decoded symbol identities without borrowing external numeric storage."""

import numpy as np

from .parameters import Parameter
from .gate_limits import current_gate_limits


def canonical_parameter_graph(circuit, symbols):
    from quantumbridge.execution.array_ownership import native_configuration_copy, _defined_type
    from quantumbridge.execution.resource_context import current_resources
    pending,seen,memo,changed=[circuit],set(),{},False
    # Only newly decoded, already-owned graphs enter this helper. Operation
    # snapshots share their decoded symbols and keep their existing arrays.
    while pending:
        item=pending.pop()
        if id(item) in seen:continue
        seen.add(id(item))
        current_gate_limits().check("max_decomposition_work",len(seen)+len(pending))
        if isinstance(item,Parameter):
            canonical=symbols.get(item.name)
            if type(canonical) is type(item) and canonical.uuid==item.uuid:
                memo[id(item)]=canonical
                changed |= canonical is not item
            continue
        if type(item) is np.ndarray:
            memo[id(item)]=item
            continue
        if type(item) is dict:
            current_gate_limits().check("max_decomposition_work",len(seen)+len(pending)+2*len(item))
            pending.extend(item.keys());children=item.values()
        elif type(item) in (tuple,list,set,frozenset):children=item
        elif _defined_type(type(item),"quantumbridge"):
            children=object.__getattribute__(item,"__dict__").values()
        else:continue
        current_gate_limits().check("max_decomposition_work",len(seen)+len(pending)+len(children))
        pending.extend(children)
    if not changed:return circuit
    return native_configuration_copy(circuit,memo=memo,category="native_decoded_parameter_interning",
        strict=current_resources() is not None)
