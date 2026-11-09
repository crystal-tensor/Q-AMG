# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
"""Native Tape/QNode dispatch without a second queue or quantum executor."""

from functools import update_wrapper

from quantumbridge.qml.tape import Tape


def bind_qnode(target, program, qnode_options=None):
    from quantumbridge.qml.qnode import QNode
    if isinstance(target, QNode):
        if qnode_options is not None:
            raise TypeError("An existing QNode already owns its device/wire configuration.")
        return target.with_transform(program)
    if callable(target) and qnode_options is not None:
        return QNode(target, **dict(qnode_options)).with_transform(program)
    raise TypeError("Transform dispatch needs a Tape or QNode; a native callable also needs explicit qnode_options including num_qubits.")


class NativeTransform:
    """A configured native transform with Tape, QNode and explicit callable dispatch."""

    def __init__(self, function, args=(), options=None, qnode_options=None):
        if not callable(function):
            raise TypeError("A native transform implementation must be callable.")
        self.function, self.args = function, tuple(args)
        self.options = dict(options or {})
        self.qnode_options = None if qnode_options is None else dict(qnode_options)
        update_wrapper(self, function)

    def __call__(self, target=None, /, *args, qnode_options=None, **options):
        arguments = self.args + args
        settings = {**self.options, **options}
        node_options = self.qnode_options if qnode_options is None else qnode_options
        if isinstance(target, Tape):
            if node_options is not None:
                raise TypeError("Tape dispatch does not accept QNode construction options.")
            from .program import check_batch, resolve_limits, copy_for_transform
            check_batch((target,),resolve_limits(settings.get("limits")))
            return self.function(copy_for_transform(target), *arguments, **settings)
        if target is None:
            return NativeTransform(self.function, arguments, settings, node_options)
        configured = NativeTransform(self.function, arguments, settings)
        from .program import TransformProgram
        return bind_qnode(target, TransformProgram((configured,)), node_options)


def transform(function):
    """Register an independently implemented Tape -> (Tapes, processor) function."""
    return function if isinstance(function, NativeTransform) else NativeTransform(function)
