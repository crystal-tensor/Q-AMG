# Copyright 2026 QuantumBridge Contributors. Apache-2.0.
"""Exact TF contractions of the existing native physical MPS tensor chain."""

import numpy as np

from quantumbridge.core.classical import UnsupportedFeature
from quantumbridge.execution.derivative_planning import tensorflow_mps_work
from quantumbridge.information.matrix_product_state import mps_routing
from quantumbridge.utils.math import gate_matrix, Hamiltonian
from .tensorflow_whole_execution import TensorFlowStatevectorPlan, _eager_admission


class TensorFlowMPSPlan(TensorFlowStatevectorPlan):
    """Readonly native operations; exact bounded index splits, no gauge SVD.

    The split M = I M (or M I) retains a full local basis. It is not a
    compression: all coefficients and their AD tangents survive, including
    at rank-deficient points. Static bond admission may conservatively refuse
    a state whose optimal Schmidt representation would fit the device.
    """

    work_planner = staticmethod(tensorflow_mps_work)

    def __init__(self, plan, *, derivative_order=2, readonly=True, allow_parameter_shift=False):
        super().__init__(plan, derivative_order=derivative_order, readonly=readonly,
            allow_parameter_shift=allow_parameter_shift)

    @staticmethod
    def _supports_fixed(operation, fixed):
        return TensorFlowStatevectorPlan._supports_fixed(operation, fixed) or (
            operation.name == "unitary" and not operation.params
            and set(operation.metadata) == {"matrix"} and 1 <= len(operation.wires) <= 2)

    @staticmethod
    def _two_site(tensors, matrix, site):
        import tensorflow as tf
        first, second = tensors[site:site+2]
        left, right = first.shape[0], second.shape[2]
        block = tf.einsum("aib,bjc->aijc", first, second)
        block = tf.einsum("ijuv,auvc->aijc", tf.reshape(matrix, (2, 2, 2, 2)), block)
        if 2*left <= 2*right:
            rank = 2*left
            tensors[site] = tf.reshape(tf.eye(rank, dtype=tf.complex128), (left, 2, rank))
            tensors[site+1] = tf.reshape(block, (rank, 2, right))
        else:
            rank = 2*right
            tensors[site] = tf.reshape(block, (left, 2, rank))
            tensors[site+1] = tf.reshape(tf.eye(rank, dtype=tf.complex128), (rank, 2, right))

    def chain(self, point):
        import tensorflow as tf
        tensors = [tf.constant([[[1.], [0.]]], tf.complex128) for _ in range(self.width)]
        scale = tf.constant(1., tf.complex128)
        for wires, matrix in self._evolution(point):
            if not wires:
                scale *= matrix
            elif len(wires) == 1:
                tensors[wires[0]] = tf.einsum("ij,ajb->aib", matrix, tensors[wires[0]])
            else:
                start, swaps = mps_routing(wires, self.width)
                swap = tf.constant(gate_matrix("swap"), tf.complex128)
                for site in swaps:
                    self._two_site(tensors, swap, site)
                self._two_site(tensors, matrix, start)
                for site in reversed(swaps):
                    self._two_site(tensors, swap, site)
        return tuple(tensors), scale

    def state(self, point):
        raise UnsupportedFeature("tensorflow_mps_full_state", "Use local MPS contractions, not a global statevector.")

    def values(self, point):
        import tensorflow as tf
        from .tensorflow_native_execution import _finite_parameters
        tensors, scale = self.chain(point)
        results = []
        for request in self._requests:
            observable = request.observable
            terms = observable.terms if isinstance(observable, Hamiltonian) else ((1., observable),)
            result = tf.constant(0., tf.float64)
            for coefficient, word in terms:
                labels = dict(word.terms)
                env = tf.ones((1, 1), tf.complex128)
                for wire, tensor in enumerate(tensors):
                    local = tf.constant(gate_matrix(labels.get(wire, "I").lower()), tf.complex128)
                    env = tf.einsum("ab,air,ij,bjs->rs", env, tf.math.conj(tensor), local, tensor)
                result += float(coefficient)*tf.math.real(env[0, 0]*tf.math.conj(scale)*scale)
            results.append(result)
        flat = tf.stack(results)
        return tf.where(_finite_parameters()(flat), flat, tf.zeros_like(flat))


def select_tensorflow_mps(kernel, packed):
    import tensorflow as tf
    from quantumbridge.devices.matrix_product_state_device import MatrixProductStateDevice
    from quantumbridge.execution.resource_context import current_resources
    from quantumbridge.execution.resource_ledger import UnifiedResourceLedger
    from .tensorflow_native_derivatives import bounded_tensor_function
    if (type(kernel.node.device) is not MatrixProductStateDevice or packed.dtype != tf.float64
            or kernel.program is not None or kernel.node.diff_method not in {"best", "mps-tangent"}):
        return None
    device = kernel.node.device
    if (device.state_options["cutoff"] != 0 or device.initial_state is not None
            or device.noise_model is not None):
        return None
    # Keep other established MPS methods and sampling paths unchanged.
    if (kernel.shots.partitions != (None,) or any(index is not None
            for references in kernel.coefficients for index in references)):
        return None
    plan = kernel.plan(np.ones(kernel.size))
    if plan.gradient_capability(kernel.node.diff_method) != "mps-tangent":
        return None
    if any(r.kind != "expval" or r.metadata or r.shots is not None for r in plan.requests):
        return None
    try:
        whole = TensorFlowMPSPlan(plan, derivative_order=kernel.node.max_diff)
    except UnsupportedFeature as error:
        # These are missing tensor rules, not approval of a different source.
        # The unchanged native host path still performs its own admission.
        if error.feature in {"tensorflow_local_power_width", "tensorflow_local_power_primitive",
                "tensorflow_local_repeated_fractional_power", "tensorflow_mps_local_width",
                "tensorflow_whole_operation"}:
            return None
        raise
    active = current_resources()
    ledger, receipt = None, {}
    if kernel.node.unified_limits is not None:
        ledger = UnifiedResourceLedger(kernel.node.unified_limits, parent=None if active is None else active.ledger)
    elif active is not None:
        ledger = active.ledger
    admission = None
    if ledger is not None:
        if not tf.executing_eagerly():
            raise UnsupportedFeature("tensorflow_managed_graph_execution", "TF MPS cached graphs cannot account runtime execution through tracing.")
        ledger.preflight(memory_bytes=whole.work["workspace_bytes"],
            derivative_work=whole.work["planned_work"], tapes=6)
        admission = _eager_admission(ledger, whole.work, receipt)
    function = bounded_tensor_function(whole.values, admission, max_order=kernel.node.max_diff)
    return function, whole.work, receipt
