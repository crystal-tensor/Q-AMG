# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
"""Device protocol dispatching to existing native execution/derivative plans."""

from dataclasses import replace

import numpy as np

from .derivative_result import DerivativeExecutionResult
from .derivative_status import classify_derivative_error, detach_status_error
from .resource_context import ResourceScope, current_resources
from .resource_ledger import UnifiedResourceLedger


def _input_tree(array, prepared):
    from quantumbridge.diff.trace import unflatten_tree
    from .array_ownership import select_columns
    items = [unflatten_tree(select_columns(array, indices), prepared.output_spec, shape)
             for shape, indices in prepared.layout]
    return items[0] if prepared.single else tuple(items)


def execute_device_derivatives(device, request, *, ledger=None, return_status=False, _prepare=None):
    """Execute native mathematical plans; no quantum SDK dependency or fallback."""
    from .derivative_planning import prepare_derivative
    from quantumbridge.diff.trace import flatten_tree, unflatten_tree
    parent = current_resources()
    source_limits = getattr(getattr(request, "circuit_or_tape", None), "unified_limits", None)
    requested_limits = getattr(request, "limits", None) or source_limits
    if ledger is None:
        ledger = UnifiedResourceLedger(requested_limits,
            parent=None if parent is None else parent.ledger)
    elif not isinstance(ledger, UnifiedResourceLedger):
        raise TypeError("Device derivative ledger must be a UnifiedResourceLedger.")
    elif requested_limits is not None:
        ledger = UnifiedResourceLedger(requested_limits, parent=ledger)
    stage = "validation"
    try:
        with ResourceScope(ledger, new_request=True) as scope:
            prepared = (prepare_derivative if _prepare is None else _prepare)(device, request)
            plan, work = prepared.plan, prepared.work
            stage = "preflight"
            ledger.preflight(memory_bytes=prepared.buffer_bytes+prepared.device_bytes,
                tapes=work["tape_count"], shots=work["physical_shots"],
                derivative_work=work["derivative_work"])
            # This conservatively holds child outputs/accumulators together;
            # transient device storage is separately reserved by each execution.
            buffers = ledger.reserve("retained_derivative_buffers", memory_bytes=prepared.buffer_bytes)
            transferred = False
            try:
                stage = "execution"
                ledger.consume(derivative_work=work["derivative_work"], category="planned_derivative_work")
                scope.derivative_work_precharged = True
                matrix = second = None
                if request.derivative_order == 2:
                    base, matrix, second, spec = plan.second_derivatives(prepared.point,
                        prepared.node.diff_method, direction=prepared.direction,
                        classical=prepared.classical, classical_second=prepared.second,
                        limits=prepared.node.higher_order_limits)
                    if request.derivative_kind == "hvp":
                        derivative = _input_tree(second[:, :, 0], prepared)
                    else:
                        blocks = tuple(tuple(unflatten_tree(second[:, left, right], spec, shape+other_shape)
                            for other_shape, right in prepared.layout) for shape, left in prepared.layout)
                        derivative = blocks[0][0] if prepared.single else blocks
                elif request.derivative_order:
                    base, matrix, spec = plan.jacobian(prepared.point, prepared.node.diff_method)
                    matrix = matrix @ prepared.classical
                    if request.derivative_kind in {"gradient", "vjp"}:
                        vector = matrix[0] if request.derivative_kind == "gradient" else flatten_tree(request.cotangent)[0] @ matrix
                        items = [vector[indices].reshape(shape) for shape, indices in prepared.layout]
                        derivative = items[0] if prepared.single else tuple(items)
                    elif request.derivative_kind == "jvp":
                        derivative = unflatten_tree(matrix @ prepared.direction, spec)
                    else:
                        derivative = _input_tree(matrix, prepared)
                else:
                    base, derivative = plan.evaluate(prepared.point), None
                if flatten_tree(base)[1] != prepared.output_spec:
                    from quantumbridge.core.classical import UnsupportedFeature
                    raise UnsupportedFeature("derivative_output_structure", "A custom transform changed the declared output structure; shape-changing derivative requests require an explicit supported output contract.")
                for value in (base, derivative):
                    if value is not None and not np.all(np.isfinite(flatten_tree(value)[0])):
                        raise FloatingPointError("Native derivative output is nonfinite.")
                actual_bytes = sum(flatten_tree(x)[0].nbytes for x in (base, derivative) if x is not None)
                from .array_ownership import native_storage_bytes, retain_reservation
                returned_arrays = (base, derivative, matrix, second)
                buffers.observe(max(actual_bytes, native_storage_bytes(returned_arrays)))
                metadata = {**plan.metadata, "device_derivative_protocol": True,
                    "requested_method": request.method, "derivative_kind": request.derivative_kind,
                    "derivative_work_semantics": "charged planned native work for the attempted request, not measured CPU operations",
                    "result_memory_semantics": "native result storage retains its conservative lease until its last array/view dies; receipts are snapshots"}
                retain_reservation(buffers, returned_arrays)
                transferred = True
            finally:
                if not transferred: buffers.release()
            result = DerivativeExecutionResult(base, derivative, metadata=metadata, jacobian=matrix, second=second)
        result.receipt = ledger.receipt()
        return result
    except Exception as error:
        failure = classify_derivative_error(error, derivative_kind=getattr(request, "derivative_kind", None),
            stage=getattr(error, "stage", stage),
            device_method=getattr(request, "method", None), receipt=ledger.receipt())
        if return_status:
            detach_status_error(error)
            detach_status_error(failure)
            # Do not keep partial successful arrays in this active frame while
            # taking the returned failure's post-cleanup snapshot.
            prepared = plan = base = matrix = second = derivative = metadata = result = returned_arrays = None
            failure.receipt = ledger.receipt()
            return DerivativeExecutionResult(status=failure.status, receipt=failure.receipt, error=failure)
        raise failure from error


class NativeDeviceDerivatives:
    """One shared protocol for the existing distinct native device engines."""

    def supports_derivatives(self, config):
        from .derivative_planning import prepare_derivative
        try:
            prepare_derivative(self, config)
        except Exception:
            return False
        return True

    def execute_derivatives(self, request, *, ledger=None, return_status=False):
        return execute_device_derivatives(self, request, ledger=ledger, return_status=return_status)

    def compute_derivatives(self, request, **options):
        return self.execute_derivatives(request, **options).derivative

    def execute_and_compute_derivatives(self, request, **options):
        result = self.execute_derivatives(request, **options)
        return result.value, result.derivative

    def compute_vjp(self, request, **options):
        return self.compute_derivatives(replace(request, derivative_kind="vjp", derivative_order=1), **options)

    def execute_and_compute_vjp(self, request, **options):
        return self.execute_and_compute_derivatives(replace(request, derivative_kind="vjp", derivative_order=1), **options)

    def compute_jvp(self, request, **options):
        return self.compute_derivatives(replace(request, derivative_kind="jvp", derivative_order=1), **options)

    def execute_and_compute_jvp(self, request, **options):
        return self.execute_and_compute_derivatives(replace(request, derivative_kind="jvp", derivative_order=1), **options)

    def compute_hessian(self, request, **options):
        return self.compute_derivatives(replace(request, derivative_kind="hessian", derivative_order=2), **options)

    def execute_and_compute_hessian(self, request, **options):
        return self.execute_and_compute_derivatives(replace(request, derivative_kind="hessian", derivative_order=2), **options)

    def compute_hvp(self, request, **options):
        return self.compute_derivatives(replace(request, derivative_kind="hvp", derivative_order=2), **options)

    def execute_and_compute_hvp(self, request, **options):
        return self.execute_and_compute_derivatives(replace(request, derivative_kind="hvp", derivative_order=2), **options)
