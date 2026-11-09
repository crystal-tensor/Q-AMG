# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is part of QuantumBridge SDK.
# This implementation is developed for the QuantumBridge native architecture.


def measurement_count(circuit) -> int:
    return len(circuit.measurements)


from .dispatch import transform


@transform
def split_hamiltonian(tape):
    """Internal expval-to-term-Tapes prototype, not split_to_single_terms parity.

    Identity and duplicate nonzero terms execute independently. Zero terms are
    validated then omitted. Coefficients are static, finite and real. Shot
    partitions are per child, not a total-shot allocation or QWC grouping.
    """
    import numpy as np
    from dataclasses import replace
    from quantumbridge.core.classical import UnsupportedFeature
    from quantumbridge.diff.qnode_engine import map_measurements
    from quantumbridge.qml.shot_vector import ShotSpec
    from quantumbridge.utils.math import Hamiltonian, PauliString
    from .program import TransformLimits, check_batch, output_tree

    limits = TransformLimits()
    check_batch((tape,), limits)
    children, slots, work = [], [], 0
    tree = output_tree(tape)

    def expand(request):
        nonlocal work
        if request.kind != "expval":
            raise UnsupportedFeature("hamiltonian_split_measurement", "B1 Hamiltonian splitting supports expval only.")
        observable = request.observable
        if hasattr(observable, "to_quantumbridge"):
            observable = observable.to_quantumbridge()
        elif hasattr(observable, "to_hamiltonian"):
            observable = observable.to_hamiltonian()
        if isinstance(observable, PauliString):
            terms = ((1., observable),)
        elif isinstance(observable, Hamiltonian):
            terms = observable.terms
        else:
            raise UnsupportedFeature("hamiltonian_split_observable", "B1 splitting requires native Pauli/Hamiltonian form.")
        work += len(terms)
        limits.check("max_expansion_steps", work)
        positions = []
        for coefficient, pauli in terms:
            from quantumbridge.diff.scalars import is_live_scalar
            if is_live_scalar(coefficient):
                raise UnsupportedFeature("prototype_coefficient_graph", "Use split_to_single_terms or QWC for coefficient derivatives, not the internal term-Tapes prototype.")
            if not np.isfinite(coefficient) or np.iscomplexobj(coefficient):
                raise ValueError("Splitting requires finite real static coefficients.")
            wires = [wire for wire, _ in pauli.terms]
            if len(set(wires)) != len(wires) or any(wire < 0 or wire >= tape.circuit.num_qubits for wire in wires):
                raise ValueError("Split Pauli terms require unique in-register wires.")
            if coefficient == 0:
                continue
            limits.check("max_tapes", len(children) + 1)
            child = tape.copy()
            terminal = replace(request, observable=pauli)
            child.measurements, child.output_tree = [terminal], terminal
            child.observable, child.measurement = pauli, "expval"
            positions.append((len(children), coefficient))
            children.append(child)
        slots.append(tuple(positions))
        return request

    map_measurements(tree, expand)
    check_batch(children, limits)

    def one_partition(values):
        values_by_request = iter(sum(coefficient * values[index] for index, coefficient in positions)
                                 for positions in slots)
        return map_measurements(tree, lambda _: next(values_by_request))

    def process(results):
        if len(results) != len(children):
            raise ValueError("Hamiltonian split result count mismatch.")
        spec = ShotSpec.parse(tape.shots)
        if spec.is_vector:
            if any(not isinstance(result, tuple) or len(result) != len(spec.partitions) for result in results):
                raise ValueError("Hamiltonian split shot partition count mismatch.")
            return tuple(one_partition(tuple(result[index] for result in results)) for index in range(len(spec.partitions)))
        return one_partition(results)

    return tuple(children), process


@transform
def split_to_single_terms(tape, *, limits=None):
    """Split linear Pauli measurements on one Tape, with global deduplication.

    Identity terms become constant offsets. Non-commuting observables remain
    separate measurements on the same Tape; this is not QWC grouping.
    """
    from dataclasses import replace
    import numpy as np
    from quantumbridge.core.classical import UnsupportedFeature
    from quantumbridge.diff.qnode_engine import map_measurements
    from quantumbridge.qml.shot_vector import ShotSpec
    from quantumbridge.utils.math import Hamiltonian, PauliString
    from .program import resolve_limits, check_batch, output_tree
    from quantumbridge.diff.scalars import CoefficientTable

    limits = resolve_limits(limits)
    check_batch((tape,),limits)
    tree = output_tree(tape)
    unique, indices, slots, work = [], {}, [], 0
    coefficients = CoefficientTable()

    def native(observable):
        if hasattr(observable,"to_quantumbridge"): return observable.to_quantumbridge()
        if hasattr(observable,"to_hamiltonian"): return observable.to_hamiltonian()
        return observable

    def key(request):
        observable = native(request.observable)
        if isinstance(observable,PauliString):
            wires = [wire for wire,_ in observable.terms]
            if len(set(wires)) != len(wires) or any(wire < 0 or wire >= tape.circuit.num_qubits for wire in wires):
                raise ValueError("Single-term measurements need distinct in-register Pauli wires.")
            identity = tuple(sorted((wire,label) for wire,label in observable.terms if label != "I"))
        elif observable is None:
            identity = None
        else:
            identity = ("opaque",len(unique))
        return (request.kind,identity,request.wires,request.shots,request.seed,
                None if not request.metadata else len(unique))

    def insert(request):
        signature = key(request)
        if signature not in indices:
            limits.check("max_postprocessing_nodes",len(unique)+1)
            indices[signature] = len(unique)
            unique.append(request)
        return indices[signature]

    def expand(request):
        nonlocal work
        observable = native(request.observable)
        if request.kind != "expval":
            if isinstance(observable,Hamiltonian):
                raise UnsupportedFeature("split_measurement_sum", "Only expectation values of sums can be split linearly.")
            slots.append(((),((insert(request),1.),),True))
            return request
        terms = observable.terms if isinstance(observable,Hamiltonian) else ((1.,observable),)
        work += len(terms)
        limits.check("max_expansion_steps",work)
        offsets, positions = [], []
        for coefficient, pauli in terms:
            reference = coefficients.add(coefficient)
            terminal = replace(request,observable=pauli)
            signature = key(terminal)
            if isinstance(pauli,PauliString) and not signature[1]:
                offsets.append(reference)
            else:
                # Even a zero coefficient retains its public-contract measurement.
                positions.append((insert(terminal),reference))
        slots.append((tuple(offsets),tuple(positions),False))
        return request

    map_measurements(tree,expand)
    child = tape.copy()
    child.measurements = unique
    child.output_tree = unique[0] if len(unique) == 1 else tuple(unique)
    child.measurement = unique[-1].kind if unique else None
    child.observable = unique[-1].observable if unique else None
    check_batch((child,),limits)

    def one(result, parameters=None):
        values = (result,) if len(unique) == 1 else result
        if len(values) != len(unique):
            raise ValueError("Single-term result width does not match its measurement list.")
        restored = iter(values[positions[0][0]] if direct else
                        sum((coefficients.resolve(coefficient,parameters)*values[index] for index,coefficient in positions),
                            sum(coefficients.resolve(coefficient,parameters) for coefficient in offsets))
                        for offsets,positions,direct in slots)
        return map_measurements(tree,lambda _:next(restored))

    def process(results, *, parameters=None, differentiating=False):
        if len(results) != 1:
            raise ValueError("split_to_single_terms expects one child Tape result.")
        spec = ShotSpec.parse(tape.shots)
        if spec.is_vector:
            if not isinstance(results[0],tuple) or len(results[0]) != len(spec.partitions):
                raise ValueError("Single-term shot partition count mismatch.")
            return tuple(one(result,parameters) for result in results[0])
        return one(results[0],parameters)

    process.native_evaluate = process
    process.native_parameters = tuple(coefficients.parameters)
    process.native_linear_quantum_outputs = True
    return (child,),process

