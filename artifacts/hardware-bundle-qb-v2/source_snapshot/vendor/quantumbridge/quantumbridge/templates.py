# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
"""Concrete executable templates sharing the existing native QML implementation."""
from .qml.template_core import NativeTemplate, TemplateLimits, wire_labels, resolve_limits, bounded_pairs
from .qml.embeddings import angle_embedding, basis_embedding, amplitude_embedding
from .qml.templates import basic_entangler_layers, _basic_entangler_layers, strongly_entangling_layers
from .qml.subroutines import iqp_embedding, qft, trotter_product


class AngleEmbedding(NativeTemplate):
    def __init__(self,features,wires,rotation="X",*,limits=None):
        self._build(angle_embedding,wires,features,rotation=rotation,limits=limits)


class BasisEmbedding(NativeTemplate):
    differentiable = False
    def __init__(self,features,wires,*,limits=None):
        self._build(basis_embedding,wires,features,limits=limits)


class AmplitudeEmbedding(NativeTemplate):
    differentiable = False
    requires_untouched_wires = True
    def __init__(self,features,wires,*,pad_with=None,normalize=False,limits=None):
        self._build(amplitude_embedding,wires,features,pad_with=pad_with,normalize=normalize,limits=limits)


class BasicEntanglerLayers(NativeTemplate):
    def __init__(self,weights,wires,rotation="X",*,limits=None):
        self._build(_basic_entangler_layers,wires,weights,rotation=rotation,limits=limits)


class StronglyEntanglingLayers(NativeTemplate):
    def __init__(self,weights,wires,ranges=None,imprimitive="cx",*,limits=None):
        self._build(strongly_entangling_layers,wires,weights,ranges=ranges,imprimitive=imprimitive,limits=limits)


class IQPEmbedding(NativeTemplate):
    def __init__(self,features,wires,n_repeats=1,pattern=None,*,limits=None):
        limits = resolve_limits(limits)
        labels = wire_labels(wires,limits)
        local = None if pattern is None else [tuple(labels.index(q) for q in pair) for pair in bounded_pairs(pattern,limits)]
        self._build(iqp_embedding,labels,features,n_repeats=n_repeats,pattern=local,limits=limits)


class QFT(NativeTemplate):
    differentiable = False
    def __init__(self,wires,inverse=False,do_swaps=True,*,limits=None):
        self._build(qft,wires,inverse=inverse,do_swaps=do_swaps,limits=limits)


class TrotterProduct(NativeTemplate):
    def __init__(self,hamiltonian,time,wires,n=1,order=1,*,limits=None):
        self._build(trotter_product,wires,hamiltonian,time,n=n,order=order,limits=limits)


__all__ = ["NativeTemplate","TemplateLimits","AngleEmbedding","BasisEmbedding","AmplitudeEmbedding",
           "BasicEntanglerLayers","StronglyEntanglingLayers","IQPEmbedding","QFT","TrotterProduct",
           "angle_embedding","basis_embedding","amplitude_embedding","basic_entangler_layers",
           "strongly_entangling_layers","iqp_embedding","qft","trotter_product"]
