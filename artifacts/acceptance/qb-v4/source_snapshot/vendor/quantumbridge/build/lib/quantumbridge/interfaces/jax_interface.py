# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is part of QuantumBridge SDK.
# This implementation is developed for the QuantumBridge native architecture.


def to_jax(value):
    try:
        import jax.numpy as jnp
    except Exception as exc:
        raise ImportError("QuantumBridge JAX interface requires the optional jax package.") from exc
    return jnp.asarray(value)

