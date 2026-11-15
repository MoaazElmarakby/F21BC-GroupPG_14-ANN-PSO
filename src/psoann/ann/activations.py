"""Activation functions and their registry. To add one: ``register_activation("name", fn)``."""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from psoann.registry import Registry

Array = NDArray[np.float64]
Activation = Callable[[Array], Array]

_REGISTRY: Registry[Activation] = Registry("activation")


def register_activation(name: str, fn: Activation, *, overwrite: bool = False) -> Activation:
    return _REGISTRY.register(name, fn, overwrite=overwrite)


def get_activation(name: str) -> Activation:
    return _REGISTRY.get(name)


def available_activations() -> list[str]:
    return _REGISTRY.names()


def identity(x: Array) -> Array:
    return x


def logistic(x: Array) -> Array:
    """1 / (1 + e^-x), with the exponent clipped so extreme weights cannot overflow."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))


def relu(x: Array) -> Array:
    return np.maximum(x, 0.0)


def leaky_relu(x: Array) -> Array:
    return np.where(x > 0.0, x, 0.01 * x)


def gaussian(x: Array) -> Array:
    return np.exp(-np.square(x))


register_activation("identity", identity)
register_activation("logistic", logistic)
register_activation("tanh", np.tanh)
register_activation("relu", relu)
register_activation("leaky_relu", leaky_relu)
register_activation("gaussian", gaussian)
