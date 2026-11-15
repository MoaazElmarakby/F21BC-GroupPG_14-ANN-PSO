"""Fitness metrics: ``metric(outputs, labels) -> float``, HIGHER IS BETTER (losses are negated).

To add one: ``register_fitness_metric("name", fn)`` then ``training.fitness = "name"``.
``outputs`` has shape (n, n_out); ``labels`` has shape (n,) with integer classes.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from psoann.registry import Registry

Metric = Callable[[NDArray[np.float64], NDArray[np.int64]], float]
_REGISTRY: Registry[Metric] = Registry("fitness metric")
_EPS = 1e-12


def register_fitness_metric(name: str, fn: Metric, *, overwrite: bool = False) -> Metric:
    return _REGISTRY.register(name, fn, overwrite=overwrite)


def get_fitness_metric(name: str) -> Metric:
    return _REGISTRY.get(name)


def available_fitness_metrics() -> list[str]:
    return _REGISTRY.names()


def predict_labels(outputs: NDArray[np.float64]) -> NDArray[np.int64]:
    """One output: class 1 if >= 0.5. Several outputs: argmax."""
    outputs = np.asarray(outputs)
    if outputs.ndim == 1 or outputs.shape[1] == 1:
        return (outputs.reshape(-1) >= 0.5).astype(np.int64)
    return np.argmax(outputs, axis=1).astype(np.int64)


def _targets(outputs: NDArray[np.float64], labels: NDArray[np.int64]) -> NDArray[np.float64]:
    """Targets shaped like ``outputs``: the label itself (one output) or a one-hot row (several)."""
    outputs = np.asarray(outputs)
    labels = np.asarray(labels)
    if outputs.ndim == 1 or outputs.shape[1] == 1:
        return labels.reshape(-1, 1).astype(np.float64)
    if labels.max() >= outputs.shape[1]:
        raise ValueError(f"label {labels.max()} does not fit {outputs.shape[1]} outputs")
    return np.eye(outputs.shape[1])[labels]


def _as_matrix(outputs: NDArray[np.float64]) -> NDArray[np.float64]:
    outputs = np.asarray(outputs, dtype=np.float64)
    return outputs.reshape(-1, 1) if outputs.ndim == 1 else outputs


def accuracy(outputs: NDArray[np.float64], labels: NDArray[np.int64]) -> float:
    return float(np.mean(predict_labels(outputs) == np.asarray(labels)))


def mse(outputs: NDArray[np.float64], labels: NDArray[np.int64]) -> float:
    """Mean squared error (lower is better; used for reporting and, negated, as a fitness)."""
    return float(np.mean((_as_matrix(outputs) - _targets(outputs, labels)) ** 2))


def neg_mse(outputs: NDArray[np.float64], labels: NDArray[np.int64]) -> float:
    return -mse(outputs, labels)


def neg_cross_entropy(outputs: NDArray[np.float64], labels: NDArray[np.int64]) -> float:
    """Negated mean binary cross-entropy (per output unit, against 0/1 or one-hot targets)."""
    p = np.clip(_as_matrix(outputs), _EPS, 1.0 - _EPS)
    t = _targets(outputs, labels)
    return float(np.mean(t * np.log(p) + (1.0 - t) * np.log(1.0 - p)))


register_fitness_metric("accuracy", accuracy)
register_fitness_metric("neg_mse", neg_mse)
register_fitness_metric("neg_cross_entropy", neg_cross_entropy)
