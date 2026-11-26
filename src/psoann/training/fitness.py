"""Turn (network, data, metric) into the single callable PSO needs: parameter vector -> score."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from psoann.ann.network import FeedForwardNetwork
from psoann.evaluation.metrics import Metric, accuracy, mse, precision_recall_f1
from psoann.pso.types import Fitness


def make_fitness(
    network: FeedForwardNetwork,
    X: NDArray[np.float64],
    y: NDArray[np.int64],
    metric: Metric,
) -> Fitness:
    def fitness(params: NDArray[np.float64]) -> float:
        return float(metric(network.forward(params, X), y))

    return fitness


def evaluate_parameters(
    network: FeedForwardNetwork,
    params: NDArray[np.float64],
    X: NDArray[np.float64],
    y: NDArray[np.int64],
) -> dict[str, float]:
    """Reporting metrics (independent of which metric was optimised). Class 1 (forged) is the positive class."""
    outputs = network.forward(params, X)
    precision, recall, f1 = precision_recall_f1(outputs, y)
    return {
        "accuracy": accuracy(outputs, y),
        "mse": mse(outputs, y),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
