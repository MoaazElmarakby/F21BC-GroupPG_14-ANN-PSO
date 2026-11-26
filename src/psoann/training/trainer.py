"""PSOTrainer: the one place where the ANN and the optimiser meet."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from psoann.ann.network import FeedForwardNetwork, build_network
from psoann.config import ANNConfig, PSOConfig, TrainingConfig
from psoann.evaluation.metrics import get_fitness_metric
from psoann.pso.optimizer import ParticleSwarmOptimizer
from psoann.pso.random_search import RandomSearchOptimizer
from psoann.pso.types import Fitness, PSOResult, SearchSpace
from psoann.training.fitness import make_fitness


class Optimizer(Protocol):
    """Anything with this method can replace the default PSO (e.g. a PSO variant)."""

    def optimise(self, fitness: Fitness, space: SearchSpace, rng: np.random.Generator) -> PSOResult: ...


@dataclass
class TrainResult:
    parameters: NDArray[np.float64]
    fitness: float
    pso: PSOResult
    network: FeedForwardNetwork


class PSOTrainer:
    def __init__(
        self,
        ann: ANNConfig,
        pso: PSOConfig,
        training: TrainingConfig,
        optimizer: Optimizer | None = None,
    ) -> None:
        self.ann = ann
        self.pso = pso
        self.training = training
        if optimizer is None:
            optimizer = (
                RandomSearchOptimizer(pso) if training.optimizer == "random_search" else ParticleSwarmOptimizer(pso)
            )
        self.optimizer: Optimizer = optimizer

    def train(self, X: NDArray[np.float64], y: NDArray[np.int64], rng: np.random.Generator) -> TrainResult:
        n_classes = max(2, self.ann.n_outputs)
        if y.min() < 0 or y.max() >= n_classes:
            raise ValueError(f"labels must be in 0..{n_classes - 1} for n_outputs={self.ann.n_outputs}")
        network = build_network(self.ann, n_inputs=X.shape[1])
        metric = get_fitness_metric(self.training.fitness)
        space = SearchSpace(*network.bounds(self.pso.bound))  # weights and biases in +-bound (+ activation genes)
        result = self.optimizer.optimise(make_fitness(network, X, y, metric), space, rng)
        return TrainResult(result.best_position, result.best_fitness, result, network)
