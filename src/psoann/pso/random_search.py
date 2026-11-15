"""Random-search control: the same evaluation budget as PSO, spent on independent uniform samples.

Not an extension of PSO: it exists so the report can show that the swarm does more than sampling
(study Q0, ``psoann sweep random_search``). It has the same ``optimise`` interface as the PSO and the
same history keys, and it consumes ``swarm_size`` samples per "iteration" so budgets are identical.
"""

from __future__ import annotations

import numpy as np

from psoann.config import PSOConfig
from psoann.pso.types import Fitness, PSOResult, SearchSpace


class RandomSearchOptimizer:
    def __init__(self, config: PSOConfig) -> None:
        self.config = config

    def optimise(self, fitness: Fitness, space: SearchSpace, rng: np.random.Generator) -> PSOResult:
        swarm, dim = self.config.swarm_size, space.dim
        keys = ("evaluations", "best", "mean", "iteration_best", "diversity", "boundary_hits")
        history: dict[str, list[float]] = {k: [] for k in keys}
        best_position = space.lower.copy()
        best_fitness = -np.inf
        evaluations = 0

        for _ in range(self.config.iterations):
            samples = space.lower + rng.random((swarm, dim)) * space.span
            scores = np.empty(swarm)
            for i, sample in enumerate(samples):
                value = float(fitness(sample))
                scores[i] = -np.inf if np.isnan(value) else value
            evaluations += swarm
            leader = int(np.argmax(scores))
            if scores[leader] > best_fitness:
                best_fitness = float(scores[leader])
                best_position = samples[leader].copy()
            history["evaluations"].append(evaluations)
            history["best"].append(best_fitness)
            history["mean"].append(float(np.mean(scores)))
            history["iteration_best"].append(float(scores[leader]))
            history["diversity"].append(float(np.mean(np.std(samples, axis=0))))
            history["boundary_hits"].append(0.0)

        return PSOResult(
            best_position=best_position,
            best_fitness=float(best_fitness),
            evaluations=evaluations,
            history={k: np.asarray(v, dtype=np.float64) for k, v in history.items()},
        )
