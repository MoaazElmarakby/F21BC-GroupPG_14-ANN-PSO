"""Particle Swarm Optimisation: Algorithm 39 of Luke, *Essentials of Metaheuristics* (2nd ed., 2013).

Comments of the form ``[Alg. 39, line N]`` refer to the book's pseudo-code (checked against the book
text supplied with the project). The optimiser maximises ``fitness`` and knows nothing about neural
networks: it only sees a callable ``vector -> float`` and a ``SearchSpace``.

Assumptions the pseudo-code leaves open (fixed here and documented in docs/design.md):

* a particle's informants include itself (the book says so in the prose);
* a personal or global best is replaced only on a *strict* improvement (line 14 uses ">");
* every iteration assesses the whole swarm, but the last one does not move it afterwards, so the
  number of fitness calls is exactly ``swarm_size * iterations``;
* the velocity update's random numbers ``b, c, d`` are always drawn (even if beta/gamma/delta are 0)
  so changing a coefficient never changes the random stream: sweeps compare like with like.

Deliberate departures from the pseudo-code, selectable in ``PSOConfig``: boundary handling
(``bound_handling``), optional velocity clamping (``max_velocity_fraction``) and optional per-iteration
informant redrawing (``regenerate_informants``).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from psoann.config import PSOConfig
from psoann.pso.bounds import BoundaryHandler, make_boundary_handler
from psoann.pso.informants import InformantStrategy, make_informant_strategy
from psoann.pso.types import Fitness, PSOResult, SearchSpace

Matrix = NDArray[np.float64]


class ParticleSwarmOptimizer:
    def __init__(
        self,
        config: PSOConfig,
        informants: InformantStrategy | None = None,
        boundary: BoundaryHandler | None = None,
    ) -> None:
        self.config = config
        self.informants = informants if informants is not None else make_informant_strategy(config)
        self.boundary = boundary if boundary is not None else make_boundary_handler(config.bound_handling)

    # ------------------------------------------------------------------------------------------
    def optimise(self, fitness: Fitness, space: SearchSpace, rng: np.random.Generator) -> PSOResult:
        cfg = self.config
        swarm, dim = cfg.swarm_size, space.dim

        # [Alg. 39, lines 7-9] P <- swarmsize random particles, each with a random velocity
        positions, velocities = self._initialise_swarm(space, rng)
        informants = self.informants.build(swarm, rng)

        pbest_position = positions.copy()  # x*: each particle's previous fittest location
        pbest_fitness = np.full(swarm, -np.inf)
        gbest_position = positions[0].copy()  # x!: fittest location any particle has found
        gbest_fitness = -np.inf  # [Alg. 39, line 10] Best <- (nothing yet)
        max_velocity = None if cfg.max_velocity_fraction is None else cfg.max_velocity_fraction * space.span

        keys = ("evaluations", "best", "mean", "iteration_best", "diversity")
        history: dict[str, list[float]] = {k: [] for k in keys}
        evaluations = 0

        for iteration in range(cfg.iterations):  # [Alg. 39, line 11] repeat ... until out of time
            # [Alg. 39, lines 12-13] assess the fitness of every particle
            scores = self._assess(fitness, positions)
            evaluations += swarm

            # personal bests (strict improvement only)
            improved = scores > pbest_fitness
            pbest_position[improved] = positions[improved]
            pbest_fitness[improved] = scores[improved]

            # [Alg. 39, lines 14-15] if Fitness(x) > Fitness(Best) then Best <- x
            leader = int(np.argmax(scores))  # first maximum, same tie-break as the sequential loop
            if scores[leader] > gbest_fitness:
                gbest_fitness = float(scores[leader])
                gbest_position = positions[leader].copy()

            history["evaluations"].append(evaluations)
            history["best"].append(gbest_fitness)
            history["mean"].append(float(np.mean(scores)))
            history["iteration_best"].append(float(scores[leader]))
            history["diversity"].append(float(np.mean(np.std(positions, axis=0))))

            if iteration == cfg.iterations - 1:
                break  # budget spent: [Alg. 39, line 27] the loop ends before another move

            if cfg.regenerate_informants:
                informants = self.informants.regenerate(informants, swarm, rng)

            # [Alg. 39, lines 16-19] for each particle: x*, x+ (best among informants), x!
            informant_best = self._informant_best(informants, pbest_position, pbest_fitness)

            # [Alg. 39, lines 20-24] per-dimension random mixing and velocity update
            randoms = rng.random((3, swarm, dim))
            velocities = self._update_velocity(
                velocities, positions, pbest_position, informant_best, gbest_position, randoms
            )
            if max_velocity is not None:
                np.clip(velocities, -max_velocity, max_velocity, out=velocities)

            # [Alg. 39, lines 25-26] x <- x + epsilon * v   (then boundary handling)
            positions = self._move(positions, velocities)
            self.boundary.apply(positions, velocities, space)

        # [Alg. 39, line 28] return Best
        return PSOResult(
            best_position=gbest_position,
            best_fitness=float(gbest_fitness),
            evaluations=evaluations,
            history={k: np.asarray(v, dtype=np.float64) for k, v in history.items()},
        )

    # ------------------------------------------------------------------------------------------
    def _initialise_swarm(self, space: SearchSpace, rng: np.random.Generator) -> tuple[Matrix, Matrix]:
        """Random positions; velocity = half the vector between two random points (Luke's default)."""
        swarm, dim = self.config.swarm_size, space.dim
        positions = space.lower + rng.random((swarm, dim)) * space.span
        a = space.lower + rng.random((swarm, dim)) * space.span  # always drawn: keeps the stream fixed
        b = space.lower + rng.random((swarm, dim)) * space.span
        velocities = 0.5 * (a - b) if self.config.initial_velocity == "half_diff" else np.zeros((swarm, dim))
        return positions, velocities

    @staticmethod
    def _assess(fitness: Fitness, positions: Matrix) -> NDArray[np.float64]:
        scores = np.empty(len(positions), dtype=np.float64)
        for i, position in enumerate(positions):
            value = float(fitness(position))
            scores[i] = -np.inf if np.isnan(value) else value  # a NaN score is "worst", not "unordered"
        return scores

    @staticmethod
    def _informant_best(
        informants: NDArray[np.int64], pbest_position: Matrix, pbest_fitness: NDArray[np.float64]
    ) -> Matrix:
        """x+: for each particle, the fittest personal-best location among its informants."""
        choice = np.argmax(pbest_fitness[informants], axis=1)  # ties -> earliest column (the particle itself)
        winners = informants[np.arange(len(informants)), choice]
        return pbest_position[winners]

    def _update_velocity(
        self,
        velocities: Matrix,
        positions: Matrix,
        personal_best: Matrix,
        informant_best: Matrix,
        global_best: NDArray[np.float64],
        randoms: NDArray[np.float64],
    ) -> Matrix:
        """[Alg. 39, line 24]  v_i <- alpha*v_i + b(x*_i - x_i) + c(x+_i - x_i) + d(x!_i - x_i)

        with b ~ U(0, beta), c ~ U(0, gamma), d ~ U(0, delta) drawn per particle and per dimension.
        """
        cfg = self.config
        b = cfg.beta * randoms[0]  # [Alg. 39, line 21]
        c = cfg.gamma * randoms[1]  # [Alg. 39, line 22]
        d = cfg.delta * randoms[2]  # [Alg. 39, line 23]
        return (
            cfg.alpha * velocities
            + b * (personal_best - positions)
            + c * (informant_best - positions)
            + d * (global_best - positions)
        )

    def _move(self, positions: Matrix, velocities: Matrix) -> Matrix:
        """[Alg. 39, line 26]  x <- x + epsilon * v"""
        return positions + self.config.epsilon * velocities
