"""Particle Swarm Optimisation: Algorithm 39 of Luke, *Essentials of Metaheuristics* (2nd ed., 2013).

Comments of the form ``[Alg. 39, line N]`` refer to the line numbers of the book's pseudo-code. The
optimiser maximises ``fitness`` and knows nothing about neural networks: it only sees a callable
``vector -> float`` and a ``SearchSpace``.

Assumptions the pseudo-code leaves open (fixed here and documented in docs/design.md):

* a particle's informants include itself (the book says so in the prose);
* a personal or global best is replaced only on a *strict* improvement (line 14 uses ">");
* every iteration assesses the whole swarm, but the last one does not move it afterwards, so the
  number of fitness calls is exactly ``swarm_size * iterations``;
* the velocity update's random numbers ``b, c, d`` are always drawn (even if beta/gamma/delta are 0)
  so changing a coefficient never changes the random stream: sweeps compare like with like.

Deliberate departures from the pseudo-code, selectable in ``PSOConfig``: boundary handling
(``bound_handling``: clamp, reject, reflect, wrap, random, none), optional velocity clamping
(``max_velocity_fraction``), optional per-iteration informant redrawing (``regenerate_informants``),
position-based informants (``informants = "nearest"``, recomputed every iteration) and coefficients
that change during the run (``schedule``; the default ``constant`` is Algorithm 39 as written).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from psoann.config import PSOConfig
from psoann.pso.bounds import BoundaryHandler, make_boundary_handler
from psoann.pso.informants import InformantStrategy, make_informant_strategy
from psoann.pso.schedules import Coefficients, CoefficientSchedule, make_coefficient_schedule
from psoann.pso.types import Fitness, PSOResult, SearchSpace

Matrix = NDArray[np.float64]


class ParticleSwarmOptimizer:
    """Algorithm 39 with pluggable informant strategy, boundary rule and coefficient schedule.

    By default the three components are built from ``config``; pass an instance to override one of them.
    """

    def __init__(
        self,
        config: PSOConfig,
        informants: InformantStrategy | None = None,
        boundary: BoundaryHandler | None = None,
        schedule: CoefficientSchedule | None = None,
    ) -> None:
        self.config = config
        self.informants = informants if informants is not None else make_informant_strategy(config)
        self.boundary = boundary if boundary is not None else make_boundary_handler(config.bound_handling)
        self.schedule = schedule if schedule is not None else make_coefficient_schedule(config)

    def optimise(self, fitness: Fitness, space: SearchSpace, rng: np.random.Generator) -> PSOResult:
        cfg = self.config
        swarm, dim = cfg.swarm_size, space.dim

        # [Alg. 39, lines 7-9] P <- swarmsize random particles, each with a random velocity
        positions, velocities = self._initialise_swarm(space, rng)
        informants = self._build_informants(positions, swarm, rng)

        pbest_position = positions.copy()  # x*: each particle's previous fittest location
        pbest_fitness = np.full(swarm, -np.inf)
        gbest_position = positions[0].copy()  # x!: fittest location any particle has found
        gbest_fitness = -np.inf  # [Alg. 39, line 10] Best <- (nothing yet)
        max_velocity = None if cfg.max_velocity_fraction is None else cfg.max_velocity_fraction * space.span

        keys = ("evaluations", "best", "mean", "iteration_best", "diversity", "boundary_hits")
        history: dict[str, list[float]] = {k: [] for k in keys}
        evaluations = 0
        boundary_hits = 0  # coordinates that left the search space in the move that led to this iteration

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
            history["boundary_hits"].append(float(boundary_hits))

            if iteration == cfg.iterations - 1:
                break  # budget spent: [Alg. 39, line 27] the loop ends before another move

            if cfg.regenerate_informants or self.informants.uses_positions:
                informants = self._regenerate_informants(informants, positions, swarm, rng)

            # [Alg. 39, lines 16-19] for each particle: x*, x+ (best among informants), x!
            informant_best = self._informant_best(informants, pbest_position, pbest_fitness)

            # [Alg. 39, lines 2-5] alpha, beta, gamma, delta for this update (fixed unless a schedule is set);
            # progress runs from 0 at the first move to 1 at the last
            coefficients = self.schedule.at(iteration / max(1, cfg.iterations - 2))

            # [Alg. 39, lines 20-24] per-dimension random mixing and velocity update
            randoms = rng.random((3, swarm, dim))
            velocities = self._update_velocity(
                velocities, positions, pbest_position, informant_best, gbest_position, randoms, coefficients
            )
            if max_velocity is not None:
                np.clip(velocities, -max_velocity, max_velocity, out=velocities)

            # [Alg. 39, lines 25-26] x <- x + epsilon * v   (then boundary handling)
            previous = positions
            positions = self._move(positions, velocities)
            noise = rng.random((swarm, dim))  # always drawn, so the stream is the same for every handler
            boundary_hits = int(np.sum((positions < space.lower) | (positions > space.upper)))
            self.boundary.apply(positions, velocities, space, noise, previous)

        # [Alg. 39, line 28] return Best
        return PSOResult(
            best_position=gbest_position,
            best_fitness=float(gbest_fitness),
            evaluations=evaluations,
            history={k: np.asarray(v, dtype=np.float64) for k, v in history.items()},
        )

    def _build_informants(self, positions: Matrix, swarm: int, rng: np.random.Generator) -> NDArray[np.int64]:
        if self.informants.uses_positions:
            return self.informants.build(swarm, rng, positions)  # type: ignore[call-arg]
        return self.informants.build(swarm, rng)

    def _regenerate_informants(
        self, informants: NDArray[np.int64], positions: Matrix, swarm: int, rng: np.random.Generator
    ) -> NDArray[np.int64]:
        if self.informants.uses_positions:
            return self.informants.regenerate(informants, swarm, rng, positions)  # type: ignore[call-arg]
        return self.informants.regenerate(informants, swarm, rng)

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
        coefficients: Coefficients | None = None,
    ) -> Matrix:
        """[Alg. 39, line 24]  v_i <- alpha*v_i + b(x*_i - x_i) + c(x+_i - x_i) + d(x!_i - x_i)

        with b ~ U(0, beta), c ~ U(0, gamma), d ~ U(0, delta) drawn per particle and per dimension.
        ``coefficients`` defaults to the fixed values in the config.
        """
        k = coefficients if coefficients is not None else Coefficients.from_config(self.config)
        b = k.beta * randoms[0]  # [Alg. 39, line 21]
        c = k.gamma * randoms[1]  # [Alg. 39, line 22]
        d = k.delta * randoms[2]  # [Alg. 39, line 23]
        return (
            k.alpha * velocities
            + b * (personal_best - positions)
            + c * (informant_best - positions)
            + d * (global_best - positions)
        )

    def _move(self, positions: Matrix, velocities: Matrix) -> Matrix:
        """[Alg. 39, line 26]  x <- x + epsilon * v"""
        return positions + self.config.epsilon * velocities
