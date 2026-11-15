"""Types shared by the optimiser and its callers. Nothing here knows about neural networks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from numpy.typing import NDArray

# parameter vector -> score, HIGHER IS BETTER. Must not modify its argument.
Fitness = Callable[[NDArray[np.float64]], float]


@dataclass(frozen=True)
class SearchSpace:
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]

    def __post_init__(self) -> None:
        lower = np.asarray(self.lower, dtype=np.float64)
        upper = np.asarray(self.upper, dtype=np.float64)
        if lower.ndim != 1 or lower.shape != upper.shape or lower.size == 0:
            raise ValueError("lower and upper must be 1-D arrays of the same non-zero length")
        if not np.all(upper > lower):
            raise ValueError("every upper bound must exceed its lower bound")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @property
    def dim(self) -> int:
        return int(self.lower.size)

    @property
    def span(self) -> NDArray[np.float64]:
        return self.upper - self.lower

    @classmethod
    def symmetric(cls, dim: int, bound: float) -> "SearchSpace":
        return cls(np.full(dim, -bound), np.full(dim, bound))


@dataclass
class PSOResult:
    best_position: NDArray[np.float64]
    best_fitness: float
    evaluations: int
    # One entry per iteration: "evaluations" (cumulative), "best" (best so far), "mean" and
    # "iteration_best" (this iteration's swarm), "diversity" (mean per-dimension std of positions) and
    # "boundary_hits" (coordinates that left the search space in the move before this iteration).
    history: dict[str, NDArray[np.float64]] = field(default_factory=dict)

    def evaluations_to_reach(self, target: float) -> int | None:
        """Cumulative evaluations at the first iteration whose best-so-far is >= ``target`` (None if never)."""
        best = self.history.get("best")
        if best is None:
            return None
        hits = np.flatnonzero(best >= target)
        return int(self.history["evaluations"][hits[0]]) if hits.size else None

    def evaluations_to_final_best(self) -> int | None:
        """Cumulative evaluations at the first iteration that reached the final best fitness."""
        return self.evaluations_to_reach(self.best_fitness)
