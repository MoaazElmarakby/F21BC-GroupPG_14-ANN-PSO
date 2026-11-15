"""Informant topologies (who tells a particle about good locations).

A strategy returns an integer matrix of shape (swarm_size, m): row i lists the particles that inform
particle i. Every row includes i itself (Luke: "x is always one of its own informants") and, for
reproducibility, in column 0 so that ties in fitness are resolved in favour of the particle itself.

To add a topology: subclass ``InformantStrategy`` and ``register_informant_strategy(name, factory)``
where ``factory(pso_config) -> strategy``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from psoann.config import PSOConfig
from psoann.registry import Registry

IndexMatrix = NDArray[np.int64]


class InformantStrategy(ABC):
    @abstractmethod
    def build(self, swarm_size: int, rng: np.random.Generator) -> IndexMatrix:
        """Create the informant matrix (called once at the start)."""

    def regenerate(self, previous: IndexMatrix, swarm_size: int, rng: np.random.Generator) -> IndexMatrix:
        """Redraw informants (called each iteration only if ``regenerate_informants``). Default: rebuild."""
        return self.build(swarm_size, rng)


class RandomInformants(InformantStrategy):
    """Each particle gets itself plus ``k`` distinct other particles chosen uniformly at random.

    Always draws a (S x S) block of uniforms, whatever ``k`` is, so sweeping ``k`` changes only how
    many of the same random scores are used.
    """

    def __init__(self, k: int) -> None:
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k

    def build(self, swarm_size: int, rng: np.random.Generator) -> IndexMatrix:
        if self.k > swarm_size - 1:
            raise ValueError(f"n_informants={self.k} needs a swarm of at least {self.k + 1} particles")
        scores = rng.random((swarm_size, swarm_size))
        scores[np.arange(swarm_size), np.arange(swarm_size)] = 2.0  # self sorts last -> never picked as "other"
        others = np.argsort(scores, axis=1)[:, : self.k]
        return np.concatenate([np.arange(swarm_size)[:, None], others], axis=1).astype(np.int64)


class RingInformants(InformantStrategy):
    """Fixed ring: each particle hears from ``radius`` neighbours on each side (no randomness)."""

    def __init__(self, radius: int = 1) -> None:
        if radius < 1:
            raise ValueError("radius must be >= 1")
        self.radius = radius

    def build(self, swarm_size: int, rng: np.random.Generator) -> IndexMatrix:
        if 2 * self.radius + 1 > swarm_size:
            raise ValueError("ring neighbourhood is larger than the swarm")
        offsets = [0] + [s * d for d in range(1, self.radius + 1) for s in (-1, 1)]
        idx = (np.arange(swarm_size)[:, None] + np.array(offsets)[None, :]) % swarm_size
        return idx.astype(np.int64)

    def regenerate(self, previous: IndexMatrix, swarm_size: int, rng: np.random.Generator) -> IndexMatrix:
        return previous


class GlobalInformants(InformantStrategy):
    """Everyone informs everyone (classic gbest topology)."""

    def build(self, swarm_size: int, rng: np.random.Generator) -> IndexMatrix:
        others = np.array([[j for j in range(swarm_size) if j != i] for i in range(swarm_size)], dtype=np.int64)
        return np.concatenate([np.arange(swarm_size)[:, None], others], axis=1)

    def regenerate(self, previous: IndexMatrix, swarm_size: int, rng: np.random.Generator) -> IndexMatrix:
        return previous


StrategyFactory = Callable[[PSOConfig], InformantStrategy]
_REGISTRY: Registry[StrategyFactory] = Registry("informant strategy")


def register_informant_strategy(name: str, factory: StrategyFactory, *, overwrite: bool = False) -> StrategyFactory:
    return _REGISTRY.register(name, factory, overwrite=overwrite)


def available_informant_strategies() -> list[str]:
    return _REGISTRY.names()


def make_informant_strategy(config: PSOConfig) -> InformantStrategy:
    return _REGISTRY.get(config.informants)(config)


register_informant_strategy("random", lambda cfg: RandomInformants(cfg.n_informants))
register_informant_strategy("ring", lambda cfg: RingInformants(max(1, cfg.n_informants // 2)))
register_informant_strategy("global", lambda cfg: GlobalInformants())
register_informant_strategy("all", lambda cfg: GlobalInformants())
