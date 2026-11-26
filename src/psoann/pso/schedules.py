"""Coefficient schedules: how alpha, beta, gamma and delta change during a run.

Algorithm 39 keeps the four coefficients fixed (``constant``, the default). The other schedules are
extensions from the PSO literature that move a coefficient linearly from a start to an end value:

* ``linear_inertia``: alpha falls from 0.9 to 0.4, wide steps first and fine steps later
  (Shi and Eberhart, 1999);
* ``tvac`` (time-varying acceleration coefficients): beta falls from 2.5 to 0.5 while gamma rises from
  0.5 to 2.5, so particles trust themselves early and their informants late (Ratnaweera, Halgamuge
  and Watson, 2004).

A schedule only changes the *values* of the coefficients. It draws no random numbers, so the
optimiser's random stream is the same for every schedule (see docs/design.md, section 3).

To add one: subclass ``CoefficientSchedule`` and ``register_coefficient_schedule(name, factory)``
where ``factory(pso_config) -> schedule``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Final

from psoann.config import PSOConfig
from psoann.registry import Registry

INERTIA_START: Final[float] = 0.9  # Shi and Eberhart (1999)
INERTIA_END: Final[float] = 0.4
TVAC_COGNITIVE_START: Final[float] = 2.5  # Ratnaweera et al. (2004): beta, pull to the personal best
TVAC_COGNITIVE_END: Final[float] = 0.5
TVAC_SOCIAL_START: Final[float] = 0.5  # gamma, pull to the informants' best
TVAC_SOCIAL_END: Final[float] = 2.5


@dataclass(frozen=True)
class Coefficients:
    """The four weights of the velocity update [Alg. 39, lines 2-5]."""

    alpha: float  # proportion of velocity retained
    beta: float  # proportion of personal best mixed in
    gamma: float  # proportion of informants' best mixed in
    delta: float  # proportion of global best mixed in

    @classmethod
    def from_config(cls, config: PSOConfig) -> Coefficients:
        return cls(config.alpha, config.beta, config.gamma, config.delta)


class CoefficientSchedule(ABC):
    @abstractmethod
    def at(self, progress: float) -> Coefficients:
        """Coefficients for one velocity update; ``progress`` runs from 0.0 (first) to 1.0 (last)."""


class ConstantSchedule(CoefficientSchedule):
    """Algorithm 39 as written: the same coefficients for the whole run."""

    def __init__(self, coefficients: Coefficients) -> None:
        self.coefficients = coefficients

    def at(self, progress: float) -> Coefficients:
        return self.coefficients


class LinearSchedule(CoefficientSchedule):
    """Every coefficient moves in a straight line from its ``start`` value to its ``end`` value."""

    def __init__(self, start: Coefficients, end: Coefficients) -> None:
        self.start = start
        self.end = end

    def at(self, progress: float) -> Coefficients:
        t = min(1.0, max(0.0, progress))
        s, e = self.start, self.end
        return Coefficients(
            alpha=s.alpha + (e.alpha - s.alpha) * t,
            beta=s.beta + (e.beta - s.beta) * t,
            gamma=s.gamma + (e.gamma - s.gamma) * t,
            delta=s.delta + (e.delta - s.delta) * t,
        )


def _linear_inertia(config: PSOConfig) -> CoefficientSchedule:
    base = Coefficients.from_config(config)
    return LinearSchedule(replace(base, alpha=INERTIA_START), replace(base, alpha=INERTIA_END))


def _tvac(config: PSOConfig) -> CoefficientSchedule:
    base = Coefficients.from_config(config)
    return LinearSchedule(
        replace(base, beta=TVAC_COGNITIVE_START, gamma=TVAC_SOCIAL_START),
        replace(base, beta=TVAC_COGNITIVE_END, gamma=TVAC_SOCIAL_END),
    )


ScheduleFactory = Callable[[PSOConfig], CoefficientSchedule]
_REGISTRY: Final[Registry[ScheduleFactory]] = Registry("coefficient schedule")


def register_coefficient_schedule(name: str, factory: ScheduleFactory, *, overwrite: bool = False) -> ScheduleFactory:
    return _REGISTRY.register(name, factory, overwrite=overwrite)


def available_coefficient_schedules() -> list[str]:
    return _REGISTRY.names()


def make_coefficient_schedule(config: PSOConfig) -> CoefficientSchedule:
    return _REGISTRY.get(config.schedule)(config)


register_coefficient_schedule("constant", lambda cfg: ConstantSchedule(Coefficients.from_config(cfg)))
register_coefficient_schedule("linear_inertia", _linear_inertia)
register_coefficient_schedule("tvac", _tvac)
