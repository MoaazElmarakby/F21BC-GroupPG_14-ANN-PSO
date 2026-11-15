"""Boundary handling: what happens when a particle's move would leave the search space.

``apply`` modifies ``positions`` (and ``velocities``) in place. Handlers must not draw random numbers,
so that the optimiser's random stream does not depend on the handler (see design doc, section 3).

To add one: subclass ``BoundaryHandler`` and ``register_boundary_handler("name", cls)``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from psoann.pso.types import SearchSpace
from psoann.registry import Registry

Matrix = NDArray[np.float64]


class BoundaryHandler(ABC):
    @abstractmethod
    def apply(self, positions: Matrix, velocities: Matrix, space: SearchSpace) -> None:
        """Bring every row of ``positions`` back inside ``space`` (in place)."""


class ClampBoundary(BoundaryHandler):
    """Absorbing wall: put the particle on the boundary and zero that velocity component."""

    def apply(self, positions: Matrix, velocities: Matrix, space: SearchSpace) -> None:
        outside = (positions < space.lower) | (positions > space.upper)
        np.clip(positions, space.lower, space.upper, out=positions)
        velocities[outside] = 0.0


class ReflectBoundary(BoundaryHandler):
    """Elastic wall: mirror the overshoot back inside and reverse that velocity component."""

    def apply(self, positions: Matrix, velocities: Matrix, space: SearchSpace) -> None:
        over = positions > space.upper
        under = positions < space.lower
        positions[over] = (2.0 * np.broadcast_to(space.upper, positions.shape) - positions)[over]
        positions[under] = (2.0 * np.broadcast_to(space.lower, positions.shape) - positions)[under]
        velocities[over | under] *= -1.0
        np.clip(positions, space.lower, space.upper, out=positions)  # overshoot larger than the span


class NoBoundary(BoundaryHandler):
    """Unbounded search (the space only fixes the initial sampling region)."""

    def apply(self, positions: Matrix, velocities: Matrix, space: SearchSpace) -> None:
        return None


_REGISTRY: Registry[type[BoundaryHandler]] = Registry("boundary handler")


def register_boundary_handler(name: str, cls: type[BoundaryHandler], *, overwrite: bool = False) -> type[BoundaryHandler]:
    return _REGISTRY.register(name, cls, overwrite=overwrite)


def available_boundary_handlers() -> list[str]:
    return _REGISTRY.names()


def make_boundary_handler(name: str) -> BoundaryHandler:
    return _REGISTRY.get(name)()


register_boundary_handler("clamp", ClampBoundary)
register_boundary_handler("reflect", ReflectBoundary)
register_boundary_handler("none", NoBoundary)
