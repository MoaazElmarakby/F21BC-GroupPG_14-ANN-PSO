"""Boundary handling: what happens when a particle's move would leave the search space.

``apply`` modifies ``positions`` (and ``velocities``) in place. Handlers never draw random numbers
themselves: a handler that needs randomness (``random``) receives a pre-drawn ``noise`` matrix of
uniforms with the shape of ``positions``. The optimiser always draws that matrix, so its random
stream does not depend on the handler (see design doc, section 3).

Strategies (compared in the boundary study, ``psoann sweep bounds``):

* ``clamp``   - absorbing wall: stop on the boundary, zero that velocity component;
* ``reflect`` - elastic wall: mirror the overshoot back inside, reverse that velocity component;
* ``wrap``    - periodic space: leave on one side, re-enter on the other (velocity unchanged);
* ``random``  - re-initialise the offending coordinate uniformly inside the bounds, zero its velocity;
* ``none``    - unbounded (the space only fixes the initial sampling region).

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
    def apply(self, positions: Matrix, velocities: Matrix, space: SearchSpace, noise: Matrix | None = None) -> None:
        """Bring every row of ``positions`` back inside ``space`` (in place)."""


class ClampBoundary(BoundaryHandler):
    """Absorbing wall: put the particle on the boundary and zero that velocity component."""

    def apply(self, positions: Matrix, velocities: Matrix, space: SearchSpace, noise: Matrix | None = None) -> None:
        outside = (positions < space.lower) | (positions > space.upper)
        np.clip(positions, space.lower, space.upper, out=positions)
        velocities[outside] = 0.0


class ReflectBoundary(BoundaryHandler):
    """Elastic wall: mirror the overshoot back inside and reverse that velocity component."""

    def apply(self, positions: Matrix, velocities: Matrix, space: SearchSpace, noise: Matrix | None = None) -> None:
        over = positions > space.upper
        under = positions < space.lower
        positions[over] = (2.0 * np.broadcast_to(space.upper, positions.shape) - positions)[over]
        positions[under] = (2.0 * np.broadcast_to(space.lower, positions.shape) - positions)[under]
        velocities[over | under] *= -1.0
        np.clip(positions, space.lower, space.upper, out=positions)  # overshoot larger than the span


class WrapBoundary(BoundaryHandler):
    """Periodic space (a torus): a coordinate leaving one side re-enters from the opposite side."""

    def apply(self, positions: Matrix, velocities: Matrix, space: SearchSpace, noise: Matrix | None = None) -> None:
        outside = (positions < space.lower) | (positions > space.upper)
        if outside.any():
            wrapped = space.lower + np.mod(positions - space.lower, space.span)
            positions[outside] = np.broadcast_to(wrapped, positions.shape)[outside]
        np.clip(positions, space.lower, space.upper, out=positions)  # guard against rounding at the upper edge


class RandomBoundary(BoundaryHandler):
    """Random re-initialisation: an offending coordinate is redrawn uniformly in the bounds, velocity zeroed."""

    def apply(self, positions: Matrix, velocities: Matrix, space: SearchSpace, noise: Matrix | None = None) -> None:
        if noise is None or noise.shape != positions.shape:
            raise ValueError("the 'random' boundary handler needs a noise matrix shaped like the positions")
        outside = (positions < space.lower) | (positions > space.upper)
        fresh = space.lower + noise * space.span
        positions[outside] = np.broadcast_to(fresh, positions.shape)[outside]
        velocities[outside] = 0.0


class NoBoundary(BoundaryHandler):
    """Unbounded search (the space only fixes the initial sampling region)."""

    def apply(self, positions: Matrix, velocities: Matrix, space: SearchSpace, noise: Matrix | None = None) -> None:
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
register_boundary_handler("wrap", WrapBoundary)
register_boundary_handler("random", RandomBoundary)
register_boundary_handler("none", NoBoundary)
