"""Particle Swarm Optimisation (Algorithm 39). Independent of the ANN package."""

from psoann.pso.bounds import (
    BoundaryHandler,
    available_boundary_handlers,
    make_boundary_handler,
    register_boundary_handler,
)
from psoann.pso.informants import (
    InformantStrategy,
    available_informant_strategies,
    make_informant_strategy,
    register_informant_strategy,
)
from psoann.pso.optimizer import ParticleSwarmOptimizer
from psoann.pso.random_search import RandomSearchOptimizer
from psoann.pso.schedules import (
    CoefficientSchedule,
    Coefficients,
    available_coefficient_schedules,
    make_coefficient_schedule,
    register_coefficient_schedule,
)
from psoann.pso.types import Fitness, PSOResult, SearchSpace

__all__ = [
    "BoundaryHandler",
    "CoefficientSchedule",
    "Coefficients",
    "Fitness",
    "InformantStrategy",
    "PSOResult",
    "ParticleSwarmOptimizer",
    "RandomSearchOptimizer",
    "SearchSpace",
    "available_boundary_handlers",
    "available_coefficient_schedules",
    "available_informant_strategies",
    "make_boundary_handler",
    "make_coefficient_schedule",
    "make_informant_strategy",
    "register_boundary_handler",
    "register_coefficient_schedule",
    "register_informant_strategy",
]
