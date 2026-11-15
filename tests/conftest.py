from pathlib import Path

import numpy as np
import pytest

from psoann.config import ExperimentConfig

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


@pytest.fixture
def tiny_config() -> ExperimentConfig:
    """A fast configuration for end-to-end tests."""
    return ExperimentConfig.from_dict(
        {
            "name": "tiny",
            "n_runs": 3,
            "pso": {"swarm_size": 8, "iterations": 6},
        }
    )
