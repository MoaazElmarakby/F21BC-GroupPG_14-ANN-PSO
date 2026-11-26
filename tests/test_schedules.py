"""Tests for coefficient schedules: fixed coefficients (Algorithm 39) and the time-varying extensions."""

import numpy as np
import pytest

from psoann.cli import main
from psoann.config import ExperimentConfig, PSOConfig
from psoann.experiments import SWEEPS
from psoann.pso import (
    Coefficients,
    CoefficientSchedule,
    ParticleSwarmOptimizer,
    SearchSpace,
    available_coefficient_schedules,
    make_coefficient_schedule,
    register_coefficient_schedule,
)
from psoann.pso.schedules import ConstantSchedule, LinearSchedule


def sphere(x):
    return -float(np.sum(x**2))


def run(cfg: PSOConfig, seed=0):
    rng = np.random.default_rng(seed)
    return ParticleSwarmOptimizer(cfg).optimise(sphere, SearchSpace.symmetric(5, 5.0), rng), rng


def test_default_schedule_is_constant_and_returns_the_config_coefficients():
    cfg = PSOConfig(alpha=0.5, beta=1.0, gamma=2.0, delta=0.25)
    assert cfg.schedule == "constant" and ExperimentConfig().pso.schedule == "constant"
    schedule = make_coefficient_schedule(cfg)
    assert isinstance(schedule, ConstantSchedule)
    assert schedule.at(0.0) == schedule.at(1.0) == Coefficients(0.5, 1.0, 2.0, 0.25)


def test_linear_schedule_interpolates_and_clips_progress():
    schedule = LinearSchedule(Coefficients(0.9, 2.5, 0.5, 0.0), Coefficients(0.4, 0.5, 2.5, 1.0))
    assert schedule.at(0.0) == Coefficients(0.9, 2.5, 0.5, 0.0)
    assert schedule.at(1.0) == schedule.at(7.0) == Coefficients(0.4, 0.5, 2.5, 1.0)
    middle = schedule.at(0.5)
    np.testing.assert_allclose([middle.alpha, middle.beta, middle.gamma, middle.delta], [0.65, 1.5, 1.5, 0.5])


def test_literature_schedules_change_only_their_own_coefficients():
    base = {"alpha": 0.7, "beta": 1.2, "gamma": 1.3, "delta": 0.4}
    inertia = make_coefficient_schedule(PSOConfig(**base, schedule="linear_inertia"))
    assert inertia.at(0.0) == Coefficients(0.9, 1.2, 1.3, 0.4) and inertia.at(1.0) == Coefficients(0.4, 1.2, 1.3, 0.4)
    tvac = make_coefficient_schedule(PSOConfig(**base, schedule="tvac"))
    assert tvac.at(0.0) == Coefficients(0.7, 2.5, 0.5, 0.4) and tvac.at(1.0) == Coefficients(0.7, 0.5, 2.5, 0.4)


def test_optimiser_asks_the_schedule_once_per_move_from_start_to_end():
    asked = []

    class Spy(CoefficientSchedule):
        def at(self, progress):
            asked.append(progress)
            return Coefficients(0.7, 1.5, 1.5, 0.0)

    cfg = PSOConfig(swarm_size=5, iterations=6)
    ParticleSwarmOptimizer(cfg, schedule=Spy()).optimise(
        sphere, SearchSpace.symmetric(2, 1.0), np.random.default_rng(0)
    )
    np.testing.assert_allclose(asked, [0.0, 0.25, 0.5, 0.75, 1.0])  # 6 assessments -> 5 moves


def test_constant_schedule_reproduces_the_fixed_coefficient_update_exactly():
    cfg = PSOConfig(swarm_size=12, iterations=40)
    explicit = ConstantSchedule(Coefficients.from_config(cfg))
    a, _ = run(cfg)
    rng = np.random.default_rng(0)
    b = ParticleSwarmOptimizer(cfg, schedule=explicit).optimise(sphere, SearchSpace.symmetric(5, 5.0), rng)
    np.testing.assert_array_equal(a.history["best"], b.history["best"])
    np.testing.assert_array_equal(a.best_position, b.best_position)


@pytest.mark.parametrize("name", ["linear_inertia", "tvac"])
def test_scheduled_runs_converge_and_keep_the_random_stream(name):
    base, rng_base = run(PSOConfig(swarm_size=30, iterations=200))
    scheduled, rng_scheduled = run(PSOConfig(swarm_size=30, iterations=200, schedule=name))
    assert scheduled.best_fitness > -1e-3
    assert rng_base.bit_generator.state == rng_scheduled.bit_generator.state
    assert not np.array_equal(base.history["best"], scheduled.history["best"])


def test_schedule_registry_and_custom_schedule():
    assert {"constant", "linear_inertia", "tvac"} <= set(available_coefficient_schedules())
    register_coefficient_schedule(
        "test_frozen", lambda cfg: ConstantSchedule(Coefficients(0.0, 0.0, 0.0, 0.0)), overwrite=True
    )
    result, _ = run(PSOConfig(swarm_size=6, iterations=10, schedule="test_frozen"))
    assert result.history["best"][0] == result.best_fitness  # zero coefficients: nobody moves
    with pytest.raises(ValueError, match="unknown coefficient schedule"):
        make_coefficient_schedule(PSOConfig(schedule="nope"))


def test_schedule_study_and_command_line(tmp_path, capsys):
    labels = [point.config.pso.schedule for point in SWEEPS["schedule"].expand()]
    assert labels == ["constant", "linear_inertia", "tvac"]
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    assert "tvac" in out and "reject" in out
    fast = ["--set", "pso.swarm_size=6", "--set", "pso.iterations=4"]
    assert main(["run", "--runs", "1", *fast, "--set", "pso.schedule=tvac", "--out", str(tmp_path)]) == 0
