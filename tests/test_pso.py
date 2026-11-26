import numpy as np
import pytest

from psoann.config import PSOConfig
from psoann.pso import (
    InformantStrategy,
    ParticleSwarmOptimizer,
    SearchSpace,
    available_boundary_handlers,
    available_informant_strategies,
    make_boundary_handler,
    make_informant_strategy,
    register_informant_strategy,
)
from psoann.pso.informants import GlobalInformants, RandomInformants, RingInformants


def sphere(x):
    return -float(np.sum(x**2))  # maximised at the origin, where it equals 0


def run(cfg: PSOConfig, fitness=sphere, dim=5, bound=5.0, seed=0):
    rng = np.random.default_rng(seed)
    return ParticleSwarmOptimizer(cfg).optimise(fitness, SearchSpace.symmetric(dim, bound), rng), rng


def test_converges_on_the_sphere_function_without_any_neural_network():
    result, _ = run(PSOConfig(swarm_size=30, iterations=200))
    assert result.best_fitness > -1e-3
    assert np.linalg.norm(result.best_position) < 0.05


def test_ring_and_global_topologies_also_converge():
    for name in ("ring", "global"):
        result, _ = run(PSOConfig(swarm_size=30, iterations=300, informants=name))
        assert result.best_fitness > -0.05, name


def test_budget_is_exactly_swarm_size_times_iterations():
    calls = []

    def counting(x):
        calls.append(1)
        return sphere(x)

    result, _ = run(PSOConfig(swarm_size=7, iterations=11), fitness=counting)
    assert len(calls) == result.evaluations == 77


def test_same_seed_same_result_different_seed_different_result():
    cfg = PSOConfig(swarm_size=10, iterations=20)
    a, _ = run(cfg, seed=3)
    b, _ = run(cfg, seed=3)
    c, _ = run(cfg, seed=4)
    np.testing.assert_array_equal(a.best_position, b.best_position)
    np.testing.assert_array_equal(a.history["best"], b.history["best"])
    assert not np.array_equal(a.best_position, c.best_position)


def test_random_stream_does_not_depend_on_coefficients():
    base = PSOConfig(swarm_size=10, iterations=15)
    states = []
    firsts = []
    for cfg in (base, PSOConfig(swarm_size=10, iterations=15, alpha=0.4, beta=0.5, gamma=2.0, delta=0.5)):
        seen = []

        def spy(x, seen=seen):
            seen.append(x.copy())
            return sphere(x)

        _, rng = run(cfg, fitness=spy)
        states.append(rng.bit_generator.state)
        firsts.append(np.array(seen[:10]))
    assert states[0] == states[1]  # identical number of random draws
    np.testing.assert_array_equal(firsts[0], firsts[1])  # identical initial swarm


def test_best_so_far_never_decreases_and_history_is_complete():
    result, _ = run(PSOConfig(swarm_size=10, iterations=40))
    best = result.history["best"]
    assert np.all(np.diff(best) >= 0)
    assert best[-1] == result.best_fitness
    assert set(result.history) == {"evaluations", "best", "mean", "iteration_best", "diversity", "boundary_hits"}
    assert all(len(v) == 40 for v in result.history.values())
    np.testing.assert_array_equal(result.history["evaluations"], np.arange(1, 41) * 10)
    assert np.all(result.history["iteration_best"] <= result.history["best"])
    assert sphere(result.best_position) == result.best_fitness


def test_evaluations_to_reach():
    result, _ = run(PSOConfig(swarm_size=10, iterations=100))
    first = result.evaluations_to_reach(-1.0)
    assert first is not None and first % 10 == 0 and first <= result.evaluations
    assert result.evaluations_to_reach(1.0) is None  # unreachable
    assert result.evaluations_to_final_best() <= result.evaluations


@pytest.mark.parametrize("handler", ["clamp", "reflect"])
def test_bounded_handlers_keep_every_evaluated_point_inside(handler):
    seen = []

    def spy(x):
        seen.append(x.copy())
        return -float(np.sum((x - 1.9) ** 2))  # optimum near the +2 wall

    cfg = PSOConfig(swarm_size=15, iterations=60, bound_handling=handler, bound=2.0)
    run(cfg, fitness=spy, dim=4, bound=2.0)
    points = np.array(seen)
    assert points.min() >= -2.0 and points.max() <= 2.0


def test_unbounded_handler_can_leave_the_region():
    seen = []

    def spy(x):
        seen.append(x.copy())
        return sphere(x - 10.0)  # optimum far outside +-1

    run(PSOConfig(swarm_size=10, iterations=40, bound_handling="none"), fitness=spy, dim=2, bound=1.0)
    assert np.abs(np.array(seen)).max() > 1.0


def test_clamp_zeroes_velocity_and_reflect_reverses_it():
    space = SearchSpace.symmetric(2, 1.0)
    pos = np.array([[1.5, 0.0]])
    vel = np.array([[2.0, 3.0]])
    make_boundary_handler("clamp").apply(pos, vel, space)
    np.testing.assert_array_equal(pos, [[1.0, 0.0]])
    np.testing.assert_array_equal(vel, [[0.0, 3.0]])
    pos = np.array([[1.25, -1.5]])
    vel = np.array([[2.0, -3.0]])
    make_boundary_handler("reflect").apply(pos, vel, space)
    np.testing.assert_allclose(pos, [[0.75, -0.5]])
    np.testing.assert_array_equal(vel, [[-2.0, 3.0]])


def test_velocity_update_matches_algorithm_39_line_24():
    cfg = PSOConfig(alpha=0.5, beta=2.0, gamma=3.0, delta=4.0)
    opt = ParticleSwarmOptimizer(cfg)
    v = np.array([[1.0, -1.0]])
    x = np.array([[0.0, 0.0]])
    pbest = np.array([[1.0, 1.0]])
    ibest = np.array([[2.0, 0.0]])
    gbest = np.array([-1.0, 3.0])
    r = np.array([[[0.5, 1.0]], [[0.25, 0.0]], [[1.0, 0.5]]])  # b/beta, c/gamma, d/delta draws
    expected = 0.5 * v + (2.0 * r[0]) * (pbest - x) + (3.0 * r[1]) * (ibest - x) + (4.0 * r[2]) * (gbest - x)
    np.testing.assert_allclose(opt._update_velocity(v, x, pbest, ibest, gbest, r), expected)
    np.testing.assert_allclose(expected, [[0.5 + 1.0 + 1.5 - 4.0, -0.5 + 2.0 + 0.0 + 6.0]])


def test_move_uses_epsilon():
    opt = ParticleSwarmOptimizer(PSOConfig(epsilon=0.5))
    np.testing.assert_allclose(opt._move(np.array([[1.0]]), np.array([[4.0]])), [[3.0]])


def test_informant_best_picks_fittest_informant_with_self_winning_ties():
    informants = np.array([[0, 1, 2], [1, 0, 2], [2, 0, 1]])
    pbest_fit = np.array([1.0, 5.0, 5.0])
    pbest_pos = np.array([[0.0], [10.0], [20.0]])
    out = ParticleSwarmOptimizer._informant_best(informants, pbest_pos, pbest_fit)
    np.testing.assert_array_equal(out.ravel(), [10.0, 10.0, 20.0])  # particle 2 keeps itself on a tie


def test_nan_fitness_counts_as_worst():
    def flaky(x):
        return float("nan") if x[0] > 0 else sphere(x)

    result, _ = run(PSOConfig(swarm_size=10, iterations=30), fitness=flaky, dim=2)
    assert np.isfinite(result.best_fitness) and result.best_position[0] <= 0


def test_max_velocity_fraction_limits_step_size():
    def first_step(fraction):
        seen = []

        def spy(x):
            seen.append(x.copy())
            return sphere(x)

        cfg = PSOConfig(
            swarm_size=5,
            iterations=2,
            alpha=1.0,
            beta=0,
            gamma=0,
            delta=0,
            bound=10.0,
            max_velocity_fraction=fraction,
            bound_handling="none",
        )
        run(cfg, fitness=spy, dim=3, bound=10.0)
        return np.abs(np.array(seen[5:]) - np.array(seen[:5]))

    unclamped = first_step(None)
    clamped = first_step(0.01)  # |v_d| <= 0.01 * span (= 0.2)
    assert unclamped.max() > 0.2  # the initial half-difference velocities really are larger
    assert 0.0 < clamped.max() <= 0.2 + 1e-12


def test_regenerating_informants_changes_the_random_stream_but_not_the_budget():
    a, rng_a = run(PSOConfig(swarm_size=10, iterations=10))
    b, rng_b = run(PSOConfig(swarm_size=10, iterations=10, regenerate_informants=True))
    assert a.evaluations == b.evaluations == 100
    assert rng_a.bit_generator.state != rng_b.bit_generator.state


# ---- informant strategies -----------------------------------------------------------------------
def test_random_informants_include_self_first_and_are_distinct(rng):
    idx = RandomInformants(3).build(20, rng)
    assert idx.shape == (20, 4)
    np.testing.assert_array_equal(idx[:, 0], np.arange(20))
    assert all(len(set(row)) == 4 for row in idx)


def test_random_informants_rejects_k_larger_than_swarm(rng):
    with pytest.raises(ValueError, match="at least"):
        RandomInformants(10).build(10, rng)


def test_ring_and_global_informants(rng):
    ring = RingInformants(1).build(5, rng)
    np.testing.assert_array_equal(ring[0], [0, 4, 1])
    np.testing.assert_array_equal(ring[4], [4, 3, 0])
    everyone = GlobalInformants().build(4, rng)
    assert everyone.shape == (4, 4) and all(set(row) == {0, 1, 2, 3} for row in everyone)
    np.testing.assert_array_equal(everyone[:, 0], np.arange(4))
    with pytest.raises(ValueError):
        RingInformants(3).build(5, rng)


def test_informant_registry_and_custom_strategy(rng):
    assert {"random", "ring", "global", "all"} <= set(available_informant_strategies())
    assert {"clamp", "reflect", "none"} <= set(available_boundary_handlers())

    class Pairs(InformantStrategy):
        def build(self, swarm_size, rng):
            return np.array([[i, i ^ 1] for i in range(swarm_size)])

    register_informant_strategy("test_pairs", lambda cfg: Pairs(), overwrite=True)
    strategy = make_informant_strategy(PSOConfig(informants="test_pairs"))
    assert isinstance(strategy, Pairs)
    result, _ = run(PSOConfig(swarm_size=10, iterations=50, informants="test_pairs"))
    assert result.best_fitness > -1.0
    with pytest.raises(ValueError, match="unknown informant strategy"):
        make_informant_strategy(PSOConfig(informants="nope"))


def test_search_space_validation():
    with pytest.raises(ValueError):
        SearchSpace(np.array([0.0, 1.0]), np.array([1.0, 1.0]))
    with pytest.raises(ValueError):
        SearchSpace(np.zeros(2), np.ones(3))
    space = SearchSpace.symmetric(3, 2.0)
    assert space.dim == 3 and np.all(space.span == 4.0)
