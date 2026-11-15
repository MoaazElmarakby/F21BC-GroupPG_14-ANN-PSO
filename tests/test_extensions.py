"""Tests for the extensions: metrics, boundary handlers, nearest informants, evolved activations,
random search, sweep points, the study presets and the statistics."""

import numpy as np
import pandas as pd
import pytest

from psoann.analysis.plots import plot_boxplot, plot_heatmap
from psoann.analysis.stats import a12, effect_label, holm, kruskal_wallis, pairwise_tests
from psoann.analysis.summary import summarise
from psoann.ann import EvolvedActivationNetwork, build_network
from psoann.ann.activations import get_activation
from psoann.cli import main
from psoann.config import ANNConfig, ExperimentConfig, PSOConfig, TrainingConfig
from psoann.evaluation import confusion_counts, precision_recall_f1
from psoann.experiments import SWEEPS, Sweep, run_experiment, run_single
from psoann.pso import ParticleSwarmOptimizer, RandomSearchOptimizer, SearchSpace, make_boundary_handler
from psoann.pso.informants import NearestInformants, make_informant_strategy
from psoann.training import PSOTrainer


# ---- metrics ----------------------------------------------------------------------------------------
def test_precision_recall_f1_on_a_known_confusion_matrix():
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    outputs = np.array([0.9, 0.8, 0.7, 0.1, 0.9, 0.2, 0.1, 0.3, 0.4, 0.0])  # tp=3 fn=1 fp=1 tn=5
    assert confusion_counts(outputs, labels) == (3, 1, 1, 5)
    p, r, f1 = precision_recall_f1(outputs, labels)
    assert (p, r) == (0.75, 0.75) and f1 == pytest.approx(0.75)


def test_f1_is_zero_when_nothing_is_predicted_positive():
    assert precision_recall_f1(np.zeros(4), np.array([1, 0, 1, 0])) == (0.0, 0.0, 0.0)


def test_run_rows_report_f1_and_target_columns():
    cfg = ExperimentConfig(n_runs=1, target_fitness=0.5, pso=PSOConfig(swarm_size=8, iterations=5))
    row, _ = run_single(cfg, 0)
    assert 0.0 <= row["test_f1"] <= 1.0 and 0.0 <= row["train_recall"] <= 1.0
    assert row["reached_target"] in (0, 1) and row["hidden_activations"] == "tanh"


# ---- boundary handlers ------------------------------------------------------------------------------
@pytest.fixture
def space():
    return SearchSpace.symmetric(2, 1.0)


def test_wrap_reenters_from_the_opposite_side(space):
    pos = np.array([[1.25, -1.5]])
    vel = np.array([[2.0, -3.0]])
    make_boundary_handler("wrap").apply(pos, vel, space)
    np.testing.assert_allclose(pos, [[-0.75, 0.5]])
    np.testing.assert_array_equal(vel, [[2.0, -3.0]])  # velocity untouched


def test_wrap_handles_overshoot_of_several_spans(space):
    pos = np.array([[7.0, -9.0]])
    make_boundary_handler("wrap").apply(pos, np.zeros((1, 2)), space)
    assert np.all(pos >= -1.0) and np.all(pos <= 1.0)


def test_random_boundary_redraws_only_offending_coordinates(space):
    pos = np.array([[1.5, 0.25]])
    vel = np.array([[2.0, 3.0]])
    noise = np.array([[0.75, 0.9]])
    make_boundary_handler("random").apply(pos, vel, space, noise)
    np.testing.assert_allclose(pos, [[0.5, 0.25]])  # -1 + 0.75 * 2
    np.testing.assert_array_equal(vel, [[0.0, 3.0]])


def test_random_boundary_needs_noise(space):
    with pytest.raises(ValueError, match="noise"):
        make_boundary_handler("random").apply(np.array([[2.0, 0.0]]), np.zeros((1, 2)), space)


@pytest.mark.parametrize("handler", ["clamp", "reflect", "wrap", "random"])
def test_every_bounded_handler_keeps_the_swarm_inside_and_counts_hits(handler):
    seen = []

    def spy(x):
        seen.append(x.copy())
        return -float(np.sum((x - 5.0) ** 2))  # optimum far outside +-2: constant pressure on the wall

    result = ParticleSwarmOptimizer(PSOConfig(swarm_size=10, iterations=30, bound_handling=handler)).optimise(
        spy, SearchSpace.symmetric(3, 2.0), np.random.default_rng(0)
    )
    assert np.abs(np.array(seen)).max() <= 2.0
    assert result.history["boundary_hits"].sum() > 0 and result.history["boundary_hits"][0] == 0


def test_random_stream_is_the_same_for_every_boundary_handler():
    """Same seed, tiny steps so nobody reaches a wall: results must not depend on the handler."""
    runs = [
        ParticleSwarmOptimizer(
            PSOConfig(swarm_size=8, iterations=10, bound_handling=h, max_velocity_fraction=1e-6)
        ).optimise(lambda x: -float(np.sum(x**2)), SearchSpace.symmetric(3, 2.0), np.random.default_rng(3))
        for h in ("clamp", "reflect", "wrap", "random", "none")
    ]
    assert all(run.history["boundary_hits"].sum() == 0 for run in runs)
    for other in runs[1:]:
        np.testing.assert_array_equal(runs[0].history["best"], other.history["best"])


# ---- informants -------------------------------------------------------------------------------------
def test_nearest_informants_pick_the_closest_particles():
    positions = np.array([[0.0], [1.0], [1.5], [10.0]])
    idx = NearestInformants(2).build(4, np.random.default_rng(0), positions)
    assert idx[:, 0].tolist() == [0, 1, 2, 3]  # self first
    assert set(idx[0, 1:]) == {1, 2} and set(idx[1, 1:]) == {0, 2} and set(idx[3, 1:]) == {2, 1}


def test_nearest_informants_validate_input(rng):
    with pytest.raises(ValueError, match="positions"):
        NearestInformants(1).build(4, rng)
    with pytest.raises(ValueError, match="at least"):
        NearestInformants(4).build(4, rng, np.zeros((4, 2)))


def test_nearest_informants_are_recomputed_each_iteration_and_use_the_budget():
    cfg = PSOConfig(swarm_size=10, iterations=15, informants="nearest", n_informants=3)
    strategy = make_informant_strategy(cfg)
    assert strategy.uses_positions
    result = ParticleSwarmOptimizer(cfg).optimise(lambda x: -float(np.sum(x**2)), SearchSpace.symmetric(4, 2.0), np.random.default_rng(1))
    assert result.evaluations == 150 and np.all(np.diff(result.history["best"]) >= 0)


# ---- evolved activations ----------------------------------------------------------------------------
def evolved_net(hidden=(5, 3)):
    cfg = ANNConfig(hidden_layers=hidden, evolve_activations=True)
    return build_network(cfg, 4), cfg


def test_evolved_network_has_one_gene_per_hidden_layer():
    net, _ = evolved_net()
    assert isinstance(net, EvolvedActivationNetwork)
    assert net.n_weights == (4 * 5 + 5) + (5 * 3 + 3) + (3 * 1 + 1) and net.n_parameters == net.n_weights + 2
    lower, upper = net.bounds(2.0)
    assert lower.shape == upper.shape == (net.n_parameters,)
    np.testing.assert_array_equal(lower[-2:], [0, 0]) and np.testing.assert_array_equal(upper[-2:], [3, 3])
    assert upper[0] == 2.0 and lower[0] == -2.0


def test_gene_integer_part_selects_the_activation(rng):
    net, cfg = evolved_net(hidden=(4,))
    X = rng.normal(size=(6, 4))
    weights = rng.normal(size=net.n_weights)
    for gene, name in ((0.2, "logistic"), (1.7, "tanh"), (2.99, "relu"), (3.0, "relu")):
        fixed = build_network(ANNConfig(hidden_layers=(4,), hidden_activation=name), 4)
        np.testing.assert_allclose(net.forward(np.append(weights, gene), X), fixed.forward(weights, X))
        assert net.describe_activations(np.append(weights, gene)) == name


def test_evolving_needs_a_hidden_layer_and_the_right_length(rng):
    with pytest.raises(ValueError, match="hidden layer"):
        ANNConfig(hidden_layers=(), evolve_activations=True)
    net, _ = evolved_net()
    with pytest.raises(ValueError, match="length"):
        net.forward(np.zeros(net.n_parameters - 1), np.zeros((2, 4)))


def test_pso_can_train_a_network_with_evolved_activations():
    cfg = ExperimentConfig.from_dict(
        {"n_runs": 2, "pso": {"swarm_size": 10, "iterations": 8}, "ann": {"hidden_layers": [4, 4], "evolve_activations": True}}
    )
    rows = run_experiment(cfg).rows
    assert all(r["n_parameters"] == (4 * 4 + 4) + (4 * 4 + 4) + (4 + 1) + 2 for r in rows)
    assert all(set(r["hidden_activations"].split("-")) <= {"logistic", "tanh", "relu"} for r in rows)


def test_genes_stay_inside_their_range_during_training(rng):
    net, cfg = evolved_net(hidden=(3,))
    seen = []

    def spy(x):
        seen.append(x[net.n_weights :].copy())
        return float(rng.random())

    from psoann.pso import SearchSpace as S

    ParticleSwarmOptimizer(PSOConfig(swarm_size=8, iterations=20)).optimise(spy, S(*net.bounds(2.0)), np.random.default_rng(0))
    genes = np.array(seen)
    assert genes.min() >= 0.0 and genes.max() <= 3.0


# ---- random search control --------------------------------------------------------------------------
def test_random_search_uses_the_same_budget_and_history_keys_as_pso():
    cfg = PSOConfig(swarm_size=10, iterations=7)
    space = SearchSpace.symmetric(3, 1.0)
    f = lambda x: -float(np.sum(x**2))  # noqa: E731
    rs = RandomSearchOptimizer(cfg).optimise(f, space, np.random.default_rng(0))
    pso = ParticleSwarmOptimizer(cfg).optimise(f, space, np.random.default_rng(0))
    assert rs.evaluations == pso.evaluations == 70 and set(rs.history) == set(pso.history)
    assert np.all(np.diff(rs.history["best"]) >= 0) and rs.best_fitness == rs.history["best"][-1]
    np.testing.assert_array_equal(rs.history["evaluations"], pso.history["evaluations"])


def test_training_optimizer_option_selects_random_search(tiny_config):
    cfg = tiny_config.with_overrides({"training.optimizer": "random_search", "n_runs": 1})
    assert isinstance(PSOTrainer(cfg.ann, cfg.pso, cfg.training).optimizer, RandomSearchOptimizer)
    assert run_experiment(cfg).rows[0]["evaluations"] == 48
    with pytest.raises(ValueError, match="optimizer"):
        TrainingConfig(optimizer="nope")


def test_pso_beats_random_search_on_a_smooth_landscape():
    cfg = PSOConfig(swarm_size=20, iterations=100)
    space = SearchSpace.symmetric(10, 2.0)
    f = lambda x: -float(np.sum(x**2))  # noqa: E731
    pso = [ParticleSwarmOptimizer(cfg).optimise(f, space, np.random.default_rng(s)).best_fitness for s in range(5)]
    rs = [RandomSearchOptimizer(cfg).optimise(f, space, np.random.default_rng(s)).best_fitness for s in range(5)]
    assert min(pso) > max(rs)


# ---- sweeps and presets -----------------------------------------------------------------------------
def test_points_are_labelled_and_crossed_with_the_grid():
    sweep = Sweep.from_dict(
        {
            "name": "s",
            "grid": {"pso.alpha": [0.4, 0.9]},
            "points": [{"label": "a", "pso.n_informants": 2}, {"label": "b", "pso.informants": "global"}],
        }
    )
    points = sweep.expand()
    assert [p.point_id for p in points] == ["alpha=0.4__a", "alpha=0.4__b", "alpha=0.9__a", "alpha=0.9__b"]
    assert points[1].config.pso.informants == "global" and points[2].config.pso.alpha == 0.9


@pytest.mark.parametrize(
    "bad",
    [
        {"name": "x", "points": []},
        {"name": "x", "points": [{"label": "only a label"}]},
        {"name": "x", "points": [{"label": "a", "pso.alpha": 1}, {"label": "a", "pso.alpha": 2}]},
        {"name": "x", "grid": {"pso.alpha": [1]}, "points": [{"pso.alpha": 2}]},
    ],
)
def test_bad_points_are_rejected(bad):
    with pytest.raises(ValueError):
        Sweep.from_dict(bad)


def sizes():
    return {name: len(sweep.expand()) for name, sweep in SWEEPS.items()}


def test_study_presets_match_the_study_plan():
    n = sizes()
    assert n["architecture"] == 13 and n["swarm_vs_iterations"] == 12 and n["coefficients"] == 15
    assert n["informants"] == 16 and n["bounds"] == 5 and n["activations"] == 5 and n["fitness"] == 3
    assert n["random_search"] == 2


def test_study_presets_use_30_runs_and_a_fixed_budget():
    for name in ("architecture", "coefficients", "informants", "bounds", "activations", "fitness", "random_search"):
        for point in SWEEPS[name].expand():
            assert point.config.n_runs == 30 and point.config.pso.budget == 2000, (name, point.point_id)
    budgets = {p.config.pso.budget for p in SWEEPS["swarm_vs_iterations"].expand()}
    assert budgets == {500, 2000}


def test_coefficient_points_sum_to_four_and_cover_all_whole_splits():
    coefficients = {(p.config.pso.beta, p.config.pso.gamma, p.config.pso.delta) for p in SWEEPS["coefficients"].expand()}
    assert len(coefficients) == 15 and all(sum(c) == 4.0 for c in coefficients)


def test_architecture_points_include_the_linear_baseline_and_depth_three():
    layers = {p.point_id: p.config.ann.hidden_layers for p in SWEEPS["architecture"].expand()}
    assert layers["linear"] == () and layers["3x16"] == (16, 16, 16) and layers["2x4"] == (4, 4)


def test_every_informant_point_builds_with_the_study_swarm():
    for point in SWEEPS["informants"].expand():
        cfg = point.config.pso
        strategy = make_informant_strategy(cfg)
        positions = np.random.default_rng(0).random((cfg.swarm_size, 5))
        idx = strategy.build(cfg.swarm_size, np.random.default_rng(0), positions) if strategy.uses_positions else strategy.build(cfg.swarm_size, np.random.default_rng(0))
        assert idx.shape[0] == cfg.swarm_size


def test_a_study_point_runs_end_to_end():
    sweep = SWEEPS["activations"]
    point = next(p for p in sweep.expand() if p.point_id == "evolved")
    row, _ = run_single(point.config.with_overrides({"pso.swarm_size": 6, "pso.iterations": 3}), 0)
    assert row["n_parameters"] == (4 * 8 + 8) + (8 * 8 + 8) + (8 + 1) + 2


# ---- statistics -------------------------------------------------------------------------------------
def test_a12_known_values():
    assert a12([3, 4, 5], [0, 1, 2]) == 1.0 and a12([0, 1, 2], [3, 4, 5]) == 0.0
    assert a12([1, 1], [1, 1]) == 0.5 and a12([1, 2, 3], [2]) == 0.5
    assert effect_label(0.5) == "negligible" and effect_label(0.9) == "large" and effect_label(0.35) == "medium"


def test_holm_matches_hand_calculation_and_is_monotone():
    adjusted = holm([0.01, 0.04, 0.03])
    np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])
    assert np.all(holm([0.5, 0.9]) <= 1.0)


@pytest.fixture
def groups():
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "point": np.repeat(["a", "b", "c"], 30),
            "test_accuracy": np.concatenate([rng.normal(0.90, 0.01, 30), rng.normal(0.95, 0.01, 30), rng.normal(0.90, 0.01, 30)]),
        }
    )


def test_kruskal_wallis_detects_a_difference(groups):
    out = kruskal_wallis(groups, "point", "test_accuracy")
    assert out["p"] < 1e-6 and out["groups"] == 3 and out["n"] == 90


def test_pairwise_tests_separate_the_different_group(groups):
    table = pairwise_tests(groups, "point", "test_accuracy")
    assert len(table) == 3
    ab = table[(table.group_a == "a") & (table.group_b == "b")].iloc[0]
    ac = table[(table.group_a == "a") & (table.group_b == "c")].iloc[0]
    assert ab.p_holm < 1e-6 and ab.a12 < 0.1 and ab.effect == "large"
    assert ac.p_holm > 0.05 and (table.p_holm >= table.p).all()
    control = pairwise_tests(groups, "point", "test_accuracy", control="a")
    assert set(control.group_a) == {"a"} and len(control) == 2


def test_identical_constant_groups_are_not_significant():
    df = pd.DataFrame({"point": ["a"] * 5 + ["b"] * 5, "test_accuracy": [1.0] * 10})
    assert kruskal_wallis(df, "point", "test_accuracy")["p"] == 1.0
    assert pairwise_tests(df, "point", "test_accuracy")["p"].iloc[0] == 1.0


def test_stats_errors(groups):
    with pytest.raises(KeyError):
        kruskal_wallis(groups, "nope", "test_accuracy")
    with pytest.raises(KeyError, match="control"):
        pairwise_tests(groups, "point", "test_accuracy", control="zzz")
    with pytest.raises(ValueError):
        kruskal_wallis(groups[groups.point == "a"], "point", "test_accuracy")


def test_summarise_includes_f1_and_target_columns_when_present():
    df = pd.DataFrame(
        {"experiment": ["e"] * 4, "train_accuracy": [1.0] * 4, "test_accuracy": [0.9] * 4, "test_f1": [0.9] * 4,
         "train_mse": [0.1] * 4, "test_mse": [0.1] * 4, "evals_to_final_best": [10] * 4,
         "evals_to_target": [5, None, 7, None], "reached_target": [1, 0, 1, 0]}
    )
    table = summarise(df)
    assert table.loc[0, "reached_target_mean"] == 0.5 and table.loc[0, "evals_to_target_mean"] == 6.0 and "test_f1_mean" in table


# ---- CLI end to end ---------------------------------------------------------------------------------
def test_stats_and_new_plots_from_the_command_line(tmp_path, capsys):
    fast = ["--set", "pso.swarm_size=6", "--set", "pso.iterations=3"]
    assert main(["sweep", "random_search", "--runs", "4", *fast, "--out", str(tmp_path)]) == 0
    csv = tmp_path / "sweep_random_search.csv"
    assert main(["stats", str(csv), "--by", "point", "--metric", "test_accuracy"]) == 0
    out = capsys.readouterr().out
    assert "Kruskal-Wallis" in out and "p_holm" in out and "pso" in out
    assert main(["stats", str(csv), "--control", "random_search"]) == 0
    png = tmp_path / "box.png"
    assert main(["plot-box", str(csv), "--out", str(png)]) == 0 and png.stat().st_size > 1000
    heat = tmp_path / "heat.png"
    df = pd.read_csv(csv)
    assert plot_heatmap(df, "point", "cfg.pso.swarm_size", "test_accuracy", heat).exists()
    assert plot_boxplot(df, "point", "test_f1", tmp_path / "b2.png").exists()
    with pytest.raises(KeyError):
        plot_heatmap(df, "nope", "point", "test_accuracy", heat)


def test_boundary_hits_can_be_plotted(tmp_path):
    assert main(["run", "--runs", "2", "--set", "pso.swarm_size=6", "--set", "pso.iterations=4", "--out", str(tmp_path)]) == 0
    png = tmp_path / "hits.png"
    assert main(["plot-convergence", str(tmp_path / "default.histories.jsonl"), "--key", "boundary_hits", "--out", str(png)]) == 0
