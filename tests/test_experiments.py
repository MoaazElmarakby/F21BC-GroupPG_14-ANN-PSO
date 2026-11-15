import json

import numpy as np
import pytest

from psoann.config import ExperimentConfig
from psoann.data import load_csv
from psoann.experiments import (
    Sweep,
    read_csv,
    read_jsonl,
    run_experiment,
    run_seed,
    run_single,
    run_sweep,
    write_results,
)
from psoann.experiments.runner import split_for_run
from psoann.pso import PSOResult
from psoann.training import PSOTrainer
from tests.conftest import REPO

METRIC_COLUMNS = ["best_fitness", "train_accuracy", "test_accuracy", "train_mse", "test_mse", "evaluations"]


def metrics(row):
    return {k: row[k] for k in METRIC_COLUMNS}


def test_experiment_produces_one_complete_row_per_run(tiny_config):
    result = run_experiment(tiny_config)
    assert len(result.rows) == len(result.histories) == 3
    row = result.rows[0]
    assert row["n_parameters"] == 49 and row["evaluations"] == 8 * 6
    assert 0.0 <= row["test_accuracy"] <= 1.0
    assert row["cfg.pso.swarm_size"] == 8 and row["cfg.ann.hidden_layers"] == "8"
    assert ExperimentConfig.from_json(row["config_json"]) == tiny_config  # row carries the full config
    assert [r["run"] for r in result.rows] == [0, 1, 2]
    hist = result.histories[0]
    assert hist["run"] == 0 and len(hist["history"]["best"]) == 6


def test_any_row_can_be_regenerated_from_config_and_run_index(tiny_config):
    result = run_experiment(tiny_config)
    for i in (0, 2):
        regenerated, _ = run_single(ExperimentConfig.from_json(result.rows[i]["config_json"]), i)
        assert metrics(regenerated) == metrics(result.rows[i])


def test_run_results_do_not_depend_on_n_runs(tiny_config):
    few = run_experiment(tiny_config.with_overrides({"n_runs": 2}))
    many = run_experiment(tiny_config.with_overrides({"n_runs": 4}))
    for i in range(2):
        assert metrics(few.rows[i]) == metrics(many.rows[i])


def test_run_seed_is_child_i_of_spawn():
    children = np.random.SeedSequence(42).spawn(5)
    for i, child in enumerate(children):
        np.testing.assert_array_equal(run_seed(42, i).generate_state(4), child.generate_state(4))


def test_runs_differ_from_each_other_and_seeds_change_results(tiny_config):
    rows = run_experiment(tiny_config).rows
    assert len({row["best_fitness"] for row in rows} | {row["train_mse"] for row in rows}) > 1
    other = run_experiment(tiny_config.with_overrides({"base_seed": 99})).rows
    assert [r["train_mse"] for r in rows] != [r["train_mse"] for r in other]


def test_fixed_split_is_shared_and_per_run_split_is_not(tiny_config):
    X, y = load_csv(tiny_config.data.path)
    fixed = [split_for_run(tiny_config, i, X, y).test_index for i in range(3)]
    assert all(np.array_equal(fixed[0], other) for other in fixed)
    per_run_cfg = tiny_config.with_overrides({"data.split_policy": "per_run"})
    varying = [split_for_run(per_run_cfg, i, X, y).test_index for i in range(3)]
    assert not np.array_equal(varying[0], varying[1])
    for split in (split_for_run(per_run_cfg, 0, X, y), split_for_run(tiny_config, 0, X, y)):
        assert not set(split.train_index) & set(split.test_index)


def test_training_improves_on_a_random_network(tiny_config):
    cfg = tiny_config.with_overrides({"pso.swarm_size": 30, "pso.iterations": 60, "n_runs": 1})
    row = run_experiment(cfg).rows[0]
    assert row["train_accuracy"] > 0.9 and row["test_accuracy"] > 0.9


def test_target_fitness_reports_evaluations_to_target(tiny_config):
    cfg = tiny_config.with_overrides({"target_fitness": 0.0, "n_runs": 1})
    assert run_experiment(cfg).rows[0]["evals_to_target"] == 8  # accuracy >= 0 is true after the first iteration
    assert run_experiment(tiny_config.with_overrides({"n_runs": 1})).rows[0]["evals_to_target"] is None


def test_custom_optimizer_can_replace_pso(tiny_config):
    class Constant:
        def optimise(self, fitness, space, rng):
            position = np.zeros(space.dim)
            return PSOResult(position, fitness(position), 1, {"evaluations": np.array([1.0]), "best": np.array([0.0])})

    row, _ = run_single(tiny_config, 0, optimizer=Constant())
    assert row["evaluations"] == 1


def test_trainer_rejects_labels_that_do_not_fit_the_output_layer(tiny_config, rng):
    trainer = PSOTrainer(tiny_config.ann, tiny_config.pso, tiny_config.training)
    with pytest.raises(ValueError, match="labels must be in"):
        trainer.train(rng.normal(size=(6, 4)), np.array([0, 1, 2, 0, 1, 2]), rng)


def test_other_fitness_metrics_and_architectures_run(tiny_config):
    for overrides in (
        {"training.fitness": "neg_mse"},
        {"training.fitness": "neg_cross_entropy"},
        {"ann.hidden_layers": [6, 3], "ann.hidden_activation": "relu"},
        {"ann.hidden_layers": []},
    ):
        row = run_experiment(tiny_config.with_overrides({**overrides, "n_runs": 1})).rows[0]
        assert 0.0 <= row["test_accuracy"] <= 1.0


def test_multi_output_uses_argmax(tiny_config):
    row = run_experiment(tiny_config.with_overrides({"ann.n_outputs": 2, "n_runs": 1})).rows[0]
    assert row["n_parameters"] == (4 * 8 + 8) + (8 * 2 + 2)
    assert 0.0 <= row["test_accuracy"] <= 1.0


def test_unknown_names_fail_loudly(tiny_config):
    for key, value, message in (
        ("ann.hidden_activation", "nope", "unknown activation"),
        ("training.fitness", "nope", "unknown fitness metric"),
        ("pso.informants", "nope", "unknown informant strategy"),
        ("pso.bound_handling", "nope", "unknown boundary handler"),
    ):
        with pytest.raises(ValueError, match=message):
            run_single(tiny_config.with_overrides({key: value}), 0)


# ---- result files -----------------------------------------------------------------------------
def test_results_round_trip_through_csv_and_jsonl(tiny_config, tmp_path):
    result = run_experiment(tiny_config)
    csv_path, jsonl_path = write_results(tmp_path / "out", "tiny", result.rows, result.histories)
    rows = read_csv(csv_path)
    assert len(rows) == 3 and float(rows[0]["test_accuracy"]) == result.rows[0]["test_accuracy"]
    assert json.loads(rows[1]["config_json"])["pso"]["swarm_size"] == 8
    histories = read_jsonl(jsonl_path)
    assert len(histories) == 3 and histories[0]["history"]["best"] == result.histories[0]["history"]["best"]


# ---- sweeps ------------------------------------------------------------------------------------
def test_grid_expansion_is_the_cartesian_product():
    sweep = Sweep.from_dict({"name": "s", "grid": {"pso.alpha": [0.4, 0.9], "pso.n_informants": [2, 3, 4]}})
    points = sweep.expand()
    assert len(points) == 6
    assert {(p.config.pso.alpha, p.config.pso.n_informants) for p in points} == {
        (a, k) for a in (0.4, 0.9) for k in (2, 3, 4)
    }
    assert len({p.point_id for p in points}) == 6 and all(p.config.name.startswith("s/") for p in points)


def test_fixed_budget_pairs_keep_the_evaluation_budget_equal():
    sweep = Sweep.from_dict({"name": "b", "fixed_budget": {"budget": 3000, "swarm_sizes": [10, 30, 100]}})
    assert [(p.config.pso.swarm_size, p.config.pso.iterations) for p in sweep.expand()] == [(10, 300), (30, 100), (100, 30)]
    assert all(p.config.pso.budget == 3000 for p in sweep.expand())


@pytest.mark.parametrize(
    "bad",
    [
        {"name": "x", "bogus": 1},
        {"grid": {"pso.alpha": [1]}},
        {"name": "x", "grid": {"pso.alpha": []}},
        {"name": "x", "grid": {"pso.nonexistent": [1]}},
        {"name": "x", "fixed_budget": {"budget": 100}},
        {"name": "x", "fixed_budget": {"budget": 100, "swarm_sizes": [10]}, "grid": {"pso.swarm_size": [5]}},
    ],
)
def test_bad_sweep_definitions_are_rejected(bad):
    with pytest.raises(ValueError):
        Sweep.from_dict(bad).expand()


@pytest.mark.parametrize("path", sorted((REPO / "configs" / "sweeps").glob("*.toml")), ids=lambda p: p.stem)
def test_every_shipped_sweep_parses_and_expands_to_valid_configs(path):
    points = Sweep.from_toml(path).expand()
    assert points and len({p.point_id for p in points}) == len(points)


def test_running_a_small_sweep_labels_rows_and_histories(tiny_config):
    sweep = Sweep.from_dict({"name": "mini", "grid": {"pso.n_informants": [2, 3]}})
    sweep = Sweep(sweep.name, tiny_config.with_overrides({"n_runs": 2}), sweep.grid)
    rows, histories = run_sweep(sweep)
    assert len(rows) == len(histories) == 4
    assert {r["point"] for r in rows} == {"n_informants=2", "n_informants=3"}
    assert {r["cfg.pso.n_informants"] for r in rows} == {2, 3} and all(r["sweep"] == "mini" for r in rows)
