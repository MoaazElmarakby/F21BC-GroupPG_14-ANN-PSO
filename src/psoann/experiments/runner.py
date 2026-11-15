"""Run experiments reproducibly.

Seeding (design doc, section 3):

* run ``i`` uses ``SeedSequence(base_seed, spawn_key=(i,))`` -- exactly child ``i`` of
  ``SeedSequence(base_seed).spawn(n_runs)`` -- so it does not depend on ``n_runs``;
* each run spawns two independent streams from that seed: one for the split, one for PSO;
* with ``split_policy = "fixed"`` the single split uses ``SeedSequence([base_seed, 1])``, a stream
  that cannot coincide with any run's.

Every result row carries the full configuration (flattened columns plus ``config_json``), so any row
can be regenerated with ``run_single(config, run_index)``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from psoann.config import ExperimentConfig
from psoann.data.loader import load_csv
from psoann.data.split import DataSplit, make_split
from psoann.pso.types import PSOResult
from psoann.training.fitness import evaluate_parameters
from psoann.training.trainer import Optimizer, PSOTrainer

Row = dict[str, Any]
History = dict[str, Any]


@dataclass
class ExperimentResult:
    config: ExperimentConfig
    rows: list[Row] = field(default_factory=list)
    histories: list[History] = field(default_factory=list)


def run_seed(base_seed: int, run_index: int) -> np.random.SeedSequence:
    return np.random.SeedSequence(base_seed, spawn_key=(run_index,))


def fixed_split_seed(base_seed: int) -> np.random.SeedSequence:
    return np.random.SeedSequence([base_seed, 1])


def split_for_run(
    config: ExperimentConfig, run_index: int, X: NDArray[np.float64], y: NDArray[np.int64]
) -> DataSplit:
    if config.data.split_policy == "fixed":
        rng = np.random.default_rng(fixed_split_seed(config.base_seed))
    else:
        split_seed, _ = run_seed(config.base_seed, run_index).spawn(2)
        rng = np.random.default_rng(split_seed)
    return make_split(X, y, config.data, rng)


def run_single(
    config: ExperimentConfig,
    run_index: int,
    dataset: tuple[NDArray[np.float64], NDArray[np.int64]] | None = None,
    optimizer: Optimizer | None = None,
) -> tuple[Row, History]:
    """Run number ``run_index`` of ``config``. Deterministic apart from the ``runtime_sec`` column."""
    X, y = dataset if dataset is not None else load_csv(config.data.path)
    split = split_for_run(config, run_index, X, y)
    _, pso_seed = run_seed(config.base_seed, run_index).spawn(2)

    trainer = PSOTrainer(config.ann, config.pso, config.training, optimizer=optimizer)
    started = time.perf_counter()
    trained = trainer.train(split.X_train, split.y_train, np.random.default_rng(pso_seed))
    runtime = time.perf_counter() - started

    train_metrics = evaluate_parameters(trained.network, trained.parameters, split.X_train, split.y_train)
    test_metrics = evaluate_parameters(trained.network, trained.parameters, split.X_test, split.y_test)
    evals_to_target = _evals_to_target(trained.pso, config.target_fitness)
    row: Row = {
        "experiment": config.name,
        "config_hash": config.config_hash(),
        "run": run_index,
        "base_seed": config.base_seed,
        "n_parameters": trained.network.n_parameters,
        "evaluations": trained.pso.evaluations,
        "best_fitness": trained.fitness,
        "hidden_activations": trained.network.describe_activations(trained.parameters),
        "train_accuracy": train_metrics["accuracy"],
        "test_accuracy": test_metrics["accuracy"],
        "train_precision": train_metrics["precision"],
        "test_precision": test_metrics["precision"],
        "train_recall": train_metrics["recall"],
        "test_recall": test_metrics["recall"],
        "train_f1": train_metrics["f1"],
        "test_f1": test_metrics["f1"],
        "train_mse": train_metrics["mse"],
        "test_mse": test_metrics["mse"],
        "evals_to_final_best": trained.pso.evaluations_to_final_best(),
        "evals_to_target": evals_to_target,
        "reached_target": None if evals_to_target is None and config.target_fitness is None else int(evals_to_target is not None),
        "runtime_sec": runtime,
    }
    row.update({f"cfg.{key}": value for key, value in config.flatten().items()})
    row["config_json"] = config.to_json()

    history: History = {
        "experiment": config.name,
        "config_hash": config.config_hash(),
        "run": run_index,
        "history": {key: values.tolist() for key, values in trained.pso.history.items()},
    }
    return row, history


def run_experiment(
    config: ExperimentConfig,
    progress: Callable[[int, int], None] | None = None,
    optimizer: Optimizer | None = None,
) -> ExperimentResult:
    dataset = load_csv(config.data.path)
    result = ExperimentResult(config=config)
    for run_index in range(config.n_runs):
        row, history = run_single(config, run_index, dataset=dataset, optimizer=optimizer)
        result.rows.append(row)
        result.histories.append(history)
        if progress is not None:
            progress(run_index + 1, config.n_runs)
    return result


def _evals_to_target(result: PSOResult, target: float | None) -> int | None:
    return None if target is None else result.evaluations_to_reach(target)
