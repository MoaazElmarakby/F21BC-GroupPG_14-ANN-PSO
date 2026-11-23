"""Built-in studies, selected by name: ``psoann sweep architecture`` (or ``psoann sweep all``).

One preset per question of the study plan (docs/STUDIES.md). Every study uses the same fixed stratified
split, 30 runs per setting with the same seeds across settings (paired comparisons), and a budget of
20 particles x 100 iterations = 2000 fitness evaluations unless the study itself varies the budget.
``target_fitness`` is the training accuracy used for "evaluations to target" (convergence speed), because
final accuracy on this dataset sits near the ceiling; change it after a pilot if it is reached by every
setting or by none.

    q0  random_search        equal-budget random-search control
    Q1  architecture         width x depth (+ no-hidden-layer baseline)
    Q2  swarm_vs_iterations  how to split a fixed budget (500 and 2000 evaluations)
    Q3  coefficients         social / cognitive balance (beta + gamma + delta = 4)
    Q4  informants           strategy x number of informants
    Q5  bounds               boundary handling rule
    Q6  activations          fixed activations against evolved activation genes
    Q7  fitness              error rate (accuracy), MSE, cross-entropy as the fitness
    Q8  schedule             fixed coefficients against time-varying ones (inertia, TVAC)
    --  pilot_bound, pilot_velocity   hidden hyperparameters, judged on training fitness only
"""

from __future__ import annotations

from typing import Any

from psoann.experiments.sweep import Sweep

STUDY_BASE: dict[str, Any] = {
    "n_runs": 30,
    "pso.swarm_size": 20,
    "pso.iterations": 100,
    "target_fitness": 0.98,
}


def _study(name: str, **spec: Any) -> Sweep:
    overrides = {**STUDY_BASE, **spec.pop("overrides", {})}
    return Sweep.from_dict({"name": name, "overrides": overrides, **spec})


def _architecture_points() -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = [{"label": "linear", "ann.hidden_layers": []}]
    for depth in (1, 2, 3):
        for width in (2, 4, 8, 16):
            points.append({"label": f"{depth}x{width}", "ann.hidden_layers": [width] * depth})
    return points


def _budget_points() -> list[dict[str, Any]]:
    return [
        {"label": f"b{budget}-s{swarm}", "pso.swarm_size": swarm, "pso.iterations": budget // swarm}
        for budget in (500, 2000)
        for swarm in (5, 10, 20, 25, 50, 100)
    ]


def _coefficient_points() -> list[dict[str, Any]]:
    """Every (beta, gamma, delta) of whole numbers summing to 4: the cognitive/social/global balance."""
    return [
        {"label": f"b{b}-g{g}-d{4 - b - g}", "pso.beta": float(b), "pso.gamma": float(g), "pso.delta": float(4 - b - g)}
        for b in range(5)
        for g in range(5 - b)
    ]


def _informant_points() -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = [{"label": "global", "pso.informants": "global"}]
    for k in (1, 2, 3, 5, 10):
        points.append({"label": f"random-k{k}", "pso.informants": "random", "pso.n_informants": k, "pso.regenerate_informants": False})
    for k in (1, 2, 3, 5, 10):
        points.append({"label": f"regen-k{k}", "pso.informants": "random", "pso.n_informants": k, "pso.regenerate_informants": True})
    for k in (1, 2, 3, 5, 10):
        points.append({"label": f"nearest-k{k}", "pso.informants": "nearest", "pso.n_informants": k})
    return points


SWEEPS: dict[str, Sweep] = {
    sweep.name: sweep
    for sweep in (
        _study(
            "random_search",
            points=[
                {"label": "pso", "training.optimizer": "pso"},
                {"label": "random_search", "training.optimizer": "random_search"},
            ],
        ),
        _study("architecture", points=_architecture_points()),
        _study("swarm_vs_iterations", points=_budget_points()),
        _study("coefficients", points=_coefficient_points()),
        _study("informants", points=_informant_points()),
        _study("bounds", grid={"pso.bound_handling": ["clamp", "reject", "reflect", "wrap", "random", "none"]}),
        _study(
            "activations",
            overrides={"ann.hidden_layers": [8, 8]},
            points=[
                {"label": "logistic", "ann.hidden_activation": "logistic"},
                {"label": "tanh", "ann.hidden_activation": "tanh"},
                {"label": "relu", "ann.hidden_activation": "relu"},
                {"label": "leaky_relu", "ann.hidden_activation": "leaky_relu"},
                {"label": "evolved", "ann.evolve_activations": True},
            ],
        ),
        _study("fitness", grid={"training.fitness": ["accuracy", "neg_mse", "neg_cross_entropy"]}),
        _study("schedule", grid={"pso.schedule": ["constant", "linear_inertia", "tvac"]}),
        # Pilots for hidden hyperparameters: choose from training fitness only (never the test set).
        _study("pilot_bound", overrides={"n_runs": 10}, grid={"pso.bound": [0.5, 1.0, 2.0, 4.0, 8.0]}),
        _study("pilot_velocity", overrides={"n_runs": 10}, grid={"pso.initial_velocity": ["half_diff", "zero"]}),
    )
}


def get_sweep(name: str) -> Sweep:
    try:
        return SWEEPS[name]
    except KeyError:
        raise KeyError(f"unknown sweep {name!r}; available: {', '.join(sorted(SWEEPS))}") from None
