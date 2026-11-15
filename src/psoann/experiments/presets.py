"""Built-in sweeps, selected by name: ``psoann sweep topology`` (or ``psoann sweep all``).

Each sweep starts from the default ``ExperimentConfig`` and crosses the dotted keys in ``grid``.
"""

from __future__ import annotations

from psoann.experiments.sweep import Sweep

SWEEPS: dict[str, Sweep] = {
    sweep.name: sweep
    for sweep in (
        # Architecture study (D7): depth, width and hidden activation.
        Sweep.from_dict(
            {
                "name": "architecture",
                "grid": {
                    "ann.hidden_layers": [[2], [4], [8], [16], [8, 4]],
                    "ann.hidden_activation": ["tanh", "relu", "logistic"],
                },
            }
        ),
        # Weight-bound study (D5) and boundary handling.
        Sweep.from_dict(
            {
                "name": "bounds",
                "grid": {"pso.bound": [0.5, 1.0, 2.0, 4.0, 8.0], "pso.bound_handling": ["clamp", "reflect"]},
            }
        ),
        # Coefficient study (D3): inertia, personal, informant and global pulls.
        Sweep.from_dict(
            {
                "name": "coefficients",
                "grid": {
                    "pso.alpha": [0.4, 0.7298, 0.9],
                    "pso.beta": [0.5, 1.4962, 2.0],
                    "pso.gamma": [0.5, 1.4962, 2.0],
                    "pso.delta": [0.0, 0.5],
                },
            }
        ),
        # Informant study (D4): K random informants per particle.
        Sweep.from_dict({"name": "informants", "grid": {"pso.n_informants": [2, 3, 4, 6, 8]}}),
        # Equal evaluation budget, different swarm size / iteration trade-offs.
        Sweep.from_dict(
            {"name": "swarm_vs_iterations", "fixed_budget": {"budget": 3000, "swarm_sizes": [10, 20, 30, 60, 100]}}
        ),
        # Compare informant topologies at equal budget.
        Sweep.from_dict({"name": "topology", "grid": {"pso.informants": ["random", "ring", "global"]}}),
    )
}


def get_sweep(name: str) -> Sweep:
    try:
        return SWEEPS[name]
    except KeyError:
        raise KeyError(f"unknown sweep {name!r}; available: {', '.join(sorted(SWEEPS))}") from None
