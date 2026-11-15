"""Parameter sweeps described by a plain dict (see ``presets.py`` for the shipped ones).

    {
        "name": "informants",
        "overrides": {"n_runs": 10},                      # optional dotted-key overrides of the defaults
        "grid": {"pso.n_informants": [2, 3, 4, 6, 8]},    # full cartesian product of these dotted keys
        "fixed_budget": {"budget": 3000,                  # optional: swarm_size x iterations pairs
                         "swarm_sizes": [10, 20, 30]},    # with equal evaluations
        "points": [{"label": "global", "pso.informants": "global"},   # optional: explicit settings, each a
                   {"label": "k3", "pso.n_informants": 3}],           # dict of dotted overrides (+ a label)
    }

The grid, the fixed-budget pairs and the explicit points are crossed. Use ``points`` where the factors
are not a full grid (e.g. the "global" topology has no number of informants). Every point becomes an
ordinary experiment, so each result row still carries its full configuration.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from psoann.config import ExperimentConfig
from psoann.experiments.runner import History, Row, run_experiment

_ALLOWED_KEYS = {"name", "overrides", "grid", "fixed_budget", "points"}


@dataclass(frozen=True)
class SweepPoint:
    point_id: str
    params: dict[str, Any]
    config: ExperimentConfig


@dataclass(frozen=True)
class Sweep:
    name: str
    base: ExperimentConfig
    grid: dict[str, list[Any]] = field(default_factory=dict)
    swarm_sizes: tuple[int, ...] = ()
    budget: int | None = None
    points: tuple[dict[str, Any], ...] = ()  # explicit settings; optional "label" names the point

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Sweep":
        unknown = sorted(set(data) - _ALLOWED_KEYS)
        if unknown:
            raise ValueError(f"unknown sweep key(s) {unknown}; allowed: {sorted(_ALLOWED_KEYS)}")
        if "name" not in data:
            raise ValueError("a sweep needs a 'name'")
        base = ExperimentConfig().with_overrides(dict(data.get("overrides", {})))

        grid = {key: list(values) for key, values in data.get("grid", {}).items()}
        for key, values in grid.items():
            if not values:
                raise ValueError(f"grid key {key!r} has no values")
        fixed = dict(data.get("fixed_budget", {}))
        unknown_fixed = sorted(set(fixed) - {"budget", "swarm_sizes"})
        if unknown_fixed:
            raise ValueError(f"unknown fixed_budget key(s) {unknown_fixed}")
        if fixed and not ("budget" in fixed and fixed.get("swarm_sizes")):
            raise ValueError("fixed_budget needs both 'budget' and a non-empty 'swarm_sizes'")
        if fixed and any(key in grid for key in ("pso.swarm_size", "pso.iterations")):
            raise ValueError("fixed_budget already sets pso.swarm_size and pso.iterations; remove them from [grid]")
        points = [dict(point) for point in data.get("points", [])]
        if "points" in data and not points:
            raise ValueError("'points' needs at least one point")
        for point in points:
            overrides = {k: v for k, v in point.items() if k != "label"}
            if not overrides:
                raise ValueError("every point needs at least one override")
            if any(key in grid for key in overrides):
                raise ValueError(f"point keys {sorted(set(overrides) & set(grid))} are also in the grid")
        labels = [p["label"] for p in points if "label" in p]
        if len(set(labels)) != len(labels):
            raise ValueError("point labels must be unique")
        return cls(
            name=str(data["name"]),
            base=base,
            grid=grid,
            swarm_sizes=tuple(int(n) for n in fixed.get("swarm_sizes", ())),
            budget=int(fixed["budget"]) if fixed else None,
            points=tuple(points),
        )

    def expand(self) -> list[SweepPoint]:
        # Each axis is a list of (params, overrides, label) options; params name the point, overrides configure it.
        axes: list[list[tuple[dict[str, Any], dict[str, Any], str | None]]] = [
            [({key: v}, {key: v}, None) for v in values] for key, values in self.grid.items()
        ]
        if self.budget is not None:
            options = []
            for swarm in self.swarm_sizes:
                if swarm > self.budget:
                    raise ValueError(f"swarm size {swarm} exceeds the budget {self.budget}")
                options.append(({"pso.swarm_size": swarm}, {"pso.swarm_size": swarm, "pso.iterations": self.budget // swarm}, None))
            axes.append(options)
        if self.points:
            axes.append(
                [
                    (overrides, overrides, point.get("label"))
                    for point in self.points
                    for overrides in [{k: v for k, v in point.items() if k != "label"}]
                ]
            )
        points: list[SweepPoint] = []
        for combo in itertools.product(*axes) if axes else [()]:
            params: dict[str, Any] = {}
            overrides: dict[str, Any] = {}
            labels: list[str] = []
            for option_params, option_overrides, label in combo:
                params.update(option_params)
                overrides.update(option_overrides)
                labels.append(label if label is not None else "__".join(f"{k.split('.')[-1]}={_label(v)}" for k, v in option_params.items()))
            point_id = "__".join(labels) or "base"
            config = self.base.with_overrides({**overrides, "name": f"{self.name}/{point_id}"})
            points.append(SweepPoint(point_id, params, config))
        return points


def _label(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "-".join(str(v) for v in value) or "none"
    return str(value)


def run_sweep(
    sweep: Sweep,
    progress: Callable[[int, int, SweepPoint], None] | None = None,
) -> tuple[list[Row], list[History]]:
    rows: list[Row] = []
    histories: list[History] = []
    points = sweep.expand()
    for index, point in enumerate(points, start=1):
        result = run_experiment(point.config)
        for row in result.rows:
            row["sweep"] = sweep.name
            row["point"] = point.point_id
        rows.extend(result.rows)
        histories.extend(
            {**h, "sweep": sweep.name, "point": point.point_id} for h in result.histories
        )
        if progress is not None:
            progress(index, len(points), point)
    return rows, histories
