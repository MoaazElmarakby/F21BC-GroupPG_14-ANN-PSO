"""Experiment configuration: frozen, validated dataclasses (leaf module: imports nothing from psoann).

* Unknown keys are rejected so a typo in a config fails loudly instead of silently using a default.
* Numeric ranges are validated here. *Names* (activation, informant topology, ...) are validated when
  the component is built, because the registries live in the packages that depend on this module.
* ``ExperimentConfig.flatten()`` gives the dotted, CSV-friendly view; ``to_json()`` the exact one.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field, fields
from typing import Any, Mapping

_SPLIT_POLICIES = ("fixed", "per_run")
_INITIAL_VELOCITIES = ("half_diff", "zero")
_OPTIMIZERS = ("pso", "random_search")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _as_float(obj: Any, name: str) -> float:
    value = getattr(obj, name)
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{name} must be a number, got {value!r}")
    value = float(value)
    object.__setattr__(obj, name, value)
    return value


def _build(cls: type, data: Mapping[str, Any], nested: Mapping[str, type] | None = None) -> Any:
    _require(isinstance(data, Mapping), f"{cls.__name__} expects a table/dict, got {type(data).__name__}")
    allowed = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - allowed)
    _require(not unknown, f"unknown key(s) {unknown} for {cls.__name__}; allowed: {sorted(allowed)}")
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        if nested and key in nested and isinstance(value, Mapping):
            kwargs[key] = nested[key].from_dict(value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


@dataclass(frozen=True)
class DataConfig:
    path: str = "data/raw/banknote.csv"
    train_fraction: float = 0.7
    standardise: bool = True  # D6: z-score with *training* statistics only
    split_policy: str = "fixed"  # "fixed": one split for all runs; "per_run": new split each run

    def __post_init__(self) -> None:
        _require(isinstance(self.path, str) and self.path, "data.path must be a non-empty string")
        f = _as_float(self, "train_fraction")
        _require(0.0 < f < 1.0, f"data.train_fraction must be in (0, 1), got {f}")
        _require(isinstance(self.standardise, bool), "data.standardise must be a bool")
        _require(self.split_policy in _SPLIT_POLICIES, f"data.split_policy must be one of {_SPLIT_POLICIES}")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DataConfig":
        return _build(cls, data)


@dataclass(frozen=True)
class ANNConfig:
    hidden_layers: tuple[int, ...] = (8,)  # D7: 4-8-1 by default
    hidden_activation: str = "tanh"
    output_activation: str = "logistic"
    n_outputs: int = 1  # 1 = one logistic unit (predict 1 if >= 0.5); >1 = argmax
    # Extension: one extra gene per hidden layer selects that layer's activation from ``activation_choices``
    # (``hidden_activation`` is then ignored). The output layer keeps ``output_activation``.
    evolve_activations: bool = False
    activation_choices: tuple[str, ...] = ("logistic", "tanh", "relu")

    def __post_init__(self) -> None:
        layers = self.hidden_layers
        _require(isinstance(layers, (list, tuple)), "ann.hidden_layers must be a list of layer widths")
        layers = tuple(layers)
        _require(all(_is_int(n) and n >= 1 for n in layers), f"ann.hidden_layers must be positive ints, got {layers}")
        object.__setattr__(self, "hidden_layers", layers)
        _require(isinstance(self.hidden_activation, str) and self.hidden_activation, "ann.hidden_activation must be a name")
        _require(isinstance(self.output_activation, str) and self.output_activation, "ann.output_activation must be a name")
        _require(_is_int(self.n_outputs) and self.n_outputs >= 1, "ann.n_outputs must be an int >= 1")
        _require(isinstance(self.evolve_activations, bool), "ann.evolve_activations must be a bool")
        choices = self.activation_choices
        _require(isinstance(choices, (list, tuple)) and len(choices) >= 1, "ann.activation_choices must be a non-empty list of names")
        _require(all(isinstance(c, str) and c for c in choices), "ann.activation_choices must contain activation names")
        object.__setattr__(self, "activation_choices", tuple(choices))
        _require(not (self.evolve_activations and not layers), "ann.evolve_activations needs at least one hidden layer")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ANNConfig":
        return _build(cls, data)


@dataclass(frozen=True)
class PSOConfig:
    swarm_size: int = 30
    iterations: int = 100
    # Algorithm 39 coefficients (D3: constriction-equivalent values, delta = 0 -> informants only)
    alpha: float = 0.7298  # proportion of velocity retained
    beta: float = 1.4962  # proportion of personal best mixed in
    gamma: float = 1.4962  # proportion of informants' best mixed in
    delta: float = 0.0  # proportion of global best mixed in
    epsilon: float = 1.0  # jump size
    informants: str = "random"  # D4: topology name from the informant registry
    n_informants: int = 3  # random topology: number of *other* particles per particle (self is always added)
    regenerate_informants: bool = False  # redraw the random informants every iteration
    bound: float = 2.0  # D5: every weight/bias is searched in [-bound, +bound]
    bound_handling: str = "clamp"
    max_velocity_fraction: float | None = None  # optional |v_d| <= fraction * (upper_d - lower_d)
    initial_velocity: str = "half_diff"

    def __post_init__(self) -> None:
        _require(_is_int(self.swarm_size) and self.swarm_size >= 2, "pso.swarm_size must be an int >= 2")
        _require(_is_int(self.iterations) and self.iterations >= 1, "pso.iterations must be an int >= 1")
        for name in ("alpha", "beta", "gamma", "delta"):
            _require(_as_float(self, name) >= 0.0, f"pso.{name} must be >= 0")
        _require(_as_float(self, "epsilon") > 0.0, "pso.epsilon must be > 0")
        _require(_as_float(self, "bound") > 0.0, "pso.bound must be > 0")
        _require(isinstance(self.informants, str) and self.informants, "pso.informants must be a name")
        _require(_is_int(self.n_informants) and self.n_informants >= 1, "pso.n_informants must be an int >= 1")
        _require(isinstance(self.regenerate_informants, bool), "pso.regenerate_informants must be a bool")
        _require(isinstance(self.bound_handling, str) and self.bound_handling, "pso.bound_handling must be a name")
        if self.max_velocity_fraction is not None:
            _require(_as_float(self, "max_velocity_fraction") > 0.0, "pso.max_velocity_fraction must be > 0 or unset")
        _require(self.initial_velocity in _INITIAL_VELOCITIES, f"pso.initial_velocity must be one of {_INITIAL_VELOCITIES}")

    @property
    def budget(self) -> int:
        """Total fitness evaluations: exactly ``swarm_size * iterations``."""
        return self.swarm_size * self.iterations

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PSOConfig":
        return _build(cls, data)


@dataclass(frozen=True)
class TrainingConfig:
    fitness: str = "accuracy"  # D2: training accuracy by default
    optimizer: str = "pso"  # "random_search" is the equal-budget control (study Q0)

    def __post_init__(self) -> None:
        _require(isinstance(self.fitness, str) and self.fitness, "training.fitness must be a metric name")
        _require(self.optimizer in _OPTIMIZERS, f"training.optimizer must be one of {_OPTIMIZERS}")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrainingConfig":
        return _build(cls, data)


_SECTIONS = {"data": DataConfig, "ann": ANNConfig, "pso": PSOConfig, "training": TrainingConfig}


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "experiment"
    base_seed: int = 0
    n_runs: int = 10
    target_fitness: float | None = None  # if set, results report evaluations needed to first reach it
    data: DataConfig = field(default_factory=DataConfig)
    ann: ANNConfig = field(default_factory=ANNConfig)
    pso: PSOConfig = field(default_factory=PSOConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self) -> None:
        _require(isinstance(self.name, str) and self.name, "name must be a non-empty string")
        _require(_is_int(self.base_seed) and self.base_seed >= 0, "base_seed must be an int >= 0")
        _require(_is_int(self.n_runs) and self.n_runs >= 1, "n_runs must be an int >= 1")
        if self.target_fitness is not None:
            _as_float(self, "target_fitness")
        for key, cls in _SECTIONS.items():
            _require(isinstance(getattr(self, key), cls), f"{key} must be a {cls.__name__}")

    # ---- construction -------------------------------------------------------------------------
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentConfig":
        return _build(cls, data, nested=_SECTIONS)

    @classmethod
    def from_json(cls, text: str) -> "ExperimentConfig":
        return cls.from_dict(json.loads(text))

    def with_overrides(self, overrides: Mapping[str, Any]) -> "ExperimentConfig":
        """Return a copy with dotted keys replaced, e.g. ``{"pso.alpha": 0.5, "n_runs": 3}``."""
        data = self.to_dict()
        for key, value in overrides.items():
            parts = key.split(".")
            node = data
            for part in parts[:-1]:
                _require(isinstance(node.get(part), dict), f"unknown config key {key!r}")
                node = node[part]
            _require(parts[-1] in node and not isinstance(node[parts[-1]], dict), f"unknown config key {key!r}")
            node[parts[-1]] = value
        return ExperimentConfig.from_dict(data)

    # ---- serialisation ------------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def config_hash(self) -> str:
        return hashlib.sha1(self.to_json().encode("utf-8")).hexdigest()[:10]

    def flatten(self) -> dict[str, Any]:
        """Dotted keys with scalar values (lists become ``"8-4"``, an empty list ``"none"``), ready to be CSV columns."""
        out: dict[str, Any] = {}

        def walk(prefix: str, node: Mapping[str, Any]) -> None:
            for key, value in node.items():
                full = f"{prefix}{key}"
                if isinstance(value, Mapping):
                    walk(full + ".", value)
                elif isinstance(value, (list, tuple)):
                    out[full] = "-".join(str(v) for v in value) or "none"
                else:
                    out[full] = value

        walk("", self.to_dict())
        return out
