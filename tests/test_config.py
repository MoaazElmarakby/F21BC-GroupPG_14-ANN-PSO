import pytest

from psoann.config import ANNConfig, DataConfig, ExperimentConfig, PSOConfig


def test_defaults_are_the_documented_ones():
    cfg = ExperimentConfig()
    assert (cfg.pso.alpha, cfg.pso.beta, cfg.pso.gamma, cfg.pso.delta) == (0.7298, 1.4962, 1.4962, 0.0)
    assert cfg.pso.n_informants == 3 and cfg.pso.informants == "random"
    assert cfg.pso.bound == 2.0 and cfg.pso.bound_handling == "clamp"
    assert cfg.data.standardise and cfg.data.split_policy == "fixed"
    assert cfg.ann.hidden_layers == (8,) and cfg.ann.hidden_activation == "tanh"
    assert cfg.training.fitness == "accuracy"


def test_budget_is_swarm_times_iterations():
    assert PSOConfig(swarm_size=20, iterations=50).budget == 1000


def test_unknown_keys_are_rejected():
    with pytest.raises(ValueError, match="unknown key"):
        PSOConfig.from_dict({"swarm_sise": 10})
    with pytest.raises(ValueError, match="unknown key"):
        ExperimentConfig.from_dict({"pso": {"alpha": 0.5, "typo": 1}})
    with pytest.raises(ValueError, match="unknown key"):
        ExperimentConfig.from_dict({"nonsense": 1})


@pytest.mark.parametrize(
    "bad",
    [
        lambda: PSOConfig(swarm_size=1),
        lambda: PSOConfig(iterations=0),
        lambda: PSOConfig(alpha=-0.1),
        lambda: PSOConfig(epsilon=0),
        lambda: PSOConfig(bound=0),
        lambda: PSOConfig(n_informants=0),
        lambda: PSOConfig(max_velocity_fraction=-1),
        lambda: PSOConfig(initial_velocity="random"),
        lambda: DataConfig(train_fraction=1.0),
        lambda: DataConfig(split_policy="sometimes"),
        lambda: ANNConfig(hidden_layers=(0,)),
        lambda: ANNConfig(n_outputs=0),
        lambda: ExperimentConfig(n_runs=0),
        lambda: ExperimentConfig(base_seed=-1),
    ],
)
def test_invalid_values_are_rejected(bad):
    with pytest.raises(ValueError):
        bad()


def test_lists_become_tuples_and_ints_become_floats():
    assert ANNConfig(hidden_layers=[8, 4]).hidden_layers == (8, 4)
    assert isinstance(PSOConfig(delta=0).delta, float)


def test_json_round_trip_and_hash(tiny_config):
    again = ExperimentConfig.from_json(tiny_config.to_json())
    assert again == tiny_config and again.config_hash() == tiny_config.config_hash()
    assert tiny_config.with_overrides({"pso.alpha": 0.5}).config_hash() != tiny_config.config_hash()


def test_with_overrides_and_unknown_override_keys(tiny_config):
    changed = tiny_config.with_overrides({"pso.alpha": 0.5, "ann.hidden_layers": [4, 4], "n_runs": 7})
    assert (changed.pso.alpha, changed.ann.hidden_layers, changed.n_runs) == (0.5, (4, 4), 7)
    assert tiny_config.pso.alpha == 0.7298  # original untouched (frozen)
    with pytest.raises(ValueError, match="unknown config key"):
        tiny_config.with_overrides({"pso.alfa": 1})
    with pytest.raises(ValueError, match="unknown config key"):
        tiny_config.with_overrides({"pso": 1})


def test_flatten_gives_scalar_dotted_keys():
    flat = ExperimentConfig(ann=ANNConfig(hidden_layers=(8, 4))).flatten()
    assert flat["ann.hidden_layers"] == "8-4" and flat["pso.alpha"] == 0.7298 and flat["n_runs"] == 10
    assert all(not isinstance(v, (list, tuple, dict)) for v in flat.values())
