import numpy as np
import pytest

from psoann.ann import FeedForwardNetwork, LayerParams, ParameterCodec, build_network, get_activation, register_activation
from psoann.config import ANNConfig


def test_default_network_has_49_parameters():  # D7: 4-8-1 -> (4*8+8) + (8*1+1)
    net = build_network(ANNConfig(), n_inputs=4)
    assert net.layer_sizes == (4, 8, 1) and net.n_parameters == 49


def test_codec_layout_weights_row_major_then_biases():
    codec = ParameterCodec([2, 3, 1])
    vec = np.arange(codec.n_parameters, dtype=float)
    first, second = codec.decode(vec)
    np.testing.assert_array_equal(first.weights, [[0, 1, 2], [3, 4, 5]])
    np.testing.assert_array_equal(first.biases, [6, 7, 8])
    np.testing.assert_array_equal(second.weights, [[9], [10], [11]])
    np.testing.assert_array_equal(second.biases, [12])


def test_codec_round_trip_and_validation(rng):
    codec = ParameterCodec([4, 5, 3, 1])
    vec = rng.normal(size=codec.n_parameters)
    np.testing.assert_array_equal(codec.encode(codec.decode(vec)), vec)
    with pytest.raises(ValueError):
        codec.decode(vec[:-1])
    with pytest.raises(ValueError):
        codec.encode([LayerParams(np.zeros((1, 1)), np.zeros(1))])
    with pytest.raises(ValueError):
        ParameterCodec([3])


def test_temperature_conversion_is_one_linear_neuron():
    net = FeedForwardNetwork([1, 1], [get_activation("identity")])
    out = net.forward(np.array([1.8, 32.0]), np.array([[0.0], [100.0], [-40.0]]))
    np.testing.assert_allclose(out.ravel(), [32.0, 212.0, -40.0])


def test_full_wave_rectifier_needs_two_relu_neurons():
    relu, ident = get_activation("relu"), get_activation("identity")
    net = FeedForwardNetwork([1, 2, 1], [relu, ident])
    # hidden weights (+1, -1), hidden biases 0, output weights (1, 1), output bias 0
    params = np.array([1.0, -1.0, 0.0, 0.0, 1.0, 1.0, 0.0])
    v = np.array([[-3.0], [2.0], [0.0], [-0.5]])
    np.testing.assert_allclose(net.forward(params, v).ravel(), np.abs(v.ravel()))


def test_rotation_is_two_linear_neurons():
    theta = np.pi / 2
    c, s = np.cos(theta), np.sin(theta)
    net = FeedForwardNetwork([2, 2], [get_activation("identity")])
    params = np.array([c, s, -s, c, 0.0, 0.0])  # weights (2x2 row-major), biases
    np.testing.assert_allclose(net.forward(params, np.array([[1.0, 0.0]])), [[0.0, 1.0]], atol=1e-12)


def test_forward_checks_shapes(rng):
    net = build_network(ANNConfig(), n_inputs=4)
    with pytest.raises(ValueError):
        net.forward(np.zeros(49), np.zeros((3, 5)))
    with pytest.raises(ValueError):
        net.forward(np.zeros(48), np.zeros((3, 4)))
    with pytest.raises(ValueError):
        FeedForwardNetwork([2, 2, 1], [get_activation("tanh")])


def test_logistic_is_stable_for_extreme_inputs():
    out = get_activation("logistic")(np.array([-1e6, 0.0, 1e6]))
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out, [0.0, 0.5, 1.0], atol=1e-200)


def test_register_activation_and_unknown_name():
    register_activation("test_square", np.square, overwrite=True)
    net = build_network(ANNConfig(hidden_activation="test_square"), n_inputs=2)
    assert net.activations[0] is np.square
    with pytest.raises(ValueError, match="unknown activation"):
        build_network(ANNConfig(hidden_activation="nope"), n_inputs=2)
    with pytest.raises(ValueError, match="already registered"):
        register_activation("tanh", np.tanh)
