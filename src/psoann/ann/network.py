"""Feed-forward network: activations are per layer; weights are supplied as one flat vector."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from psoann.ann.activations import Activation, get_activation
from psoann.ann.codec import ParameterCodec
from psoann.config import ANNConfig


class FeedForwardNetwork:
    """``forward(params, X)`` maps (n, n_in) inputs to (n, n_out) outputs. Stateless in the weights."""

    def __init__(self, layer_sizes: Sequence[int], activations: Sequence[Callable]) -> None:
        self.codec = ParameterCodec(layer_sizes)
        if len(activations) != len(self.codec.layer_sizes) - 1:
            raise ValueError(
                f"need one activation per non-input layer ({len(self.codec.layer_sizes) - 1}), got {len(activations)}"
            )
        self.activations: tuple[Activation, ...] = tuple(activations)

    @property
    def layer_sizes(self) -> tuple[int, ...]:
        return self.codec.layer_sizes

    @property
    def n_parameters(self) -> int:
        return self.codec.n_parameters

    def forward(self, params: NDArray[np.float64], X: NDArray[np.float64]) -> NDArray[np.float64]:
        a = np.asarray(X, dtype=np.float64)
        if a.ndim != 2 or a.shape[1] != self.layer_sizes[0]:
            raise ValueError(f"expected X of shape (n, {self.layer_sizes[0]}), got {a.shape}")
        for layer, activation in zip(self.codec.decode(params), self.activations):
            a = activation(a @ layer.weights + layer.biases)
        return a


def build_network(config: ANNConfig, n_inputs: int) -> FeedForwardNetwork:
    """Layer sizes are ``(n_inputs, *hidden_layers, n_outputs)``; names are looked up in the registry."""
    sizes = (n_inputs, *config.hidden_layers, config.n_outputs)
    hidden = get_activation(config.hidden_activation)
    output = get_activation(config.output_activation)
    return FeedForwardNetwork(sizes, [hidden] * len(config.hidden_layers) + [output])
