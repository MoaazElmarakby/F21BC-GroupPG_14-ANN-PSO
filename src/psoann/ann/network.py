"""Feed-forward network: activations are per layer; weights are supplied as one flat vector.

Two variants share one ``forward``:

* ``FeedForwardNetwork`` - fixed activations; the vector holds weights and biases only;
* ``EvolvedActivationNetwork`` - the vector additionally holds one *gene* per hidden layer, in
  ``[0, K)``, whose integer part selects that layer's activation from ``K`` candidates. The particle
  stays a fixed-length vector and the architecture stays fixed: only the choice of activation is searched.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from psoann.ann.activations import Activation, get_activation
from psoann.ann.codec import ParameterCodec
from psoann.config import ANNConfig


class FeedForwardNetwork:
    """``forward(params, X)`` maps (n, n_in) inputs to (n, n_out) outputs. Stateless in the weights."""

    def __init__(
        self,
        layer_sizes: Sequence[int],
        activations: Sequence[Callable],
        activation_names: Sequence[str] | None = None,
    ) -> None:
        self.codec = ParameterCodec(layer_sizes)
        if len(activations) != len(self.codec.layer_sizes) - 1:
            raise ValueError(
                f"need one activation per non-input layer ({len(self.codec.layer_sizes) - 1}), got {len(activations)}"
            )
        self.activations: tuple[Activation, ...] = tuple(activations)
        self.activation_names: tuple[str, ...] = tuple(activation_names) if activation_names else ()

    @property
    def layer_sizes(self) -> tuple[int, ...]:
        return self.codec.layer_sizes

    @property
    def n_parameters(self) -> int:
        return self.codec.n_parameters

    def bounds(self, weight_bound: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Per-dimension (lower, upper) search bounds: every weight and bias in ``[-b, +b]``."""
        return np.full(self.n_parameters, -weight_bound), np.full(self.n_parameters, weight_bound)

    def describe_activations(self, params: NDArray[np.float64]) -> str:
        """Hidden-layer activation names for this parameter vector, e.g. ``"tanh-relu"`` (``"none"`` if no hidden layer)."""
        hidden = self.activation_names[:-1]
        return "-".join(hidden) if hidden else "none"

    def _split(self, params: NDArray[np.float64]) -> tuple[NDArray[np.float64], tuple[Activation, ...]]:
        """Return (weights-and-biases vector, per-layer activations) for this parameter vector."""
        return params, self.activations

    def forward(self, params: NDArray[np.float64], X: NDArray[np.float64]) -> NDArray[np.float64]:
        a = np.asarray(X, dtype=np.float64)
        if a.ndim != 2 or a.shape[1] != self.layer_sizes[0]:
            raise ValueError(f"expected X of shape (n, {self.layer_sizes[0]}), got {a.shape}")
        weights, activations = self._split(np.asarray(params, dtype=np.float64))
        for layer, activation in zip(self.codec.decode(weights), activations):
            a = activation(a @ layer.weights + layer.biases)
        return a


class EvolvedActivationNetwork(FeedForwardNetwork):
    """Hidden-layer activations are chosen by extra genes at the end of the parameter vector."""

    def __init__(
        self,
        layer_sizes: Sequence[int],
        output_activation: Callable,
        choices: Sequence[tuple[str, Callable]],
        output_name: str = "logistic",
    ) -> None:
        n_hidden = len(layer_sizes) - 2
        if n_hidden < 1:
            raise ValueError("evolving activations needs at least one hidden layer")
        if not choices:
            raise ValueError("need at least one activation choice")
        self.choice_names = tuple(name for name, _ in choices)
        self.choice_functions = tuple(fn for _, fn in choices)
        self.n_hidden = n_hidden
        placeholders = [self.choice_functions[0]] * n_hidden + [output_activation]
        super().__init__(layer_sizes, placeholders, [self.choice_names[0]] * n_hidden + [output_name])

    @property
    def n_weights(self) -> int:
        return self.codec.n_parameters

    @property
    def n_parameters(self) -> int:
        return self.codec.n_parameters + self.n_hidden

    def bounds(self, weight_bound: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        lower = np.concatenate([np.full(self.n_weights, -weight_bound), np.zeros(self.n_hidden)])
        upper = np.concatenate([np.full(self.n_weights, weight_bound), np.full(self.n_hidden, float(len(self.choice_names)))])
        return lower, upper

    def activation_indices(self, params: NDArray[np.float64]) -> NDArray[np.int64]:
        """Gene g -> activation index floor(g), clipped to the valid range (g = K maps to the last one)."""
        genes = np.asarray(params, dtype=np.float64)[self.n_weights :]
        return np.clip(np.floor(genes), 0, len(self.choice_names) - 1).astype(np.int64)

    def describe_activations(self, params: NDArray[np.float64]) -> str:
        return "-".join(self.choice_names[i] for i in self.activation_indices(params))

    def _split(self, params: NDArray[np.float64]) -> tuple[NDArray[np.float64], tuple[Activation, ...]]:
        if params.shape != (self.n_parameters,):
            raise ValueError(f"expected a vector of length {self.n_parameters}, got shape {params.shape}")
        hidden = tuple(self.choice_functions[i] for i in self.activation_indices(params))
        return params[: self.n_weights], hidden + (self.activations[-1],)


def build_network(config: ANNConfig, n_inputs: int) -> FeedForwardNetwork:
    """Layer sizes are ``(n_inputs, *hidden_layers, n_outputs)``; names are looked up in the registry."""
    sizes = (n_inputs, *config.hidden_layers, config.n_outputs)
    output = get_activation(config.output_activation)
    if config.evolve_activations:
        choices = [(name, get_activation(name)) for name in config.activation_choices]
        return EvolvedActivationNetwork(sizes, output, choices, output_name=config.output_activation)
    hidden = get_activation(config.hidden_activation)
    n_hidden = len(config.hidden_layers)
    names = [config.hidden_activation] * n_hidden + [config.output_activation]
    return FeedForwardNetwork(sizes, [hidden] * n_hidden + [output], names)
