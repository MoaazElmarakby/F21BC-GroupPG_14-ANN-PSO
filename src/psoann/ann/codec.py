"""Flat parameter vector <-> per-layer weights and biases.

Layout: for each layer in order, the weight matrix (n_in x n_out, row-major) followed by the bias
vector (n_out). Length = sum(n_in * n_out + n_out).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class LayerParams:
    weights: NDArray[np.float64]  # shape (n_in, n_out)
    biases: NDArray[np.float64]  # shape (n_out,)


class ParameterCodec:
    def __init__(self, layer_sizes: Sequence[int]) -> None:
        sizes = tuple(int(n) for n in layer_sizes)
        if len(sizes) < 2 or any(n < 1 for n in sizes):
            raise ValueError(f"layer_sizes needs >= 2 positive entries, got {sizes}")
        self.layer_sizes = sizes
        self._slices: list[tuple[slice, tuple[int, int], slice]] = []
        offset = 0
        for n_in, n_out in pairwise(sizes):
            w = slice(offset, offset + n_in * n_out)
            offset = w.stop
            b = slice(offset, offset + n_out)
            offset = b.stop
            self._slices.append((w, (n_in, n_out), b))
        self.n_parameters = offset

    def decode(self, vector: NDArray[np.float64]) -> list[LayerParams]:
        """Views into ``vector`` (no copy). Callers must not modify the result."""
        vector = np.asarray(vector, dtype=np.float64)
        if vector.shape != (self.n_parameters,):
            raise ValueError(f"expected a vector of length {self.n_parameters}, got shape {vector.shape}")
        return [LayerParams(vector[w].reshape(shape), vector[b]) for w, shape, b in self._slices]

    def encode(self, layers: Sequence[LayerParams]) -> NDArray[np.float64]:
        if len(layers) != len(self._slices):
            raise ValueError(f"expected {len(self._slices)} layers, got {len(layers)}")
        vector = np.empty(self.n_parameters, dtype=np.float64)
        for layer, (w, shape, b) in zip(layers, self._slices, strict=True):
            if layer.weights.shape != shape or layer.biases.shape != (shape[1],):
                raise ValueError("layer shapes do not match the codec")
            vector[w] = layer.weights.reshape(-1)
            vector[b] = layer.biases
        return vector
