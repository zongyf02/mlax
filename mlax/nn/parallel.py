import jax
from jax import random, tree_util as jtu
from dataclasses import dataclass
from typing import Any
from mlax import Parameter, Module, is_mlax_module
from mlax._utils import _needs_rng

@dataclass
class _Ret:
    activation: jax.Array
    layer: Any

class Parallel(Module):
    """Combination of layers that do not require rng in parallel."""
    def __init__(
        self,
        layers
    ):
        """Initialize a parallel layer.

        :param layers: PyTree of layers to combine in parallel.
        """
        super().__init__()
        self.layers = Parameter(
            trainable=True,
            data=layers
        )
        
    
    def __call__(self, x, rng=None, inference_mode=False):
        """Apply layers that do not require rng in parallel.
        
        :param self: Parallel layer.
        :param x: PyTree of input features, same structure as ``layers`` with
            MLAX Modules as leaves.
        :param rng: PRNG key. Ignored. Default: None.
        :param inference_mode: Whether in inference or training mode. Default:
            False.
        
        :returns: Output of ``layers`` applied in parallel on ``x``.
        :returns: Parallel layer with updated state. Possibly the same object as
            ``self``.
        """
        def mapped_fn(layer, input):
            acts, layer = layer(input, None, inference_mode)
            return _Ret(acts, layer)

        combined = jtu.tree_map(
            mapped_fn, self.layers.data, x, is_leaf=is_mlax_module
        )

        self.layers.data = jtu.tree_map(
            lambda ret: ret.layer,
            combined
        )
        return jtu.tree_map(
            lambda ret: ret.activation,
            combined
        ), self

class ParallelRng(Module):
    """Combination of layers that may require rng in parallel."""
    def __init__(
        self,
        layers
    ):
        """Initialize a parallel layer.

        :param layers: PyTree of layers to combine in parallel.
        """
        super().__init__()
        self.layers = Parameter(
            trainable=True,
            data=layers
        )
    
    def __call__(self, x, rng, inference_mode=False):
        """Apply layers that may not require rng in parallel.
        
        :param self: ParallelRng layer.
        :param x: PyTree of input features, same structure as ``layers`` with
            MLAX Modules as leaves.
        :param rng: PRNG key.
        :param inference_mode: Whether in inference or training mode. Default:
            False.
        
        :returns: Output of ``layers`` applied in parallel on ``x``.
        :returns: ParallelRng layer with updated state. Possibly the same object
            as ``self``.
        """
        needs_rngs = jtu.tree_map(
            _needs_rng,
            self.layers.data,
            is_leaf=is_mlax_module
        )
        num_rngs = jtu.tree_reduce(
            lambda accum, needs_rng: accum + needs_rng,
            needs_rngs
        )
        if num_rngs > 1:
            rng_iter = iter(random.split(rng, num_rngs))
        else:
            rng_iter = iter([rng])

        def mapped_fn(layer, needs_rng, x):
            if needs_rng:
                acts, layer = layer(x, next(rng_iter), inference_mode)
            else:
                acts, layer = layer(x, None, inference_mode)
            return _Ret(acts, layer)

        combined = jtu.tree_map(
            mapped_fn,
            self.layers.data,
            needs_rngs,
            x,
            is_leaf=is_mlax_module
        )

        self.layers.data = jtu.tree_map(
            lambda ret: ret.layer,
            combined
        )
        return jtu.tree_map(
            lambda ret: ret.activation,
            combined
        ), self
