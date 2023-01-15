from jax import random
from typing import Sequence
from mlax import Parameter, Module
from mlax._utils import _needs_rng

class Series(Module):
    """Combination of layers that do not require rng in series."""
    def __init__(
        self,
        layers: Sequence[Module]
    ):
        """Initialize a series layer.

        :param layers: Layers to combine in series.
        """
        super().__init__()
        self.layers = Parameter(
            trainable=True,
            data=layers
        )
        
    
    def __call__(self, x, rng=None, inference_mode=False):
        """Apply layers that do not require rng in series.
        
        :param self: Series layer.
        :param x: Input features.
        :param rng: PRNG key. Ignored. Default: None.
        :param inference_mode: Whether in inference or training mode. Default:
            False.
        
        :returns: Output of ``layers`` applied in series on ``x``.
        :returns: Series layer with updated state. Possibly the same object as
            ``self``.
        """
        new_layers = []
        for layer in self.layers.data:
            x, layer = layer(x, None, inference_mode)
            new_layers.append(layer)
        self.layers.data = new_layers
        return x, self

class SeriesRng(Module):
    """Combination of layers that may require rng in series."""
    def __init__(
        self,
        layers: Sequence[Module]
    ):
        """Initialize a series layer.

        :param layers: Layers to combine in series.
        """
        super().__init__()
        self.layers = Parameter(
            trainable=True,
            data=layers
        )
    
    def __call__(self, x, rng, inference_mode=False):
        """Apply layers that may not require rng in series.
        
        :param self: SeriesRng layer.
        :param x: Input features.
        :param rng: PRNG key.
        :param inference_mode: Whether in inference or training mode. Default:
            False.
        
        :returns: Output of ``layers`` applied in series on ``x``.
        :returns: SeriesRng layer with updated state. Possibly the same object
            as ``self``.
        """
        layers = self.layers.data
        needs_rngs = [_needs_rng(layer) for layer in layers]
        num_rngs = sum(needs_rngs)
        if num_rngs > 1:
            rng_iter = iter(random.split(rng, num_rngs))
        else:
            rng_iter = iter([rng])

        new_layers = []
        for needs_rng, layer in zip(needs_rngs, layers):
            if needs_rng:
                x, layer = layer(x, next(rng_iter), inference_mode)
            else:
                x, layer = layer(x, None, inference_mode)
            new_layers.append(layer)

        self.layers.data = layers
        return x, self
