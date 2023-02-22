from jax import random
from typing import Iterable
from mlax import Module, ModuleSeq
from mlax._utils import _needs_rng

class Series(Module):
    """Combination of layers that do not require rng in series."""
    def __init__(self, layers: Iterable[Module]):
        """Initialize a Series layer.

        :param layers: Layers to combine in series.
        """
        super().__init__()
        self.layers = ModuleSeq(submodules=layers)
        
    
    def __call__(self, x, rng=None, inference_mode=False):
        """Apply layers that do not require rng in series.

        :param x: Input features.
        :param rng: PRNG key. Ignored. Default: None.
        :param inference_mode: Whether in inference or training mode. Default:
            False.
        
        :returns: Output of ``layers`` applied in series on ``x``.
        :returns: Series layer with updated state. Possibly the same object as
            ``self``.
        """
        for i, layer in enumerate(self.layers):
            x, self.layers[i] = layer(x, None, inference_mode)
        return x, self

class SeriesRng(Module):
    """Combination of layers that may require rng in series."""
    def __init__(self, layers: Iterable[Module]):
        """Initialize a SeriesRNG layer.

        :param layers: Layers to combine in series.
        """
        super().__init__()
        self.layers = ModuleSeq(submodules=layers)
    
    def __call__(self, x, rng, inference_mode=False):
        """Apply layers that may not require rng in series.

        :param x: Input features.
        :param rng: PRNG key.
        :param inference_mode: Whether in inference or training mode. Default:
            False.
        
        :returns: Output of ``layers`` applied in series on ``x``.
        :returns: SeriesRng layer with updated state. Possibly the same object
            as ``self``.
        """
        needs_rngs = [_needs_rng(layer) for layer in self.layers]
        num_rngs = sum(needs_rngs)
        if num_rngs > 1:
            rng_iter = iter(random.split(rng, num_rngs))
        else:
            rng_iter = iter([rng])

        for i, (needs_rng, layer) in enumerate(zip(needs_rngs, self.layers)):
            if needs_rng:
                x, self.layers[i] = layer(x, next(rng_iter), inference_mode)
            else:
                x, self.layers[i] = layer(x, None, inference_mode)
        return x, self
