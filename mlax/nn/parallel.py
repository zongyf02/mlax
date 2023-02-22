from jax import random
from typing import Sequence, Iterable
from mlax import Module, ModuleSeq
from mlax._utils import _needs_rng

class Parallel(Module):
    """Combination of layers that do not require rng in parallel."""
    def __init__(self, layers: Iterable[Module]):
        """Initialize a Parallel layer.

        :param layers: Layers to combine in parallel.
        """
        super().__init__()
        self.layers = ModuleSeq(submodules=layers)        
    
    def __call__(self, x: Sequence, rng=None, inference_mode=False):
        """Apply layers that do not require rng in parallel.

        :param x: Sequence of input features, one for each layer.
        :param rng: PRNG key. Ignored. Default: None.
        :param inference_mode: Whether in inference or training mode. Default:
            False.
        
        :returns: List of outputs from each layer.
        :returns: Parallel layer with updated state. Possibly the same object as
            ``self``.
        """
        res = []
        for i, (_x, layer) in enumerate(zip(x, self.layers)):
            _x, self.layers[i] = layer(_x, None, inference_mode)
            res.append(_x)
        return res, self

class ParallelRng(Module):
    """Combination of layers that may require rng in parallel."""
    def __init__(self, layers: Iterable[Module]):
        """Initialize a ParallelRng layer.

        :param layers: PyTree of layers to combine in parallel.
        """
        super().__init__()
        self.layers = ModuleSeq(submodules=layers) 
    
    def __call__(self, x: Sequence, rng, inference_mode=False):
        """Apply layers that may not require rng in parallel.

        :param x: Sequence of input features, one for each layer.
        :param rng: PRNG key.
        :param inference_mode: Whether in inference or training mode. Default:
            False.
        
        :returns: List of outputs from each layer.
        :returns: ParallelRng layer with updated state. Possibly the same object
            as ``self``.
        """
        needs_rngs = [_needs_rng(layer) for layer in self.layers]
        num_rngs = sum(needs_rngs)
        if num_rngs > 1:
            rng_iter = iter(random.split(rng, num_rngs))
        else:
            rng_iter = iter([rng])

        res = []
        for i, (_x, needs_rng, layer) in enumerate(
            zip(x, needs_rngs, self.layers)
        ):
            if needs_rng:
                _x, self.layers[i] = layer(_x, next(rng_iter), inference_mode)
            else:
                _x, self.layers[i] = layer(_x, None, inference_mode)
            res.append(_x)
        return res, self
