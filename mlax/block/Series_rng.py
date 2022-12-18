import jax
from jax import (
    random
)
from typing import Tuple, Any, NamedTuple
from functools import reduce
from mlax._utils import _get_fwd, _needs_key

class Hyperparams(NamedTuple):
    layers: Tuple

def init(
    *layers
) -> Tuple[Tuple, Tuple, Hyperparams]:
    """Initialize parameters and hyperparameters for a layer that applies
    sub-layers that may consume a PRNG key in series.

    :param layers: Initialized parameters and hyperparameters from each of the
        sub-layers.

    :returns trainables: Tuple of trainable weights from each of the
        sub-layers.
    :returns non_trainables: Tuple of non-trainable weights from each of the 
        sub-layers.
    :returns hyperparams: NamedTuple containing the hyperparameters.
    """
    trainables, non_trainables, hyperparams = zip(*layers)
    return trainables, non_trainables, Hyperparams(hyperparams)

def fwd(
    x: jax.Array,
    trainables: Tuple[Any],
    non_trainables: Tuple[Any],
    key: Any,
    hyperparams: Hyperparams,
    inference_mode: bool=False
) -> Tuple[jax.Array, Tuple]:
    """Apply a series of layers that may consume a PRNG key.
    
    .. note::
        ``series_fwd`` is usually faster than ``series_rng_fwd`` because the
        former does not need to split keys. Therefore, use ``series_fwd`` over
        ``series_rng_fwd`` when possible.

    :param x: Input features.
    :param trainables: Tuple of trainable weights from each of the layers.
    :param non_trainables: Tuple of non-trainable weights from each of the 
        layers.
    :param key: PRNG key.
    :param hyperparams: NamedTuple containing the hyperparameters.
    :param inference_mode: Whether in inference or training mode. Default:
        False, training mode.

    :returns y: ``x`` with the layers applied.
    :returns non_trainables: Updated ``non_trainables`` from each of the
        layers.
    """
    fwds = []
    needs_keys = []
    def reduce_fn1(accum, hp):
        fwd = _get_fwd(hp)
        needs_key = _needs_key(fwd)
        fwds.append(fwd)
        needs_keys.append(needs_key)
        return accum + needs_key

    n_keys = reduce(reduce_fn1, hyperparams.layers, 0)
    if n_keys > 1:
        keys_iter = iter(random.split(key, n_keys))
    else:
        keys_iter = iter((key,))
    
    new_ntrs = []
    def reduce_fn2(x, params):
        fwd, needs_key, tr, ntr, hp = params
        if needs_key:
            x, new_ntr = fwd(
                x, tr, ntr, next(keys_iter), hp, inference_mode
            )
        else:
            x, new_ntr = fwd(
                x, tr, ntr, hp, inference_mode
            )
        new_ntrs.append(new_ntr)
        return x
    
    x = reduce(
        reduce_fn2,
        zip(fwds, needs_keys, trainables, non_trainables, hyperparams.layers),
        x
    )
    return x, Hyperparams(new_ntrs)
