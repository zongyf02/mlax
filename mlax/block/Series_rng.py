import jax
from jax import (
    random
)
from typing import Tuple, Union, Any
from functools import reduce
from collections import namedtuple
from mlax._utils import _get_fwd, _needs_key

def init(
    *layers
) -> Tuple[Union[Tuple, jax.Array], Union[Tuple, jax.Array], Any]:
    """Initialize parameters and hyperparameters for a block that applies
    layers that may consume a PRNG key in series.

    :param layers: Initialized parameters and hyperparameters from each of the
        layers.

    :returns trainables: Named tuple of trainable weights from each of the
        layers or trainables weights from the layer if there's only one layer.
    :returns non_trainables: Named tuple of non-trainable weights from each of
        the layers or non-trainables weights from the layer if there's only one
        layer.
    :returns hyperparams: Named tuple of hyperparams from each of the layers or
        hyperparams of the layer if there is only one layer.
    """
    if len(layers) <= 1:
        return layers[0]
    else:
        SeriesRng = namedtuple(
            "SeriesRng",
            (f"layer{i}" for i in range(len(layers)))
        )
        trainables, non_trainables, hyperparams = zip(*layers)
        return (
            SeriesRng(*trainables),
            SeriesRng(*non_trainables),
            SeriesRng(*hyperparams)
        )

def fwd(
    x: jax.Array,
    trainables: Union[Tuple, jax.Array],
    non_trainables: Union[Tuple, jax.Array],
    key: Any,
    hyperparams: Any,
    inference_mode: bool=False
) -> Tuple[jax.Array, Tuple]:
    """Apply a series of layers that may consume a PRNG key.
    
    .. note::
        ``series_fwd`` is usually faster than ``series_rng_fwd`` because the
        former does not need to split keys. Therefore, use ``series_fwd`` over
        ``series_rng_fwd`` when possible.

    :param x: Input features.
    :param trainables: Named tuple of trainable weights from each of the
        layers or trainables weights from the layer if there's only one layer.
    :param non_trainables: Named tuple of non-trainable weights from each of
        the layers or non-trainables weights from the layer if there's only one
        layer.
    :param key: PRNG key.
    :param hyperparams: Named tuple of hyperparams from each of the layers or
        hyperparams of the layer if there is only one layer.
    :param inference_mode: Whether in inference or training mode. Default:
        False, training mode.

    :returns y: ``x`` with the layers applied.
    :returns non_trainables: Updated ``non_trainables``.
    """
    if isinstance(hyperparams, tuple):
        fwds = tuple(map(_get_fwd, hyperparams))
        needs_keys = tuple(map(_needs_key, fwds))
        n_keys = sum(needs_keys)
        if n_keys > 1:
            keys_iter = iter(random.split(key, n_keys))
        else:
            keys_iter = iter((key,))
        
        new_ntrs = []
        def reduce_fn(x, params):
            tr, ntr, hp, fwd, needs_key = params
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
            reduce_fn,
            zip(trainables, non_trainables, hyperparams, fwds, needs_keys),
            x
        )
        return x, non_trainables.__class__(*new_ntrs)
    else:
        fwd = _get_fwd(hyperparams)
        needs_key = _needs_key(fwd)
        if needs_key:
            return fwd(
                x, trainables, non_trainables, key, hyperparams, inference_mode
            )
        else:
            return fwd(
                x, trainables, non_trainables, hyperparams, inference_mode
            )