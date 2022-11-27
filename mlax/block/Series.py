import jax
from jax import (
    random
)
import sys
from typing import Tuple, Any
from functools import reduce
from inspect import signature

def series_fwd(
    x: jax.Array,
    trainables: Tuple[Any],
    non_trainables: Tuple[Any],
    hyperparams: Tuple[Any],
    inference_mode: bool=False
):
    """Apply a series of sub-transformations that do not require PRNG keys.

    :param x: Input features.
    :param trainables: Tuple of trainable weights from each of the
        sub-transformations.
    :param non_trainables: Tuple of non-trainable weights from each of the 
        sub-transformations.
    :param hyperparams: Tuple of hyperparameters from each of the
        sub-transformations.
    :param inference_mode: Whether in inference or training mode. Default:
        False, training mode.

    :returns y: ``x`` with the sub-transformations applied in series.
    :returns non_trainables: Updated ``non_trainables`` from each of the
        sub-transformation.
    """
    def reduce_fn(accum, params):
        tr, ntr, hp = params
        x, new_ntrs = accum
        x, new_ntr = sys.modules[hp.__module__].fwd(
            x, tr, ntr, hp, inference_mode
        )
        new_ntrs.append(new_ntr)
        return x, new_ntrs

    x, non_trainables = reduce(
        reduce_fn, zip(trainables, non_trainables, hyperparams), (x, [])
    )
    return x, tuple(non_trainables)


def series_rng_fwd(
    x: jax.Array,
    trainables: Tuple[Any],
    non_trainables: Tuple[Any],
    key: Any,
    hyperparams: Tuple[Any],
    inference_mode: bool=False
):
    """Apply a series of sub-transformations that may consume a PRNG key.
    
    .. note::
        ``series_fwd`` is usually faster than ``series_rng_fwd`` because the
        former does not need to split keys. Therefore, use ``series_fwd`` over
        ``series_rng_fwd`` when possible.

    :param x: Input features.
    :param trainables: Tuple of trainable weights from each of the
        sub-transformations.
    :param non_trainables: Tuple of non-trainable weights from each of the 
        sub-transformations.
    :param key: PRNG key.
    :param hyperparams: Tuple of hyperparameters from each of the
        sub-transformations.
    :param inference_mode: Whether in inference or training mode. Default:
        False, training mode.


    :returns y: ``x`` with the sub-transformations applied.
    :returns non_trainables: Updated ``non_trainables`` from each of the
        sub-transformation.
    """
    def get_fwd_nkey(state, hp):
        fwds, has_keys, n_keys = state 
        fwd = sys.modules[hp.__module__].fwd
        has_key = signature(fwd).parameters.__contains__("key")
        fwds.append(fwd)
        has_keys.append(has_key)
        return fwds, has_keys, n_keys + has_key

    fwds, has_keys, n_keys = reduce(
        get_fwd_nkey, hyperparams, ([], [], 0)
    )

    if n_keys > 1:
        keys_iter = iter(random.split(key, n_keys))
    else:
        keys_iter = iter((key,))
    
    def reduce_fn(accum, params):
        fwd, needs_key, tr, ntr, hp = params
        x, new_ntrs = accum
        if needs_key:
            x, new_ntr = fwd(
                x, tr, ntr, next(keys_iter), hp, inference_mode
            )
        else:
            x, new_ntr = fwd(
                x, tr, ntr, hp, inference_mode
            )
        new_ntrs.append(new_ntr)
        return x, new_ntrs
    
    x, non_trainables = reduce(
        reduce_fn,
        zip(fwds, has_keys, trainables, non_trainables, hyperparams),
        (x, [])
    )
    return x, tuple(non_trainables)
